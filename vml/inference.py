from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn

from .dataset_index import ImageRecord, KIND_TO_CLASS_ID, read_jsonl
from .image_pipeline import ImagePipelineConfig, load_image_uint8_transformed
from .torch_data import TorchDataConfig, make_dataloader
from .torch_models import (
    TorchCNNArch,
    TorchCNNClassifier,
    TorchMLPArch,
    TorchMLPClassifier,
    TorchLinearClassifier,
    class_names_from_map,
    id_to_name_from_map,
    infer_num_classes,
)
from .torch_training import resolve_device


@dataclass(frozen=True)
class LoadedModel:
    model_type: str  # "softmax_regression" | "mlp" | "cnn"
    input_shape: tuple[int, ...]
    channels: str
    class_map: dict[str, int]
    train_config: dict
    device: str  # "cpu" | "gpu"
    net: nn.Module


def _read_meta(model_path: Path) -> dict:
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing model meta json: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _to_tuple_int(seq: Sequence) -> tuple[int, ...]:
    return tuple(int(x) for x in seq)


def load_model(model_path: Path, *, device: str = "cpu") -> LoadedModel:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    meta = _read_meta(model_path)
    model_type = str(meta.get("model_type"))
    input_shape = _to_tuple_int(meta.get("input_shape", []))
    channels = str(meta.get("channels"))
    class_map = meta.get("class_map") or dict(KIND_TO_CLASS_ID)
    train_config = meta.get("train_config") or {}

    num_classes = infer_num_classes({str(k): int(v) for k, v in class_map.items()})

    if model_type == "cnn":
        arch_meta = meta.get("arch") or {}
        if not isinstance(arch_meta, dict):
            raise ValueError("Invalid cnn meta: missing 'arch' dict")
        arch = TorchCNNArch(
            conv_out_channels=int(arch_meta.get("conv_out_channels", 32)),
            conv2_out_channels=int(arch_meta.get("conv2_out_channels", 64)),
            kernel_size=int(arch_meta.get("kernel_size", 3)),
            conv_stride=int(arch_meta.get("conv_stride", 1)),
            pool_size=int(arch_meta.get("pool_size", 2)),
            hidden_size=int(arch_meta.get("hidden_size", 256)),
            dropout=float(arch_meta.get("dropout", 0.0)),
            batch_norm=bool(arch_meta.get("batch_norm", True)),
        )
        net: nn.Module = TorchCNNClassifier(
            input_shape=input_shape,
            channels=channels,
            num_classes=num_classes,
            arch=arch,
        )
    elif model_type == "mlp":
        arch_meta = meta.get("arch") or {}
        if not isinstance(arch_meta, dict):
            raise ValueError("Invalid mlp meta: missing 'arch' dict")
        arch = TorchMLPArch(
            hidden_sizes=tuple(int(x) for x in (arch_meta.get("hidden_sizes") or [256])),
            dropout=float(arch_meta.get("dropout", 0.0)),
        )
        net = TorchMLPClassifier(
            input_shape=input_shape,
            channels=channels,
            num_classes=num_classes,
            arch=arch,
        )
    elif model_type == "softmax_regression":
        net = TorchLinearClassifier(
            input_shape=input_shape,
            channels=channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Invalid model file: expected a state_dict dict")
    net.load_state_dict(state)

    device = str(device)
    torch_device = resolve_device(device)
    net.to(torch_device)
    net.eval()

    return LoadedModel(
        model_type=model_type,
        input_shape=input_shape,
        channels=channels,
        class_map={str(k): int(v) for k, v in class_map.items()},
        train_config=dict(train_config),
        device=device,
        net=net,
    )


def choose_shape_mode(model: LoadedModel, override: str) -> str:
    if override != "auto":
        return override
    shape_mode = model.train_config.get("shape_mode")
    if shape_mode in {"strict", "pad_or_crop", "resize"}:
        return str(shape_mode)
    return "strict"


def make_eval_pipeline_config(
    model: LoadedModel,
    *,
    shape_mode: str,
) -> ImagePipelineConfig:
    if shape_mode not in {"strict", "pad_or_crop", "resize"}:
        raise ValueError(f"Unsupported shape_mode: {shape_mode}")

    image_size = None
    if shape_mode in {"pad_or_crop", "resize"}:
        if len(model.input_shape) < 2:
            raise ValueError(f"Invalid input_shape: {model.input_shape}")
        image_size = (int(model.input_shape[0]), int(model.input_shape[1]))

    return ImagePipelineConfig(
        channels=model.channels,
        shape_mode=shape_mode,
        image_size=image_size,
        cache_dir=model.train_config.get("cache_dir"),
        augment_hflip=False,
        augment_random_resized_crop=False,
        augment_brightness=0.0,
        augment_contrast=0.0,
        augment_cutout=False,
    )


def iter_records_from_jsonl(path: Path, *, limit: int | None = None) -> Iterator[ImageRecord]:
    n = 0
    for rec in read_jsonl(path):
        if rec.class_id is None:
            continue
        yield rec
        n += 1
        if limit is not None and n >= int(limit):
            break


def _iter_batches(records: Iterable[ImageRecord], *, batch_size: int) -> Iterator[list[ImageRecord]]:
    batch: list[ImageRecord] = []
    for rec in records:
        batch.append(rec)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def summarize_confusion_matrix(cm: np.ndarray) -> dict:
    num_classes = int(cm.shape[0])
    per_class = []
    for i in range(num_classes):
        tp = int(cm[i, i])
        support = int(np.sum(cm[i, :]))
        pred_count = int(np.sum(cm[:, i]))
        recall = tp / support if support else None
        precision = tp / pred_count if pred_count else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class.append(
            {
                "class_id": i,
                "support": support,
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {"per_class": per_class}


def format_confusion_matrix(cm: np.ndarray, class_names: Sequence[str]) -> str:
    names = list(class_names)
    n = len(names)
    if cm.shape != (n, n):
        raise ValueError("cm shape must match class_names length")

    name_width = max(len("true\\pred"), max((len(x) for x in names), default=0))
    cell_width = max(5, max(len(str(int(v))) for v in cm.flatten()))

    header = " " * (name_width + 2) + " ".join(name.ljust(cell_width) for name in names)
    rows = [header]
    for i, name in enumerate(names):
        row_vals = " ".join(str(int(v)).ljust(cell_width) for v in cm[i])
        rows.append(name.ljust(name_width) + "  " + row_vals)
    return "\n".join(rows)


def evaluate(
    model: LoadedModel,
    *,
    records: Iterable[ImageRecord],
    batch_size: int = 64,
    shape_mode: str = "auto",
    return_predictions: bool = False,
    out_predictions: Path | None = None,
) -> dict:
    actual_shape_mode = choose_shape_mode(model, shape_mode)
    pipeline_config = make_eval_pipeline_config(model, shape_mode=actual_shape_mode)

    recs = [r for r in records if r.class_id is not None]
    if not recs:
        return {
            "n": 0,
            "loss": None,
            "accuracy": None,
            "confusion_matrix": np.zeros((len(model.class_map), len(model.class_map)), dtype=np.int64).tolist(),
            "class_names": class_names_from_map(model.class_map),
            "shape_mode": actual_shape_mode,
            **summarize_confusion_matrix(np.zeros((len(model.class_map), len(model.class_map)), dtype=np.int64)),
        }

    device = resolve_device(model.device)
    data_eval = TorchDataConfig(
        batch_size=int(batch_size),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        prefetch_factor=None,
        sample_strategy="shuffle",
    )
    loader = make_dataloader(
        recs,
        pipeline_config=pipeline_config,
        expected_shape=model.input_shape,
        is_train=False,
        data=data_eval,
    )

    num_classes = infer_num_classes(model.class_map)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_n = 0

    class_names = class_names_from_map(model.class_map)
    id_to_name = id_to_name_from_map(model.class_map)

    predictions: list[dict] | None = [] if return_predictions else None
    pred_f = None
    if out_predictions is not None:
        out_predictions = Path(out_predictions)
        out_predictions.parent.mkdir(parents=True, exist_ok=True)
        pred_f = out_predictions.open("w", encoding="utf-8")

    criterion = nn.CrossEntropyLoss()

    try:
        model.net.eval()
        with torch.no_grad():
            for X, y, metas in loader:
                X = X.to(device=device, non_blocking=True)
                y = y.to(device=device, non_blocking=True)
                logits = model.net(X)
                loss = criterion(logits, y)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)

                y_cpu = y.detach().cpu().numpy().astype(np.int64, copy=False)
                p_cpu = pred.detach().cpu().numpy().astype(np.int64, copy=False)
                idx = y_cpu * num_classes + p_cpu
                cm += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

                n = int(y.shape[0])
                total_loss += float(loss.item()) * n
                total_n += n

                if predictions is not None or pred_f is not None:
                    probs_cpu = probs.detach().cpu().numpy()
                    for meta, true_id, pred_id, prob_vec in zip(metas, y_cpu.tolist(), p_cpu.tolist(), probs_cpu.tolist()):
                        true_id_i = int(true_id)
                        pred_id_i = int(pred_id)
                        true_name = id_to_name.get(true_id_i, str(true_id_i))
                        pred_name = id_to_name.get(pred_id_i, str(pred_id_i))
                        p_true = float(prob_vec[true_id_i]) if 0 <= true_id_i < len(prob_vec) else None
                        p_pred = float(prob_vec[pred_id_i]) if 0 <= pred_id_i < len(prob_vec) else None
                        margin = None if p_true is None or p_pred is None else float(p_pred - p_true)

                        top2 = sorted(
                            [
                                {
                                    "class_id": int(i),
                                    "class_name": id_to_name.get(int(i), str(int(i))),
                                    "probability": float(p),
                                }
                                for i, p in enumerate(prob_vec)
                            ],
                            key=lambda d: d["probability"],
                            reverse=True,
                        )[:2]

                        item = {
                            "path": meta.get("path"),
                            "relpath": meta.get("relpath"),
                            "character": meta.get("character"),
                            "kind": meta.get("kind"),
                            "true_class_id": true_id_i,
                            "true_class_name": true_name,
                            "predicted_class_id": pred_id_i,
                            "predicted_class_name": pred_name,
                            "correct": bool(true_id_i == pred_id_i),
                            "true_probability": p_true,
                            "predicted_probability": p_pred,
                            "margin": margin,
                            "probabilities": {
                                id_to_name.get(int(i), str(int(i))): float(prob_vec[i]) for i in range(len(prob_vec))
                            },
                            "top2": top2,
                        }
                        if predictions is not None:
                            predictions.append(item)
                        if pred_f is not None:
                            pred_f.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        if pred_f is not None:
            pred_f.close()

    acc = float(np.trace(cm) / total_n) if total_n else None
    loss = float(total_loss / total_n) if total_n else None

    result = {
        "n": total_n,
        "loss": loss,
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "shape_mode": actual_shape_mode,
        **summarize_confusion_matrix(cm),
    }
    if predictions is not None:
        result["predictions"] = predictions
    if out_predictions is not None:
        result["predictions_jsonl"] = str(out_predictions)
    return result


def predict_image(
    model: LoadedModel,
    *,
    image_path: Path,
    shape_mode: str = "auto",
) -> dict:
    actual_shape_mode = choose_shape_mode(model, shape_mode)
    pipeline_config = make_eval_pipeline_config(model, shape_mode=actual_shape_mode)

    arr = load_image_uint8_transformed(
        str(image_path),
        key=str(image_path),
        expected_shape=model.input_shape,
        config=pipeline_config,
        rng=random,
        is_train=False,
    )
    arr = np.asarray(arr, dtype=np.uint8)
    if (not arr.flags.writeable) or any(int(s) < 0 for s in arr.strides):
        arr = arr.copy()
    if arr.ndim == 2:
        x = torch.from_numpy(arr)[None, :, :]
    else:
        x = torch.from_numpy(arr).permute(2, 0, 1)
    x = x.to(dtype=torch.float32).div_(255.0)[None, ...]

    device = resolve_device(model.device)
    x = x.to(device=device)
    model.net.eval()
    with torch.no_grad():
        logits = model.net(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().tolist()

    inv = id_to_name_from_map(model.class_map)
    pred_id = int(np.argmax(np.asarray(probs)))
    pred_name = inv.get(pred_id, str(pred_id))

    return {
        "model_type": model.model_type,
        "image": str(image_path),
        "predicted_class_id": pred_id,
        "predicted_class_name": pred_name,
        "probabilities": {inv.get(i, str(i)): float(probs[i]) for i in range(len(probs))},
        "shape_mode": actual_shape_mode,
    }
