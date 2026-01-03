from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .dataset_index import ImageRecord, KIND_TO_CLASS_ID, read_jsonl
from .image_pipeline import ImagePipelineConfig
from .torch_data import TorchDataConfig, infer_pipeline_and_shape, make_dataloader
from .torch_models import TorchCNNArch, TorchCNNClassifier, infer_num_classes, num_trainable_params
from .torch_training import TorchTrainLoopConfig, eval_loss_acc, seed_everything, train_loop


@dataclass(frozen=True)
class CNNTrainConfig:
    # train loop
    epochs: int = 10
    batch_size: int = 64
    sample_strategy: str = "shuffle"  # shuffle|balanced_class|balanced_character|balanced_character_class
    lr: float = 0.03
    reg: float = 0.0  # legacy alias (maps to weight_decay when weight_decay==0)
    optimizer: str = "sgd"  # sgd|adamw
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_schedule: str = "constant"  # constant|step
    lr_decay_factor: float = 0.1
    lr_decay_epochs: int = 10
    label_smoothing: float = 0.0
    save_best_val: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "loss"  # loss|accuracy
    restore_best_val: bool = True
    seed: int = 42
    device: str = "cpu"  # cpu|gpu (gpu=CUDA)
    amp: bool = False
    max_grad_norm: float | None = None

    # input pipeline
    channels: str = "grayscale"  # grayscale|rgb
    dtype: str = "float32"  # kept for CLI compatibility (torch uses float32)
    shape_mode: str = "resize"  # strict|pad_or_crop|resize
    image_size: tuple[int, int] | None = (128, 128)
    cache_dir: str | None = None
    augment_hflip: bool = False
    augment_random_resized_crop: bool = False
    augment_rrc_prob: float = 1.0
    augment_rrc_scale: tuple[float, float] = (0.8, 1.0)
    augment_brightness: float = 0.0
    augment_contrast: float = 0.0
    augment_cutout: bool = False
    augment_cutout_prob: float = 0.5
    augment_cutout_scale: tuple[float, float] = (0.02, 0.2)

    # model arch
    conv_out_channels: int = 32
    conv2_out_channels: int = 64  # 0 = disable
    kernel_size: int = 3
    conv_stride: int = 1
    pool_size: int = 2
    hidden_size: int = 256
    dropout: float = 0.0
    batch_norm: bool = True

    # dataloader perf
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    # debug limits
    max_train: int | None = None
    max_val: int | None = None
    max_test: int | None = None
    verbose: bool = True


def _limit_sample(records: list[ImageRecord], max_n: int | None, *, rng: random.Random) -> list[ImageRecord]:
    if max_n is None:
        return list(records)
    max_n = max(0, int(max_n))
    if max_n >= len(records):
        return list(records)
    return rng.sample(list(records), k=max_n)


def _save_model(path: Path, model: torch.nn.Module, *, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_cnn(
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    test_records: Sequence[ImageRecord],
    *,
    config: CNNTrainConfig,
) -> tuple[torch.nn.Module, dict]:
    if config.channels not in {"grayscale", "rgb"}:
        raise ValueError(f"Unsupported channels: {config.channels}")
    if config.shape_mode not in {"strict", "pad_or_crop", "resize"}:
        raise ValueError(f"Unsupported shape_mode: {config.shape_mode}")
    if config.shape_mode in {"pad_or_crop", "resize"} and config.image_size is None:
        raise ValueError("image_size must be provided when shape_mode is pad_or_crop/resize")

    # unknown kind filtered out (safety)
    train_known = [r for r in train_records if r.class_id is not None]
    val_known = [r for r in val_records if r.class_id is not None]
    test_known = [r for r in test_records if r.class_id is not None]
    if not train_known:
        raise ValueError("No train samples with known class_id.")

    seed_everything(int(config.seed))
    rng = random.Random(int(config.seed))

    train_pipeline, expected_shape = infer_pipeline_and_shape(
        train_known,
        channels=str(config.channels),
        shape_mode=str(config.shape_mode),
        image_size=tuple(config.image_size) if config.image_size is not None else None,
        cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        augment_hflip=bool(config.augment_hflip),
        augment_random_resized_crop=bool(config.augment_random_resized_crop),
        augment_rrc_prob=float(config.augment_rrc_prob),
        augment_rrc_scale=tuple(float(x) for x in config.augment_rrc_scale),
        augment_brightness=float(config.augment_brightness),
        augment_contrast=float(config.augment_contrast),
        augment_cutout=bool(config.augment_cutout),
        augment_cutout_prob=float(config.augment_cutout_prob),
        augment_cutout_scale=tuple(float(x) for x in config.augment_cutout_scale),
    )

    eval_pipeline = ImagePipelineConfig(
        channels=str(config.channels),
        shape_mode=str(config.shape_mode),
        image_size=tuple(config.image_size) if config.image_size is not None else None,
        cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        augment_hflip=False,
        augment_random_resized_crop=False,
        augment_brightness=0.0,
        augment_contrast=0.0,
        augment_cutout=False,
    )

    data_train = TorchDataConfig(
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers),
        prefetch_factor=config.prefetch_factor,
        sample_strategy=str(config.sample_strategy),
    )
    data_eval = TorchDataConfig(
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory),
        persistent_workers=bool(config.persistent_workers),
        prefetch_factor=config.prefetch_factor,
        sample_strategy="shuffle",
    )

    train_loader = make_dataloader(
        train_known,
        pipeline_config=train_pipeline,
        expected_shape=expected_shape,
        is_train=True,
        data=data_train,
    )
    val_loader = (
        make_dataloader(
            val_known,
            pipeline_config=eval_pipeline,
            expected_shape=expected_shape,
            is_train=False,
            data=data_eval,
        )
        if val_known
        else None
    )
    test_loader = (
        make_dataloader(
            test_known,
            pipeline_config=eval_pipeline,
            expected_shape=expected_shape,
            is_train=False,
            data=data_eval,
        )
        if test_known
        else None
    )

    arch = TorchCNNArch(
        conv_out_channels=int(config.conv_out_channels),
        conv2_out_channels=int(config.conv2_out_channels),
        kernel_size=int(config.kernel_size),
        conv_stride=int(config.conv_stride),
        pool_size=int(config.pool_size),
        hidden_size=int(config.hidden_size),
        dropout=float(config.dropout),
        batch_norm=bool(config.batch_norm),
    )

    model = TorchCNNClassifier(
        input_shape=expected_shape,
        channels=str(config.channels),
        num_classes=infer_num_classes(KIND_TO_CLASS_ID),
        arch=arch,
    )

    loop_cfg = TorchTrainLoopConfig(
        epochs=int(config.epochs),
        lr=float(config.lr),
        optimizer=str(config.optimizer),
        momentum=float(config.momentum),
        weight_decay=float(config.weight_decay) if float(config.weight_decay) != 0.0 else float(config.reg),
        lr_schedule=str(config.lr_schedule),
        lr_decay_factor=float(config.lr_decay_factor),
        lr_decay_epochs=int(config.lr_decay_epochs),
        label_smoothing=float(config.label_smoothing),
        save_best_val=bool(config.save_best_val),
        early_stopping_patience=int(config.early_stopping_patience),
        early_stopping_min_delta=float(config.early_stopping_min_delta),
        early_stopping_metric=str(config.early_stopping_metric),
        restore_best_val=bool(config.restore_best_val),
        seed=int(config.seed),
        device=str(config.device),
        amp=bool(config.amp),
        max_grad_norm=config.max_grad_norm,
        verbose=bool(config.verbose),
    )

    train_loop_metrics = train_loop(model, train_loader=train_loader, val_loader=val_loader, config=loop_cfg)

    # Final eval (no label smoothing in reported loss)
    device = torch.device("cpu") if str(config.device) == "cpu" else torch.device("cuda")
    criterion = torch.nn.CrossEntropyLoss()
    train_metrics = eval_loss_acc(model, train_loader, device=device, criterion=criterion)
    val_metrics = eval_loss_acc(model, val_loader, device=device, criterion=criterion) if val_loader is not None else {"n": 0, "loss": None, "accuracy": None}
    test_metrics = eval_loss_acc(model, test_loader, device=device, criterion=criterion) if test_loader is not None else {"n": 0, "loss": None, "accuracy": None}

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        **train_loop_metrics,
    }
    return model, metrics


def train_cnn_from_splits(
    *,
    splits_dir: Path,
    out_model_path: Path,
    config: CNNTrainConfig,
) -> dict:
    rng = random.Random(int(config.seed))
    train_path = Path(splits_dir) / "train.jsonl"
    val_path = Path(splits_dir) / "val.jsonl"
    test_path = Path(splits_dir) / "test.jsonl"

    train_records = _limit_sample(list(read_jsonl(train_path)), config.max_train, rng=rng)
    val_records = _limit_sample(list(read_jsonl(val_path)), config.max_val, rng=rng) if val_path.exists() else []
    test_records = _limit_sample(list(read_jsonl(test_path)), config.max_test, rng=rng) if test_path.exists() else []

    model, metrics = train_cnn(train_records, val_records, test_records, config=config)

    meta = {
        "framework": "pytorch",
        "model_type": "cnn",
        "input_shape": list(metrics.get("input_shape") or []),
        "channels": str(config.channels),
        "class_map": dict(KIND_TO_CLASS_ID),
        "arch": asdict(
            TorchCNNArch(
                conv_out_channels=int(config.conv_out_channels),
                conv2_out_channels=int(config.conv2_out_channels),
                kernel_size=int(config.kernel_size),
                conv_stride=int(config.conv_stride),
                pool_size=int(config.pool_size),
                hidden_size=int(config.hidden_size),
                dropout=float(config.dropout),
                batch_norm=bool(config.batch_norm),
            )
        ),
        "train_config": asdict(config),
        "metrics": metrics,
        "num_trainable_params": num_trainable_params(model),
        "torch": {"version": torch.__version__},
    }
    # Persist expected input shape for loader (H,W[,C]) so eval can reproduce.
    # We store it under meta["input_shape"].
    train_known = [r for r in train_records if r.class_id is not None]
    pipeline, expected_shape = infer_pipeline_and_shape(
        train_known,
        channels=str(config.channels),
        shape_mode=str(config.shape_mode),
        image_size=tuple(config.image_size) if config.image_size is not None else None,
        cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        augment_hflip=False,
        augment_random_resized_crop=False,
        augment_rrc_prob=1.0,
        augment_rrc_scale=(0.8, 1.0),
        augment_brightness=0.0,
        augment_contrast=0.0,
        augment_cutout=False,
        augment_cutout_prob=0.5,
        augment_cutout_scale=(0.02, 0.2),
    )
    _ = pipeline
    meta["input_shape"] = list(expected_shape)

    _save_model(Path(out_model_path), model, meta=meta)
    return metrics
