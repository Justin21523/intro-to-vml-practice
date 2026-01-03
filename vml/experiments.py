from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset_index import ImageRecord
from .error_analysis import ErrorAnalysisConfig, analyze_errors
from .inference import evaluate, iter_records_from_jsonl, load_model
from .train_cnn import CNNTrainConfig, train_cnn_from_splits


@dataclass(frozen=True)
class ExperimentIOConfig:
    eval_batch_size: int = 64
    eval_limit: int | None = None
    analyze_errors: bool = True
    errors_limit: int | None = None
    errors_thumb_size: int = 224
    errors_max_per_pair: int = 36
    seed: int = 0


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def cnn_presets(*, device: str = "cpu") -> dict[str, CNNTrainConfig]:
    """
    這些 preset 的目標是：
    - baseline：容易過擬合（用來當對照）
    - regularized：加入強一點的正則化/增強/早停，目標是縮小 train↔val/test gap
    """
    device = str(device)
    if device not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported device: {device}")

    common = dict(
        seed=42,
        device=device,
        channels="grayscale",
        dtype="float32",
        shape_mode="resize",
        image_size=(128, 128),
        cache_dir="data/cache/resize128_grayscale",
        conv_out_channels=32,
        conv2_out_channels=64,
        kernel_size=3,
        conv_stride=1,
        pool_size=2,
        hidden_size=256,
        batch_norm=True,
        optimizer="sgd",
        amp=False,
        max_grad_norm=None,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
    )

    baseline = CNNTrainConfig(
        **common,
        epochs=10,
        batch_size=64,
        lr=0.03,
        reg=0.0,
        momentum=0.0,
        weight_decay=0.0,
        lr_schedule="constant",
        save_best_val=False,
        early_stopping_patience=0,
        label_smoothing=0.0,
        dropout=0.0,
        sample_strategy="shuffle",
        augment_hflip=False,
        augment_random_resized_crop=False,
        augment_brightness=0.0,
        augment_contrast=0.0,
        augment_cutout=False,
        verbose=True,
    )

    regularized = CNNTrainConfig(
        **common,
        epochs=40,
        batch_size=64,
        lr=0.03,
        reg=0.0,
        momentum=0.9,
        weight_decay=5e-4,
        lr_schedule="step",
        lr_decay_epochs=10,
        lr_decay_factor=0.5,
        save_best_val=True,
        early_stopping_patience=5,
        early_stopping_min_delta=0.0,
        early_stopping_metric="loss",
        restore_best_val=True,
        dropout=0.3,
        label_smoothing=0.05,
        sample_strategy="balanced_character_class",
        augment_hflip=True,
        augment_random_resized_crop=True,
        augment_rrc_prob=1.0,
        augment_rrc_scale=(0.8, 1.0),
        augment_brightness=0.1,
        augment_contrast=0.1,
        augment_cutout=True,
        augment_cutout_prob=0.5,
        augment_cutout_scale=(0.02, 0.15),
        verbose=True,
    )

    return {
        "baseline": baseline,
        "regularized": regularized,
    }


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary_from_eval(eval_train: dict, eval_val: dict, eval_test: dict) -> dict:
    train_acc = eval_train.get("accuracy")
    val_acc = eval_val.get("accuracy")
    test_acc = eval_test.get("accuracy")

    def _gap(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    return {
        "accuracy_train": train_acc,
        "accuracy_val": val_acc,
        "accuracy_test": test_acc,
        "gap_train_minus_val": _gap(train_acc, val_acc),
        "gap_train_minus_test": _gap(train_acc, test_acc),
    }


def run_cnn_experiment(
    *,
    splits_dir: Path,
    out_dir: Path,
    train_config: CNNTrainConfig,
    io: ExperimentIOConfig,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    train_metrics = train_cnn_from_splits(
        splits_dir=Path(splits_dir),
        out_model_path=model_path,
        config=train_config,
    )
    _write_json(out_dir / "train_metrics.json", train_metrics)
    _write_json(out_dir / "train_config.json", asdict(train_config))

    model = load_model(model_path, device=str(train_config.device))

    eval_results: dict[str, dict] = {}
    for split_name in ["train", "val", "test"]:
        split_path = Path(splits_dir) / f"{split_name}.jsonl"
        records = iter_records_from_jsonl(split_path, limit=io.eval_limit)
        eval_results[split_name] = evaluate(
            model,
            records=records,
            batch_size=int(io.eval_batch_size),
            shape_mode="auto",
        )
        _write_json(out_dir / f"eval_{split_name}.json", eval_results[split_name])

    errors_summary = None
    if io.analyze_errors:
        test_path = Path(splits_dir) / "test.jsonl"
        records = iter_records_from_jsonl(test_path, limit=io.errors_limit)
        errors_summary = analyze_errors(
            model,
            records=records,
            out_dir=out_dir / "errors_test",
            config=ErrorAnalysisConfig(
                batch_size=int(io.eval_batch_size),
                shape_mode="auto",
                max_per_pair=int(io.errors_max_per_pair),
                thumb_size=int(io.errors_thumb_size),
                seed=int(io.seed),
            ),
        )
        _write_json(out_dir / "errors_test_summary.json", errors_summary)

    summary = {
        "splits_dir": str(Path(splits_dir)),
        "out_dir": str(out_dir),
        "model": str(model_path),
        "timestamp": _timestamp(),
        "train_config": asdict(train_config),
        "train_metrics": {
            "train": train_metrics.get("train"),
            "val": train_metrics.get("val"),
            "test": train_metrics.get("test"),
            "best_val": train_metrics.get("best_val"),
            "early_stopping": train_metrics.get("early_stopping"),
        },
        "eval": eval_results,
        "summary": _summary_from_eval(eval_results["train"], eval_results["val"], eval_results["test"]),
        "errors_test": {"enabled": bool(io.analyze_errors), "out_dir": str(out_dir / "errors_test")}
        if io.analyze_errors
        else {"enabled": False},
    }
    if errors_summary is not None:
        summary["errors_test"]["artifacts"] = errors_summary.get("artifacts")

    _write_json(out_dir / "summary.json", summary)
    return summary


def _list_fold_dirs(folds_dir: Path, *, max_folds: int | None = None) -> list[Path]:
    folds_dir = Path(folds_dir)
    meta_path = folds_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        k = int(meta.get("folds", 0))
        fold_dirs = [folds_dir / f"fold_{i:02d}" for i in range(k)]
    else:
        fold_dirs = sorted([p for p in folds_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])

    if max_folds is not None:
        fold_dirs = fold_dirs[: max(0, int(max_folds))]
    return fold_dirs


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": int(arr.size)}


def run_cnn_kfold_experiment(
    *,
    folds_dir: Path,
    out_dir: Path,
    train_config: CNNTrainConfig,
    io: ExperimentIOConfig,
    max_folds: int | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = _list_fold_dirs(Path(folds_dir), max_folds=max_folds)
    if not fold_dirs:
        raise ValueError(f"No fold directories found under: {folds_dir}")

    fold_summaries: list[dict] = []
    for fold_dir in fold_dirs:
        fold_out = out_dir / fold_dir.name
        fold_summaries.append(
            run_cnn_experiment(
                splits_dir=fold_dir,
                out_dir=fold_out,
                train_config=train_config,
                io=io,
            )
        )

    test_accs: list[float] = []
    val_accs: list[float] = []
    gaps: list[float] = []
    for s in fold_summaries:
        summ = s.get("summary") or {}
        if summ.get("accuracy_test") is not None:
            test_accs.append(float(summ["accuracy_test"]))
        if summ.get("accuracy_val") is not None:
            val_accs.append(float(summ["accuracy_val"]))
        if summ.get("gap_train_minus_test") is not None:
            gaps.append(float(summ["gap_train_minus_test"]))

    aggregate = {
        "folds_dir": str(Path(folds_dir)),
        "out_dir": str(out_dir),
        "timestamp": _timestamp(),
        "train_config": asdict(train_config),
        "folds": [Path(s["out_dir"]).name for s in fold_summaries],
        "aggregate": {
            "accuracy_test": _mean_std(test_accs),
            "accuracy_val": _mean_std(val_accs),
            "gap_train_minus_test": _mean_std(gaps),
        },
    }

    _write_json(out_dir / "kfold_summary.json", aggregate)
    return {"aggregate": aggregate, "folds": fold_summaries}


def run_experiment_suite(
    *,
    splits_dir: Path | None = None,
    folds_dir: Path | None = None,
    out_dir: Path,
    presets: Iterable[str],
    device: str,
    io: ExperimentIOConfig,
    max_folds: int | None = None,
) -> dict:
    if (splits_dir is None) == (folds_dir is None):
        raise ValueError("Provide exactly one of splits_dir or folds_dir.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = cnn_presets(device=device)
    preset_names = [str(p) for p in presets]
    for p in preset_names:
        if p not in available:
            raise ValueError(f"Unknown preset: {p}. Available: {sorted(available.keys())}")

    results: dict[str, dict] = {}
    for name in preset_names:
        cfg = available[name]
        if splits_dir is not None:
            results[name] = run_cnn_experiment(
                splits_dir=Path(splits_dir),
                out_dir=out_dir / name,
                train_config=cfg,
                io=io,
            )
        else:
            results[name] = run_cnn_kfold_experiment(
                folds_dir=Path(folds_dir),
                out_dir=out_dir / name,
                train_config=cfg,
                io=io,
                max_folds=max_folds,
            )

    comparisons: dict[str, dict] = {}
    if "baseline" in results:
        base = results["baseline"]
        if splits_dir is not None:
            base_summary = (base.get("summary") or {}) if isinstance(base, dict) else {}

            for name in preset_names:
                if name == "baseline":
                    continue
                curr = results[name]
                curr_summary = (curr.get("summary") or {}) if isinstance(curr, dict) else {}

                def _delta(key: str):
                    a = base_summary.get(key)
                    b = curr_summary.get(key)
                    if a is None or b is None:
                        return None
                    return float(b) - float(a)

                comparisons[name] = {
                    "baseline": "baseline",
                    "candidate": name,
                    "delta_accuracy_test": _delta("accuracy_test"),
                    "delta_accuracy_val": _delta("accuracy_val"),
                    "delta_gap_train_minus_test": _delta("gap_train_minus_test"),
                    "delta_gap_train_minus_val": _delta("gap_train_minus_val"),
                }
        else:
            base_agg = base.get("aggregate", {}).get("aggregate", {}) if isinstance(base, dict) else {}

            def _mean(obj: dict, key: str) -> float | None:
                v = obj.get(key)
                if not isinstance(v, dict):
                    return None
                m = v.get("mean")
                return None if m is None else float(m)

            for name in preset_names:
                if name == "baseline":
                    continue
                curr = results[name]
                curr_agg = curr.get("aggregate", {}).get("aggregate", {}) if isinstance(curr, dict) else {}

                def _delta_mean(metric: str) -> float | None:
                    a = _mean(base_agg, metric)
                    b = _mean(curr_agg, metric)
                    if a is None or b is None:
                        return None
                    return float(b) - float(a)

                comparisons[name] = {
                    "baseline": "baseline",
                    "candidate": name,
                    "delta_accuracy_test_mean": _delta_mean("accuracy_test"),
                    "delta_accuracy_val_mean": _delta_mean("accuracy_val"),
                    "delta_gap_train_minus_test_mean": _delta_mean("gap_train_minus_test"),
                }

    suite = {
        "timestamp": _timestamp(),
        "out_dir": str(out_dir),
        "mode": "kfold" if folds_dir is not None else "single",
        "splits_dir": str(Path(splits_dir)) if splits_dir is not None else None,
        "folds_dir": str(Path(folds_dir)) if folds_dir is not None else None,
        "presets": preset_names,
        "results": results,
        "comparisons": comparisons,
    }

    _write_json(out_dir / "suite_summary.json", suite)
    return suite
