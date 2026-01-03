from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vml.dataset_index import ImageRecord, kfold_split_records_by_character, write_jsonl
from vml.experiments import ExperimentIOConfig, run_cnn_experiment, run_cnn_kfold_experiment
from vml.train_cnn import CNNTrainConfig


def _write_spot(path: Path, *, row: int, col: int, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[int(row), int(col)] = 255
    Image.fromarray(arr).save(path)


def _make_record(root: Path, character: str, kind: str, idx: int, *, row: int, col: int) -> ImageRecord:
    img_path = root / character / kind / f"img_{idx:04d}.png"
    _write_spot(img_path, row=row, col=col)
    return ImageRecord(
        path=str(img_path),
        relpath=str(img_path.relative_to(root)),
        character=character,
        kind=kind,
        label=None,
        class_name=kind,
        class_id={"action": 0, "pose": 1, "expression": 2}[kind],
    )


class TestExperiments(unittest.TestCase):
    def test_run_cnn_experiment_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            out_dir = root / "out"

            train: list[ImageRecord] = []
            val: list[ImageRecord] = []
            test: list[ImageRecord] = []

            spots = {"action": (0, 0), "pose": (0, 4), "expression": (4, 0)}
            for kind, (r, c) in spots.items():
                for i in range(3):
                    train.append(_make_record(root, "charA", kind, i, row=r, col=c))
                for i in range(1):
                    val.append(_make_record(root, "charB", kind, i, row=r, col=c))
                    test.append(_make_record(root, "charC", kind, i, row=r, col=c))

            splits_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(splits_dir / "train.jsonl", train)
            write_jsonl(splits_dir / "val.jsonl", val)
            write_jsonl(splits_dir / "test.jsonl", test)

            cfg = CNNTrainConfig(
                epochs=1,
                batch_size=4,
                lr=0.1,
                reg=0.0,
                seed=0,
                device="cpu",
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                augment_hflip=False,
                conv_out_channels=4,
                conv2_out_channels=0,
                kernel_size=3,
                conv_stride=1,
                pool_size=2,
                hidden_size=8,
                verbose=False,
            )
            summary = run_cnn_experiment(
                splits_dir=splits_dir,
                out_dir=out_dir,
                train_config=cfg,
                io=ExperimentIOConfig(analyze_errors=False, eval_batch_size=4),
            )
            self.assertTrue((out_dir / "model.pt").exists())
            self.assertTrue((out_dir / "model.json").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertIn("eval", summary)
            self.assertIn("summary", summary)

    def test_run_cnn_kfold_experiment_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folds_dir = root / "folds"
            out_dir = root / "out"

            records: list[ImageRecord] = []
            spots = {"action": (0, 0), "pose": (0, 4), "expression": (4, 0)}
            for ch in ["charA", "charB", "charC", "charD", "charE", "charF"]:
                for kind, (r, c) in spots.items():
                    records.append(_make_record(root, ch, kind, 0, row=r, col=c))

            fold_splits, _ = kfold_split_records_by_character(records, folds=3, seed=0, val_fold_offset=1)
            folds_dir.mkdir(parents=True, exist_ok=True)
            for i, splits in enumerate(fold_splits):
                fold_dir = folds_dir / f"fold_{i:02d}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(fold_dir / "train.jsonl", splits["train"])
                write_jsonl(fold_dir / "val.jsonl", splits["val"])
                write_jsonl(fold_dir / "test.jsonl", splits["test"])

            cfg = CNNTrainConfig(
                epochs=1,
                batch_size=4,
                lr=0.1,
                reg=0.0,
                seed=0,
                device="cpu",
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                augment_hflip=False,
                conv_out_channels=4,
                conv2_out_channels=0,
                kernel_size=3,
                conv_stride=1,
                pool_size=2,
                hidden_size=8,
                verbose=False,
            )

            result = run_cnn_kfold_experiment(
                folds_dir=folds_dir,
                out_dir=out_dir,
                train_config=cfg,
                io=ExperimentIOConfig(analyze_errors=False, eval_batch_size=4),
                max_folds=2,
            )
            self.assertIn("aggregate", result)
            self.assertTrue((out_dir / "kfold_summary.json").exists())
