from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vml.dataset_index import ImageRecord, write_jsonl
from vml.train_cnn import CNNTrainConfig, train_cnn_from_splits


def _write_spot_image(path: Path, *, row: int, col: int, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[int(row), int(col)] = 255
    Image.fromarray(arr).save(path)


class TestTrainCNN(unittest.TestCase):
    def test_train_cnn_overfits_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "model_cnn.pt"

            train: list[ImageRecord] = []
            val: list[ImageRecord] = []
            test: list[ImageRecord] = []

            spots = {0: (0, 0), 1: (0, 4), 2: (4, 0)}
            kinds = [("action", 0), ("pose", 1), ("expression", 2)]

            for kind, class_id in kinds:
                row, col = spots[class_id]
                for i in range(8):
                    img_path = root / "charA" / kind / f"train_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    train.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )
                for i in range(2):
                    img_path = root / "charA" / kind / f"val_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    val.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )
                for i in range(2):
                    img_path = root / "charA" / kind / f"test_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    test.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )

            splits_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(splits_dir / "train.jsonl", train)
            write_jsonl(splits_dir / "val.jsonl", val)
            write_jsonl(splits_dir / "test.jsonl", test)

            cfg = CNNTrainConfig(
                epochs=10,
                batch_size=8,
                lr=0.1,
                reg=0.0,
                seed=0,
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                conv_out_channels=4,
                kernel_size=3,
                pool_size=2,
                hidden_size=16,
                verbose=False,
            )

            metrics = train_cnn_from_splits(
                splits_dir=splits_dir,
                out_model_path=model_path,
                config=cfg,
            )

            self.assertIsInstance(metrics, dict)
            self.assertGreaterEqual(metrics["train"]["accuracy"], 0.99)
            self.assertGreaterEqual(metrics["val"]["accuracy"], 0.99)
            self.assertGreaterEqual(metrics["test"]["accuracy"], 0.99)

            self.assertTrue(model_path.exists())
            self.assertTrue(model_path.with_suffix(".json").exists())

            meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(meta["model_type"], "cnn")

    def test_train_cnn_early_stopping_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "model_cnn_es.pt"

            train: list[ImageRecord] = []
            val: list[ImageRecord] = []
            test: list[ImageRecord] = []

            spots = {0: (0, 0), 1: (0, 4), 2: (4, 0)}
            kinds = [("action", 0), ("pose", 1), ("expression", 2)]

            for kind, class_id in kinds:
                row, col = spots[class_id]
                for i in range(6):
                    img_path = root / "charA" / kind / f"train_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    train.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )
                for i in range(2):
                    img_path = root / "charA" / kind / f"val_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    val.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )
                for i in range(2):
                    img_path = root / "charA" / kind / f"test_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    test.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )

            splits_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(splits_dir / "train.jsonl", train)
            write_jsonl(splits_dir / "val.jsonl", val)
            write_jsonl(splits_dir / "test.jsonl", test)

            cfg = CNNTrainConfig(
                epochs=50,
                batch_size=8,
                lr=0.1,
                reg=0.0,
                seed=0,
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                conv_out_channels=4,
                kernel_size=3,
                pool_size=2,
                hidden_size=16,
                early_stopping_patience=1,
                early_stopping_min_delta=1e9,
                early_stopping_metric="loss",
                verbose=False,
            )

            metrics = train_cnn_from_splits(
                splits_dir=splits_dir,
                out_model_path=model_path,
                config=cfg,
            )
            self.assertTrue(metrics["early_stopping"]["stopped_early"])
            self.assertEqual(int(metrics["early_stopping"]["stopped_epoch"]), 2)
            self.assertLessEqual(len(metrics["history"]), 2)

    def test_train_cnn_with_conv2_overfits_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "model_cnn_conv2.pt"

            train: list[ImageRecord] = []
            val: list[ImageRecord] = []
            test: list[ImageRecord] = []

            spots = {0: (0, 0), 1: (0, 4), 2: (4, 0)}
            kinds = [("action", 0), ("pose", 1), ("expression", 2)]

            for kind, class_id in kinds:
                row, col = spots[class_id]
                for i in range(8):
                    img_path = root / "charA" / kind / f"train_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    train.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )
                for i in range(2):
                    img_path = root / "charA" / kind / f"val_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    val.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )
                for i in range(2):
                    img_path = root / "charA" / kind / f"test_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col)
                    test.append(
                        ImageRecord(
                            path=str(img_path),
                            relpath=str(img_path.relative_to(root)),
                            character="charA",
                            kind=kind,
                            label=None,
                            class_name=kind,
                            class_id=class_id,
                        )
                    )

            splits_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(splits_dir / "train.jsonl", train)
            write_jsonl(splits_dir / "val.jsonl", val)
            write_jsonl(splits_dir / "test.jsonl", test)

            cfg = CNNTrainConfig(
                epochs=10,
                batch_size=8,
                lr=0.1,
                reg=0.0,
                seed=0,
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                conv_out_channels=4,
                conv2_out_channels=8,
                kernel_size=3,
                pool_size=2,
                hidden_size=16,
                verbose=False,
            )

            metrics = train_cnn_from_splits(
                splits_dir=splits_dir,
                out_model_path=model_path,
                config=cfg,
            )

            self.assertGreaterEqual(metrics["train"]["accuracy"], 0.99)
            self.assertGreaterEqual(metrics["val"]["accuracy"], 0.99)
            self.assertGreaterEqual(metrics["test"]["accuracy"], 0.99)

            meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(int(meta["arch"]["conv2_out_channels"]), 8)

            import torch

            state = torch.load(model_path, map_location="cpu")
            self.assertIn("features.4.weight", state)
