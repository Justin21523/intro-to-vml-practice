from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vml.dataset_index import ImageRecord, write_jsonl
from vml.train_softmax import SoftmaxTrainConfig, train_softmax_from_splits


def _write_pattern_image(path: Path, *, which: int, size: tuple[int, int] = (4, 4)) -> None:
    """
    建立一張簡單可線性分離的灰階圖：
    - class 0: (0,0)=255
    - class 1: (0,1)=255
    - class 2: (0,2)=255
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[0, which] = 255
    Image.fromarray(arr).save(path)


def _write_constant_image(path: Path, *, value: int, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full(size, int(value), dtype=np.uint8)
    Image.fromarray(arr).save(path)


class TestTrainSoftmax(unittest.TestCase):
    def test_train_softmax_overfits_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "model.pt"

            train: list[ImageRecord] = []
            val: list[ImageRecord] = []
            test: list[ImageRecord] = []

            kinds = [("action", 0), ("pose", 1), ("expression", 2)]

            # 5 張 train / 2 張 val / 2 張 test per class
            for kind, class_id in kinds:
                for i in range(5):
                    img_path = root / "charA" / kind / f"train_{i:04d}.png"
                    _write_pattern_image(img_path, which=class_id)
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
                    _write_pattern_image(img_path, which=class_id)
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
                    _write_pattern_image(img_path, which=class_id)
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

            cfg = SoftmaxTrainConfig(
                epochs=25,
                batch_size=8,
                lr=0.5,
                reg=0.0,
                seed=0,
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                verbose=False,
            )

            metrics = train_softmax_from_splits(
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
            self.assertEqual(meta["model_type"], "softmax_regression")

    def test_train_softmax_pad_or_crop_handles_variable_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "model_pad_or_crop.pt"

            train: list[ImageRecord] = []
            val: list[ImageRecord] = []
            test: list[ImageRecord] = []

            kinds = [("action", 0, 64), ("pose", 1, 128), ("expression", 2, 192)]

            # 混合大小：大圖會被裁切，小圖會被 padding
            for kind, class_id, value in kinds:
                for i in range(4):
                    size = (6, 6) if i < 2 else (2, 2)
                    img_path = root / "charA" / kind / f"train_{i:04d}.png"
                    _write_constant_image(img_path, value=value, size=size)
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
                    size = (6, 6) if i == 0 else (2, 2)
                    img_path = root / "charA" / kind / f"val_{i:04d}.png"
                    _write_constant_image(img_path, value=value, size=size)
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
                    size = (6, 6) if i == 0 else (2, 2)
                    img_path = root / "charA" / kind / f"test_{i:04d}.png"
                    _write_constant_image(img_path, value=value, size=size)
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

            cfg = SoftmaxTrainConfig(
                epochs=200,
                batch_size=12,
                lr=0.5,
                reg=0.0,
                seed=0,
                channels="grayscale",
                dtype="float32",
                shape_mode="pad_or_crop",
                image_size=(4, 4),
                augment_hflip=True,
                verbose=False,
            )

            metrics = train_softmax_from_splits(
                splits_dir=splits_dir,
                out_model_path=model_path,
                config=cfg,
            )

            self.assertGreaterEqual(metrics["train"]["accuracy"], 0.9)
            self.assertGreaterEqual(metrics["val"]["accuracy"], 0.9)
            self.assertGreaterEqual(metrics["test"]["accuracy"], 0.9)
            self.assertTrue(model_path.exists())

    def test_train_softmax_strict_raises_on_variable_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "model_strict.pt"

            img_a = root / "charA" / "action" / "a.png"
            img_b = root / "charA" / "pose" / "b.png"
            _write_constant_image(img_a, value=50, size=(6, 6))
            _write_constant_image(img_b, value=200, size=(2, 2))

            train = [
                ImageRecord(
                    path=str(img_a),
                    relpath=str(img_a.relative_to(root)),
                    character="charA",
                    kind="action",
                    label=None,
                    class_name="action",
                    class_id=0,
                ),
                ImageRecord(
                    path=str(img_b),
                    relpath=str(img_b.relative_to(root)),
                    character="charA",
                    kind="pose",
                    label=None,
                    class_name="pose",
                    class_id=1,
                ),
            ]

            splits_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(splits_dir / "train.jsonl", train)
            write_jsonl(splits_dir / "val.jsonl", [])
            write_jsonl(splits_dir / "test.jsonl", [])

            cfg = SoftmaxTrainConfig(
                epochs=1,
                batch_size=2,
                lr=0.1,
                seed=0,
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                verbose=False,
            )

            with self.assertRaises(ValueError) as ctx:
                _ = train_softmax_from_splits(
                    splits_dir=splits_dir,
                    out_model_path=model_path,
                    config=cfg,
                )
            self.assertIn("Inconsistent image shape", str(ctx.exception))
