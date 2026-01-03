from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vml.dataset_index import ImageRecord, write_jsonl
from vml.train_mlp import MLPTrainConfig, train_mlp_from_splits


def _write_pattern_image(path: Path, *, which: int, size: tuple[int, int] = (4, 4)) -> None:
    """
    建立一張簡單可分類的灰階圖：
    - class 0: (0,0)=255
    - class 1: (0,1)=255
    - class 2: (0,2)=255
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[0, which] = 255
    Image.fromarray(arr).save(path)


class TestTrainMLP(unittest.TestCase):
    def test_train_mlp_overfits_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "model_mlp.pt"

            train: list[ImageRecord] = []
            val: list[ImageRecord] = []
            test: list[ImageRecord] = []

            kinds = [("action", 0), ("pose", 1), ("expression", 2)]

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

            cfg = MLPTrainConfig(
                epochs=50,
                batch_size=8,
                lr=0.5,
                reg=0.0,
                seed=0,
                channels="grayscale",
                dtype="float32",
                shape_mode="strict",
                hidden_sizes=(16,),
                verbose=False,
            )

            metrics = train_mlp_from_splits(
                splits_dir=splits_dir,
                out_model_path=model_path,
                config=cfg,
            )

            self.assertGreaterEqual(metrics["train"]["accuracy"], 0.99)
            self.assertGreaterEqual(metrics["val"]["accuracy"], 0.99)
            self.assertGreaterEqual(metrics["test"]["accuracy"], 0.99)

            self.assertTrue(model_path.exists())
            self.assertTrue(model_path.with_suffix(".json").exists())

            meta = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(meta["model_type"], "mlp")
