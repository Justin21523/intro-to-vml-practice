from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import json

from vml.dataset_index import ImageRecord, write_jsonl
from vml.inference import evaluate, load_model, predict_image
from vml.train_cnn import CNNTrainConfig, train_cnn_from_splits
from vml.train_mlp import MLPTrainConfig, train_mlp_from_splits
from vml.train_softmax import SoftmaxTrainConfig, train_softmax_from_splits


def _write_pattern_image(path: Path, *, which: int, size: tuple[int, int] = (4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[0, which] = 255
    Image.fromarray(arr).save(path)


def _write_spot_image(path: Path, *, row: int, col: int, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[int(row), int(col)] = 255
    Image.fromarray(arr).save(path)


class TestInferenceEvalPredict(unittest.TestCase):
    def test_eval_and_predict_softmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "softmax.pt"

            train: list[ImageRecord] = []
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
            write_jsonl(splits_dir / "val.jsonl", [])
            write_jsonl(splits_dir / "test.jsonl", test)

            cfg = SoftmaxTrainConfig(epochs=25, batch_size=8, lr=0.5, seed=0, shape_mode="strict", verbose=False)
            _ = train_softmax_from_splits(splits_dir=splits_dir, out_model_path=model_path, config=cfg)

            model = load_model(model_path)
            result = evaluate(model, records=test, batch_size=4, shape_mode="auto")
            self.assertGreaterEqual(result["accuracy"], 0.99)
            cm = np.array(result["confusion_matrix"], dtype=int)
            self.assertEqual(int(np.trace(cm)), len(test))

            preds_out = root / "predictions.jsonl"
            detailed = evaluate(
                model,
                records=test,
                batch_size=4,
                shape_mode="auto",
                return_predictions=True,
                out_predictions=preds_out,
            )
            self.assertIn("predictions", detailed)
            self.assertEqual(len(detailed["predictions"]), len(test))
            self.assertTrue(preds_out.exists())
            lines = preds_out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(test))
            row = json.loads(lines[0])
            self.assertIn("true_class_id", row)
            self.assertIn("predicted_class_id", row)
            self.assertIn("probabilities", row)

            pred = predict_image(model, image_path=Path(test[0].path), shape_mode="auto")
            self.assertIn("predicted_class_id", pred)
            self.assertIn("probabilities", pred)

    def test_eval_and_predict_cnn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "cnn.pt"

            train: list[ImageRecord] = []
            test: list[ImageRecord] = []

            spots = {0: (0, 0), 1: (0, 4), 2: (4, 0)}
            kinds = [("action", 0), ("pose", 1), ("expression", 2)]
            for kind, class_id in kinds:
                row, col = spots[class_id]
                for i in range(6):
                    img_path = root / "charA" / kind / f"train_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col, size=(8, 8))
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
                    img_path = root / "charA" / kind / f"test_{i:04d}.png"
                    _write_spot_image(img_path, row=row, col=col, size=(8, 8))
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
            write_jsonl(splits_dir / "val.jsonl", [])
            write_jsonl(splits_dir / "test.jsonl", test)

            cfg = CNNTrainConfig(
                epochs=10,
                batch_size=8,
                lr=0.1,
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
            _ = train_cnn_from_splits(splits_dir=splits_dir, out_model_path=model_path, config=cfg)

            model = load_model(model_path)
            result = evaluate(model, records=test, batch_size=4, shape_mode="auto")
            self.assertGreaterEqual(result["accuracy"], 0.99)
            cm = np.array(result["confusion_matrix"], dtype=int)
            self.assertEqual(int(np.trace(cm)), len(test))

            pred = predict_image(model, image_path=Path(test[0].path), shape_mode="auto")
            self.assertIn("predicted_class_id", pred)
            self.assertIn("probabilities", pred)

    def test_eval_mlp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits_dir = root / "splits"
            model_path = root / "mlp.pt"

            train: list[ImageRecord] = []
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
            write_jsonl(splits_dir / "val.jsonl", [])
            write_jsonl(splits_dir / "test.jsonl", test)

            cfg = MLPTrainConfig(
                epochs=50,
                batch_size=8,
                lr=0.5,
                seed=0,
                hidden_sizes=(16,),
                shape_mode="strict",
                verbose=False,
            )
            _ = train_mlp_from_splits(splits_dir=splits_dir, out_model_path=model_path, config=cfg)

            model = load_model(model_path)
            result = evaluate(model, records=test, batch_size=4, shape_mode="auto")
            self.assertGreaterEqual(result["accuracy"], 0.99)
