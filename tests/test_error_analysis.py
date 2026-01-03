from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from vml.dataset_index import ImageRecord
from vml.error_analysis import ErrorAnalysisConfig, analyze_errors
from vml.inference import load_model


def _write_gray_spot(path: Path, *, row: int, col: int, size: tuple[int, int] = (4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[int(row), int(col)] = 255
    Image.fromarray(arr).save(path)


class TestErrorAnalysis(unittest.TestCase):
    def test_analyze_errors_writes_report_and_grids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "softmax.pt"
            meta_path = model_path.with_suffix(".json")

            # A deterministic "always predict action" linear model: W=0, b=0 => argmax=0.
            state = {
                "net.1.weight": torch.zeros((3, 16), dtype=torch.float32),
                "net.1.bias": torch.zeros((3,), dtype=torch.float32),
            }
            torch.save(state, model_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "model_type": "softmax_regression",
                        "input_shape": [4, 4],
                        "channels": "grayscale",
                        "class_map": {"action": 0, "pose": 1, "expression": 2},
                        "train_config": {"shape_mode": "strict"},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            # Build a tiny dataset with at least one pose sample (to trigger pose->action focus pair).
            records: list[ImageRecord] = []
            kinds = [("action", 0, (0, 0)), ("pose", 1, (0, 1)), ("expression", 2, (1, 0))]
            for kind, class_id, (r, c) in kinds:
                for i in range(2):
                    img_path = root / "charA" / kind / f"img_{i:04d}.png"
                    _write_gray_spot(img_path, row=r, col=c, size=(4, 4))
                    records.append(
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

            model = load_model(model_path)
            out_dir = root / "report"
            summary = analyze_errors(
                model,
                records=records,
                out_dir=out_dir,
                config=ErrorAnalysisConfig(batch_size=4, shape_mode="auto", max_per_pair=8, thumb_size=32, seed=0),
            )
            self.assertIn("overall", summary)
            self.assertTrue((out_dir / "report.html").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "predictions.jsonl").exists())

            # focus pair grid for pose->action should exist
            self.assertTrue((out_dir / "grids" / "global" / "pose_to_action.png").exists())
            self.assertTrue((out_dir / "grids" / "by_character" / "charA" / "pose_to_action.png").exists())
