from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vml.dataset_index import DatasetScanOptions, scan_dataset


def _write_dummy_image(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


class TestDatasetIndex(unittest.TestCase):
    def test_scan_dataset_parses_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dummy_image(root / "charA" / "action" / "wave" / "0001.png")
            _write_dummy_image(root / "charA" / "pose" / "standing" / "0002.jpg")
            _write_dummy_image(root / "charB" / "expression" / "happy" / "0003.webp")
            (root / "charB" / "expression" / "happy" / "ignore.txt").write_text("x", encoding="utf-8")

            options = DatasetScanOptions(root=root, with_size=True)
            records = sorted(scan_dataset(options), key=lambda r: r.relpath)

            self.assertEqual(len(records), 3)

            r0 = records[0]
            self.assertEqual(r0.character, "charA")
            self.assertEqual(r0.kind, "action")
            self.assertEqual(r0.label, "wave")
            self.assertEqual(r0.class_name, "action")
            self.assertEqual(r0.class_id, 0)
            self.assertEqual((r0.width, r0.height), (8, 6))

            r1 = records[1]
            self.assertEqual(r1.character, "charA")
            self.assertEqual(r1.kind, "pose")
            self.assertEqual(r1.label, "standing")
            self.assertEqual(r1.class_name, "pose")
            self.assertEqual(r1.class_id, 1)

            r2 = records[2]
            self.assertEqual(r2.character, "charB")
            self.assertEqual(r2.kind, "expression")
            self.assertEqual(r2.label, "happy")
            self.assertEqual(r2.class_name, "expression")
            self.assertEqual(r2.class_id, 2)

    def test_scan_dataset_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(5):
                _write_dummy_image(root / "charA" / "action" / "wave" / f"{i:04d}.png")

            options = DatasetScanOptions(root=root, limit=2)
            records = list(scan_dataset(options))
            self.assertEqual(len(records), 2)
