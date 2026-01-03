from __future__ import annotations

import unittest

from vml.dataset_index import ImageRecord, kfold_split_records_by_character, split_records


def _make_record(character: str, kind: str, idx: int) -> ImageRecord:
    return ImageRecord(
        path=f"/abs/{character}/{kind}/{idx:04d}.png",
        relpath=f"{character}/{kind}/{idx:04d}.png",
        character=character,
        kind=kind,
        label=None,
        class_name=kind,
        class_id={"action": 0, "pose": 1, "expression": 2}[kind],
        width=None,
        height=None,
    )


class TestDatasetSplit(unittest.TestCase):
    def test_split_random_stratified_counts(self) -> None:
        records: list[ImageRecord] = []
        for kind in ["action", "pose", "expression"]:
            for i in range(10):
                records.append(_make_record("charA", kind, i))

        splits, meta = split_records(
            records,
            val_ratio=0.2,
            test_ratio=0.3,
            seed=123,
            strategy="random",
        )

        self.assertEqual(meta["counts"]["train"] + meta["counts"]["val"] + meta["counts"]["test"], 30)

        # 每個 class 10 張，test=3、val=2、train=5
        def count_kind(split_name: str, kind: str) -> int:
            return sum(1 for r in splits[split_name] if r.kind == kind)

        for kind in ["action", "pose", "expression"]:
            self.assertEqual(count_kind("test", kind), 3)
            self.assertEqual(count_kind("val", kind), 2)
            self.assertEqual(count_kind("train", kind), 5)

    def test_split_by_character_no_leak(self) -> None:
        records: list[ImageRecord] = []
        for character in ["A", "B", "C", "D", "E"]:
            for kind in ["action", "pose", "expression"]:
                for i in range(2):
                    records.append(_make_record(f"char{character}", kind, i))

        splits, meta = split_records(
            records,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            strategy="by-character",
        )

        seen: dict[str, str] = {}
        for split_name in ["train", "val", "test"]:
            for rec in splits[split_name]:
                if rec.character is None:
                    continue
                if rec.character in seen and seen[rec.character] != split_name:
                    self.fail(f"character leak: {rec.character} in {seen[rec.character]} and {split_name}")
                seen[rec.character] = split_name

        self.assertIn("characters_train", meta)
        self.assertIn("characters_val", meta)
        self.assertIn("characters_test", meta)

    def test_kfold_by_character_no_leak(self) -> None:
        records: list[ImageRecord] = []
        for character in ["A", "B", "C", "D", "E", "F"]:
            for kind in ["action", "pose", "expression"]:
                for i in range(2):
                    records.append(_make_record(f"char{character}", kind, i))

        fold_splits, meta = kfold_split_records_by_character(
            records,
            folds=3,
            seed=123,
            val_fold_offset=1,
            keep_unknown=False,
        )

        self.assertEqual(int(meta["folds"]), 3)
        self.assertEqual(len(fold_splits), 3)

        for fold_idx, splits in enumerate(fold_splits):
            seen: dict[str, str] = {}
            for split_name in ["train", "val", "test"]:
                for rec in splits[split_name]:
                    if rec.character is None:
                        continue
                    if rec.character in seen and seen[rec.character] != split_name:
                        self.fail(
                            f"character leak in fold {fold_idx}: {rec.character} in {seen[rec.character]} and {split_name}"
                        )
                    seen[rec.character] = split_name

            fold_meta = meta["folds_meta"][fold_idx]
            self.assertEqual(set(fold_meta["characters_train"]), {c for c, s in seen.items() if s == "train"})
            self.assertEqual(set(fold_meta["characters_val"]), {c for c, s in seen.items() if s == "val"})
            self.assertEqual(set(fold_meta["characters_test"]), {c for c, s in seen.items() if s == "test"})
