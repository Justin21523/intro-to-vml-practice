from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image

DEFAULT_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
KNOWN_KINDS = {"action", "pose", "expression"}
KIND_TO_CLASS_ID = {"action": 0, "pose": 1, "expression": 2}


@dataclass(frozen=True)
class DatasetScanOptions:
    root: Path
    with_size: bool = False
    limit: int | None = None
    image_exts: frozenset[str] = frozenset(DEFAULT_IMAGE_EXTS)


@dataclass(frozen=True)
class ImageRecord:
    path: str
    relpath: str
    character: str | None
    kind: str | None
    label: str | None
    class_name: str | None = None
    class_id: int | None = None
    width: int | None = None
    height: int | None = None


def _iter_image_paths(root: Path, image_exts: frozenset[str]) -> Iterator[Path]:
    root = root.resolve()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in image_exts:
                yield path


def _read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return width, height


def _build_record(root: Path, path: Path, *, with_size: bool) -> ImageRecord:
    rel = path.resolve().relative_to(root.resolve())
    parts = rel.parts

    # 預期資料夾結構：
    # - root/<character>/<kind>/.../<image>
    # 其中 kind ∈ {"action", "pose", "expression"}
    character: str | None
    kind: str | None
    if len(parts) >= 2 and parts[1] in KNOWN_KINDS:
        character = parts[0]
        kind = parts[1]
        label_parts = parts[2:-1]
    elif len(parts) >= 1 and parts[0] in KNOWN_KINDS:
        character = None
        kind = parts[0]
        label_parts = parts[1:-1]
    else:
        character = parts[0] if len(parts) >= 2 else None
        kind = None
        label_parts = parts[1:-1]

    label = "/".join(label_parts) if label_parts else None
    class_name = kind
    class_id = KIND_TO_CLASS_ID.get(kind) if kind is not None else None

    width = height = None
    if with_size:
        width, height = _read_size(path)

    return ImageRecord(
        path=str(path),
        relpath=str(rel),
        character=character,
        kind=kind,
        label=label,
        class_name=class_name,
        class_id=class_id,
        width=width,
        height=height,
    )


def scan_dataset(options: DatasetScanOptions) -> Iterator[ImageRecord]:
    root = options.root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    count = 0
    for path in _iter_image_paths(root, options.image_exts):
        yield _build_record(root, path, with_size=options.with_size)
        count += 1
        if options.limit is not None and count >= options.limit:
            break


def summarize_records(records: Iterable[ImageRecord]) -> dict:
    total = 0
    by_character: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    unknown_kind_images = 0
    unknown_kind_examples: list[str] = []

    for rec in records:
        total += 1
        if rec.character is not None:
            by_character[rec.character] = by_character.get(rec.character, 0) + 1
        if rec.kind is not None:
            by_kind[rec.kind] = by_kind.get(rec.kind, 0) + 1
        else:
            unknown_kind_images += 1
            if len(unknown_kind_examples) < 5:
                unknown_kind_examples.append(rec.relpath)

    return {
        "total_images": total,
        "characters": len(by_character),
        "by_character_top10": sorted(by_character.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: kv[0])),
        "class_map": KIND_TO_CLASS_ID,
        "unknown_kind_images": unknown_kind_images,
        "unknown_kind_examples": unknown_kind_examples,
    }


def write_jsonl(out_path: Path, records: Iterable[ImageRecord]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[ImageRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    allowed = set(ImageRecord.__dataclass_fields__.keys())
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSONL line (expected object): {line[:200]}")
            filtered = {k: v for k, v in obj.items() if k in allowed}
            yield ImageRecord(**filtered)


def _validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")
    if not (0.0 <= test_ratio <= 1.0):
        raise ValueError(f"test_ratio must be in [0, 1], got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}")


def _count_from_ratio(n: int, ratio: float) -> int:
    if ratio <= 0.0 or n <= 0:
        return 0
    return max(1, int(round(ratio * n)))


def _split_list(
    items: Sequence[ImageRecord],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[ImageRecord]]:
    items_list = list(items)
    rng.shuffle(items_list)

    n = len(items_list)
    n_test = int(round(test_ratio * n))
    n_val = int(round(val_ratio * n))
    if n_test + n_val > n:
        n_val = max(0, n - n_test)

    test = items_list[:n_test]
    val = items_list[n_test : n_test + n_val]
    train = items_list[n_test + n_val :]

    return {"train": train, "val": val, "test": test}


def split_records(
    records: Sequence[ImageRecord],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    strategy: str = "random",
    keep_unknown: bool = False,
) -> tuple[dict[str, list[ImageRecord]], dict]:
    """
    切分 train/val/test。

    - random：依 class_id（action/pose/expression）分層隨機切分
    - by-character：以 character 為單位切分（避免同角色同時出現在 train/val/test）
    """
    _validate_split_ratios(val_ratio, test_ratio)
    if strategy not in {"random", "by-character"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    rng = random.Random(seed)

    known: list[ImageRecord] = [r for r in records if r.class_id is not None]
    unknown: list[ImageRecord] = [r for r in records if r.class_id is None]

    splits: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    meta: dict = {
        "strategy": strategy,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "class_map": KIND_TO_CLASS_ID,
        "input_total": len(records),
        "input_known": len(known),
        "input_unknown": len(unknown),
        "keep_unknown": keep_unknown,
    }

    if strategy == "random":
        by_class: dict[int, list[ImageRecord]] = {}
        for rec in known:
            by_class.setdefault(int(rec.class_id), []).append(rec)

        for _, group in sorted(by_class.items(), key=lambda kv: kv[0]):
            group_splits = _split_list(group, val_ratio=val_ratio, test_ratio=test_ratio, rng=rng)
            for split_name in ["train", "val", "test"]:
                splits[split_name].extend(group_splits[split_name])

        if keep_unknown and unknown:
            unknown_splits = _split_list(unknown, val_ratio=val_ratio, test_ratio=test_ratio, rng=rng)
            for split_name in ["train", "val", "test"]:
                splits[split_name].extend(unknown_splits[split_name])
        elif unknown:
            splits["unknown"] = list(unknown)

    if strategy == "by-character":
        characters = sorted({r.character for r in known if r.character is not None})
        rng.shuffle(characters)

        n_chars = len(characters)
        n_test_chars = _count_from_ratio(n_chars, test_ratio)
        n_val_chars = _count_from_ratio(n_chars, val_ratio)
        if n_test_chars + n_val_chars > n_chars:
            n_val_chars = max(0, n_chars - n_test_chars)

        test_chars = set(characters[:n_test_chars])
        val_chars = set(characters[n_test_chars : n_test_chars + n_val_chars])
        train_chars = set(characters[n_test_chars + n_val_chars :])

        meta.update(
            {
                "characters_total": n_chars,
                "characters_train": sorted(train_chars),
                "characters_val": sorted(val_chars),
                "characters_test": sorted(test_chars),
            }
        )

        for rec in known:
            if rec.character in test_chars:
                splits["test"].append(rec)
            elif rec.character in val_chars:
                splits["val"].append(rec)
            else:
                splits["train"].append(rec)

        if keep_unknown and unknown:
            for rec in unknown:
                if rec.character in test_chars:
                    splits["test"].append(rec)
                elif rec.character in val_chars:
                    splits["val"].append(rec)
                else:
                    splits["train"].append(rec)
        elif unknown:
            splits["unknown"] = list(unknown)

    meta["counts"] = {k: len(v) for k, v in splits.items()}
    return splits, meta


def kfold_split_records_by_character(
    records: Sequence[ImageRecord],
    *,
    folds: int,
    seed: int = 42,
    val_fold_offset: int = 1,
    balance_folds: bool = False,
    keep_unknown: bool = False,
) -> tuple[list[dict[str, list[ImageRecord]]], dict]:
    """
    以 character 為單位做 K-fold 切分（用於「跨角色泛化」評估）。

    - 每個 fold 都會產生 train/val/test 三份：
      - test = 第 i 個 fold 的角色
      - val = 第 (i + val_fold_offset) 個 fold 的角色
      - train = 其餘角色
    - val/test 角色都不會出現在 train（避免 leakage）

    注意：val/test 都是「未看過的角色」，這是刻意設計（domain generalization）。
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if val_fold_offset <= 0:
        raise ValueError("val_fold_offset must be >= 1")

    rng = random.Random(seed)

    known: list[ImageRecord] = [r for r in records if r.class_id is not None]
    unknown: list[ImageRecord] = [r for r in records if r.class_id is None]

    characters = sorted({r.character for r in known if r.character is not None})
    if len(characters) < 2:
        raise ValueError("Need at least 2 characters for by-character k-fold splitting.")

    rng.shuffle(characters)

    n_chars = len(characters)
    fold_sizes = [n_chars // folds + (1 if i < (n_chars % folds) else 0) for i in range(folds)]

    if balance_folds:
        # Approximate StratifiedGroupKFold:
        # - group = character
        # - target = each fold has similar per-class distribution and total images
        by_char_counts: dict[str, list[int]] = {c: [0] * len(KIND_TO_CLASS_ID) for c in characters}
        for rec in known:
            if rec.character is None or rec.class_id is None:
                continue
            if rec.character not in by_char_counts:
                continue
            cid = int(rec.class_id)
            if 0 <= cid < len(KIND_TO_CLASS_ID):
                by_char_counts[rec.character][cid] += 1

        total_by_class = [0] * len(KIND_TO_CLASS_ID)
        for c in characters:
            for cid, n in enumerate(by_char_counts.get(c, [0] * len(KIND_TO_CLASS_ID))):
                total_by_class[cid] += int(n)
        total_all = int(sum(total_by_class))

        # Stable sort: shuffle first (seeded), then sort by size desc so ties are randomized but reproducible.
        characters_sorted = list(characters)
        rng.shuffle(characters_sorted)
        characters_sorted.sort(key=lambda ch: -int(sum(by_char_counts.get(ch, []))))

        fold_chars: list[list[str]] = [[] for _ in range(folds)]
        fold_char_counts = [0] * folds
        fold_counts = [[0] * len(KIND_TO_CLASS_ID) for _ in range(folds)]
        fold_totals = [0] * folds
        target = 1.0 / float(folds)

        def _score(fold_idx: int, ch: str) -> float:
            ch_counts = by_char_counts.get(ch, [0] * len(KIND_TO_CLASS_ID))
            new_total = fold_totals[fold_idx] + int(sum(ch_counts))
            score = 0.0
            for cid in range(len(KIND_TO_CLASS_ID)):
                denom = float(total_by_class[cid]) if total_by_class[cid] > 0 else 1.0
                frac = float(fold_counts[fold_idx][cid] + int(ch_counts[cid])) / denom
                score += (frac - target) ** 2
            denom_all = float(total_all) if total_all > 0 else 1.0
            score += 0.5 * ((float(new_total) / denom_all) - target) ** 2
            return score

        for ch in characters_sorted:
            candidates = [i for i in range(folds) if fold_char_counts[i] < fold_sizes[i]]
            if not candidates:
                raise RuntimeError("No available fold capacity during balancing; please check folds.")
            best = min(candidates, key=lambda i: (_score(i, ch), fold_totals[i], i))
            fold_chars[best].append(ch)
            fold_char_counts[best] += 1
            ch_counts = by_char_counts.get(ch, [0] * len(KIND_TO_CLASS_ID))
            for cid in range(len(KIND_TO_CLASS_ID)):
                fold_counts[best][cid] += int(ch_counts[cid])
            fold_totals[best] += int(sum(ch_counts))
    else:
        fold_chars = []
        start = 0
        for size in fold_sizes:
            fold_chars.append(list(characters[start : start + size]))
            start += size

    if any(len(fc) == 0 for fc in fold_chars):
        raise ValueError(
            f"Too many folds={folds} for characters_total={n_chars}. Reduce folds or add more characters."
        )

    all_chars = set(characters)
    fold_splits: list[dict[str, list[ImageRecord]]] = []
    fold_meta: list[dict] = []

    for i in range(folds):
        test_chars = set(fold_chars[i])
        val_chars = set(fold_chars[(i + val_fold_offset) % folds])
        if val_chars & test_chars:
            raise RuntimeError("val/test overlap; please adjust val_fold_offset or folds")
        train_chars = all_chars - test_chars - val_chars
        if not train_chars:
            raise ValueError("Empty train characters; please reduce folds/val_fold_offset or add more characters.")

        splits: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
        for rec in known:
            if rec.character in test_chars:
                splits["test"].append(rec)
            elif rec.character in val_chars:
                splits["val"].append(rec)
            else:
                # character is either in train_chars, or None/unknown => default to train
                splits["train"].append(rec)

        if keep_unknown and unknown:
            for rec in unknown:
                if rec.character in test_chars:
                    splits["test"].append(rec)
                elif rec.character in val_chars:
                    splits["val"].append(rec)
                else:
                    splits["train"].append(rec)
        elif unknown:
            splits["unknown"] = list(unknown)

        fold_splits.append(splits)
        fold_meta.append(
            {
                "fold": i,
                "characters_train": sorted(train_chars),
                "characters_val": sorted(val_chars),
                "characters_test": sorted(test_chars),
                "counts": {k: len(v) for k, v in splits.items()},
            }
        )

    meta = {
        "strategy": "by-character-kfold",
        "seed": seed,
        "folds": folds,
        "val_fold_offset": val_fold_offset,
        "balance_folds": bool(balance_folds),
        "class_map": KIND_TO_CLASS_ID,
        "input_total": len(records),
        "input_known": len(known),
        "input_unknown": len(unknown),
        "keep_unknown": keep_unknown,
        "characters_total": len(characters),
        "folds_characters": fold_chars,
        "folds_meta": fold_meta,
    }
    return fold_splits, meta
