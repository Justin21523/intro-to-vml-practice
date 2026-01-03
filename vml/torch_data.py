from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset_index import ImageRecord
from .image_pipeline import ImagePipelineConfig, infer_expected_shape, load_image_uint8_transformed


@dataclass(frozen=True)
class TorchDataConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None  # None = torch default
    sample_strategy: str = "shuffle"  # shuffle|balanced_class|balanced_character|balanced_character_class


def _as_meta(rec: ImageRecord) -> dict:
    return {
        "path": rec.path,
        "relpath": rec.relpath,
        "character": rec.character,
        "kind": rec.kind,
    }


class ImageRecordTorchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        *,
        expected_shape: tuple[int, ...],
        pipeline_config: ImagePipelineConfig,
        is_train: bool,
    ) -> None:
        self.records = [r for r in records if r.class_id is not None]
        self.expected_shape = tuple(int(x) for x in expected_shape)
        self.pipeline_config = pipeline_config
        self.is_train = bool(is_train)

        if not self.records:
            raise ValueError("No records with known class_id.")

    def __len__(self) -> int:  # type: ignore[override]
        return int(len(self.records))

    def __getitem__(self, idx: int):  # type: ignore[override]
        rec = self.records[int(idx)]
        arr = load_image_uint8_transformed(
            rec.path,
            key=str(rec.relpath or rec.path),
            expected_shape=self.expected_shape,
            config=self.pipeline_config,
            rng=random,
            is_train=self.is_train,
        )
        arr = np.asarray(arr, dtype=np.uint8)
        if (not arr.flags.writeable) or any(int(s) < 0 for s in arr.strides):
            arr = arr.copy()
        if arr.ndim == 2:
            x = torch.from_numpy(arr)[None, :, :]
        elif arr.ndim == 3:
            x = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image ndim: {arr.ndim}")

        x = x.to(dtype=torch.float32).div_(255.0)
        y = int(rec.class_id)
        return x, y, _as_meta(rec)


def collate_with_meta(batch: list[tuple[torch.Tensor, int, dict]]):
    xs, ys, metas = zip(*batch)
    X = torch.stack(list(xs), dim=0)
    y = torch.tensor(list(ys), dtype=torch.int64)
    return X, y, list(metas)


def _sample_key(rec: ImageRecord, strategy: str) -> Any | None:
    if rec.class_id is None:
        return None
    if strategy == "balanced_class":
        return int(rec.class_id)
    character = rec.character or "unknown"
    if strategy == "balanced_character":
        return str(character)
    if strategy == "balanced_character_class":
        return (str(character), int(rec.class_id))
    raise ValueError(f"Unsupported sample_strategy: {strategy}")


def compute_sample_weights(records: Sequence[ImageRecord], *, sample_strategy: str) -> list[float] | None:
    sample_strategy = str(sample_strategy)
    if sample_strategy == "shuffle":
        return None
    if sample_strategy not in {"balanced_class", "balanced_character", "balanced_character_class"}:
        raise ValueError(f"Unsupported sample_strategy: {sample_strategy}")

    keys = [_sample_key(r, sample_strategy) for r in records]
    counts: dict[Any, int] = {}
    for k in keys:
        if k is None:
            continue
        counts[k] = counts.get(k, 0) + 1
    if not counts:
        return None
    return [0.0 if k is None else (1.0 / float(counts[k])) for k in keys]


def make_dataloader(
    records: Sequence[ImageRecord],
    *,
    pipeline_config: ImagePipelineConfig,
    expected_shape: tuple[int, ...],
    is_train: bool,
    data: TorchDataConfig,
) -> DataLoader:
    dataset = ImageRecordTorchDataset(
        records,
        expected_shape=expected_shape,
        pipeline_config=pipeline_config,
        is_train=is_train,
    )

    weights = compute_sample_weights(dataset.records, sample_strategy=str(data.sample_strategy))
    sampler = None
    shuffle = bool(is_train)
    if weights is not None:
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    kwargs: dict[str, Any] = {}
    if int(data.num_workers) > 0 and data.prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(data.prefetch_factor)

    return DataLoader(
        dataset,
        batch_size=int(data.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data.num_workers),
        pin_memory=bool(data.pin_memory) and torch.cuda.is_available(),
        persistent_workers=bool(data.persistent_workers) if int(data.num_workers) > 0 else False,
        collate_fn=collate_with_meta,
        **kwargs,
    )


def infer_pipeline_and_shape(
    records: Sequence[ImageRecord],
    *,
    channels: str,
    shape_mode: str,
    image_size: tuple[int, int] | None,
    cache_dir: str | None,
    augment_hflip: bool,
    augment_random_resized_crop: bool,
    augment_rrc_prob: float,
    augment_rrc_scale: tuple[float, float],
    augment_brightness: float,
    augment_contrast: float,
    augment_cutout: bool,
    augment_cutout_prob: float,
    augment_cutout_scale: tuple[float, float],
) -> tuple[ImagePipelineConfig, tuple[int, ...]]:
    pipeline = ImagePipelineConfig(
        channels=str(channels),
        shape_mode=str(shape_mode),
        image_size=image_size,
        cache_dir=cache_dir,
        augment_hflip=bool(augment_hflip),
        augment_random_resized_crop=bool(augment_random_resized_crop),
        augment_rrc_prob=float(augment_rrc_prob),
        augment_rrc_scale=tuple(float(x) for x in augment_rrc_scale),
        augment_brightness=float(augment_brightness),
        augment_contrast=float(augment_contrast),
        augment_cutout=bool(augment_cutout),
        augment_cutout_prob=float(augment_cutout_prob),
        augment_cutout_scale=tuple(float(x) for x in augment_cutout_scale),
    )
    expected_shape = infer_expected_shape(records, config=pipeline)
    return pipeline, expected_shape


def iter_records(records: Iterable[ImageRecord], *, limit: int | None = None) -> list[ImageRecord]:
    out: list[ImageRecord] = []
    n = 0
    for rec in records:
        if rec.class_id is None:
            continue
        out.append(rec)
        n += 1
        if limit is not None and n >= int(limit):
            break
    return out
