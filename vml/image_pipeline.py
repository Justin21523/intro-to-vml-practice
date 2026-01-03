from __future__ import annotations

import math
import random
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .dataset_index import ImageRecord


@dataclass(frozen=True)
class ImagePipelineConfig:
    channels: str = "grayscale"  # "grayscale" | "rgb"
    shape_mode: str = "strict"  # "strict" | "pad_or_crop" | "resize"
    image_size: tuple[int, int] | None = None  # (H, W) when shape_mode != strict
    cache_dir: str | None = None  # resize 後的影像快取資料夾（僅在 shape_mode=resize 時生效）
    augment_hflip: bool = False
    augment_random_resized_crop: bool = False
    augment_rrc_prob: float = 1.0
    augment_rrc_scale: tuple[float, float] = (0.8, 1.0)  # area scale range
    augment_brightness: float = 0.0  # 0.0=off, e.g. 0.2 means factor in [0.8, 1.2]
    augment_contrast: float = 0.0  # 0.0=off, e.g. 0.2 means factor in [0.8, 1.2]
    augment_cutout: bool = False
    augment_cutout_prob: float = 0.5
    augment_cutout_scale: tuple[float, float] = (0.02, 0.2)  # area scale range


def _load_image_uint8(path: str, *, channels: str) -> np.ndarray:
    with Image.open(path) as img:
        if channels == "grayscale":
            img = img.convert("L")
        elif channels == "rgb":
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported channels: {channels}")
        return np.asarray(img, dtype=np.uint8)


def _load_image_uint8_resized(path: str, *, channels: str, target_h: int, target_w: int) -> np.ndarray:
    with Image.open(path) as img:
        if channels == "grayscale":
            img = img.convert("L")
        elif channels == "rgb":
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported channels: {channels}")

        img = img.resize((target_w, target_h), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def _cache_path_for_key(
    *,
    cache_dir: str,
    key: str,
    channels: str,
    target_h: int,
    target_w: int,
) -> Path:
    base = Path(cache_dir)
    key_path = Path(str(key))

    size_tag = f"{int(target_h)}x{int(target_w)}"
    chan_tag = str(channels)

    if key_path.is_absolute():
        digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:16]
        return base / "abs" / f"{digest}.{chan_tag}.{size_tag}.npy"

    rel = key_path
    name = f"{rel.name}.{chan_tag}.{size_tag}.npy"
    return base / rel.parent / name


def _load_image_uint8_resized_cached(
    path: str,
    *,
    key: str,
    channels: str,
    target_h: int,
    target_w: int,
    cache_dir: str,
) -> np.ndarray:
    cache_path = _cache_path_for_key(
        cache_dir=cache_dir,
        key=key,
        channels=channels,
        target_h=target_h,
        target_w=target_w,
    )
    if cache_path.exists():
        try:
            arr = np.load(cache_path, allow_pickle=False)
            arr = np.asarray(arr, dtype=np.uint8)
            return arr
        except Exception:
            try:
                cache_path.unlink()
            except Exception:
                pass

    arr = _load_image_uint8_resized(path, channels=channels, target_h=target_h, target_w=target_w)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy")
    np.save(tmp_file, arr)
    tmp_file.replace(cache_path)
    return arr


def _random_resized_crop_from_array(
    arr: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    rng: random.Random,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")

    if h <= 1 or w <= 1:
        img = Image.fromarray(arr)
        resized = img.resize((target_w, target_h), resample=Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    scale_min = float(scale_min)
    scale_max = float(scale_max)
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError(f"Invalid augment_rrc_scale: {(scale_min, scale_max)}")

    desired_aspect = target_w / target_h
    area_total = float(h * w)

    for _ in range(10):
        scale = rng.uniform(scale_min, scale_max)
        crop_area = area_total * scale
        crop_w = int(round(math.sqrt(crop_area * desired_aspect)))
        crop_h = int(round(math.sqrt(crop_area / desired_aspect)))
        crop_w = max(1, min(w, crop_w))
        crop_h = max(1, min(h, crop_h))
        if crop_w > w or crop_h > h:
            continue
        left = rng.randrange(0, w - crop_w + 1)
        top = rng.randrange(0, h - crop_h + 1)
        img = Image.fromarray(arr)
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        resized = cropped.resize((target_w, target_h), resample=Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    resized = Image.fromarray(arr).resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _center_crop_resize(img: Image.Image, *, target_h: int, target_w: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img.resize((target_w, target_h), resample=Image.BILINEAR)

    desired_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect > desired_aspect:
        crop_h = h
        crop_w = max(1, int(round(h * desired_aspect)))
    else:
        crop_w = w
        crop_h = max(1, int(round(w / desired_aspect)))

    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((target_w, target_h), resample=Image.BILINEAR)


def _load_image_uint8_random_resized_crop(
    path: str,
    *,
    channels: str,
    target_h: int,
    target_w: int,
    rng: random.Random,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    with Image.open(path) as img:
        if channels == "grayscale":
            img = img.convert("L")
        elif channels == "rgb":
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported channels: {channels}")

        w, h = img.size
        if w <= 1 or h <= 1:
            resized = img.resize((target_w, target_h), resample=Image.BILINEAR)
            return np.asarray(resized, dtype=np.uint8)

        scale_min = float(scale_min)
        scale_max = float(scale_max)
        if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
            raise ValueError(f"Invalid augment_rrc_scale: {(scale_min, scale_max)}")

        desired_aspect = target_w / target_h
        area_total = float(w * h)

        for _ in range(10):
            scale = rng.uniform(scale_min, scale_max)
            crop_area = area_total * scale
            crop_w = int(round(math.sqrt(crop_area * desired_aspect)))
            crop_h = int(round(math.sqrt(crop_area / desired_aspect)))
            if crop_w <= 0 or crop_h <= 0:
                continue
            if crop_w > w or crop_h > h:
                continue
            left = rng.randrange(0, w - crop_w + 1)
            top = rng.randrange(0, h - crop_h + 1)
            cropped = img.crop((left, top, left + crop_w, top + crop_h))
            resized = cropped.resize((target_w, target_h), resample=Image.BILINEAR)
            return np.asarray(resized, dtype=np.uint8)

        resized = _center_crop_resize(img, target_h=target_h, target_w=target_w)
        return np.asarray(resized, dtype=np.uint8)


def _hflip(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, ::-1]
    if arr.ndim == 3:
        return arr[:, ::-1, :]
    raise ValueError(f"Unsupported image ndim: {arr.ndim}")


def _pad_or_crop_to_size(
    arr: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    is_train: bool,
    rng: random.Random,
) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")

    if h > target_h:
        top = rng.randrange(0, h - target_h + 1) if is_train else (h - target_h) // 2
        arr = arr[top : top + target_h, ...]
        h = target_h
    if w > target_w:
        left = rng.randrange(0, w - target_w + 1) if is_train else (w - target_w) // 2
        arr = arr[:, left : left + target_w, ...]
        w = target_w

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if arr.ndim == 2:
        return np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
    return np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _brightness_contrast(arr: np.ndarray, *, rng: random.Random, brightness: float, contrast: float) -> np.ndarray:
    if brightness <= 0.0 and contrast <= 0.0:
        return arr

    x = arr.astype(np.float32)

    if brightness > 0.0:
        lo = max(0.0, 1.0 - float(brightness))
        hi = 1.0 + float(brightness)
        x = x * float(rng.uniform(lo, hi))

    if contrast > 0.0:
        lo = max(0.0, 1.0 - float(contrast))
        hi = 1.0 + float(contrast)
        factor = float(rng.uniform(lo, hi))
        if x.ndim == 2:
            mean = float(np.mean(x))
        elif x.ndim == 3:
            mean = np.mean(x, axis=(0, 1), keepdims=True)
        else:
            raise ValueError(f"Unsupported image ndim: {x.ndim}")
        x = mean + factor * (x - mean)

    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _cutout(
    arr: np.ndarray,
    *,
    rng: random.Random,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    scale_min = float(scale_min)
    scale_max = float(scale_max)
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError(f"Invalid augment_cutout_scale: {(scale_min, scale_max)}")

    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")

    if h <= 1 or w <= 1:
        return arr

    area_total = float(h * w)
    frac = float(rng.uniform(scale_min, scale_max))
    cut_area = max(1.0, area_total * frac)
    aspect = float(rng.uniform(0.3, 3.3))
    cut_w = int(round(math.sqrt(cut_area * aspect)))
    cut_h = int(round(math.sqrt(cut_area / aspect)))
    cut_w = max(1, min(w, cut_w))
    cut_h = max(1, min(h, cut_h))

    left = rng.randrange(0, w - cut_w + 1)
    top = rng.randrange(0, h - cut_h + 1)

    out = arr.copy()
    if out.ndim == 2:
        out[top : top + cut_h, left : left + cut_w] = 0
    else:
        out[top : top + cut_h, left : left + cut_w, :] = 0
    return out


def infer_expected_shape(records: Sequence[ImageRecord], *, config: ImagePipelineConfig) -> tuple[int, ...]:
    if config.shape_mode in {"pad_or_crop", "resize"} and config.image_size is None:
        raise ValueError("image_size must be provided when shape_mode is pad_or_crop/resize")

    if config.shape_mode in {"pad_or_crop", "resize"}:
        assert config.image_size is not None
        target_h, target_w = config.image_size
        if config.channels == "grayscale":
            return (int(target_h), int(target_w))
        if config.channels == "rgb":
            return (int(target_h), int(target_w), 3)
        raise ValueError(f"Unsupported channels: {config.channels}")

    for rec in records:
        if rec.path:
            arr = _load_image_uint8(rec.path, channels=config.channels)
            return tuple(arr.shape)
    raise ValueError("No records to infer input shape.")


def load_image_uint8_transformed(
    path: str,
    *,
    key: str,
    expected_shape: tuple[int, ...],
    config: ImagePipelineConfig,
    rng: random.Random,
    is_train: bool,
) -> np.ndarray:
    """
    讀取單張圖片並套用與 load_batch 相同的尺寸處理/augmentation，回傳 uint8 array（未正規化）。
    """
    if is_train and config.augment_random_resized_crop and config.shape_mode != "resize":
        raise ValueError("augment_random_resized_crop requires shape_mode='resize' (so output shape stays fixed).")

    target_h, target_w = int(expected_shape[0]), int(expected_shape[1])

    if config.shape_mode == "resize":
        if config.cache_dir:
            base = _load_image_uint8_resized_cached(
                path,
                key=str(key or path),
                channels=config.channels,
                target_h=target_h,
                target_w=target_w,
                cache_dir=str(config.cache_dir),
            )
            if is_train and config.augment_random_resized_crop and rng.random() < float(config.augment_rrc_prob):
                arr = _random_resized_crop_from_array(
                    base,
                    target_h=target_h,
                    target_w=target_w,
                    rng=rng,
                    scale_min=float(config.augment_rrc_scale[0]),
                    scale_max=float(config.augment_rrc_scale[1]),
                )
            else:
                arr = base
        else:
            if is_train and config.augment_random_resized_crop and rng.random() < float(config.augment_rrc_prob):
                arr = _load_image_uint8_random_resized_crop(
                    path,
                    channels=config.channels,
                    target_h=target_h,
                    target_w=target_w,
                    rng=rng,
                    scale_min=float(config.augment_rrc_scale[0]),
                    scale_max=float(config.augment_rrc_scale[1]),
                )
            else:
                arr = _load_image_uint8_resized(path, channels=config.channels, target_h=target_h, target_w=target_w)
    else:
        arr = _load_image_uint8(path, channels=config.channels)

    if is_train and config.augment_hflip and rng.random() < 0.5:
        arr = _hflip(arr)

    if config.shape_mode == "pad_or_crop":
        arr = _pad_or_crop_to_size(arr, target_h=target_h, target_w=target_w, is_train=is_train, rng=rng)

    if tuple(arr.shape) != expected_shape:
        if config.shape_mode == "strict":
            raise ValueError(
                "Inconsistent image shape. "
                f"Expected {expected_shape}, got {tuple(arr.shape)} at {path}. "
                "Tip: use --shape-mode pad_or_crop/resize with --image-size H W."
            )
        raise ValueError(
            "Image transform produced unexpected shape. "
            f"Expected {expected_shape}, got {tuple(arr.shape)} at {path}."
        )

    if is_train and (config.augment_brightness > 0.0 or config.augment_contrast > 0.0):
        arr = _brightness_contrast(
            arr,
            rng=rng,
            brightness=float(config.augment_brightness),
            contrast=float(config.augment_contrast),
        )

    if is_train and config.augment_cutout and rng.random() < float(config.augment_cutout_prob):
        arr = _cutout(
            arr,
            rng=rng,
            scale_min=float(config.augment_cutout_scale[0]),
            scale_max=float(config.augment_cutout_scale[1]),
        )

    return arr


def load_batch(
    batch: Sequence[ImageRecord],
    *,
    expected_shape: tuple[int, ...],
    dtype: np.dtype,
    config: ImagePipelineConfig,
    rng: random.Random,
    is_train: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not batch:
        raise ValueError("Empty batch")

    if is_train and config.augment_random_resized_crop and config.shape_mode != "resize":
        raise ValueError("augment_random_resized_crop requires shape_mode='resize' (so output shape stays fixed).")

    feature_dim = int(np.prod(expected_shape))
    X = np.empty((len(batch), feature_dim), dtype=dtype)
    y = np.empty((len(batch),), dtype=np.int64)

    target_h, target_w = int(expected_shape[0]), int(expected_shape[1])

    for i, rec in enumerate(batch):
        if rec.class_id is None:
            raise ValueError(f"Record has no class_id (unknown kind): {rec.relpath}")
        arr = load_image_uint8_transformed(
            rec.path,
            key=str(rec.relpath or rec.path),
            expected_shape=expected_shape,
            config=config,
            rng=rng,
            is_train=is_train,
        )

        X[i] = (arr.astype(dtype) / 255.0).reshape(-1)
        y[i] = int(rec.class_id)

    return X, y


def load_features(
    paths: Sequence[str],
    *,
    expected_shape: tuple[int, ...],
    dtype: np.dtype,
    config: ImagePipelineConfig,
    rng: random.Random,
    is_train: bool = False,
) -> np.ndarray:
    """
    讀取一組影像路徑，回傳 flatten 後的特徵（不需要 label）。
    """
    if not paths:
        raise ValueError("Empty paths")

    if is_train and config.augment_random_resized_crop and config.shape_mode != "resize":
        raise ValueError("augment_random_resized_crop requires shape_mode='resize' (so output shape stays fixed).")

    feature_dim = int(np.prod(expected_shape))
    X = np.empty((len(paths), feature_dim), dtype=dtype)

    target_h, target_w = int(expected_shape[0]), int(expected_shape[1])

    for i, path in enumerate(paths):
        arr = load_image_uint8_transformed(
            path,
            key=str(path),
            expected_shape=expected_shape,
            config=config,
            rng=rng,
            is_train=is_train,
        )

        X[i] = (arr.astype(dtype) / 255.0).reshape(-1)

    return X
