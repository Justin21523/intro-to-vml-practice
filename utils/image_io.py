"""
Image I/O utilities - 影像讀寫工具
==================================

最小化的圖片讀寫功能，只依賴 PIL 和 numpy。
這是為了讓你專注於演算法實作，而不是處理檔案格式。

Minimal image I/O using only PIL and numpy.
Allows you to focus on algorithm implementation, not file formats.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    讀取圖片並轉換為 numpy array (RGB format)

    Load an image and convert to numpy array in RGB format.

    Parameters
    ----------
    path : str or Path
        圖片檔案路徑

    Returns
    -------
    np.ndarray
        Shape: (H, W, 3), dtype: uint8, range: [0, 255]
        如果是灰階圖會自動轉成 RGB（三個 channel 相同）

    Example
    -------
    >>> img = load_image('test.jpg')
    >>> print(img.shape)  # (480, 640, 3)
    >>> print(img.dtype)  # uint8
    """
    img = Image.open(path)

    # 轉換成 RGB（處理 RGBA, L, P 等格式）
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return np.array(img)


def load_grayscale(path: Union[str, Path]) -> np.ndarray:
    """
    讀取圖片並轉換為灰階 numpy array

    Load an image and convert to grayscale numpy array.

    Parameters
    ----------
    path : str or Path
        圖片檔案路徑

    Returns
    -------
    np.ndarray
        Shape: (H, W), dtype: uint8, range: [0, 255]

    Example
    -------
    >>> img = load_grayscale('test.jpg')
    >>> print(img.shape)  # (480, 640)
    """
    img = Image.open(path).convert('L')
    return np.array(img)


def save_image(arr: np.ndarray, path: Union[str, Path]) -> None:
    """
    將 numpy array 儲存為圖片

    Save a numpy array as an image file.

    Parameters
    ----------
    arr : np.ndarray
        圖片資料，可以是：
        - (H, W): 灰階圖
        - (H, W, 3): RGB 圖
        - (H, W, 4): RGBA 圖

        dtype 可以是 uint8 [0, 255] 或 float [0, 1]

    path : str or Path
        輸出檔案路徑，副檔名決定格式 (.png, .jpg, etc.)

    Example
    -------
    >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> save_image(img, 'random.png')
    """
    # 處理 float 格式（假設範圍是 [0, 1]）
    if arr.dtype in [np.float32, np.float64]:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)

    # 確保是 uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # 根據 shape 決定 mode
    if arr.ndim == 2:
        mode = 'L'
    elif arr.shape[2] == 3:
        mode = 'RGB'
    elif arr.shape[2] == 4:
        mode = 'RGBA'
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    img = Image.fromarray(arr, mode=mode)
    img.save(path)


def normalize_to_float(arr: np.ndarray) -> np.ndarray:
    """
    將 uint8 [0, 255] 圖片轉換為 float [0, 1]

    Convert uint8 image to float [0, 1].

    Parameters
    ----------
    arr : np.ndarray
        uint8 圖片，range [0, 255]

    Returns
    -------
    np.ndarray
        float64 圖片，range [0, 1]
    """
    return arr.astype(np.float64) / 255.0


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    將 float [0, 1] 圖片轉換為 uint8 [0, 255]

    Convert float image to uint8 [0, 255].

    Parameters
    ----------
    arr : np.ndarray
        float 圖片，range [0, 1]

    Returns
    -------
    np.ndarray
        uint8 圖片，range [0, 255]
    """
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)
