# Module 1: Basic Image Processing - 基礎影像處理

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：用純 numpy 實作基本影像處理（卷積、濾波、幾何變換、直方圖）。

我已經有 utils/image_io.py 可以讀寫圖片，請在適當時候提醒我使用。

### 第一部分：2D Convolution（卷積）

請用以下順序教我：

1. 卷積的直覺
   - 什麼是「sliding window」
   - kernel/filter 的概念
   - 卷積 vs 相關（correlation）的差異

2. Naive 實作
   - 最簡單的雙重 for-loop 版本
   - 先不考慮 padding、stride
   - 讓我自己寫出來，你來 review

3. Padding 和 Stride
   - 為什麼需要 padding（保持尺寸、邊緣資訊）
   - valid, same, full padding
   - stride 的效果

4. 向量化
   - 把 for-loop 改成 numpy 操作
   - 或者用 im2col 的概念
   - 比較效能差異

### 第二部分：常見 Filters

用我自己寫的 convolution 實作以下濾波器：

1. 平均濾波（Box filter / Mean filter）
   - 最簡單的平滑

2. Gaussian 濾波
   - Gaussian kernel 的生成
   - σ (sigma) 的影響

3. Sobel 邊緣偵測
   - Sobel x 和 Sobel y
   - 計算 gradient magnitude
   - 這是後面 Canny 的基礎

4. Sharpen filter
   - Laplacian kernel
   - 如何增強邊緣

### 第三部分：直方圖操作

1. 計算直方圖
   - 從零用 numpy 計算（不用 np.histogram）
   - 256 bins 的灰階直方圖

2. 直方圖均衡（Histogram Equalization）
   - 累積分布函數（CDF）
   - 映射公式
   - 實作並觀察效果

### 第四部分：幾何變換

1. 影像旋轉
   - 旋轉矩陣的數學
   - 最近鄰插值（Nearest Neighbor）
   - 雙線性插值（Bilinear Interpolation）

2. 仿射變換（選做）
   - 6 個參數
   - 用矩陣表示

### 教學要求

- 每種演算法先講直覺與公式，再拆成 3~5 個明確的小步驟
- 每個步驟給我一個小任務，等我把 code 貼上來後，再幫我檢查與優化
- 不要使用 OpenCV 或 scipy.ndimage，只能用 numpy + PIL（只用來讀寫圖片）
- 讓我用真實圖片測試每個實作的效果
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 從零寫出 2D convolution
2. 理解 padding 和 stride 的作用
3. 實作 Gaussian blur, Sobel edge detection, sharpen
4. 實作直方圖均衡
5. 實作影像旋轉（含雙線性插值）

---

## 對應資源

- **Gonzalez & Woods Ch. 3**: Intensity Transformations and Spatial Filtering
- **Gonzalez & Woods Ch. 4**: Filtering in the Frequency Domain
- **Goodfellow Ch. 9.1-9.2**: Convolution Operation

---

## 程式碼骨架

### convolution.py

```python
import numpy as np

def conv2d_naive(image, kernel, padding='valid'):
    """
    Naive 2D convolution implementation.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) for grayscale
    kernel : np.ndarray
        Convolution kernel, shape (kH, kW)
    padding : str
        'valid' (no padding) or 'same' (output same size as input)

    Returns
    -------
    np.ndarray
        Convolved image
    """
    # TODO: 實作
    pass


def gaussian_kernel(size, sigma):
    """
    Generate a Gaussian kernel.

    Parameters
    ----------
    size : int
        Kernel size (should be odd)
    sigma : float
        Standard deviation

    Returns
    -------
    np.ndarray
        Gaussian kernel, shape (size, size), normalized to sum to 1
    """
    # TODO: 實作
    pass


def sobel_edge_detection(image):
    """
    Compute gradient magnitude using Sobel operators.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W)

    Returns
    -------
    magnitude : np.ndarray
        Gradient magnitude
    direction : np.ndarray
        Gradient direction in radians
    """
    # TODO: 實作
    pass
```

### histogram.py

```python
import numpy as np

def compute_histogram(image, bins=256):
    """
    Compute histogram of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, uint8
    bins : int
        Number of bins

    Returns
    -------
    hist : np.ndarray
        Histogram, shape (bins,)
    """
    # TODO: 不使用 np.histogram 實作
    pass


def histogram_equalization(image):
    """
    Perform histogram equalization.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, uint8

    Returns
    -------
    np.ndarray
        Equalized image, uint8
    """
    # TODO: 實作
    pass
```

### transforms.py

```python
import numpy as np

def rotate_image(image, angle, interpolation='nearest'):
    """
    Rotate an image by the given angle.

    Parameters
    ----------
    image : np.ndarray
        Input image
    angle : float
        Rotation angle in degrees (counter-clockwise)
    interpolation : str
        'nearest' or 'bilinear'

    Returns
    -------
    np.ndarray
        Rotated image
    """
    # TODO: 實作
    pass
```
