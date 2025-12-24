# Module 3: Feature Descriptors - 特徵描述子

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：關鍵點偵測與特徵描述子（Harris、SIFT、HOG）。

我已經完成 Module 1 和 Module 2，有自己的 convolution、Gaussian blur、Sobel、Canny 實作。

### 第一部分：Harris Corner Detector

1. 問題定義
   - Corner（角點）的直覺：往任何方向移動，灰階都會變化
   - Edge：只有一個方向變化
   - Flat：任何方向都不變

2. Structure Tensor（結構張量）
   - M = [Σ Ix²    Σ IxIy]
       [Σ IxIy   Σ Iy² ]
   - 其中 Ix, Iy 是圖像的 x, y 方向梯度
   - Σ 是在 window 內的加權和（通常用 Gaussian 加權）

3. Corner Response Function
   - Harris 用 R = det(M) - k * trace(M)²
   - k 通常取 0.04-0.06
   - R 大的地方是角點

4. 實作步驟
   - 計算 Ix, Iy（用 Sobel）
   - 計算 Ix², Iy², IxIy
   - 用 Gaussian 對上述三個做平滑
   - 計算 R
   - Non-maximum suppression 找局部最大值

### 第二部分：簡化版 SIFT

不需要完整實作 SIFT（那很複雜），但要理解核心概念：

1. 尺度空間（Scale Space）
   - 為什麼需要：物件在不同距離下大小不同
   - Gaussian pyramid：不同 σ 的 Gaussian blur
   - 實作：生成一個簡單的 3-4 層 pyramid

2. DoG (Difference of Gaussian)
   - 相鄰尺度的 Gaussian 影像相減
   - 近似 Laplacian of Gaussian
   - 找 DoG 的極值點

3. 方向直方圖（Orientation Histogram）
   - 在關鍵點周圍 16x16 區域
   - 計算梯度方向直方圖（8 bins）
   - 找主方向（peak）

4. 簡化版 Descriptor
   - 以關鍵點為中心，取 4x4 的區域
   - 每個區域計算 8-bin 方向直方圖
   - 總共得到 4x4x8 = 128 維描述子

### 第三部分：HOG (Histograms of Oriented Gradients)

1. HOG 的應用
   - 最初用於行人偵測
   - 能抓住物體的形狀信息

2. Cell 和 Block
   - 把圖像分成小的 cell（例如 8x8 pixels）
   - 每個 cell 計算梯度方向直方圖（9 bins, 0°-180°）
   - 用 2x2 cells 組成 block

3. Block Normalization
   - 為什麼需要：處理光照變化
   - L2-norm: v / √(||v||² + ε)

4. 完整 HOG Pipeline
   - (Optional) Gamma correction
   - Compute gradients
   - Compute cell histograms
   - Block normalize
   - Collect into feature vector

### 教學要求

- Harris 要讓我完整實作
- SIFT 只需要實作簡化版：Gaussian pyramid + DoG + 找極值
- HOG 要完整實作，因為後面 Module 4 要用 HOG+SVM 做行人偵測
- 每一種方法都要連結回原始論文的想法，但用直覺語言解釋
- 請幫我把「公式 → 程式碼」的對照寫清楚
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 實作 Harris corner detector
2. 理解 SIFT 的尺度空間和 DoG 原理
3. 完整實作 HOG 特徵擷取
4. 理解為什麼這些手工特徵能描述影像內容

---

## 對應資源

- **SIFT 論文**: Lowe, IJCV 2004
- **HOG 論文**: Dalal & Triggs, CVPR 2005
- **Szeliski Ch. 7**: Feature Detection and Matching

---

## 程式碼骨架

### harris.py

```python
import numpy as np

def harris_corner_detector(image, window_size=5, k=0.05, threshold=0.01):
    """
    Detect corners using Harris corner detector.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W)
    window_size : int
        Size of the Gaussian window for computing structure tensor
    k : float
        Harris parameter (typically 0.04-0.06)
    threshold : float
        Threshold for corner response (as fraction of max response)

    Returns
    -------
    corners : list of (y, x)
        List of corner coordinates
    response : np.ndarray
        Corner response map
    """
    # TODO: 實作
    pass
```

### sift_simplified.py

```python
import numpy as np

def build_gaussian_pyramid(image, num_octaves=4, num_scales=5, sigma=1.6):
    """
    Build Gaussian pyramid.

    Parameters
    ----------
    image : np.ndarray
        Input image
    num_octaves : int
        Number of octaves (image sizes)
    num_scales : int
        Number of scales per octave
    sigma : float
        Base sigma

    Returns
    -------
    pyramid : list of list of np.ndarray
        pyramid[octave][scale] is a blurred image
    """
    # TODO: 實作
    pass


def compute_dog(pyramid):
    """
    Compute Difference of Gaussian from Gaussian pyramid.

    Parameters
    ----------
    pyramid : list of list of np.ndarray
        Gaussian pyramid

    Returns
    -------
    dog : list of list of np.ndarray
        DoG pyramid
    """
    # TODO: 實作
    pass


def find_extrema(dog):
    """
    Find local extrema in DoG scale space.

    Parameters
    ----------
    dog : list of list of np.ndarray
        DoG pyramid

    Returns
    -------
    keypoints : list of (octave, scale, y, x)
        Detected keypoint locations
    """
    # TODO: 實作
    pass
```

### hog.py

```python
import numpy as np

def compute_hog(image, cell_size=8, block_size=2, num_bins=9):
    """
    Compute HOG features.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W)
        H and W should be multiples of cell_size
    cell_size : int
        Size of each cell in pixels
    block_size : int
        Size of each block in cells
    num_bins : int
        Number of orientation bins

    Returns
    -------
    features : np.ndarray
        HOG feature vector
    """
    # TODO: 實作
    pass


def compute_cell_histogram(magnitude, direction, num_bins=9):
    """
    Compute orientation histogram for a cell.

    Parameters
    ----------
    magnitude : np.ndarray
        Gradient magnitude for the cell, shape (cell_size, cell_size)
    direction : np.ndarray
        Gradient direction in degrees (0-180), shape (cell_size, cell_size)
    num_bins : int
        Number of bins

    Returns
    -------
    hist : np.ndarray
        Histogram, shape (num_bins,)
    """
    # TODO: 實作
    pass


def normalize_block(block, eps=1e-5):
    """
    L2-normalize a block of cell histograms.

    Parameters
    ----------
    block : np.ndarray
        Block features
    eps : float
        Small constant for numerical stability

    Returns
    -------
    np.ndarray
        Normalized block
    """
    # TODO: 實作
    pass
```
