# Module 2: Edge Detection & Segmentation - 邊緣偵測與分割

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：從零手刻 Otsu 閾值分割與 Canny 邊緣偵測。

我已經完成 Module 1，有自己的 convolution、Gaussian blur、Sobel 實作。

### 第一部分：Otsu's Method（大津法）

1. 問題定義
   - 二值化分割：找一個閾值 T，把像素分成兩類
   - 好的閾值應該讓類內變異最小（或類間變異最大）

2. 數學推導
   - 帶我推導類間變異數 (between-class variance) σ_B²
   - 說明為什麼最大化 σ_B² 等於最小化類內變異 σ_W²
   - 公式：σ_B²(T) = ω₀(T) ω₁(T) [μ₀(T) - μ₁(T)]²

3. 實作
   - 遍歷所有可能的 T (0-255)
   - 計算每個 T 對應的 σ_B²
   - 找最大值對應的 T

4. 測試
   - 用不同圖片測試
   - 和我預期的分割結果比較

### 第二部分：Canny Edge Detection

請用以下順序帶我完成完整的 Canny 實作：

#### Step 1: Gaussian Smoothing
- 使用我 Module 1 寫的 Gaussian blur
- 為什麼要先平滑：減少噪點對梯度的影響

#### Step 2: Gradient Computation
- 使用 Sobel 計算 Gx, Gy
- 計算 magnitude: |G| = √(Gx² + Gy²)
- 計算 direction: θ = atan2(Gy, Gx)

#### Step 3: Non-Maximum Suppression (NMS)
- 這一步的目的：把粗邊緣細化成單像素寬
- 直覺：只保留梯度方向上的局部最大值
- 把梯度方向量化成 4 個方向（0°, 45°, 90°, 135°）
- 比較當前像素和它梯度方向上兩個鄰居的大小

#### Step 4: Hysteresis Thresholding（雙閾值）
- 使用兩個閾值：high threshold 和 low threshold
- Strong edge: magnitude > high
- Weak edge: low < magnitude <= high
- 連接規則：weak edge 只有在連接到 strong edge 時才保留

### 教學要求

- Step 3 (NMS) 是最難的部分，請用小矩陣例子讓我手算一次
- Step 4 可以用 BFS/DFS 或者迭代方式實作
- 每一步都讓我先實作，你來 review
- 不要直接給完整程式，只給關鍵片段與框架
- 每一步都要解釋「為什麼 Canny 要這樣做」
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 解釋 Otsu 法的數學原理並實作
2. 完整實作 Canny edge detector 的所有步驟
3. 理解 non-maximum suppression 的作用
4. 理解 hysteresis thresholding 如何處理 weak edges

---

## 對應資源

- **Canny 論文**: A Computational Approach to Edge Detection, IEEE TPAMI 1986
- **Otsu 論文**: A Threshold Selection Method from Gray-Level Histograms, IEEE TSMC 1979
- **Gonzalez & Woods Ch. 10.3**: Thresholding

---

## 程式碼骨架

### otsu.py

```python
import numpy as np

def otsu_threshold(image):
    """
    Find optimal threshold using Otsu's method.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, uint8, shape (H, W)

    Returns
    -------
    threshold : int
        Optimal threshold value (0-255)

    Algorithm
    ---------
    1. Compute histogram
    2. For each possible threshold T:
       - Compute weights ω₀, ω₁
       - Compute means μ₀, μ₁
       - Compute between-class variance σ_B²
    3. Return T that maximizes σ_B²
    """
    # TODO: 實作
    pass


def apply_threshold(image, threshold):
    """
    Apply binary threshold to image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image
    threshold : int
        Threshold value

    Returns
    -------
    np.ndarray
        Binary image (0 or 255)
    """
    # TODO: 實作
    pass
```

### canny.py

```python
import numpy as np

def canny_edge_detector(image, sigma=1.0, low_threshold=0.1, high_threshold=0.3):
    """
    Canny edge detection.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W)
    sigma : float
        Gaussian kernel sigma for smoothing
    low_threshold : float
        Low threshold for hysteresis (as fraction of max gradient)
    high_threshold : float
        High threshold for hysteresis (as fraction of max gradient)

    Returns
    -------
    edges : np.ndarray
        Binary edge map
    """
    # Step 1: Gaussian smoothing
    smoothed = gaussian_blur(image, sigma)

    # Step 2: Gradient computation
    magnitude, direction = compute_gradient(smoothed)

    # Step 3: Non-maximum suppression
    nms = non_maximum_suppression(magnitude, direction)

    # Step 4: Hysteresis thresholding
    edges = hysteresis_threshold(nms, low_threshold, high_threshold)

    return edges


def non_maximum_suppression(magnitude, direction):
    """
    Apply non-maximum suppression to thin edges.

    Parameters
    ----------
    magnitude : np.ndarray
        Gradient magnitude, shape (H, W)
    direction : np.ndarray
        Gradient direction in radians, shape (H, W)

    Returns
    -------
    np.ndarray
        Thinned edge map

    Algorithm
    ---------
    For each pixel:
    1. Quantize gradient direction to 0°, 45°, 90°, 135°
    2. Compare with two neighbors along gradient direction
    3. Keep only if it's the local maximum
    """
    # TODO: 實作
    pass


def hysteresis_threshold(image, low_thresh, high_thresh):
    """
    Apply hysteresis thresholding.

    Parameters
    ----------
    image : np.ndarray
        NMS result
    low_thresh : float
        Low threshold
    high_thresh : float
        High threshold

    Returns
    -------
    np.ndarray
        Binary edge map

    Algorithm
    ---------
    1. Mark strong edges (> high_thresh)
    2. Mark weak edges (low_thresh < x <= high_thresh)
    3. Keep weak edges only if connected to strong edges
    """
    # TODO: 實作
    pass
```

---

## NMS 手算範例

考慮這個 3x3 的 magnitude 和 direction：

```
Magnitude:          Direction (degrees):
10  20  15          45   0   45
25  50  30          90   0   90
15  20  10          135  0   135
```

對於中心像素 (magnitude=50, direction=0°)：
- 梯度方向是水平（0°）
- 要和左右鄰居比較：50 vs 25, 50 vs 30
- 50 > 25 且 50 > 30，所以保留

對於右上角像素 (magnitude=15, direction=45°)：
- 梯度方向是 45°
- 要和左下、右上鄰居比較
- ...
