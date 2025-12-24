# Module 2: Edge Detection & Segmentation - 邊緣偵測與分割

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **Otsu's Method**
   - 理解類內/類間變異數的概念
   - 推導 Otsu 閾值選擇公式
   - 實作自動閾值分割

2. **Canny Edge Detection**
   - 理解 Canny 的四個步驟
   - 實作 non-maximum suppression
   - 實作 hysteresis thresholding

## 核心概念 (Core Concepts)

### Otsu's Method

目標：找閾值 T 最大化類間變異數

```
σ_B²(T) = ω₀(T) × ω₁(T) × [μ₀(T) - μ₁(T)]²
```

其中：
- ω₀, ω₁：兩類的權重（像素比例）
- μ₀, μ₁：兩類的平均灰度值

### Canny Edge Detection Pipeline

```
1. Gaussian Smoothing
   ↓
2. Gradient Computation (Sobel)
   ↓
3. Non-Maximum Suppression
   ↓
4. Hysteresis Thresholding
```

### Non-Maximum Suppression

目的：把粗邊緣細化成單像素寬

```
對每個像素：
1. 找梯度方向（量化到 0°, 45°, 90°, 135°）
2. 和該方向上的兩個鄰居比較
3. 只保留局部最大值
```

### Hysteresis Thresholding

```
- Strong edge: magnitude > high_threshold
- Weak edge: low_threshold < magnitude ≤ high_threshold
- Weak edge 只有連接到 strong edge 時才保留
```

## 實作任務 (Implementation Tasks)

- [ ] `otsu_threshold(image)` - Otsu 自動閾值
- [ ] `apply_threshold(image, threshold)` - 二值化
- [ ] `non_maximum_suppression(magnitude, direction)` - NMS
- [ ] `hysteresis_threshold(image, low, high)` - 雙閾值
- [ ] `canny_edge_detector(image, sigma, low, high)` - 完整 Canny

## 檔案結構

```
m2_edge_seg/
├── README.md
├── otsu.py             # Otsu 閾值
├── canny.py            # Canny 邊緣偵測
└── tests/
    ├── test_otsu.py
    └── test_canny.py
```

## 開始學習

1. 確保已完成 Module 1（需要 convolution, Sobel）
2. 閱讀 `prompts/m2_edge_seg.md`
3. 先從 Otsu 開始，比較簡單
4. Canny 的 NMS 是最難的部分，用小矩陣手算驗證

## 測試你的實作

```python
from utils.image_io import load_grayscale, save_image
from m2_edge_seg.otsu import otsu_threshold, apply_threshold
from m2_edge_seg.canny import canny_edge_detector

img = load_grayscale('data/test.jpg')

# Otsu
threshold = otsu_threshold(img)
binary = apply_threshold(img, threshold)
save_image(binary, 'output/otsu.jpg')

# Canny
edges = canny_edge_detector(img, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
save_image(edges, 'output/canny.jpg')
```

## 參考資源

- Canny 論文: IEEE TPAMI 1986
- Otsu 論文: IEEE TSMC 1979
- Gonzalez & Woods Ch. 10: Image Segmentation
