# Module 3: Feature Descriptors - 特徵描述子

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **Harris Corner Detector**
   - 理解結構張量的意義
   - 實作 Harris response function
   - 找出角點位置

2. **SIFT (簡化版)**
   - 理解尺度空間的概念
   - 建立 Gaussian pyramid
   - 計算 DoG 並找極值

3. **HOG**
   - 完整實作 HOG 特徵
   - 理解 cell, block, normalization

## 核心概念 (Core Concepts)

### Harris Corner Detector

結構張量：
```
M = Σ w(x,y) × [Ix²    IxIy]
               [IxIy   Iy² ]
```

Harris response：
```
R = det(M) - k × trace(M)²
  = λ₁λ₂ - k(λ₁ + λ₂)²
```

### Scale Space

為什麼需要：物體在不同距離下大小不同

Gaussian Pyramid：
```
L(x, y, σ) = G(x, y, σ) * I(x, y)
```

DoG 近似 LoG：
```
DoG = L(x, y, kσ) - L(x, y, σ)
```

### HOG Pipeline

```
1. (Optional) Gamma correction
2. Compute gradients (Gx, Gy)
3. Compute magnitude and direction
4. Divide into cells (e.g., 8×8 pixels)
5. For each cell: compute orientation histogram (9 bins)
6. Group cells into blocks (e.g., 2×2 cells)
7. Normalize each block
8. Concatenate all normalized blocks
```

## 實作任務 (Implementation Tasks)

### Harris
- [ ] `compute_structure_tensor(image)` - 計算結構張量
- [ ] `harris_response(M, k)` - 計算 Harris response
- [ ] `harris_corner_detector(image)` - 完整 Harris

### SIFT (簡化版)
- [ ] `build_gaussian_pyramid(image)` - 建立 Gaussian pyramid
- [ ] `compute_dog(pyramid)` - 計算 DoG
- [ ] `find_extrema(dog)` - 找尺度空間極值

### HOG
- [ ] `compute_cell_histogram(magnitude, direction)` - 計算 cell 直方圖
- [ ] `normalize_block(block)` - Block normalization
- [ ] `compute_hog(image)` - 完整 HOG

## 檔案結構

```
m3_features/
├── README.md
├── harris.py           # Harris corner detector
├── sift_simplified.py  # 簡化版 SIFT
├── hog.py              # HOG 特徵
└── tests/
    └── test_hog.py
```

## 開始學習

1. 確保已完成 Module 1 和 2
2. 閱讀 `prompts/m3_features.md`
3. Harris 相對簡單，先完成它
4. HOG 是最重要的，Module 4 會用到

## 視覺化 HOG

```python
def visualize_hog(image, cell_size=8, num_bins=9):
    """
    視覺化 HOG 特徵（每個 cell 畫出主要梯度方向）
    """
    # TODO: 實作視覺化
    pass
```

## 參考資源

- SIFT: Lowe, IJCV 2004
- HOG: Dalal & Triggs, CVPR 2005
- Szeliski Ch. 7: Feature Detection and Matching
