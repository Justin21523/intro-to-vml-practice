# Module 1: Basic Image Processing - 基礎影像處理

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **2D Convolution**
   - 理解卷積的直覺（滑動視窗 + 加權和）
   - 實作 naive 版本（雙重迴圈）
   - 理解 padding 和 stride 的作用
   - 向量化實作

2. **濾波器**
   - 實作 Gaussian blur
   - 實作 Sobel 邊緣偵測
   - 實作 sharpening filter

3. **直方圖操作**
   - 計算灰階直方圖
   - 實作直方圖均衡

4. **幾何變換**
   - 實作影像旋轉
   - 理解最近鄰插值和雙線性插值

## 核心概念 (Core Concepts)

### 2D Convolution
```
output[i,j] = Σₘ Σₙ input[i+m, j+n] × kernel[m, n]
```

### Gaussian Kernel
```
G(x,y) = (1/2πσ²) × exp(-(x² + y²)/(2σ²))
```

### Sobel Operators
```
Sx = [-1 0 1]    Sy = [-1 -2 -1]
     [-2 0 2]         [ 0  0  0]
     [-1 0 1]         [ 1  2  1]
```

## 實作任務 (Implementation Tasks)

- [ ] `conv2d_naive(image, kernel)` - 基本卷積
- [ ] `conv2d_vectorized(image, kernel)` - 向量化卷積
- [ ] `gaussian_kernel(size, sigma)` - 生成 Gaussian kernel
- [ ] `gaussian_blur(image, sigma)` - Gaussian 模糊
- [ ] `sobel_edge_detection(image)` - Sobel 邊緣偵測
- [ ] `compute_histogram(image)` - 計算直方圖
- [ ] `histogram_equalization(image)` - 直方圖均衡
- [ ] `rotate_image(image, angle)` - 影像旋轉

## 檔案結構

```
m1_basic_ip/
├── README.md
├── convolution.py      # 卷積實作
├── filters.py          # 各種濾波器
├── histogram.py        # 直方圖操作
├── transforms.py       # 幾何變換
└── tests/
    └── test_conv.py
```

## 開始學習

1. 確保已完成 Module 0
2. 閱讀 `prompts/m1_basic_ip.md`
3. 準備幾張測試圖片放在 `data/` 資料夾
4. 使用 `utils/image_io.py` 讀取圖片

## 測試你的實作

```python
from utils.image_io import load_grayscale, save_image
from m1_basic_ip.convolution import conv2d_naive, gaussian_kernel

# 讀取圖片
img = load_grayscale('data/test.jpg')

# 生成 Gaussian kernel
kernel = gaussian_kernel(5, 1.0)

# 卷積
blurred = conv2d_naive(img.astype(float), kernel)

# 儲存結果
save_image(blurred, 'output/blurred.jpg')
```

## 參考資源

- Gonzalez & Woods Ch. 3: Spatial Filtering
- Gonzalez & Woods Ch. 4: Frequency Domain Filtering
