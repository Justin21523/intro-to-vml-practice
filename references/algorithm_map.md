# Algorithm Map - 影像與機器學習演算法地圖

這份文件整理了影像處理與機器學習的主要演算法家族，作為學習路線的參考。

---

## 1. Traditional Image Processing（傳統影像處理）

### 1.1 Spatial Domain Filtering（空間域濾波）

| Category | Methods | Description |
|----------|---------|-------------|
| Linear Filters | Gaussian, Box, Mean | 平滑去噪，卷積操作 |
| Nonlinear Filters | Median, Bilateral | 保邊去噪 |
| Sharpening | Laplacian, Unsharp mask | 增強邊緣 |

### 1.2 Frequency Domain（頻率域）

| Method | Application |
|--------|-------------|
| 2D DFT/FFT | 頻譜分析 |
| Low-pass filter | 去除高頻噪訊 |
| High-pass filter | 邊緣增強 |
| Butterworth / Gaussian | 平滑截止的濾波器 |

### 1.3 Histogram Operations（直方圖操作）

| Method | Purpose |
|--------|---------|
| Histogram Equalization | 增強對比度 |
| Histogram Specification | 匹配目標分布 |
| Adaptive Histogram Equalization (CLAHE) | 局部對比增強 |

### 1.4 Edge Detection（邊緣偵測）

| Method | Type | Key Idea |
|--------|------|----------|
| Sobel, Prewitt, Roberts | First-order gradient | 一階微分估計 |
| LoG (Laplacian of Gaussian) | Second-order | 二階微分找零交叉 |
| **Canny** | Multi-stage | Gaussian → Gradient → NMS → Hysteresis |
| Structured Edge | Learning-based | Random Forest + 邊緣特徵 |

### 1.5 Image Segmentation（影像分割）

| Method | Approach |
|--------|----------|
| **Otsu's Method** | 最大化類間變異的自動閾值 |
| Region Growing | 從種子點向外擴展相似區域 |
| Watershed | 把梯度圖當地形，找分水嶺 |
| Graph Cut | 最小割最大流，能量最小化 |
| Mean Shift | 模態搜尋，找密度峰值 |

### 1.6 Morphological Operations（形態學）

| Operation | Effect |
|-----------|--------|
| Erosion | 縮小前景 |
| Dilation | 擴大前景 |
| Opening | Erosion → Dilation，去除小雜點 |
| Closing | Dilation → Erosion，填補小洞 |

### 1.7 Geometric Transforms（幾何變換）

| Transform | Parameters |
|-----------|------------|
| Translation | tx, ty |
| Rotation | θ |
| Scaling | sx, sy |
| Affine | 6 parameters (保持平行線) |
| Projective (Homography) | 8 parameters |

---

## 2. Feature Extraction（特徵擷取）

### 2.1 Statistical Features（統計特徵）

| Feature | Description |
|---------|-------------|
| Color Histogram | RGB/HSV/Lab 顏色分布 |
| GLCM | Gray-Level Co-occurrence Matrix，紋理特徵 |
| Haralick Features | 從 GLCM 導出的統計量（對比、能量、熵等）|
| LBP | Local Binary Pattern，局部紋理描述 |

### 2.2 Keypoint Detectors（關鍵點偵測）

| Detector | Key Idea |
|----------|----------|
| Harris Corner | 結構張量的特徵值分析 |
| Shi-Tomasi | 改良 Harris，選「好追蹤」的點 |
| FAST | 比較圓周像素，快速 corner 檢測 |
| DoG (Difference of Gaussian) | SIFT 用的尺度空間極值 |

### 2.3 Feature Descriptors（特徵描述子）

| Descriptor | Dim | Key Idea |
|------------|-----|----------|
| **SIFT** | 128 | 尺度不變，方向直方圖 |
| SURF | 64/128 | 用積分影像加速的 SIFT-like |
| **HOG** | varies | 梯度方向直方圖，適合物件偵測 |
| BRIEF | 128/256 bits | Binary descriptor，快速比對 |
| ORB | 256 bits | Oriented FAST + rotated BRIEF |

---

## 3. Classical Machine Learning（傳統機器學習）

### 3.1 Supervised Learning（監督式學習）

| Model | Type | Key Idea |
|-------|------|----------|
| Linear Regression | Regression | 最小平方法 |
| Logistic Regression | Classification | Sigmoid + cross-entropy |
| k-NN | Classification | 最近鄰投票 |
| Naive Bayes | Classification | 假設特徵獨立 |
| **SVM** | Classification | 最大化 margin，kernel trick |
| Decision Tree | Both | 資訊增益分裂 |
| Random Forest | Both | Bagging + 隨機特徵子集 |
| AdaBoost | Both | 提升弱分類器 |
| Gradient Boosting | Both | 擬合殘差 |

### 3.2 Unsupervised Learning（非監督式學習）

| Model | Purpose |
|-------|---------|
| k-Means | Clustering |
| GMM + EM | Soft clustering |
| DBSCAN | Density-based clustering |
| Hierarchical Clustering | 階層式聚類 |

### 3.3 Dimensionality Reduction（降維）

| Method | Type |
|--------|------|
| PCA | Linear，最大化變異 |
| LDA | Linear，最大化類間/類內比 |
| Kernel PCA | Nonlinear |
| t-SNE | Visualization |
| UMAP | Visualization |

### 3.4 Classic Object Detection（經典物件偵測）

| Method | Key Components |
|--------|----------------|
| **Viola-Jones** | Haar features + Integral Image + AdaBoost Cascade |
| HOG + SVM | HOG 特徵 + 線性 SVM + 滑動視窗 |
| DPM | Deformable Part Models |

---

## 4. Deep Learning for Vision（深度學習影像）

### 4.1 CNN Building Blocks（CNN 基礎元件）

| Component | Function |
|-----------|----------|
| Conv2D | 特徵擷取，局部連接 |
| Pooling (Max/Avg) | 降維，空間不變性 |
| ReLU | 非線性，解決梯度消失 |
| BatchNorm | 加速訓練，正則化 |
| Dropout | 正則化，防止過擬合 |

### 4.2 Image Classification Architectures（分類架構）

| Model | Year | Innovation |
|-------|------|------------|
| LeNet-5 | 1998 | 第一個成功的 CNN |
| AlexNet | 2012 | ImageNet 突破，ReLU + Dropout |
| VGG | 2014 | 很深的 3x3 卷積堆疊 |
| GoogLeNet/Inception | 2014 | Inception module |
| **ResNet** | 2015 | Residual connection，超深網路 |
| DenseNet | 2017 | Dense connection |
| EfficientNet | 2019 | NAS + compound scaling |
| **ViT** | 2020 | Vision Transformer |

### 4.3 Object Detection（物件偵測）

| Type | Models |
|------|--------|
| Two-stage | R-CNN, Fast R-CNN, Faster R-CNN |
| One-stage | YOLO, SSD, RetinaNet |
| Anchor-free | CenterNet, FCOS |

### 4.4 Semantic Segmentation（語義分割）

| Model | Key Idea |
|-------|----------|
| FCN | Fully Convolutional |
| **U-Net** | Encoder-Decoder + Skip connections |
| DeepLab | Atrous convolution, ASPP |
| PSPNet | Pyramid Pooling |

### 4.5 Generative Models（生成模型）

| Model | Type |
|-------|------|
| Autoencoder | Reconstruction |
| VAE | Variational latent space |
| GAN | Adversarial training |
| Diffusion Models | Denoising process |

### 4.6 Self-Supervised Learning（自監督學習）

| Method | Approach |
|--------|----------|
| SimCLR | Contrastive learning |
| MoCo | Momentum contrast |
| BYOL | No negative samples |
| MAE | Masked autoencoder |

---

## 5. Motion & Video（運動與視頻）

### 5.1 Optical Flow（光流）

| Method | Type |
|--------|------|
| Lucas-Kanade | Local (sparse) |
| Horn-Schunck | Global (dense) |
| FlowNet | CNN-based |
| RAFT | Recurrent, state-of-the-art |

### 5.2 Object Tracking（物件追蹤）

| Type | Methods |
|------|---------|
| Traditional | KCF, MOSSE |
| Deep | SiamFC, SiamRPN |
| Transformer | TransTrack |

---

## Learning Path Recommendation（學習路徑建議）

```
Module 0: Math Foundations
    ↓
Module 1: Basic IP (Convolution, Filtering)
    ↓
Module 2: Edge Detection & Segmentation (Otsu, Canny)
    ↓
Module 3: Feature Descriptors (Harris, SIFT, HOG)
    ↓
Module 4: Classical ML (SVM, k-means, GMM)
    ↓
Module 5: CNN from Scratch (LeNet-like)
    ↓
Module 6: Advanced (ResNet, U-Net)
    ↓
Module 7: CPU Optimization
```
