# Textbooks - 教科書推薦與章節對照

這些是學習影像處理與機器學習的核心教科書，依照本課程的 module 對應到相關章節。

---

## 1. Digital Image Processing（影像處理）

### Gonzalez & Woods - Digital Image Processing (3rd ed.)

經典影像處理教科書，涵蓋幾乎所有傳統影像處理技術。

**PDF**: [UOC SDE](https://sde.uoc.ac.in/sites/default/files/sde_videos/Digital%20Image%20Processing%203rd%20ed.%20-%20R.%20Gonzalez%2C%20R.%20Woods-ilovepdf-compressed.pdf)

| Module | 相關章節 |
|--------|----------|
| M1: Basic IP | Ch. 3: Intensity Transformations and Spatial Filtering |
| M1: Basic IP | Ch. 4: Filtering in the Frequency Domain |
| M1: Basic IP | Ch. 2.6: Geometric Spatial Transformations |
| M2: Edge Detection | Ch. 10: Image Segmentation |
| M2: Segmentation | Ch. 10.3: Thresholding (Otsu) |
| - | Ch. 9: Morphological Image Processing |

**學習建議**：
- Ch. 3 是最重要的起點，理解空間濾波
- Ch. 10 配合 Canny 和 Otsu 論文一起讀

---

## 2. Pattern Recognition and Machine Learning（機器學習）

### Bishop - PRML (2006)

傳統機器學習的「聖經」，理論深入但需要一定數學基礎。

**PDF**: [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

| Module | 相關章節 |
|--------|----------|
| M0: Math | Ch. 1: Introduction (probability review) |
| M0: Math | Ch. 2: Probability Distributions |
| M4: Classical ML | Ch. 3: Linear Models for Regression |
| M4: Classical ML | Ch. 4: Linear Models for Classification |
| M4: Classical ML | Ch. 7: Sparse Kernel Machines (SVM) |
| M4: Classical ML | Ch. 9: Mixture Models and EM |
| M4: Classical ML | Ch. 12: Principal Component Analysis |

**學習建議**：
- 不需要從頭讀到尾
- Ch. 1-2 可以跳過細節，理解大意即可
- 實作時配合對應章節查閱

---

## 3. Deep Learning（深度學習）

### Goodfellow, Bengio, Courville - Deep Learning (2016)

深度學習的「大百科」，前幾章還幫你複習數學。

**Website**: [deeplearningbook.org](https://www.deeplearningbook.org/)

| Module | 相關章節 |
|--------|----------|
| M0: Math | Part I: Ch. 2-4 (Linear Algebra, Probability, Numerical Computation) |
| M5: CNN | Ch. 6: Deep Feedforward Networks |
| M5: CNN | Ch. 7: Regularization |
| M5: CNN | Ch. 8: Optimization |
| M5: CNN | Ch. 9: Convolutional Networks |
| M6: Advanced | Ch. 7: Regularization (BatchNorm, Dropout) |
| - | Ch. 14: Autoencoders |
| - | Ch. 20: Generative Models |

**學習建議**：
- Part I (Ch. 2-4) 可以當數學參考書用
- Ch. 9 是 CNN 必讀章節
- Ch. 6, 8 理解 backprop 和 optimization

---

## 4. Computer Vision（電腦視覺）

### Szeliski - Computer Vision: Algorithms and Applications (2nd ed., 2022)

免費線上電子書，涵蓋傳統 CV 和部分深度學習。

**Website**: [szeliski.org](https://szeliski.org/Book/)

| Module | 相關章節 |
|--------|----------|
| M1: Basic IP | Ch. 3: Image Processing |
| M2: Edge Detection | Ch. 7: Feature Detection and Matching |
| M3: Features | Ch. 7: Feature Detection and Matching |
| - | Ch. 5: Geometric Primitives and Transformations |
| - | Ch. 6: Image Alignment |

**學習建議**：
- 寫作風格較現代，適合配合其他書一起看
- Ch. 7 的特徵匹配部分很實用

---

## 5. 中文資源補充

### 台大李宏毅 - 機器學習課程

YouTube 免費影片，中文講解，適合入門。

**Link**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)

| Module | 相關影片 |
|--------|----------|
| M0: Math | Regression, Gradient Descent |
| M4: Classical ML | SVM, Classification |
| M5: CNN | CNN, Backpropagation |

---

## 閱讀優先順序建議

### 優先讀（必讀）

1. **Gonzalez Ch. 3** - 空間濾波基礎
2. **Goodfellow Ch. 9** - CNN 架構
3. **Bishop Ch. 4** - 分類模型
4. **Bishop Ch. 7** - SVM

### 進階讀（選讀）

1. **Bishop Ch. 9** - EM 演算法
2. **Goodfellow Ch. 2-4** - 數學複習
3. **Szeliski Ch. 7** - 特徵匹配

### 參考用

這些書不需要系統性閱讀，實作時遇到問題再查：

- Gonzalez 的其他章節（形態學、頻率域）
- Bishop 的其他章節（Bayesian 相關）
- Goodfellow 的 Part III（研究主題）

---

## 如何搭配使用

```
開始新 Module
    ↓
讀對應的教科書章節（概念理解）
    ↓
讀相關論文（深入細節）
    ↓
動手實作 + 和 Claude 互動
    ↓
遇到問題回頭查書
```
