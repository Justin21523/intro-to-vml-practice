# Key Papers - 重要論文清單

按照學習順序整理的核心論文，每篇都附上簡短摘要和為什麼要讀。

---

## 1. Edge Detection & Segmentation

### Canny Edge Detection
- **Title**: A Computational Approach to Edge Detection
- **Author**: John Canny
- **Venue**: IEEE TPAMI, 1986
- **Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/B9780080515816500246)
- **為什麼讀**: 定義了好的邊緣偵測器的三個準則：低錯誤率、精確定位、單一響應。Canny 演算法至今仍是最常用的邊緣偵測方法。

### Otsu's Method
- **Title**: A Threshold Selection Method from Gray-Level Histograms
- **Author**: Nobuyuki Otsu
- **Venue**: IEEE TSMC, 1979
- **Link**: [Purdue](https://engineering.purdue.edu/kak/computervision/ECE661.08/OTSU_paper.pdf)
- **為什麼讀**: 最經典的自動閾值選擇方法，數學推導簡潔優雅，是學習變異數分解的好範例。

---

## 2. Feature Descriptors

### SIFT
- **Title**: Distinctive Image Features from Scale-Invariant Keypoints
- **Author**: David Lowe
- **Venue**: IJCV, 2004
- **Link**: [SpringerLink](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)
- **為什麼讀**: 尺度不變特徵的里程碑，詳細解釋了尺度空間、關鍵點定位、方向指派、描述子計算。雖然現在常用深度特徵，但理解 SIFT 對理解特徵設計思路很有幫助。

### HOG
- **Title**: Histograms of Oriented Gradients for Human Detection
- **Author**: Navneet Dalal, Bill Triggs
- **Venue**: CVPR, 2005
- **Link**: [INRIA](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
- **為什麼讀**: 行人偵測的經典方法，HOG+SVM 是深度學習之前的 state-of-the-art。論文詳細分析了各種設計選擇對性能的影響。

---

## 3. Classical Object Detection

### Viola-Jones Face Detector
- **Title**: Rapid Object Detection using a Boosted Cascade of Simple Features
- **Author**: Paul Viola, Michael Jones
- **Venue**: CVPR, 2001
- **Link**: [CMU](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- **為什麼讀**: 第一個實時人臉偵測器，整合了積分影像、AdaBoost、cascade 結構三個巧妙的加速技術。

---

## 4. Deep Learning Foundations

### LeNet / CNN Origins
- **Title**: Gradient-Based Learning Applied to Document Recognition
- **Author**: Yann LeCun et al.
- **Venue**: Proceedings of IEEE, 1998
- **Link**: [Stanford](https://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- **為什麼讀**: CNN 的開山之作，不只是架構，還詳細討論了訓練技術和 graph transformer networks。

### AlexNet
- **Title**: ImageNet Classification with Deep Convolutional Neural Networks
- **Author**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **Venue**: NeurIPS, 2012
- **Link**: [NeurIPS](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **為什麼讀**: 引爆深度學習革命的論文，展示了 GPU 訓練、ReLU、Dropout、Data Augmentation 的威力。

### ResNet
- **Title**: Deep Residual Learning for Image Recognition
- **Author**: Kaiming He et al.
- **Venue**: CVPR, 2016
- **Link**: [arXiv](https://arxiv.org/abs/1512.03385)
- **為什麼讀**: Residual connection 解決了深層網路訓練困難的問題，是現代網路架構的基礎。

---

## 5. Segmentation

### U-Net
- **Title**: U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Author**: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- **Venue**: MICCAI, 2015
- **Link**: [arXiv](https://arxiv.org/abs/1505.04597)
- **為什麼讀**: Encoder-decoder + skip connection 的經典設計，在醫學影像分割中極為成功，架構簡潔易懂。

---

## 6. Vision Transformer

### ViT
- **Title**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **Author**: Alexey Dosovitskiy et al.
- **Venue**: ICLR, 2021
- **Link**: [arXiv](https://arxiv.org/abs/2010.11929)
- **為什麼讀**: 證明了純 Transformer 架構可以在影像分類上達到甚至超越 CNN 的效果，開啟了 Vision Transformer 時代。

---

## 7. Surveys & Reviews（綜述）

### Traditional vs Deep Features
- **Title**: A survey of traditional and deep learning-based feature descriptors for high dimensional data in computer vision
- **Author**: Theodoros Georgiou et al.
- **Venue**: International Journal of Multimedia Information Retrieval, 2020
- **Link**: [SpringerLink](https://link.springer.com/article/10.1007/s13735-019-00183-w)
- **為什麼讀**: 系統性比較傳統特徵（SIFT, HOG 等）和深度學習特徵。

### Image Feature Extraction (2025)
- **Title**: Image feature extraction techniques: A comprehensive review
- **Author**: Vinayak Hallur et al.
- **Venue**: Intelligent Systems with Applications, 2025
- **Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2773186325001549)
- **為什麼讀**: 最新的特徵擷取方法總覽，涵蓋幾何、統計、紋理、顏色、深度特徵。

### Edge Detection Survey
- **Title**: Survey of Image Edge Detection
- **Venue**: Frontiers in Signal Processing, 2022
- **Link**: [Frontiers](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2022.826967/full)
- **為什麼讀**: 把邊緣偵測方法分成 gradient-based、LoG、multi-scale、learning-based 四類。

---

## Reading Strategy（閱讀策略）

1. **第一遍**：只讀 Abstract 和 Introduction，了解問題和貢獻
2. **第二遍**：看圖表和 Method 概述，理解整體架構
3. **第三遍**：詳細讀 Method，對照程式碼實作
4. **第四遍**：讀 Experiments 和 Ablation，理解設計選擇

不需要一次讀完所有論文。跟著 module 進度，需要時再深入閱讀。
