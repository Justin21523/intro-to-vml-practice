# Module 4: Classical ML Models - 傳統機器學習模型

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **線性模型**
   - 實作 Linear Regression（閉式解 + 梯度下降）
   - 實作 Logistic Regression
   - 實作 Softmax Regression

2. **SVM**
   - 理解 hinge loss
   - 用 subgradient descent 訓練 linear SVM

3. **聚類**
   - 實作 k-means
   - (選做) 實作 GMM + EM

4. **應用**
   - 組合 HOG + SVM 做影像分類

## 核心概念 (Core Concepts)

### Linear Regression

Loss: MSE
```
L = (1/N) Σ(y_pred - y)²
```

閉式解：
```
w = (X^T X)^{-1} X^T y
```

### Logistic Regression

Sigmoid：
```
σ(z) = 1 / (1 + e^{-z})
```

Loss: Binary Cross-Entropy
```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

### SVM

Hinge Loss：
```
L = max(0, 1 - y × f(x))
```

其中 y ∈ {-1, +1}

### k-Means

```
1. 初始化 K 個中心
2. 重複直到收斂：
   a. 把每個點分配給最近的中心
   b. 更新中心為群內平均
```

## 實作任務 (Implementation Tasks)

### 線性模型
- [ ] `LinearRegression.fit_closed_form()` - 閉式解
- [ ] `LinearRegression.fit_gradient_descent()` - 梯度下降
- [ ] `LogisticRegression.fit()` - Logistic 迴歸
- [ ] `SoftmaxRegression.fit()` - 多類別分類

### SVM
- [ ] `LinearSVM.hinge_loss()` - Hinge loss
- [ ] `LinearSVM.fit()` - Subgradient descent

### 聚類
- [ ] `KMeans.fit()` - k-means 聚類
- [ ] `KMeans._init_centroids()` - k-means++ 初始化

### 應用
- [ ] HOG + SVM 分類器

## 檔案結構

```
m4_classical_ml/
├── README.md
├── linear_models.py    # Linear/Logistic/Softmax Regression
├── svm.py              # Linear SVM
├── kmeans.py           # k-means
├── gmm.py              # (選做) GMM + EM
└── hog_svm_demo.py     # HOG + SVM 範例
```

## 開始學習

1. 確保已完成 Module 0（數學基礎）和 Module 3（HOG）
2. 閱讀 `prompts/m4_classical_ml.md`
3. 按順序：Linear → Logistic → Softmax → SVM → k-means
4. 每個模型都要用 `gradient_check` 驗證梯度

## 測試你的實作

```python
import numpy as np
from m4_classical_ml.linear_models import LogisticRegression
from m4_classical_ml.svm import LinearSVM

# 生成簡單資料
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X, y)
print(f"LR Accuracy: {(lr.predict(X) == y).mean():.2%}")

# SVM (注意 y 要轉成 {-1, +1})
y_svm = 2 * y - 1
svm = LinearSVM()
svm.fit(X, y_svm)
print(f"SVM Accuracy: {(svm.predict(X) == y_svm).mean():.2%}")
```

## 最終專案：HOG + SVM

```python
from m3_features.hog import compute_hog
from m4_classical_ml.svm import LinearSVM

# 假設你有行人圖片和背景圖片
# 1. 擷取 HOG 特徵
X_train = np.array([compute_hog(img) for img in train_images])
y_train = train_labels  # 1 for pedestrian, -1 for background

# 2. 訓練 SVM
svm = LinearSVM(C=1.0)
svm.fit(X_train, y_train)

# 3. 測試
X_test = np.array([compute_hog(img) for img in test_images])
predictions = svm.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"Test Accuracy: {accuracy:.2%}")
```

## 參考資源

- Bishop PRML Ch. 3-4: Linear Models
- Bishop PRML Ch. 7: SVM
- Bishop PRML Ch. 9: EM Algorithm
