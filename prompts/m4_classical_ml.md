# Module 4: Classical ML Models - 傳統機器學習模型

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：用純 numpy 手刻傳統機器學習模型，並應用在影像特徵上。

我已經完成前面的 modules，有自己的 HOG 特徵擷取器。現在要學習如何用 ML 模型來分類這些特徵。

### 第一部分：線性回歸

先從最簡單的模型開始：

1. 問題設定
   - y = Xw + b
   - 損失函數：MSE = (1/N) Σ(y_pred - y)²

2. 閉式解（Normal Equation）
   - w = (X^T X)^{-1} X^T y
   - 為什麼這是解（對 w 微分 = 0）
   - 什麼時候會失敗（X^T X 不可逆）

3. 梯度下降解法
   - ∂L/∂w = ?
   - 實作並和閉式解比較

4. 正則化
   - Ridge (L2): + λ||w||²
   - 對閉式解的影響：(X^T X + λI)^{-1} X^T y

### 第二部分：邏輯迴歸（Logistic Regression）

1. 二分類問題
   - Sigmoid function: σ(z) = 1 / (1 + e^{-z})
   - 輸出解釋為機率 P(y=1|x)

2. 損失函數：Binary Cross-Entropy
   - L = -[y log(ŷ) + (1-y) log(1-ŷ)]
   - 為什麼不用 MSE

3. 梯度推導
   - 推導 ∂L/∂w 和 ∂L/∂b
   - 用 gradient_check.py 驗證

4. 實作
   - 用梯度下降訓練
   - 在簡單資料集上測試

### 第三部分：Softmax Regression（多分類）

1. Softmax function
   - 將 K 個分數轉成 K 個機率
   - 數值穩定性：減去最大值

2. Cross-Entropy Loss
   - L = -Σ y_k log(ŷ_k)
   - 其中 y 是 one-hot vector

3. 梯度推導
   - ∂L/∂z = ŷ - y （很漂亮的結果！）

4. 實作
   - 用 MNIST 或類似的小型資料集測試

### 第四部分：SVM（支持向量機）

1. 線性 SVM 的直覺
   - 找最大間隔的超平面
   - Support vectors 的意義

2. Hinge Loss
   - L = max(0, 1 - y·f(x))
   - y ∈ {-1, +1}

3. SVM 優化問題
   - min (1/N) Σ max(0, 1 - y_i(w·x_i + b)) + λ||w||²
   - 用 subgradient descent 求解

4. 實作
   - 先在 2D 資料上視覺化
   - 再用 HOG 特徵做分類

### 第五部分：k-Means Clustering

1. 演算法
   - 初始化 K 個中心
   - 重複：分配點到最近中心 → 更新中心為群內平均

2. 初始化問題
   - 隨機初始化可能陷入局部最優
   - k-means++ 初始化

3. 實作與視覺化
   - 在 2D 資料上看聚類效果
   - 用在影像壓縮（color quantization）

### 第六部分：GMM + EM（選做，較進階）

1. Gaussian Mixture Model
   - K 個 Gaussian 的加權和
   - 軟分配 vs k-means 的硬分配

2. EM 演算法
   - E-step: 計算每個點屬於各 component 的責任
   - M-step: 更新參數

### 最終專案：HOG + SVM 分類器

把前面學的組合起來：
1. 用 HOG 擷取影像特徵
2. 用 SVM 分類
3. 在簡單資料集（例如行人 vs 背景）上測試

### 教學要求

- 每個模型都要先寫出 loss function
- 帶我一步步推導梯度，用 gradient_check 驗證
- 不使用 sklearn 等現成實作
- 每個模型都要有視覺化（2D 情況下畫決策邊界）
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 從零實作 Linear Regression, Logistic Regression, Softmax
2. 理解並實作 SVM (hinge loss + gradient descent)
3. 實作 k-means clustering
4. 組合 HOG + SVM 做簡單的影像分類

---

## 對應資源

- **Bishop PRML Ch. 3**: Linear Models for Regression
- **Bishop PRML Ch. 4**: Linear Models for Classification
- **Bishop PRML Ch. 7**: Sparse Kernel Machines (SVM)
- **Bishop PRML Ch. 9**: Mixture Models and EM

---

## 程式碼骨架

### linear_models.py

```python
import numpy as np

class LinearRegression:
    def __init__(self, regularization=0.0):
        self.w = None
        self.b = None
        self.reg = regularization

    def fit_closed_form(self, X, y):
        """Fit using normal equation."""
        # TODO: 實作 (X^T X + λI)^{-1} X^T y
        pass

    def fit_gradient_descent(self, X, y, lr=0.01, n_iter=1000):
        """Fit using gradient descent."""
        # TODO: 實作
        pass

    def predict(self, X):
        # TODO: 實作
        pass


class LogisticRegression:
    def __init__(self, regularization=0.0):
        self.w = None
        self.b = None
        self.reg = regularization

    def sigmoid(self, z):
        # TODO: 實作（注意數值穩定性）
        pass

    def fit(self, X, y, lr=0.01, n_iter=1000):
        """Fit using gradient descent."""
        # TODO: 實作
        pass

    def predict_proba(self, X):
        # TODO: 實作
        pass

    def predict(self, X, threshold=0.5):
        # TODO: 實作
        pass


class SoftmaxRegression:
    def __init__(self, n_classes, regularization=0.0):
        self.n_classes = n_classes
        self.W = None  # shape: (n_features, n_classes)
        self.b = None  # shape: (n_classes,)
        self.reg = regularization

    def softmax(self, z):
        # TODO: 實作（注意數值穩定性）
        pass

    def fit(self, X, y, lr=0.01, n_iter=1000):
        """
        Fit softmax regression.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)
            Class labels (integers 0 to K-1)
        """
        # TODO: 實作
        pass

    def predict(self, X):
        # TODO: 實作
        pass
```

### svm.py

```python
import numpy as np

class LinearSVM:
    def __init__(self, C=1.0):
        """
        Linear SVM with hinge loss.

        Parameters
        ----------
        C : float
            Regularization parameter (smaller = more regularization)
        """
        self.C = C
        self.w = None
        self.b = None

    def hinge_loss(self, X, y):
        """
        Compute hinge loss.

        L = (1/N) Σ max(0, 1 - y_i * (w·x_i + b)) + (1/C) * ||w||²

        Parameters
        ----------
        y : np.ndarray
            Labels in {-1, +1}
        """
        # TODO: 實作
        pass

    def fit(self, X, y, lr=0.001, n_iter=1000):
        """
        Fit using subgradient descent.

        Note: y should be in {-1, +1}
        """
        # TODO: 實作
        pass

    def predict(self, X):
        # TODO: 實作 sign(w·x + b)
        pass
```

### kmeans.py

```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init  # 'random' or 'kmeans++'
        self.centroids = None

    def _init_centroids(self, X):
        """Initialize centroids."""
        if self.init == 'random':
            # TODO: 隨機選 K 個點
            pass
        elif self.init == 'kmeans++':
            # TODO: 實作 k-means++ 初始化
            pass

    def fit(self, X):
        """
        Fit k-means.

        Algorithm:
        1. Initialize centroids
        2. Repeat until convergence:
           a. Assign each point to nearest centroid
           b. Update centroids as mean of assigned points
        """
        # TODO: 實作
        pass

    def predict(self, X):
        """Assign points to nearest centroid."""
        # TODO: 實作
        pass

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
```
