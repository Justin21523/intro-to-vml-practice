# Module 0: Math Foundations - 數學基礎

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：為影像機器學習打底的數學復習。

請你幫我複習以下幾塊，我的目標是能看懂 ML/CV 論文和教科書中的數學：

### 第一部分：向量與矩陣

1. 向量基礎
   - 向量的幾何意義（方向 + 長度）
   - 內積（dot product）：計算方式、幾何意義（投影）
   - 範數（norm）：L1, L2, Lp, L∞
   - 用影像例子說明：把 2D 灰階圖 flatten 成向量

2. 矩陣基礎
   - 矩陣乘法的意義：線性變換
   - 轉置、對稱矩陣
   - 矩陣的 rank 和 determinant（直覺即可）

3. 特徵值與特徵向量
   - 直覺：什麼方向被矩陣「放大」或「縮小」
   - 為什麼 PCA 要找 covariance matrix 的特徵向量

### 第二部分：微分與梯度

1. 純量對向量的微分
   - 如果 f(x) 是純量，x 是向量，∂f/∂x 是什麼？
   - 用最簡單的例子：f(x) = x^T A x

2. 梯度的幾何意義
   - 梯度指向函數增長最快的方向
   - 梯度下降為什麼是往 -∇f 方向走

3. 鏈式法則（chain rule）
   - 複合函數的微分
   - 為什麼反向傳播就是在套用鏈式法則

### 第三部分：機率基礎

1. 隨機變數與分布
   - 離散 vs 連續
   - PMF vs PDF
   - 期望值、變異數

2. 常見分布
   - Bernoulli, Categorical
   - Gaussian（一維和多維）
   - 為什麼 Gaussian 這麼常用

3. Bayes' theorem
   - P(A|B) = P(B|A)P(A) / P(B)
   - Prior, Likelihood, Posterior 的直覺

### 第四部分：最佳化基礎

1. 最小平方法
   - 為什麼 (X^T X)^{-1} X^T y 是解
   - 用 numpy 實作

2. 梯度下降
   - 基本演算法
   - Learning rate 的影響
   - 用 numpy 實作

### 教學要求

- 每一小節先用直覺和圖像說明，再帶入數學式
- 每一小節最後給我 2~3 個練習題，讓我用 Python / numpy 算出來
- 練習題中請包含：手算一次梯度，然後用 gradient_check.py 驗證
- 逐步進行，每次等我完成練習再往下
- 如果我哪裡卡住，用更簡單的例子解釋
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 看懂 ML 論文中的向量/矩陣表示法
2. 對任意簡單的 loss function 手推梯度
3. 用數值微分驗證你的梯度是否正確
4. 理解為什麼 Gaussian 分布這麼常見
5. 實作基本的梯度下降

---

## 對應資源

- **Goodfellow Ch. 2**: Linear Algebra
- **Goodfellow Ch. 3**: Probability and Information Theory
- **Goodfellow Ch. 4**: Numerical Computation
- **Bishop Ch. 1-2**: Introduction and Probability Distributions

---

## 練習題範例

這些是你應該能夠完成的練習：

### 練習 1: 向量運算
```python
import numpy as np

# 計算兩個向量的 L2 距離
def l2_distance(a, b):
    # TODO: 實作
    pass

# 計算 cosine similarity
def cosine_similarity(a, b):
    # TODO: 實作
    pass
```

### 練習 2: 梯度計算
```python
import numpy as np
from utils.gradient_check import gradient_check

# 對 f(x) = x^T A x 計算梯度
def quadratic_gradient(x, A):
    # TODO: 實作 ∂f/∂x = (A + A^T) x
    pass

# 驗證
A = np.random.randn(5, 5)
x = np.random.randn(5)
f = lambda x: x @ A @ x
analytic = quadratic_gradient(x, A)
gradient_check(analytic, f, x)
```

### 練習 3: 梯度下降
```python
import numpy as np

def gradient_descent(f, grad_f, x0, lr=0.01, n_iter=100):
    """
    實作基本梯度下降

    Parameters
    ----------
    f : callable
        目標函數
    grad_f : callable
        梯度函數
    x0 : np.ndarray
        初始點
    lr : float
        學習率
    n_iter : int
        迭代次數

    Returns
    -------
    x : np.ndarray
        最終解
    history : list
        每一步的 f(x) 值
    """
    # TODO: 實作
    pass
```
