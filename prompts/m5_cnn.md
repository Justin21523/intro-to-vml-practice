# Module 5: CNN from Scratch - 從零實作 CNN

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：從零實作一個簡單的 CNN（類 LeNet），包含完整的反向傳播。

我已經完成前面的 modules，有了卷積、梯度檢查等基礎工具。現在要把這些組合成一個可訓練的神經網路。

### 第一部分：理解反向傳播

1. 計算圖（Computational Graph）
   - 前向傳播：從輸入到輸出
   - 反向傳播：從 loss 到每個參數的梯度
   - Chain rule：∂L/∂x = ∂L/∂y · ∂y/∂x

2. 一維例子熱身
   - y = wx + b
   - L = (y - target)²
   - 手推 ∂L/∂w, ∂L/∂b, ∂L/∂x

3. 向量化版本
   - X shape: (N, D), W shape: (D, M)
   - 輸出 Y shape: (N, M)
   - ∂L/∂W = ?

### 第二部分：Fully Connected Layer

1. Forward
   - out = X @ W + b

2. Backward
   - 輸入：dout (∂L/∂out)
   - 輸出：dX, dW, db

3. 實作與驗證
   - 用 gradient_check 驗證每個梯度

### 第三部分：Activation Functions

1. ReLU
   - forward: max(0, x)
   - backward: 1 if x > 0 else 0

2. Sigmoid
   - forward: 1 / (1 + exp(-x))
   - backward: σ(x) · (1 - σ(x))

3. Softmax + Cross-Entropy Loss
   - 為什麼要一起算（數值穩定性）
   - 合併後的梯度很簡單：ŷ - y

### 第四部分：Conv2D Layer

這是最複雜的部分，要仔細推導。

1. Forward
   - 用你 Module 1 的 conv2d，但要處理 batch 和 多 channel
   - Input shape: (N, C_in, H, W)
   - Kernel shape: (C_out, C_in, kH, kW)
   - Output shape: (N, C_out, H', W')

2. Backward
   - ∂L/∂kernel: 把 input 和 dout 做某種「卷積」
   - ∂L/∂input: 把 kernel 翻轉後和 dout 做卷積（full convolution）

3. im2col 技巧（選做）
   - 把卷積轉成矩陣乘法
   - 這樣 backward 也變成簡單的矩陣乘法

### 第五部分：Pooling Layer

1. Max Pooling Forward
   - 在每個 window 取最大值
   - 需要記錄最大值的位置

2. Max Pooling Backward
   - 梯度只流向最大值的位置
   - 其他位置梯度為 0

### 第六部分：組裝成 Network

1. 簡單架構（類 LeNet）
   - Conv(6 filters, 5x5) → ReLU → MaxPool(2x2)
   - Conv(16 filters, 5x5) → ReLU → MaxPool(2x2)
   - Flatten
   - FC(120) → ReLU
   - FC(84) → ReLU
   - FC(10) → Softmax

2. Training Loop
   - Forward pass
   - Compute loss
   - Backward pass
   - Update parameters (SGD)

3. 測試
   - 在小型資料集上訓練（MNIST 的子集，或自己生成的資料）
   - 確認 loss 能下降

### 第七部分：優化技巧

1. Weight Initialization
   - 為什麼 Xavier/He 初始化重要

2. Momentum SGD
   - v = β·v - lr·grad
   - w = w + v

3. Learning Rate Schedule
   - 固定 vs 衰減

### 教學要求

- 每一層的 backward 都要用 gradient_check 嚴格驗證
- 不使用任何深度學習框架（PyTorch/TensorFlow 等）
- 如果數學太複雜，請用「index 展開」方式說明，而不是只丟矩陣微分公式
- 先在非常小的資料上 overfit，確認網路能學習
- Conv2D backward 是最難的部分，請特別詳細解釋
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 從零實作 FC, Conv2D, Pooling, ReLU, Softmax 的 forward 和 backward
2. 理解反向傳播的 chain rule
3. 組裝成一個完整可訓練的 CNN
4. 在小型資料集上訓練並看到 loss 下降

---

## 對應資源

- **Goodfellow Ch. 6**: Deep Feedforward Networks
- **Goodfellow Ch. 9**: Convolutional Networks
- **LeCun 1998**: Gradient-Based Learning Applied to Document Recognition
- **CS231n**: Stanford CNN course (很好的視覺化教材)

---

## 程式碼骨架

### layers.py

```python
import numpy as np

class Layer:
    """Base class for layers."""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class FullyConnected(Layer):
    def __init__(self, in_features, out_features):
        # Xavier initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.cache = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, D)

        Returns
        -------
        out : np.ndarray, shape (N, M)
        """
        # TODO: 實作
        # 記得存 cache 給 backward 用
        pass

    def backward(self, dout):
        """
        Parameters
        ----------
        dout : np.ndarray, shape (N, M)
            Gradient of loss w.r.t. output

        Returns
        -------
        dx : np.ndarray, shape (N, D)
        """
        # TODO: 實作
        # 同時計算 self.dW 和 self.db
        pass


class ReLU(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, x):
        # TODO: 實作
        pass

    def backward(self, dout):
        # TODO: 實作
        pass


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization
        self.W = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        self.cache = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, C_in, H, W)

        Returns
        -------
        out : np.ndarray, shape (N, C_out, H', W')
        """
        # TODO: 實作
        pass

    def backward(self, dout):
        """
        Parameters
        ----------
        dout : np.ndarray, shape (N, C_out, H', W')

        Returns
        -------
        dx : np.ndarray, shape (N, C_in, H, W)
        """
        # TODO: 實作
        # 這是最難的部分！
        pass


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.cache = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, C, H, W)
        """
        # TODO: 實作
        # 記得存最大值的位置
        pass

    def backward(self, dout):
        # TODO: 實作
        pass


class Flatten(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)
```

### loss.py

```python
import numpy as np

def softmax(x):
    """
    Numerically stable softmax.

    Parameters
    ----------
    x : np.ndarray, shape (N, C)
    """
    # TODO: 實作
    pass


def cross_entropy_loss(scores, y):
    """
    Softmax + Cross-Entropy loss.

    Parameters
    ----------
    scores : np.ndarray, shape (N, C)
        Raw scores (logits)
    y : np.ndarray, shape (N,)
        True labels (integers)

    Returns
    -------
    loss : float
    dscores : np.ndarray, shape (N, C)
        Gradient w.r.t. scores
    """
    # TODO: 實作
    pass
```

### network.py

```python
import numpy as np

class SimpleConvNet:
    """
    A simple CNN similar to LeNet.

    Architecture:
    Conv(6, 5x5) -> ReLU -> MaxPool(2x2) ->
    Conv(16, 5x5) -> ReLU -> MaxPool(2x2) ->
    Flatten -> FC(120) -> ReLU -> FC(84) -> ReLU -> FC(10)
    """

    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        # TODO: 初始化所有層
        pass

    def forward(self, x):
        # TODO: 前向傳播
        pass

    def backward(self, dout):
        # TODO: 反向傳播
        pass

    def get_params_and_grads(self):
        # TODO: 返回所有可訓練參數和它們的梯度
        pass


def train(model, X_train, y_train, lr=0.01, epochs=10, batch_size=32):
    """
    Training loop.
    """
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward
            scores = model.forward(X_batch)

            # Loss
            loss, dscores = cross_entropy_loss(scores, y_batch)
            total_loss += loss * len(y_batch)

            # Backward
            model.backward(dscores)

            # Update
            for param, grad in model.get_params_and_grads():
                param -= lr * grad

        print(f"Epoch {epoch+1}, Loss: {total_loss / n_samples:.4f}")
```
