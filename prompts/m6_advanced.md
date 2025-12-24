# Module 6: Advanced Architectures - 進階架構

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：在我自己寫的 CNN 上加入 ResNet block 與 U-Net 式 encoder-decoder。

我已經完成 Module 5，有自己實作的 Conv2D, MaxPool, ReLU, FC 層（含 forward 和 backward）。現在要在此基礎上加入更進階的架構元素。

### 第一部分：Batch Normalization

在學 ResNet 之前，先要有 BatchNorm。

1. Forward
   - 計算 batch 的 mean 和 variance
   - 正規化：x̂ = (x - μ) / √(σ² + ε)
   - Scale 和 shift：y = γ·x̂ + β

2. Training vs Inference
   - Training: 用當前 batch 的統計量
   - Inference: 用 running mean/variance

3. Backward
   - 這個推導比較繁瑣，但很重要
   - ∂L/∂γ, ∂L/∂β, ∂L/∂x

4. 為什麼有效
   - 減少 internal covariate shift
   - 可以用更大的 learning rate
   - 有輕微的正則化效果

### 第二部分：ResNet Block（殘差塊）

1. 問題背景
   - 深層網路難以訓練：梯度消失/爆炸
   - 更深不一定更好（degradation problem）

2. Residual Connection
   - H(x) = F(x) + x
   - 網路只需要學習 residual F(x) = H(x) - x
   - 為什麼這樣更容易學習

3. Basic Block 實作
   - Conv3x3 → BN → ReLU → Conv3x3 → BN → Add → ReLU
   - 需要處理：當維度不匹配時的 shortcut（用 1x1 conv）

4. Bottleneck Block（選做）
   - 1x1 conv → 3x3 conv → 1x1 conv
   - 為什麼這樣設計（減少計算量）

### 第三部分：U-Net 架構

1. 問題背景
   - 語義分割：輸出和輸入一樣大
   - 需要同時有 high-level 和 low-level 特徵

2. Encoder-Decoder 結構
   - Encoder：逐步降低解析度，增加 channels
   - Decoder：逐步恢復解析度

3. Skip Connections
   - 把 encoder 的特徵直接接到 decoder
   - 保留細節資訊

4. 上採樣方法
   - Nearest neighbor upsampling
   - Bilinear interpolation
   - Transposed convolution（反卷積）

5. 實作簡化版 U-Net
   - Encoder: 2-3 層 (Conv → BN → ReLU → MaxPool)
   - Decoder: 對應的 2-3 層 (Upsample → Conv → BN → ReLU)
   - Skip connections

### 第四部分：Transposed Convolution（選做）

1. 直覺
   - 一般卷積縮小，反卷積放大
   - 「反向」的卷積操作

2. Forward 和 Backward
   - Forward 其實是 backward convolution 的 forward
   - Backward 是正常的 convolution

### 第五部分：組裝與測試

1. 測試 ResNet-style 小網路
   - 在 CIFAR-10 子集上訓練
   - 比較有無 residual connection 的效果

2. 測試簡化版 U-Net
   - 在簡單的分割任務上測試
   - 例如：把圓形/方形分割出來

### 教學要求

- 所有新的 layer 都要有 forward 和 backward
- 用 gradient_check 驗證每個新 layer
- 重點解釋：為什麼 ResNet 能訓練更深的網路
- 重點解釋：Skip connection 在不同架構中的作用
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 實作 Batch Normalization（含 forward/backward）
2. 實作 ResNet Basic Block
3. 理解 encoder-decoder 架構和 skip connections
4. 實作簡化版 U-Net

---

## 對應資源

- **ResNet 論文**: He et al., "Deep Residual Learning", CVPR 2016
- **U-Net 論文**: Ronneberger et al., MICCAI 2015
- **Batch Normalization**: Ioffe & Szegedy, ICML 2015

---

## 程式碼骨架

### batchnorm.py

```python
import numpy as np

class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Batch Normalization for 2D inputs (images).

        Parameters
        ----------
        num_features : int
            Number of channels
        eps : float
            Small constant for numerical stability
        momentum : float
            Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.cache = None
        self.training = True

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, C, H, W)

        Returns
        -------
        out : np.ndarray, shape (N, C, H, W)
        """
        if self.training:
            # TODO: 計算 batch mean/var，normalize，scale & shift
            # 更新 running statistics
            pass
        else:
            # TODO: 用 running mean/var
            pass

    def backward(self, dout):
        """
        Parameters
        ----------
        dout : np.ndarray, shape (N, C, H, W)

        Returns
        -------
        dx : np.ndarray, shape (N, C, H, W)

        Also computes self.dgamma and self.dbeta
        """
        # TODO: 實作
        # 這個推導比較複雜，可以參考 CS231n 的筆記
        pass
```

### resnet_blocks.py

```python
import numpy as np

class BasicBlock:
    """
    ResNet Basic Block:
    x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        stride : int
            If stride > 1, need to downsample the shortcut
        """
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()

        # Shortcut: identity if same shape, otherwise 1x1 conv
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, 1, stride=stride)
            self.shortcut_bn = BatchNorm2D(out_channels)

        self.cache = None

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : np.ndarray, shape (N, C_in, H, W)
        """
        # TODO: 實作
        # 記得處理 shortcut
        pass

    def backward(self, dout):
        """
        Backward pass.
        """
        # TODO: 實作
        # 梯度要分到兩條路徑然後相加
        pass
```

### unet.py

```python
import numpy as np

class UNetEncoder:
    """
    U-Net encoder block:
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        # TODO: 初始化層
        pass

    def forward(self, x):
        # TODO: 返回 pool 前的特徵（給 skip connection 用）和 pool 後的
        pass

    def backward(self, dout, dskip):
        # TODO: 實作
        pass


class UNetDecoder:
    """
    U-Net decoder block:
    Upsample -> Concat(skip) -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        # in_channels 已經包含 skip connection 的 channels
        # TODO: 初始化層
        pass

    def forward(self, x, skip):
        # TODO: 實作
        pass

    def backward(self, dout):
        # TODO: 實作
        pass


class SimpleUNet:
    """
    Simplified U-Net for demonstration.

    Input: (N, 1, 64, 64)
    Output: (N, n_classes, 64, 64)
    """

    def __init__(self, in_channels=1, n_classes=2):
        # TODO: 初始化 encoder 和 decoder
        pass

    def forward(self, x):
        # TODO: 實作完整 forward
        pass

    def backward(self, dout):
        # TODO: 實作完整 backward
        pass
```

---

## ResNet 梯度流分析

為什麼 residual connection 能幫助梯度流動：

```
沒有 residual:
∂L/∂x = ∂L/∂y · ∂y/∂x
       = ∂L/∂y · ∂F(x)/∂x
如果 ∂F/∂x 很小，梯度會消失

有 residual:
y = F(x) + x
∂L/∂x = ∂L/∂y · (∂F(x)/∂x + 1)
                            ↑ 這個 1 保證梯度至少為 ∂L/∂y
```

這就是為什麼 ResNet 能訓練非常深的網路。
