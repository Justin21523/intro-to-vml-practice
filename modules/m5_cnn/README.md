# Module 5: CNN from Scratch - 從零實作 CNN

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **反向傳播**
   - 理解計算圖和鏈式法則
   - 對簡單函數手推 backward

2. **各層 Forward/Backward**
   - Fully Connected layer
   - ReLU activation
   - Conv2D (最難的部分！)
   - MaxPool2D

3. **組裝網路**
   - 建立類似 LeNet 的小型 CNN
   - 實作完整的訓練迴圈

## 核心概念 (Core Concepts)

### Backpropagation

鏈式法則：
```
∂L/∂x = ∂L/∂y × ∂y/∂x
```

對於多輸入的情況：
```
∂L/∂x = Σᵢ ∂L/∂yᵢ × ∂yᵢ/∂x
```

### Fully Connected Layer

Forward:
```
Y = X @ W + b
```

Backward:
```
dX = dY @ W^T
dW = X^T @ dY
db = sum(dY, axis=0)
```

### Conv2D Backward

這是最複雜的部分。關鍵洞察：

- `dW`: 把 input 和 dout 做某種「卷積」
- `dX`: 把 kernel 翻轉後和 dout 做 full convolution

### Max Pooling Backward

梯度只流向 forward 時取到最大值的位置。

## 實作任務 (Implementation Tasks)

### 基礎層
- [ ] `FullyConnected.forward()` 和 `.backward()`
- [ ] `ReLU.forward()` 和 `.backward()`
- [ ] `Sigmoid.forward()` 和 `.backward()`

### 卷積層
- [ ] `Conv2D.forward()` - 可以用 Module 1 的卷積
- [ ] `Conv2D.backward()` - 最難的部分！
- [ ] `MaxPool2D.forward()` 和 `.backward()`

### Loss
- [ ] `softmax()` - 數值穩定版本
- [ ] `cross_entropy_loss()` - 含梯度

### 網路
- [ ] `SimpleConvNet` - 組裝成完整網路
- [ ] `train()` - 訓練迴圈

## 檔案結構

```
m5_cnn/
├── README.md
├── layers.py           # 各種 layer
├── loss.py             # Loss functions
├── network.py          # 完整網路
├── train.py            # 訓練迴圈
└── tests/
    ├── test_fc.py
    ├── test_conv.py
    └── test_pool.py
```

## 開始學習

1. 確保已完成 Module 0 和 Module 1
2. 閱讀 `prompts/m5_cnn.md`
3. 按順序：FC → ReLU → Softmax+CE → Conv2D → MaxPool
4. **每一層都要用 `gradient_check` 驗證！**

## 驗證你的梯度

```python
from utils.gradient_check import check_layer_gradients
from m5_cnn.layers import FullyConnected, Conv2D

# 測試 FC layer
def fc_forward(x, params):
    fc = FullyConnected(10, 5)
    fc.W = params['W']
    fc.b = params['b']
    out = fc.forward(x)
    return out, fc.cache

def fc_backward(dout, cache):
    fc = FullyConnected(10, 5)
    fc.cache = cache
    dx = fc.backward(dout)
    return dx, {'W': fc.dW, 'b': fc.db}

results = check_layer_gradients(
    fc_forward, fc_backward,
    input_shape=(2, 10),
    param_shapes={'W': (10, 5), 'b': (5,)}
)
```

## LeNet-like 架構

```
Input: (N, 1, 28, 28)
    ↓
Conv(6, 5×5) → ReLU → MaxPool(2×2)
    ↓
(N, 6, 12, 12)
    ↓
Conv(16, 5×5) → ReLU → MaxPool(2×2)
    ↓
(N, 16, 4, 4)
    ↓
Flatten
    ↓
(N, 256)
    ↓
FC(120) → ReLU
    ↓
FC(84) → ReLU
    ↓
FC(10) → Softmax
    ↓
Output: (N, 10)
```

## 訓練小技巧

1. **先 overfit 小資料**：用 10-100 張圖，確認 loss 能降到很低
2. **檢查梯度**：任何新的 layer 都要驗證
3. **學習率**：從 0.01 開始，太大會爆炸，太小學不動
4. **初始化**：使用 Xavier 或 He initialization

## 參考資源

- Goodfellow Ch. 6: Deep Feedforward Networks
- Goodfellow Ch. 9: Convolutional Networks
- CS231n: Stanford CNN course
