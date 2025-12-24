# Module 6: Advanced Architectures - 進階架構

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **Batch Normalization**
   - 理解 BN 為什麼有效
   - 實作 forward（training 和 inference）
   - 實作 backward

2. **ResNet Block**
   - 理解 residual connection 的作用
   - 實作 Basic Block
   - 處理維度不匹配的情況

3. **U-Net 架構**
   - 理解 encoder-decoder 結構
   - 理解 skip connection 的作用
   - 實作簡化版 U-Net

## 核心概念 (Core Concepts)

### Batch Normalization

Forward (training):
```
μ = mean(x, axis=(0,2,3))
σ² = var(x, axis=(0,2,3))
x̂ = (x - μ) / √(σ² + ε)
y = γ × x̂ + β
```

為什麼有效：
- 減少 internal covariate shift
- 允許更大的 learning rate
- 輕微的正則化效果

### Residual Connection

```
H(x) = F(x) + x
```

為什麼有效：
```
∂L/∂x = ∂L/∂H × (∂F/∂x + 1)
                         ↑
              這個 1 保證梯度至少為 ∂L/∂H
```

### U-Net

```
Encoder                    Decoder
   ↓                          ↑
Conv → Pool ──────────→ Concat → Conv
   ↓                          ↑
Conv → Pool ──────────→ Concat → Conv
   ↓                          ↑
        Bottleneck
```

## 實作任務 (Implementation Tasks)

### Batch Normalization
- [ ] `BatchNorm2D.forward()` - training mode
- [ ] `BatchNorm2D.forward()` - inference mode
- [ ] `BatchNorm2D.backward()`

### ResNet
- [ ] `BasicBlock.forward()`
- [ ] `BasicBlock.backward()`
- [ ] 處理 shortcut 的維度轉換

### U-Net
- [ ] `UNetEncoder` - 下採樣路徑
- [ ] `UNetDecoder` - 上採樣路徑
- [ ] 上採樣方法（nearest/bilinear/transposed conv）
- [ ] `SimpleUNet` - 完整組裝

## 檔案結構

```
m6_advanced/
├── README.md
├── batchnorm.py        # Batch Normalization
├── resnet_blocks.py    # ResNet blocks
├── unet.py             # U-Net 架構
└── tests/
    ├── test_bn.py
    └── test_resnet.py
```

## 開始學習

1. 確保已完成 Module 5（有完整的 CNN 基礎）
2. 閱讀 `prompts/m6_advanced.md`
3. 按順序：BatchNorm → ResNet Block → U-Net
4. 每個新元件都要驗證梯度

## BatchNorm Backward 推導提示

這個推導比較繁瑣。關鍵步驟：

```
1. ∂L/∂γ = Σ ∂L/∂y × x̂
2. ∂L/∂β = Σ ∂L/∂y
3. ∂L/∂x̂ = ∂L/∂y × γ
4. 接下來用鏈式法則對 x̂ = (x-μ)/√(σ²+ε) 展開
```

可以參考 CS231n 的 BatchNorm backward 講解。

## ResNet 實驗

比較有無 residual connection 的訓練效果：

```python
# 沒有 residual
class PlainBlock:
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out  # 沒有加 x

# 有 residual
class ResidualBlock:
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x  # 加上 shortcut
        out = self.relu2(out)
        return out
```

## U-Net 簡化版範例

```
Input: (N, 1, 64, 64)
    ↓
Enc1: Conv(64) → BN → ReLU → Pool → (N, 64, 32, 32), skip1
    ↓
Enc2: Conv(128) → BN → ReLU → Pool → (N, 128, 16, 16), skip2
    ↓
Bottleneck: Conv(256) → BN → ReLU → (N, 256, 16, 16)
    ↓
Dec2: Upsample → Concat(skip2) → Conv(128) → (N, 128, 32, 32)
    ↓
Dec1: Upsample → Concat(skip1) → Conv(64) → (N, 64, 64, 64)
    ↓
Output: Conv(n_classes) → (N, n_classes, 64, 64)
```

## 參考資源

- ResNet: He et al., CVPR 2016
- U-Net: Ronneberger et al., MICCAI 2015
- Batch Normalization: Ioffe & Szegedy, ICML 2015
