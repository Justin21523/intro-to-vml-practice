# Module 0: Math Foundations - 數學基礎

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **向量與矩陣操作**
   - 計算內積、外積、矩陣乘法
   - 理解不同範數 (L1, L2, L∞) 的幾何意義
   - 把影像轉成向量表示

2. **微分與梯度**
   - 對向量函數求梯度
   - 理解梯度的幾何意義（函數增長最快的方向）
   - 應用鏈式法則

3. **機率基礎**
   - 理解期望值、變異數
   - 知道 Gaussian 分布的性質
   - 理解 Bayes' theorem

4. **最佳化**
   - 實作梯度下降
   - 理解學習率的影響

## 核心概念 (Core Concepts)

### 向量
- 可以表示一個點的座標
- 可以表示一個方向和長度
- 影像可以 flatten 成一個很長的向量

### 梯度
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```
梯度指向函數增長最快的方向，所以梯度下降往 `-∇f` 走。

### 鏈式法則
```
∂L/∂x = ∂L/∂y · ∂y/∂x
```
這是反向傳播的數學基礎。

## 實作任務 (Implementation Tasks)

- [ ] 向量範數計算 (`l1_norm`, `l2_norm`, `linf_norm`)
- [ ] Cosine similarity
- [ ] 對二次型 f(x) = x^T A x 求梯度
- [ ] 用數值微分驗證解析梯度
- [ ] 實作基本的梯度下降

## 檔案結構

```
m0_math/
├── README.md           # 本文件
├── vector_ops.py       # 向量運算
├── gradient.py         # 梯度計算
├── optimization.py     # 梯度下降
└── exercises/          # 練習題
    ├── ex1_vectors.py
    ├── ex2_gradient.py
    └── ex3_gd.py
```

## 開始學習

1. 閱讀 `prompts/m0_math.md`
2. 開始和 Claude 互動
3. 依序完成各個實作任務
4. 用 `utils/gradient_check.py` 驗證你的梯度

## 參考資源

- Goodfellow Ch. 2: Linear Algebra
- Goodfellow Ch. 3: Probability and Information Theory
- Goodfellow Ch. 4: Numerical Computation
