# Module 7: CPU Optimization - CPU 優化

## 學習目標 (Learning Objectives)

完成這個 module 後，你應該能夠：

1. **理解 Python 效能限制**
   - GIL 的影響
   - numpy 的多執行緒

2. **向量化**
   - 把 for-loop 改成 numpy 操作
   - Broadcasting 技巧

3. **im2col 加速**
   - 把卷積轉成矩陣乘法
   - 利用高度優化的 BLAS

4. **平行化**
   - 使用 multiprocessing
   - 共享記憶體技巧

5. **Profiling**
   - 找出程式瓶頸
   - 有系統地優化

## 核心概念 (Core Concepts)

### Python GIL

```python
# 這個不會變快，因為 GIL
import threading
threads = [threading.Thread(target=cpu_bound_task) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### numpy 多執行緒

numpy 的矩陣運算通常已經用 BLAS/LAPACK，會自動使用多核心。

```bash
# 設定 numpy 使用的執行緒數
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### im2col

把卷積轉成矩陣乘法：

```
Input: (N, C, H, W)
         ↓ im2col
Column: (N*H'*W', C*kH*kW)
         ×
Kernel: (C*kH*kW, C_out)
         ↓
Output: (N*H'*W', C_out)
         ↓ reshape
        (N, C_out, H', W')
```

### multiprocessing

```python
from multiprocessing import Pool

def process_image(img):
    return some_heavy_computation(img)

with Pool(processes=8) as pool:
    results = pool.map(process_image, images)
```

## 實作任務 (Implementation Tasks)

### 基礎
- [ ] 比較 numpy 線程數設定的影響
- [ ] 比較 for-loop vs 向量化的效能

### im2col
- [ ] `im2col(x, kernel_size)` - 展開成列
- [ ] `col2im(col, x_shape, kernel_size)` - 折回
- [ ] `conv2d_im2col()` - 用 im2col 實作卷積

### Benchmarks
- [ ] 卷積 benchmark (naive vs im2col)
- [ ] 訓練 benchmark
- [ ] 推論 benchmark (單張 vs batch)

### 平行化
- [ ] 平行化資料載入
- [ ] 平行化圖片處理

## 檔案結構

```
m7_optimization/
├── README.md
├── im2col.py           # im2col 實作
├── benchmarks.py       # 效能測試
├── profiling.py        # Profiling 工具
└── parallel_utils.py   # 平行化工具
```

## 開始學習

1. 閱讀 `prompts/m7_optimization.md`
2. 先 profile 你現有的程式碼，找出瓶頸
3. 實作 im2col 並比較效能
4. 嘗試 multiprocessing

## Benchmark 範例

```python
import time
import numpy as np

def benchmark_convolution(conv_func, x, kernel, n_runs=10):
    """
    測量卷積函數的執行時間
    """
    # Warmup
    conv_func(x, kernel)

    times = []
    for _ in range(n_runs):
        start = time.time()
        conv_func(x, kernel)
        times.append(time.time() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# 比較不同實作
x = np.random.randn(1, 3, 64, 64).astype(np.float32)
kernel = np.random.randn(16, 3, 3, 3).astype(np.float32)

print("Naive convolution:")
print(benchmark_convolution(conv2d_naive, x, kernel))

print("\nim2col convolution:")
print(benchmark_convolution(conv2d_im2col, x, kernel))
```

## 優化優先順序

```
1. 演算法層級（最重要）
   └── O(n²) → O(n log n) 的改進

2. 減少不必要計算
   └── Cache 中間結果，避免重複計算

3. 向量化
   └── Python loop → numpy operations

4. 使用 BLAS (im2col)
   └── 利用高度優化的矩陣乘法

5. 平行化 (multiprocessing)
   └── 平行處理獨立任務

6. 底層優化 (Numba/Cython)
   └── 最後手段
```

## 查看你的 numpy 配置

```python
import numpy as np
np.show_config()

# 檢查環境變數
import os
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
```

## 參考資源

- numpy 文件: Performance Tips
- Python 文件: multiprocessing
- OpenBLAS / Intel MKL 文件
