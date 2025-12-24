# Module 7: CPU Optimization - CPU 優化

先貼 `system_prompt.md` 的內容，再貼這個 module prompt。

---

## Prompt

```
主題：如何有效利用 32 線程 CPU 來訓練/執行我自己寫的影像演算法。

我的硬體配置：32 執行緒 CPU，沒有 GPU。我想盡可能榨乾 CPU 的效能。

### 第一部分：Python 效能基礎

1. Python GIL（Global Interpreter Lock）
   - GIL 是什麼，為什麼存在
   - 為什麼多執行緒對 CPU-bound 的純 Python 迴圈沒有幫助
   - 什麼情況下可以繞過 GIL

2. numpy 的內建多執行緒
   - numpy 使用 BLAS/LAPACK（如 OpenBLAS, MKL）
   - 這些庫通常已經多執行緒化
   - 如何查看/設定 numpy 用幾個執行緒：
     - OMP_NUM_THREADS
     - MKL_NUM_THREADS

3. Benchmark 方法
   - 用 time 模組測量執行時間
   - 用 %timeit (如果用 Jupyter)
   - 看 CPU 使用率驗證是否真的在用多核

### 第二部分：向量化 vs For-Loop

1. 為什麼向量化快
   - 減少 Python interpreter overhead
   - SIMD 指令
   - Cache-friendly 記憶體存取

2. 實驗：比較不同實作
   - 純 Python for-loop 的 2D convolution
   - numpy 向量化的 convolution
   - 比較執行時間

3. Broadcasting 技巧
   - 正確使用 broadcasting 可以避免顯式迴圈
   - 常見的 broadcasting 模式

### 第三部分：im2col 技巧

1. 卷積轉矩陣乘法
   - 把所有要做內積的 patch 展開成矩陣
   - 用 GEMM（矩陣乘法）計算
   - GEMM 是高度優化的，能吃到多核心

2. im2col Forward
   - 輸入 shape: (N, C, H, W)
   - 輸出 shape: (N*H'*W', C*kH*kW)

3. col2im Backward
   - 把梯度「折回」原本的 shape

4. 效能比較
   - naive convolution vs im2col
   - 在不同輸入大小下的差異

### 第四部分：multiprocessing

1. 什麼時候用 multiprocessing
   - 當任務可以完全獨立執行
   - 例如：平行處理多張圖片

2. 基本用法
   - Pool.map / Pool.starmap
   - 傳遞資料的開銷

3. 在訓練中使用
   - 資料載入平行化
   - Batch 平行計算（但要小心 gradient 怎麼合併）

4. 共享記憶體
   - multiprocessing.Array
   - numpy 的 shared memory
   - 減少資料複製的開銷

### 第五部分：Numba / Cython（選做）

如果 numpy 還不夠快：

1. Numba
   - @jit 裝飾器
   - 對 numpy 友好
   - 什麼情況下有效/無效

2. Cython
   - 寫 C extension 的簡化版
   - 適合 numpy 操作之間有很多 Python 迴圈的情況

### 第六部分：實際案例與 Benchmark

設計幾個實驗，讓我實際測量：

1. 卷積 Benchmark
   - naive for-loop
   - numpy vectorized
   - im2col
   - (選做) numba 加速的版本
   - 比較在不同輸入大小下的時間

2. 訓練 Benchmark
   - 單線程訓練
   - 資料載入平行化
   - 計算平行化（如果可行）

3. 推論 Benchmark
   - 單張圖片 vs batch
   - 平行處理多張圖片

### 教學要求

- 每個優化技術都要有「優化前 vs 優化後」的時間比較
- 解釋清楚每種技術的適用場景
- 不要過度優化：先 profile，找到瓶頸再優化
- 提醒我：最重要的優化通常是演算法層級，而不是底層實作
```

---

## 預期學習成果

完成這個 module 後，你應該能夠：

1. 理解 Python GIL 和 numpy 多執行緒
2. 把 for-loop 改寫成向量化操作
3. 實作 im2col 加速卷積
4. 使用 multiprocessing 平行化資料處理
5. 有系統地 benchmark 和 profile 程式碼

---

## 對應資源

- **numpy 文件**: Broadcasting, Performance Tips
- **Python 文件**: multiprocessing, threading
- **BLAS 文件**: OpenBLAS, Intel MKL

---

## 程式碼骨架

### im2col.py

```python
import numpy as np

def im2col(x, kernel_size, stride=1, padding=0):
    """
    Convert input to column matrix for convolution.

    Parameters
    ----------
    x : np.ndarray, shape (N, C, H, W)
    kernel_size : int
    stride : int
    padding : int

    Returns
    -------
    col : np.ndarray, shape (N*H'*W', C*kH*kW)
    """
    # TODO: 實作
    pass


def col2im(col, x_shape, kernel_size, stride=1, padding=0):
    """
    Convert column matrix back to image format.
    Used for backward pass.

    Parameters
    ----------
    col : np.ndarray, shape (N*H'*W', C*kH*kW)
    x_shape : tuple
        Original input shape (N, C, H, W)
    """
    # TODO: 實作
    pass


def conv2d_im2col(x, W, b, stride=1, padding=0):
    """
    Convolution using im2col.

    Parameters
    ----------
    x : np.ndarray, shape (N, C_in, H, W)
    W : np.ndarray, shape (C_out, C_in, kH, kW)
    b : np.ndarray, shape (C_out,)
    """
    N, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W.shape

    # im2col
    col = im2col(x, kH, stride, padding)  # (N*H'*W', C_in*kH*kW)

    # Reshape kernel
    W_col = W.reshape(C_out, -1).T  # (C_in*kH*kW, C_out)

    # Matrix multiplication
    out_col = col @ W_col + b  # (N*H'*W', C_out)

    # Reshape output
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W_in + 2*padding - kW) // stride + 1
    out = out_col.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)

    return out
```

### benchmarks.py

```python
import numpy as np
import time

def benchmark_convolution():
    """
    Compare different convolution implementations.
    """
    # Test sizes
    sizes = [(1, 3, 32, 32), (1, 3, 64, 64), (1, 3, 128, 128)]
    kernel = np.random.randn(16, 3, 3, 3).astype(np.float32)

    for size in sizes:
        x = np.random.randn(*size).astype(np.float32)

        # Naive implementation
        start = time.time()
        out1 = conv2d_naive(x, kernel)
        time_naive = time.time() - start

        # im2col implementation
        start = time.time()
        out2 = conv2d_im2col(x, kernel, np.zeros(16))
        time_im2col = time.time() - start

        print(f"Size {size}:")
        print(f"  Naive:   {time_naive:.4f}s")
        print(f"  im2col:  {time_im2col:.4f}s")
        print(f"  Speedup: {time_naive / time_im2col:.2f}x")
        print()


def benchmark_parallel_inference():
    """
    Compare single-threaded vs multi-threaded inference.
    """
    import multiprocessing as mp

    def process_single_image(args):
        image, model = args
        return model.forward(image[np.newaxis])

    n_images = 100
    images = [np.random.randn(1, 28, 28) for _ in range(n_images)]

    # Single-threaded
    start = time.time()
    for img in images:
        model.forward(img[np.newaxis])
    time_single = time.time() - start

    # Multi-threaded (using multiprocessing)
    start = time.time()
    with mp.Pool(processes=8) as pool:
        results = pool.map(process_single_image, [(img, model) for img in images])
    time_multi = time.time() - start

    print(f"Single-threaded: {time_single:.2f}s")
    print(f"Multi-threaded:  {time_multi:.2f}s")
    print(f"Speedup: {time_single / time_multi:.2f}x")
```

### profiling.py

```python
import cProfile
import pstats
import io

def profile_function(func, *args, **kwargs):
    """
    Profile a function and print stats.
    """
    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())

    return result


def check_numpy_threads():
    """
    Check how many threads numpy is using.
    """
    import numpy as np

    # Check BLAS info
    np.show_config()

    # Check OMP threads
    import os
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
```

---

## 優化優先順序

1. **演算法層級**：O(n²) → O(n log n) 的改進遠比底層優化重要
2. **減少不必要的計算**：cache 中間結果，避免重複計算
3. **向量化**：把 Python 迴圈換成 numpy 操作
4. **im2col / GEMM**：利用高度優化的矩陣乘法
5. **multiprocessing**：平行化獨立的任務
6. **numba/cython**：最後的手段

> "We should forget about small efficiencies, say about 97% of the time:
> premature optimization is the root of all evil." — Donald Knuth
