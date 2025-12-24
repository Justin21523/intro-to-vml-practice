"""
Gradient Checking Utilities - 梯度檢查工具
==========================================

用於驗證你手推的解析梯度是否正確。
這是學習反向傳播時最重要的除錯工具。

Verify that your analytical gradients are correct.
This is the most important debugging tool when learning backprop.

原理（Principle）
----------------
數值微分使用中央差分（central difference）：

    df/dx ≈ [f(x + ε) - f(x - ε)] / (2ε)

這比前向差分 [f(x + ε) - f(x)] / ε 更準確，
因為誤差是 O(ε²) 而不是 O(ε)。

如果你的解析梯度正確，它應該與數值梯度非常接近（相對誤差 < 1e-5）。
"""

import numpy as np
from typing import Callable, Union, Tuple


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    計算函數 f 在點 x 的數值梯度

    Compute numerical gradient of f at x using central difference.

    Parameters
    ----------
    f : callable
        純量函數 f(x) -> float
        注意：f 只能回傳一個數字（loss），不能是向量

    x : np.ndarray
        計算梯度的位置，可以是任意 shape

    eps : float, default=1e-5
        差分步長，太小會有數值誤差，太大會有截斷誤差
        1e-5 通常是好的選擇

    Returns
    -------
    np.ndarray
        與 x 相同 shape 的梯度

    Example
    -------
    >>> def f(x):
    ...     return np.sum(x ** 2)  # f(x) = x1² + x2² + ...
    >>> x = np.array([3.0, 4.0])
    >>> grad = numerical_gradient(f, x)
    >>> print(grad)  # [6.0, 8.0]，因為 df/dxi = 2*xi
    """
    grad = np.zeros_like(x, dtype=np.float64)

    # 用 nditer 遍歷所有元素（支援任意維度）
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]

        # f(x + eps)
        x[idx] = original_value + eps
        f_plus = f(x)

        # f(x - eps)
        x[idx] = original_value - eps
        f_minus = f(x)

        # 中央差分
        grad[idx] = (f_plus - f_minus) / (2 * eps)

        # 恢復原值
        x[idx] = original_value

        it.iternext()

    return grad


def gradient_check(
    analytic_grad: np.ndarray,
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
    tolerance: float = 1e-5,
    verbose: bool = True
) -> Tuple[bool, float]:
    """
    比較解析梯度與數值梯度

    Compare analytical gradient with numerical gradient.

    Parameters
    ----------
    analytic_grad : np.ndarray
        你計算的解析梯度（手推的公式）

    f : callable
        損失函數 f(x) -> float

    x : np.ndarray
        計算梯度的位置

    eps : float, default=1e-5
        數值微分的步長

    tolerance : float, default=1e-5
        可接受的相對誤差

    verbose : bool, default=True
        是否印出詳細資訊

    Returns
    -------
    passed : bool
        是否通過檢查

    relative_error : float
        相對誤差

    Example
    -------
    >>> def f(x):
    ...     return np.sum(x ** 2)
    >>> x = np.array([3.0, 4.0])
    >>> analytic = 2 * x  # 我們知道 df/dx = 2x
    >>> passed, error = gradient_check(analytic, f, x)
    >>> print(f"Passed: {passed}, Error: {error:.2e}")
    """
    numeric_grad = numerical_gradient(f, x.copy(), eps)

    # 計算相對誤差
    # 使用 ||a - n|| / max(||a||, ||n||) 避免除以零
    diff = np.linalg.norm(analytic_grad - numeric_grad)
    norm_a = np.linalg.norm(analytic_grad)
    norm_n = np.linalg.norm(numeric_grad)

    # 避免兩個都是零的情況
    if norm_a == 0 and norm_n == 0:
        relative_error = 0.0
    else:
        relative_error = diff / max(norm_a, norm_n)

    passed = relative_error < tolerance

    if verbose:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"Gradient Check {status}")
        print(f"  Relative error: {relative_error:.2e}")
        print(f"  Tolerance: {tolerance:.2e}")

        if not passed:
            # 找出最大差異的位置
            abs_diff = np.abs(analytic_grad - numeric_grad)
            max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            print(f"  Max difference at index {max_diff_idx}:")
            print(f"    Analytic: {analytic_grad[max_diff_idx]:.6f}")
            print(f"    Numeric:  {numeric_grad[max_diff_idx]:.6f}")

    return passed, relative_error


def check_layer_gradients(
    layer_forward: Callable,
    layer_backward: Callable,
    input_shape: Tuple[int, ...],
    param_shapes: dict = None,
    seed: int = 42
) -> dict:
    """
    檢查一個 layer 的所有梯度（輸入梯度和參數梯度）

    Check all gradients for a layer (input gradients and parameter gradients).

    這是為了方便檢查 CNN 各層的梯度而設計的 helper function。

    Parameters
    ----------
    layer_forward : callable
        前向傳播函數 (x, params) -> output

    layer_backward : callable
        反向傳播函數 (dout, cache) -> (dx, dparams)

    input_shape : tuple
        輸入的 shape

    param_shapes : dict, optional
        參數的 shapes，例如 {'W': (3, 4), 'b': (4,)}

    seed : int
        隨機種子，確保可重現

    Returns
    -------
    dict
        每個梯度的檢查結果

    Example
    -------
    >>> # 假設你實作了一個 fully connected layer
    >>> results = check_layer_gradients(
    ...     fc_forward, fc_backward,
    ...     input_shape=(2, 10),  # batch=2, features=10
    ...     param_shapes={'W': (10, 5), 'b': (5,)}
    ... )
    """
    np.random.seed(seed)

    # 生成隨機輸入
    x = np.random.randn(*input_shape)

    # 生成隨機參數
    params = {}
    if param_shapes:
        for name, shape in param_shapes.items():
            params[name] = np.random.randn(*shape) * 0.01

    # 前向傳播
    out, cache = layer_forward(x, params)

    # 生成隨機的上游梯度
    dout = np.random.randn(*out.shape)

    # 反向傳播
    dx, dparams = layer_backward(dout, cache)

    results = {}

    # 檢查輸入梯度
    def f_x(x_):
        out_, _ = layer_forward(x_, params)
        return np.sum(out_ * dout)

    passed, error = gradient_check(dx, f_x, x.copy(), verbose=False)
    results['dx'] = {'passed': passed, 'error': error}
    print(f"dx: {'✓' if passed else '✗'} (error: {error:.2e})")

    # 檢查每個參數的梯度
    for name, dparam in dparams.items():
        def f_param(p):
            params_copy = params.copy()
            params_copy[name] = p
            out_, _ = layer_forward(x, params_copy)
            return np.sum(out_ * dout)

        passed, error = gradient_check(dparam, f_param, params[name].copy(), verbose=False)
        results[f'd{name}'] = {'passed': passed, 'error': error}
        print(f"d{name}: {'✓' if passed else '✗'} (error: {error:.2e})")

    return results
