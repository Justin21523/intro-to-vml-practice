from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np


def _get_im2col_indices(
    x_shape: tuple[int, int, int, int],
    field_height: int,
    field_width: int,
    padding: int,
    stride: int,
    *,
    xp: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = xp.repeat(xp.arange(field_height), field_width)
    i0 = xp.tile(i0, C)
    i1 = stride * xp.repeat(xp.arange(out_height), out_width)

    j0 = xp.tile(xp.arange(field_width), field_height)
    j0 = xp.tile(j0, C)
    j1 = stride * xp.tile(xp.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = xp.repeat(xp.arange(C), field_height * field_width).reshape(-1, 1)
    return k, i, j


def im2col(
    x: np.ndarray,
    field_height: int,
    field_width: int,
    *,
    padding: int,
    stride: int,
    xp: Any = np,
) -> np.ndarray:
    """
    x: (N, C, H, W)
    returns cols: (C*field_height*field_width, N*out_h*out_w)
    """
    N, C, H, W = x.shape
    x_padded = xp.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    k, i, j = _get_im2col_indices((N, C, H, W), field_height, field_width, padding, stride, xp=xp)
    cols = x_padded[:, k, i, j]  # (N, C*fh*fw, out_h*out_w)
    cols = cols.transpose(1, 2, 0).reshape(C * field_height * field_width, -1)
    return cols


def col2im(
    cols: np.ndarray,
    x_shape: tuple[int, int, int, int],
    field_height: int,
    field_width: int,
    *,
    padding: int,
    stride: int,
    xp: Any = np,
) -> np.ndarray:
    """
    cols: (C*field_height*field_width, N*out_h*out_w)
    returns x: (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = _get_im2col_indices((N, C, H, W), field_height, field_width, padding, stride, xp=xp)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N).transpose(2, 0, 1)
    xp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class ReLU:
    def __init__(self, *, xp: Any = np) -> None:
        self.xp = xp
        self._mask: Any | None = None

    def forward(self, x: Any) -> Any:
        self._mask = x > 0
        return self.xp.maximum(0.0, x)

    def backward(self, dout: Any) -> Any:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        return dout * self._mask


class Flatten:
    def __init__(self) -> None:
        self._shape: tuple[int, ...] | None = None

    def forward(self, x: Any) -> Any:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: Any) -> Any:
        if self._shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        return dout.reshape(self._shape)


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rng: np.random.Generator,
        dtype: np.dtype,
        xp: Any = np,
    ) -> None:
        self.xp = xp
        W0 = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.W = xp.asarray(W0, dtype=dtype)
        self.b = xp.zeros((out_features,), dtype=dtype)
        self.dW: Any | None = None
        self.db: Any | None = None
        self._x: Any | None = None

    def forward(self, x: Any) -> Any:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: Any) -> Any:
        if self._x is None:
            raise RuntimeError("Linear.backward called before forward")
        x = self._x
        self.dW = x.T @ dout
        self.db = self.xp.sum(dout, axis=0)
        return dout @ self.W.T


class Conv2D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int,
        padding: int,
        rng: np.random.Generator,
        dtype: np.dtype,
        xp: Any = np,
    ) -> None:
        self.xp = xp
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        W0 = rng.standard_normal((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) * np.sqrt(
            2.0 / fan_in
        )
        self.W = xp.asarray(W0, dtype=dtype)
        self.b = xp.zeros((self.out_channels,), dtype=dtype)

        self.dW: Any | None = None
        self.db: Any | None = None
        self._cache: tuple[Any, Any, Any] | None = None

    def forward(self, x: Any) -> Any:
        N, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C}")

        k = self.kernel_size
        out_h = (H + 2 * self.padding - k) // self.stride + 1
        out_w = (W + 2 * self.padding - k) // self.stride + 1

        x_col = im2col(x, k, k, padding=self.padding, stride=self.stride, xp=self.xp)
        W_col = self.W.reshape(self.out_channels, -1)

        out = (W_col @ x_col) + self.b.reshape(-1, 1)
        out = out.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)

        self._cache = (x, x_col, W_col)
        return out

    def backward(self, dout: Any) -> Any:
        if self._cache is None:
            raise RuntimeError("Conv2D.backward called before forward")

        x, x_col, W_col = self._cache
        N, C, H, W = x.shape

        dout_col = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        self.db = self.xp.sum(dout, axis=(0, 2, 3))
        dW_col = dout_col @ x_col.T
        self.dW = dW_col.reshape(self.W.shape)

        dx_col = W_col.T @ dout_col
        dx = col2im(
            dx_col,
            (N, C, H, W),
            self.kernel_size,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            xp=self.xp,
        )
        return dx


class MaxPool2D:
    def __init__(self, kernel_size: int, *, stride: int | None = None, xp: Any = np) -> None:
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.xp = xp
        self._cache: tuple[Any, Any, Any] | None = None

    def forward(self, x: Any) -> Any:
        N, C, H, W = x.shape
        k = self.kernel_size
        out_h = (H - k) // self.stride + 1
        out_w = (W - k) // self.stride + 1

        x_reshaped = x.reshape(N * C, 1, H, W)
        x_col = im2col(x_reshaped, k, k, padding=0, stride=self.stride, xp=self.xp)  # (k*k, N*C*out_h*out_w)
        argmax = self.xp.argmax(x_col, axis=0)
        out = x_col[argmax, self.xp.arange(argmax.size)]
        out = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

        self._cache = (x, x_col, argmax)
        return out

    def backward(self, dout: Any) -> Any:
        if self._cache is None:
            raise RuntimeError("MaxPool2D.backward called before forward")

        x, x_col, argmax = self._cache
        N, C, H, W = x.shape
        k = self.kernel_size

        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dx_col = self.xp.zeros_like(x_col)
        dx_col[argmax, self.xp.arange(argmax.size)] = dout_flat

        dx = col2im(dx_col, (N * C, 1, H, W), k, k, padding=0, stride=self.stride, xp=self.xp)
        return dx.reshape(x.shape)


def softmax_cross_entropy(
    logits: Any,
    y: Any,
    *,
    xp: Any = np,
    as_float: bool = True,
) -> tuple[Any, Any, Any]:
    """
    logits: (N, K), y: (N,)
    returns loss, dlogits, accuracy
    """
    logits = logits - xp.max(logits, axis=1, keepdims=True)
    exp_logits = xp.exp(logits)
    probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

    n = logits.shape[0]
    correct = probs[xp.arange(n), y]
    loss = xp.mean(-xp.log(correct + 1e-12))

    pred = xp.argmax(probs, axis=1)
    acc = xp.mean(pred == y)

    dlogits = probs
    dlogits[xp.arange(n), y] -= 1.0
    dlogits = dlogits / n
    if as_float:
        loss = float(loss)
        acc = float(acc)
    return loss, dlogits, acc


@dataclass(frozen=True)
class TinyCNNArch:
    conv_out_channels: int = 8
    conv2_out_channels: int = 0  # 0 表示不使用第二層 conv
    kernel_size: int = 3
    conv_stride: int = 1
    pool_size: int = 2
    hidden_size: int = 128


class TinyConvNet:
    """
    (Conv -> ReLU -> MaxPool) x (1 or 2) -> Flatten -> FC -> ReLU -> FC
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        channels: str,
        num_classes: int,
        arch: TinyCNNArch,
        seed: int,
        dtype: np.dtype,
        xp: Any = np,
    ) -> None:
        if channels not in {"grayscale", "rgb"}:
            raise ValueError(f"Unsupported channels: {channels}")

        self.input_shape = tuple(int(x) for x in input_shape)
        self.channels = channels
        self.num_classes = int(num_classes)
        self.arch = arch
        self.xp = xp

        if len(self.input_shape) < 2:
            raise ValueError(f"Invalid input_shape: {self.input_shape}")

        H, W = int(self.input_shape[0]), int(self.input_shape[1])
        in_channels = 1 if channels == "grayscale" else 3

        if arch.kernel_size <= 0 or arch.pool_size <= 0 or arch.hidden_size <= 0 or arch.conv_out_channels <= 0:
            raise ValueError(f"Invalid arch: {arch}")
        if arch.conv_stride <= 0:
            raise ValueError(f"Invalid arch: {arch}")
        if arch.conv2_out_channels < 0:
            raise ValueError(f"Invalid arch: {arch}")

        if arch.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd (so padding=kernel_size//2 keeps spatial size)")

        rng = np.random.default_rng(seed)
        padding = arch.kernel_size // 2

        self.conv = Conv2D(
            in_channels,
            arch.conv_out_channels,
            arch.kernel_size,
            stride=arch.conv_stride,
            padding=padding,
            rng=rng,
            dtype=dtype,
            xp=xp,
        )
        self.relu1 = ReLU(xp=xp)
        self.pool = MaxPool2D(arch.pool_size, xp=xp)

        self.conv2: Conv2D | None
        self.relu2: ReLU | None
        self.pool2: MaxPool2D | None
        if arch.conv2_out_channels > 0:
            self.conv2 = Conv2D(
                arch.conv_out_channels,
                arch.conv2_out_channels,
                arch.kernel_size,
                stride=1,
                padding=padding,
                rng=rng,
                dtype=dtype,
                xp=xp,
            )
            self.relu2 = ReLU(xp=xp)
            self.pool2 = MaxPool2D(arch.pool_size, xp=xp)
        else:
            self.conv2 = None
            self.relu2 = None
            self.pool2 = None

        self.flatten = Flatten()

        H_conv = (H + 2 * padding - arch.kernel_size) // arch.conv_stride + 1
        W_conv = (W + 2 * padding - arch.kernel_size) // arch.conv_stride + 1
        if H_conv <= 0 or W_conv <= 0:
            raise ValueError("Invalid conv output shape; check input_shape / kernel_size / conv_stride")

        pool_stride = arch.pool_size  # MaxPool2D 預設 stride=kernel_size
        H_pool1 = (H_conv - arch.pool_size) // pool_stride + 1
        W_pool1 = (W_conv - arch.pool_size) // pool_stride + 1
        if H_pool1 <= 0 or W_pool1 <= 0:
            raise ValueError("Invalid pool output shape; check input_shape / pool_size / conv_stride")

        conv_out_for_fc = arch.conv_out_channels
        H_for_fc = H_pool1
        W_for_fc = W_pool1

        if arch.conv2_out_channels > 0:
            H_pool2 = (H_pool1 - arch.pool_size) // pool_stride + 1
            W_pool2 = (W_pool1 - arch.pool_size) // pool_stride + 1
            if H_pool2 <= 0 or W_pool2 <= 0:
                raise ValueError("Invalid pool2 output shape; check input_shape / pool_size / conv_stride")
            conv_out_for_fc = arch.conv2_out_channels
            H_for_fc = H_pool2
            W_for_fc = W_pool2

        flat_dim = conv_out_for_fc * H_for_fc * W_for_fc

        self.fc1 = Linear(flat_dim, arch.hidden_size, rng=rng, dtype=dtype, xp=xp)
        self.relu_fc = ReLU(xp=xp)
        self.fc2 = Linear(arch.hidden_size, self.num_classes, rng=rng, dtype=dtype, xp=xp)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self.conv.forward(x)
        out = self.relu1.forward(out)
        out = self.pool.forward(out)
        if self.conv2 is not None:
            assert self.relu2 is not None
            assert self.pool2 is not None
            out = self.conv2.forward(out)
            out = self.relu2.forward(out)
            out = self.pool2.forward(out)
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu_fc.forward(out)
        out = self.fc2.forward(out)
        return out

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        dout = self.fc2.backward(dlogits)
        dout = self.relu_fc.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)
        if self.conv2 is not None:
            assert self.relu2 is not None
            assert self.pool2 is not None
            dout = self.pool2.backward(dout)
            dout = self.relu2.backward(dout)
            dout = self.conv2.backward(dout)
        dout = self.pool.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv.backward(dout)
        return dout

    def params_and_grads(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.conv.dW is None or self.conv.db is None:
            raise RuntimeError("No gradients yet; call backward first")
        if self.conv2 is not None and (self.conv2.dW is None or self.conv2.db is None):
            raise RuntimeError("No gradients yet; call backward first")
        if self.fc1.dW is None or self.fc1.db is None:
            raise RuntimeError("No gradients yet; call backward first")
        if self.fc2.dW is None or self.fc2.db is None:
            raise RuntimeError("No gradients yet; call backward first")

        yield self.conv.W, self.conv.dW
        yield self.conv.b, self.conv.db
        if self.conv2 is not None:
            yield self.conv2.W, self.conv2.dW
            yield self.conv2.b, self.conv2.db
        yield self.fc1.W, self.fc1.dW
        yield self.fc1.b, self.fc1.db
        yield self.fc2.W, self.fc2.dW
        yield self.fc2.b, self.fc2.db
