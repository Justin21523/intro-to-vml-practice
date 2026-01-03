from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TorchCNNArch:
    conv_out_channels: int = 32
    conv2_out_channels: int = 64  # 0 = disable
    kernel_size: int = 3
    conv_stride: int = 1
    pool_size: int = 2
    hidden_size: int = 256
    dropout: float = 0.0
    batch_norm: bool = True


class TorchCNNClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        channels: str,
        num_classes: int,
        arch: TorchCNNArch,
    ) -> None:
        super().__init__()
        if channels not in {"grayscale", "rgb"}:
            raise ValueError(f"Unsupported channels: {channels}")
        if len(input_shape) < 2:
            raise ValueError(f"Invalid input_shape: {input_shape}")

        H, W = int(input_shape[0]), int(input_shape[1])
        in_channels = 1 if channels == "grayscale" else 3

        if arch.kernel_size <= 0 or arch.pool_size <= 0 or arch.hidden_size <= 0:
            raise ValueError(f"Invalid arch: {arch}")
        if arch.conv_out_channels <= 0:
            raise ValueError(f"Invalid arch: {arch}")
        if arch.conv2_out_channels < 0:
            raise ValueError(f"Invalid arch: {arch}")
        if arch.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd (padding=kernel_size//2)")
        if arch.conv_stride <= 0:
            raise ValueError(f"Invalid arch: {arch}")
        if not (0.0 <= float(arch.dropout) < 1.0):
            raise ValueError(f"Invalid dropout: {arch.dropout}")

        pad = int(arch.kernel_size // 2)

        def _block(cin: int, cout: int) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=int(arch.kernel_size),
                    stride=int(arch.conv_stride) if cin == in_channels else 1,
                    padding=pad,
                    bias=not bool(arch.batch_norm),
                )
            ]
            if arch.batch_norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=int(arch.pool_size), stride=int(arch.pool_size)))
            return layers

        feats: list[nn.Module] = []
        feats.extend(_block(in_channels, int(arch.conv_out_channels)))
        if int(arch.conv2_out_channels) > 0:
            feats.extend(_block(int(arch.conv_out_channels), int(arch.conv2_out_channels)))

        self.features = nn.Sequential(*feats)

        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, H, W), dtype=torch.float32)
            out = self.features(dummy)
            flat_dim = int(out.numel())
            if flat_dim <= 0:
                raise ValueError("Invalid feature dim; check image_size/pool/stride settings.")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, int(arch.hidden_size)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(arch.dropout)) if float(arch.dropout) > 0.0 else nn.Identity(),
            nn.Linear(int(arch.hidden_size), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass(frozen=True)
class TorchMLPArch:
    hidden_sizes: tuple[int, ...] = (256,)
    dropout: float = 0.0


class TorchMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        channels: str,
        num_classes: int,
        arch: TorchMLPArch,
    ) -> None:
        super().__init__()
        if channels not in {"grayscale", "rgb"}:
            raise ValueError(f"Unsupported channels: {channels}")
        if len(input_shape) < 2:
            raise ValueError(f"Invalid input_shape: {input_shape}")
        if not arch.hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")
        if not (0.0 <= float(arch.dropout) < 1.0):
            raise ValueError(f"Invalid dropout: {arch.dropout}")

        in_channels = 1 if channels == "grayscale" else 3
        H, W = int(input_shape[0]), int(input_shape[1])
        in_dim = int(in_channels * H * W)

        layers: list[nn.Module] = [nn.Flatten()]
        last = in_dim
        for h in arch.hidden_sizes:
            h = int(h)
            if h <= 0:
                raise ValueError(f"Invalid hidden size: {h}")
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            if float(arch.dropout) > 0.0:
                layers.append(nn.Dropout(p=float(arch.dropout)))
            last = h
        layers.append(nn.Linear(last, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchLinearClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        channels: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        if channels not in {"grayscale", "rgb"}:
            raise ValueError(f"Unsupported channels: {channels}")
        if len(input_shape) < 2:
            raise ValueError(f"Invalid input_shape: {input_shape}")

        in_channels = 1 if channels == "grayscale" else 3
        H, W = int(input_shape[0]), int(input_shape[1])
        in_dim = int(in_channels * H * W)
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, int(num_classes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def num_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def infer_num_classes(class_map: dict[str, int]) -> int:
    if not class_map:
        raise ValueError("class_map is empty")
    return int(max(int(v) for v in class_map.values()) + 1)


def class_names_from_map(class_map: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_map.items(), key=lambda kv: int(kv[1]))]


def id_to_name_from_map(class_map: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in class_map.items()}
