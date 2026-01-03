from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TorchTrainLoopConfig:
    epochs: int = 10
    lr: float = 0.03
    optimizer: str = "sgd"  # "sgd" | "adamw"
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_schedule: str = "constant"  # "constant" | "step"
    lr_decay_factor: float = 0.1
    lr_decay_epochs: int = 10
    label_smoothing: float = 0.0
    save_best_val: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "loss"  # "loss" | "accuracy"
    restore_best_val: bool = True
    seed: int = 42
    device: str = "cpu"  # "cpu" | "gpu"
    amp: bool = False
    max_grad_norm: float | None = None
    verbose: bool = True


def resolve_device(device: str) -> torch.device:
    device = str(device)
    if device == "cpu":
        return torch.device("cpu")
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested device='gpu' but torch.cuda.is_available() is False. "
                "You may need a CUDA-enabled PyTorch build."
            )
        return torch.device("cuda")
    raise ValueError(f"Unsupported device: {device}")


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _now_s() -> float:
    return time.perf_counter()


@torch.no_grad()
def eval_loss_acc(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for X, y, _ in loader:
        X = X.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == y).sum().item())
        total_n += n

    return {
        "n": total_n,
        "loss": (total_loss / total_n) if total_n else None,
        "accuracy": (total_correct / total_n) if total_n else None,
    }


def _lr_for_optimizer(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group.get("lr", 0.0))
    return 0.0


def _val_score(val_metrics: dict, *, metric: str) -> float | None:
    if metric == "loss":
        v = val_metrics.get("loss")
        return None if v is None else float(v)
    if metric == "accuracy":
        v = val_metrics.get("accuracy")
        return None if v is None else float(v)
    raise ValueError(f"Unsupported early_stopping_metric: {metric}")


def _is_improved(current: float, best: float, *, metric: str, min_delta: float) -> bool:
    min_delta = float(min_delta)
    if metric == "loss":
        return current < (best - min_delta)
    if metric == "accuracy":
        return current > (best + min_delta)
    raise ValueError(f"Unsupported early_stopping_metric: {metric}")


def make_optimizer(model: nn.Module, *, config: TorchTrainLoopConfig) -> torch.optim.Optimizer:
    opt = str(config.optimizer)
    if opt == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.lr),
            momentum=float(config.momentum),
            weight_decay=float(config.weight_decay),
        )
    if opt == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {opt}")


def make_scheduler(optimizer: torch.optim.Optimizer, *, config: TorchTrainLoopConfig):
    sched = str(config.lr_schedule)
    if sched == "constant":
        return None
    if sched == "step":
        if int(config.lr_decay_epochs) <= 0:
            raise ValueError("lr_decay_epochs must be > 0 when lr_schedule='step'")
        if float(config.lr_decay_factor) <= 0.0:
            raise ValueError("lr_decay_factor must be > 0 when lr_schedule='step'")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.lr_decay_epochs),
            gamma=float(config.lr_decay_factor),
        )
    raise ValueError(f"Unsupported lr_schedule: {sched}")


def train_loop(
    model: nn.Module,
    *,
    train_loader,
    val_loader,
    config: TorchTrainLoopConfig,
) -> dict:
    if int(config.epochs) <= 0:
        raise ValueError("epochs must be > 0")
    if float(config.lr) <= 0:
        raise ValueError("lr must be > 0")
    if float(config.label_smoothing) < 0.0 or float(config.label_smoothing) >= 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if int(config.early_stopping_patience) < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if float(config.early_stopping_min_delta) < 0.0:
        raise ValueError("early_stopping_min_delta must be >= 0")
    if str(config.early_stopping_metric) not in {"loss", "accuracy"}:
        raise ValueError(f"Unsupported early_stopping_metric: {config.early_stopping_metric}")

    device = resolve_device(config.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing))
    optimizer = make_optimizer(model, config=config)
    scheduler = make_scheduler(optimizer, config=config)

    use_amp = bool(config.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    history: list[dict] = []

    track_best = bool(config.save_best_val or int(config.early_stopping_patience) > 0)
    best_state: dict[str, torch.Tensor] | None = None
    best_score: float | None = None
    best_epoch: int | None = None
    epochs_without_improve = 0
    stopped_early = False
    stopped_epoch: int | None = None

    for epoch in range(int(config.epochs)):
        t0 = _now_s()

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for X, y, _ in train_loader:
            X = X.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()

            n = int(y.shape[0])
            total_loss += float(loss.item()) * n
            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == y).sum().item())
            total_n += n

        train_metrics = {
            "n": total_n,
            "loss": (total_loss / total_n) if total_n else None,
            "accuracy": (total_correct / total_n) if total_n else None,
        }

        val_metrics = (
            eval_loss_acc(model, val_loader, device=device, criterion=criterion)
            if val_loader is not None
            else {"n": 0, "loss": None, "accuracy": None}
        )

        if scheduler is not None:
            scheduler.step()

        dt = _now_s() - t0
        lr = _lr_for_optimizer(optimizer)
        history.append({"epoch": epoch + 1, "seconds": float(dt), "lr": float(lr), "train": train_metrics, "val": val_metrics})

        if config.verbose:
            val_loss = val_metrics.get("loss")
            val_acc = val_metrics.get("accuracy")
            val_loss_s = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            val_acc_s = f"{val_acc:.2%}" if val_acc is not None else "n/a"
            print(
                f"Epoch {epoch+1}/{config.epochs} - {dt:.2f}s - lr {lr:.6g} "
                f"- train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.2%} "
                f"- val loss {val_loss_s} acc {val_acc_s}"
            )

        if track_best:
            score = _val_score(val_metrics, metric=str(config.early_stopping_metric))
            if score is not None:
                if best_score is None or _is_improved(
                    float(score),
                    float(best_score),
                    metric=str(config.early_stopping_metric),
                    min_delta=float(config.early_stopping_min_delta),
                ):
                    best_score = float(score)
                    best_epoch = epoch + 1
                    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if int(config.early_stopping_patience) > 0 and epochs_without_improve >= int(config.early_stopping_patience):
                    stopped_early = True
                    stopped_epoch = epoch + 1
                    if config.verbose:
                        print(
                            f"Early stopping at epoch {stopped_epoch} "
                            f"(best epoch {best_epoch}, best {config.early_stopping_metric}={best_score})."
                        )
                    break

    if best_state is not None and bool(config.restore_best_val):
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_val": (
            {"epoch": best_epoch, "metric": str(config.early_stopping_metric), "score": best_score}
            if best_score is not None
            else None
        ),
        "early_stopping": {
            "enabled": bool(int(config.early_stopping_patience) > 0),
            "patience": int(config.early_stopping_patience),
            "min_delta": float(config.early_stopping_min_delta),
            "metric": str(config.early_stopping_metric),
            "stopped_early": stopped_early,
            "stopped_epoch": stopped_epoch,
        },
    }
