from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from .dataset_index import (
    DatasetScanOptions,
    kfold_split_records_by_character,
    read_jsonl,
    scan_dataset,
    split_records,
    summarize_records,
    write_jsonl,
)
from .error_analysis import ErrorAnalysisConfig, analyze_errors
from .experiments import ExperimentIOConfig, run_experiment_suite
from .inference import evaluate, format_confusion_matrix, iter_records_from_jsonl, load_model, predict_image
from .train_cnn import CNNTrainConfig, train_cnn_from_splits
from .train_mlp import MLPTrainConfig, train_mlp_from_splits
from .train_softmax import SoftmaxTrainConfig, train_softmax_from_splits


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vml", description="intro-to-vml CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="dataset 工具")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    scan = dataset_sub.add_parser("scan", help="掃描資料夾並輸出影像索引（JSONL）")
    scan.add_argument(
        "--root",
        type=Path,
        required=True,
        help="資料集根目錄（例如 /mnt/data/.../generated_data）",
    )
    scan.add_argument(
        "--out",
        type=Path,
        default=None,
        help="輸出 JSONL 檔案路徑（未指定則不寫檔，只印 summary）",
    )
    scan.add_argument(
        "--with-size",
        action="store_true",
        help="讀取每張圖的寬高（會變慢，但可順便驗證是否可開啟）",
    )
    scan.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多掃描多少張（除錯用）",
    )

    stats = dataset_sub.add_parser("stats", help="掃描資料夾並印出資料統計")
    stats.add_argument("--root", type=Path, required=True, help="資料集根目錄")
    stats.add_argument("--with-size", action="store_true", help="同 scan --with-size")
    stats.add_argument("--limit", type=int, default=None, help="同 scan --limit")

    split = dataset_sub.add_parser("split", help="切分 train/val/test（輸出 JSONL）")
    split.add_argument(
        "--index",
        type=Path,
        default=None,
        help="輸入 JSONL 索引（由 dataset scan 產生）；未指定則改用 --root 即時掃描",
    )
    split.add_argument("--root", type=Path, default=None, help="資料集根目錄（當 --index 未指定時使用）")
    split.add_argument("--out-dir", type=Path, required=True, help="輸出資料夾（內含 train/val/test.jsonl）")
    split.add_argument("--seed", type=int, default=42, help="隨機種子")
    split.add_argument("--val-ratio", type=float, default=0.1, help="驗證集比例（0~1）")
    split.add_argument("--test-ratio", type=float, default=0.1, help="測試集比例（0~1）")
    split.add_argument("--with-size", action="store_true", help="若用 --root 即時掃描，是否讀取寬高")
    split.add_argument("--limit", type=int, default=None, help="若用 --root 即時掃描，最多掃描多少張")
    split.add_argument(
        "--strategy",
        choices=["random", "by-character"],
        default="random",
        help="切分策略：random=依 class 分層隨機；by-character=以角色為單位切分",
    )
    split.add_argument(
        "--keep-unknown",
        action="store_true",
        help="保留 kind 不明的圖片（否則會輸出到 unknown.jsonl 並排除於 train/val/test）",
    )

    kfold = dataset_sub.add_parser("kfold", help="以角色為單位做 K-fold 切分（輸出 JSONL）")
    kfold.add_argument(
        "--index",
        type=Path,
        default=None,
        help="輸入 JSONL 索引（由 dataset scan 產生）；未指定則改用 --root 即時掃描",
    )
    kfold.add_argument("--root", type=Path, default=None, help="資料集根目錄（當 --index 未指定時使用）")
    kfold.add_argument("--out-dir", type=Path, required=True, help="輸出資料夾（內含 fold_*/train|val|test.jsonl）")
    kfold.add_argument("--seed", type=int, default=42, help="隨機種子")
    kfold.add_argument("--folds", type=int, default=5, help="K-fold 的 K（>=2）")
    kfold.add_argument("--val-fold-offset", type=int, default=1, help="val fold 相對於 test fold 的偏移量（>=1）")
    kfold.add_argument(
        "--balance-folds",
        action="store_true",
        help="嘗試讓每個 fold 的 class 分布更平均（近似 StratifiedGroupKFold；仍保證以角色為單位切分）",
    )
    kfold.add_argument("--with-size", action="store_true", help="若用 --root 即時掃描，是否讀取寬高")
    kfold.add_argument("--limit", type=int, default=None, help="若用 --root 即時掃描，最多掃描多少張")
    kfold.add_argument(
        "--keep-unknown",
        action="store_true",
        help="保留 kind 不明的圖片（否則會輸出到 unknown.jsonl 並排除於 train/val/test）",
    )

    train_parser = subparsers.add_parser("train", help="訓練工具")
    train_sub = train_parser.add_subparsers(dest="train_command", required=True)

    softmax = train_sub.add_parser("softmax", help="訓練 softmax regression（三分類：action/pose/expression）")
    softmax.add_argument("--splits-dir", type=Path, required=True, help="包含 train/val/test.jsonl 的資料夾")
    softmax.add_argument("--epochs", type=int, default=5, help="訓練 epoch 數")
    softmax.add_argument("--batch-size", type=int, default=64, help="batch size")
    softmax.add_argument("--lr", type=float, default=0.1, help="learning rate")
    softmax.add_argument("--reg", type=float, default=0.0, help="L2 regularization strength")
    softmax.add_argument("--seed", type=int, default=42, help="隨機種子")
    softmax.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="訓練裝置（gpu=CUDA；需 CUDA PyTorch）")
    softmax.add_argument("--channels", choices=["grayscale", "rgb"], default="grayscale", help="輸入影像通道")
    softmax.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="計算 dtype")
    softmax.add_argument(
        "--shape-mode",
        choices=["strict", "pad_or_crop", "resize"],
        default="resize",
        help="圖片尺寸不一致時的處理方式：strict=必須一致；pad_or_crop=padding/裁切到指定尺寸；resize=縮放到指定尺寸",
    )
    softmax.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(128, 128),
        help="目標尺寸（H W）。當 shape-mode=pad_or_crop/resize 時必填；用於固定輸入維度。",
    )
    softmax.add_argument(
        "--augment-hflip",
        action="store_true",
        help="訓練時隨機水平翻轉（augmentation）",
    )
    softmax.add_argument("--max-train", type=int, default=None, help="最多使用多少張 train（除錯用）")
    softmax.add_argument("--max-val", type=int, default=None, help="最多使用多少張 val（除錯用）")
    softmax.add_argument("--max-test", type=int, default=None, help="最多使用多少張 test（除錯用）")
    softmax.add_argument("--quiet", action="store_true", help="不印出每個 epoch 的進度")
    softmax.add_argument(
        "--out-model",
        type=Path,
        default=Path("data/models/softmax_regression.pt"),
        help="輸出模型（PyTorch state_dict，建議副檔名 .pt）",
    )

    mlp = train_sub.add_parser("mlp", help="訓練 MLP（三分類：action/pose/expression）")
    mlp.add_argument("--splits-dir", type=Path, required=True, help="包含 train/val/test.jsonl 的資料夾")
    mlp.add_argument("--epochs", type=int, default=10, help="訓練 epoch 數")
    mlp.add_argument("--batch-size", type=int, default=64, help="batch size")
    mlp.add_argument("--lr", type=float, default=0.1, help="learning rate")
    mlp.add_argument("--reg", type=float, default=0.0, help="L2 regularization strength")
    mlp.add_argument("--seed", type=int, default=42, help="隨機種子")
    mlp.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="訓練裝置（gpu=CUDA；需 CUDA PyTorch）")
    mlp.add_argument("--channels", choices=["grayscale", "rgb"], default="grayscale", help="輸入影像通道")
    mlp.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="計算 dtype")
    mlp.add_argument(
        "--shape-mode",
        choices=["strict", "pad_or_crop", "resize"],
        default="resize",
        help="圖片尺寸不一致時的處理方式：strict=必須一致；pad_or_crop=padding/裁切到指定尺寸；resize=縮放到指定尺寸",
    )
    mlp.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(128, 128),
        help="目標尺寸（H W）。當 shape-mode=pad_or_crop/resize 時必填；用於固定輸入維度。",
    )
    mlp.add_argument("--augment-hflip", action="store_true", help="訓練時隨機水平翻轉（augmentation）")
    mlp.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256],
        help="隱藏層尺寸（可多層），例如：--hidden-sizes 512 256",
    )
    mlp.add_argument("--max-train", type=int, default=None, help="最多使用多少張 train（除錯用）")
    mlp.add_argument("--max-val", type=int, default=None, help="最多使用多少張 val（除錯用）")
    mlp.add_argument("--max-test", type=int, default=None, help="最多使用多少張 test（除錯用）")
    mlp.add_argument("--quiet", action="store_true", help="不印出每個 epoch 的進度")
    mlp.add_argument(
        "--out-model",
        type=Path,
        default=Path("data/models/mlp.pt"),
        help="輸出模型（PyTorch state_dict，建議副檔名 .pt）",
    )

    cnn = train_sub.add_parser("cnn", help="訓練 Tiny CNN（三分類：action/pose/expression）")
    cnn.add_argument("--splits-dir", type=Path, required=True, help="包含 train/val/test.jsonl 的資料夾")
    cnn.add_argument("--epochs", type=int, default=5, help="訓練 epoch 數")
    cnn.add_argument("--batch-size", type=int, default=32, help="batch size")
    cnn.add_argument(
        "--sample-strategy",
        choices=["shuffle", "balanced_class", "balanced_character", "balanced_character_class"],
        default="shuffle",
        help="訓練資料抽樣策略（shuffle=原始洗牌；balanced_* 會用權重抽樣降低某些 class/角色過度主導）",
    )
    cnn.add_argument("--lr", type=float, default=0.05, help="learning rate")
    cnn.add_argument("--reg", type=float, default=0.0, help="L2 regularization strength")
    cnn.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd", help="optimizer")
    cnn.add_argument("--momentum", type=float, default=0.0, help="SGD momentum（0=不使用）")
    cnn.add_argument("--weight-decay", type=float, default=0.0, help="weight decay / L2（0=不使用）")
    cnn.add_argument(
        "--lr-schedule",
        choices=["constant", "step"],
        default="constant",
        help="learning rate schedule",
    )
    cnn.add_argument("--lr-decay-factor", type=float, default=0.1, help="step schedule 的衰減倍率")
    cnn.add_argument("--lr-decay-epochs", type=int, default=10, help="step schedule 每隔幾個 epoch 衰減一次")
    cnn.add_argument("--label-smoothing", type=float, default=0.0, help="CrossEntropy label smoothing（0=off）")
    cnn.add_argument("--save-best-val", action="store_true", help="訓練過程中保存最佳 val（最後會還原）")
    cnn.add_argument("--early-stopping-patience", type=int, default=0, help="early stopping patience（0=不使用）")
    cnn.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="early stopping 最小改善幅度")
    cnn.add_argument(
        "--early-stopping-metric",
        choices=["loss", "accuracy"],
        default="loss",
        help="early stopping 監控指標",
    )
    cnn.add_argument("--no-restore-best-val", action="store_true", help="不還原到最佳 val 權重（預設會還原）")
    cnn.add_argument("--seed", type=int, default=42, help="隨機種子")
    cnn.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="訓練裝置（gpu=CUDA；需 CUDA PyTorch）")
    cnn.add_argument("--channels", choices=["grayscale", "rgb"], default="grayscale", help="輸入影像通道")
    cnn.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="計算 dtype")
    cnn.add_argument(
        "--shape-mode",
        choices=["strict", "pad_or_crop", "resize"],
        default="resize",
        help="圖片尺寸不一致時的處理方式：strict=必須一致；pad_or_crop=padding/裁切到指定尺寸；resize=縮放到指定尺寸",
    )
    cnn.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(128, 128),
        help="目標尺寸（H W）。當 shape-mode=pad_or_crop/resize 時必填；用於固定輸入維度。",
    )
    cnn.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="resize 後影像快取資料夾（可大幅加速訓練/評估；僅在 shape-mode=resize 時生效）",
    )
    cnn.add_argument("--augment-hflip", action="store_true", help="訓練時隨機水平翻轉（augmentation）")
    cnn.add_argument("--augment-rrc", action="store_true", help="random resized crop（需 shape-mode=resize）")
    cnn.add_argument("--augment-rrc-prob", type=float, default=1.0, help="RRC 套用機率（0~1）")
    cnn.add_argument(
        "--augment-rrc-scale",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.8, 1.0),
        help="RRC 面積比例範圍（MIN MAX，例如 0.8 1.0）",
    )
    cnn.add_argument("--augment-brightness", type=float, default=0.0, help="亮度抖動幅度（0=off）")
    cnn.add_argument("--augment-contrast", type=float, default=0.0, help="對比抖動幅度（0=off）")
    cnn.add_argument("--augment-cutout", action="store_true", help="random erase / cutout")
    cnn.add_argument("--augment-cutout-prob", type=float, default=0.5, help="cutout 套用機率（0~1）")
    cnn.add_argument(
        "--augment-cutout-scale",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.02, 0.2),
        help="cutout 面積比例範圍（MIN MAX，例如 0.02 0.2）",
    )
    cnn.add_argument("--conv-out-channels", type=int, default=8, help="卷積輸出通道數")
    cnn.add_argument("--conv2-out-channels", type=int, default=0, help="第二層卷積輸出通道數（0=不使用）")
    cnn.add_argument("--kernel-size", type=int, default=3, help="卷積 kernel size（需為奇數）")
    cnn.add_argument("--conv-stride", type=int, default=1, help="卷積 stride（例如 1 或 2）")
    cnn.add_argument("--pool-size", type=int, default=2, help="MaxPool kernel/stride size")
    cnn.add_argument("--hidden-size", type=int, default=128, help="FC 隱藏層維度")
    cnn.add_argument("--dropout", type=float, default=0.0, help="dropout（0=off）")
    cnn.add_argument("--no-batch-norm", dest="batch_norm", action="store_false", default=True, help="關閉 BatchNorm")
    cnn.add_argument("--amp", action="store_true", help="啟用自動混合精度（僅 CUDA 生效）")
    cnn.add_argument("--max-grad-norm", type=float, default=None, help="gradient clipping（L2 norm）")
    cnn.add_argument("--num-workers", type=int, default=0, help="DataLoader workers（>0 可提升 GPU 吞吐）")
    cnn.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", default=True, help="關閉 pin_memory")
    cnn.add_argument("--persistent-workers", action="store_true", help="保持 DataLoader workers（需 num-workers>0）")
    cnn.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor（需 num-workers>0）")
    cnn.add_argument("--max-train", type=int, default=None, help="最多使用多少張 train（除錯用）")
    cnn.add_argument("--max-val", type=int, default=None, help="最多使用多少張 val（除錯用）")
    cnn.add_argument("--max-test", type=int, default=None, help="最多使用多少張 test（除錯用）")
    cnn.add_argument("--quiet", action="store_true", help="不印出每個 epoch 的進度")
    cnn.add_argument(
        "--out-model",
        type=Path,
        default=Path("data/models/cnn.pt"),
        help="輸出模型（PyTorch state_dict，建議副檔名 .pt）",
    )

    eval_parser = subparsers.add_parser("eval", help="推論/評估工具")
    eval_parser.add_argument("--model", type=Path, required=True, help="模型檔案（PyTorch state_dict，建議副檔名 .pt）")
    eval_parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="推論裝置（cnn 可用 gpu；需 CUDA 版 PyTorch）")
    eval_parser.add_argument(
        "--shape-mode",
        choices=["auto", "strict", "pad_or_crop", "resize"],
        default="auto",
        help="圖片尺寸處理方式（auto=沿用訓練設定；其餘為覆蓋）",
    )
    eval_parser.add_argument("--batch-size", type=int, default=64, help="評估 batch size")
    eval_parser.add_argument("--limit", type=int, default=None, help="最多評估多少張（除錯用）")
    eval_parser.add_argument(
        "--out-predictions",
        type=Path,
        default=None,
        help="輸出逐筆預測結果（JSONL）。可用於後續錯誤分析/視覺化。",
    )
    data_group = eval_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", type=Path, default=None, help="輸入 JSONL（內含 path/relpath/class_id...）")
    data_group.add_argument("--splits-dir", type=Path, default=None, help="包含 train/val/test.jsonl 的資料夾")
    eval_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="當使用 --splits-dir 時，要評估哪個 split",
    )
    eval_parser.add_argument("--json-only", action="store_true", help="只輸出 JSON，不印 confusion matrix 表格")

    predict_parser = subparsers.add_parser("predict", help="單張圖片推論")
    predict_parser.add_argument("--model", type=Path, required=True, help="模型檔案（PyTorch state_dict，建議副檔名 .pt）")
    predict_parser.add_argument("--image", type=Path, required=True, help="圖片路徑")
    predict_parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="推論裝置（cnn 可用 gpu；需 CUDA 版 PyTorch）",
    )
    predict_parser.add_argument(
        "--shape-mode",
        choices=["auto", "strict", "pad_or_crop", "resize"],
        default="auto",
        help="圖片尺寸處理方式（auto=沿用訓練設定；其餘為覆蓋）",
    )

    analyze_parser = subparsers.add_parser("analyze", help="分析工具")
    analyze_sub = analyze_parser.add_subparsers(dest="analyze_command", required=True)

    errors = analyze_sub.add_parser("errors", help="錯誤分析（per-character/per-class + 視覺化報表）")
    errors.add_argument("--model", type=Path, required=True, help="模型檔案（PyTorch state_dict，建議副檔名 .pt）")
    errors.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="推論裝置（cnn 可用 gpu；需 CUDA 版 PyTorch）",
    )
    errors.add_argument(
        "--shape-mode",
        choices=["auto", "strict", "pad_or_crop", "resize"],
        default="auto",
        help="圖片尺寸處理方式（auto=沿用訓練設定；其餘為覆蓋）",
    )
    errors.add_argument("--batch-size", type=int, default=64, help="推論 batch size")
    errors.add_argument("--limit", type=int, default=None, help="最多分析多少張（除錯用）")
    errors_data = errors.add_mutually_exclusive_group(required=True)
    errors_data.add_argument("--data", type=Path, default=None, help="輸入 JSONL")
    errors_data.add_argument("--splits-dir", type=Path, default=None, help="包含 train/val/test.jsonl 的資料夾")
    errors.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="當使用 --splits-dir 時，要分析哪個 split",
    )
    errors.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="輸出資料夾（內含 report.html、summary.json、thumbs/、grids/）",
    )
    errors.add_argument("--max-per-pair", type=int, default=36, help="每個混淆對（例如 action→pose）最多視覺化幾張")
    errors.add_argument("--thumb-size", type=int, default=224, help="縮圖邊長（正方形）")
    errors.add_argument("--seed", type=int, default=0, help="抽樣用隨機種子（確保可重現）")

    exp_parser = subparsers.add_parser("experiment", help="一鍵實驗流程（train + eval + report）")
    exp_data = exp_parser.add_mutually_exclusive_group(required=True)
    exp_data.add_argument("--splits-dir", type=Path, default=None, help="單次 split 目錄（包含 train/val/test.jsonl）")
    exp_data.add_argument("--folds-dir", type=Path, default=None, help="K-fold 目錄（內含 fold_*/train|val|test.jsonl）")
    exp_parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=["baseline", "regularized"],
        help="要跑哪些 preset（例如 baseline regularized）",
    )
    exp_parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="訓練裝置（preset 會套用）")
    exp_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="輸出資料夾（預設 data/experiments/<timestamp>）",
    )
    exp_parser.add_argument("--eval-batch-size", type=int, default=64, help="eval / error analysis 的 batch size")
    exp_parser.add_argument("--eval-limit", type=int, default=None, help="最多評估多少張（除錯用）")
    exp_errors = exp_parser.add_mutually_exclusive_group()
    exp_errors.add_argument(
        "--analyze-errors",
        dest="analyze_errors",
        action="store_true",
        default=None,
        help="產出 error analysis 報告（單次 split 預設開啟；K-fold 預設關閉）",
    )
    exp_errors.add_argument(
        "--no-analyze-errors",
        dest="analyze_errors",
        action="store_false",
        default=None,
        help="不產出 error analysis 報告（比較快）",
    )
    exp_parser.add_argument("--errors-limit", type=int, default=None, help="錯誤分析最多處理多少張（除錯用）")
    exp_parser.add_argument("--errors-thumb-size", type=int, default=224, help="錯誤分析縮圖邊長")
    exp_parser.add_argument("--errors-max-per-pair", type=int, default=36, help="每個混淆對最多視覺化幾張")
    exp_parser.add_argument("--seed", type=int, default=0, help="抽樣用隨機種子（確保可重現）")
    exp_parser.add_argument("--max-folds", type=int, default=None, help="K-fold 模式下最多跑幾個 fold（除錯用）")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset":
        if args.dataset_command == "scan":
            options = DatasetScanOptions(
                root=Path(args.root),
                with_size=bool(args.with_size),
                limit=args.limit,
            )
            records = list(scan_dataset(options))
            summary = summarize_records(records)
            print(json.dumps(summary, ensure_ascii=False, indent=2))

            if args.out is not None:
                write_jsonl(Path(args.out), records)
                print(f"Wrote: {args.out}")
            return 0

        if args.dataset_command == "stats":
            options = DatasetScanOptions(
                root=Path(args.root),
                with_size=bool(args.with_size),
                limit=args.limit,
            )
            records = list(scan_dataset(options))
            summary = summarize_records(records)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.dataset_command == "split":
            if args.index is None and args.root is None:
                raise SystemExit("Please provide either --index or --root.")

            if args.index is not None:
                records = list(read_jsonl(Path(args.index)))
            else:
                options = DatasetScanOptions(
                    root=Path(args.root),
                    with_size=bool(args.with_size),
                    limit=args.limit,
                )
                records = list(scan_dataset(options))

            splits, meta = split_records(
                records,
                val_ratio=float(args.val_ratio),
                test_ratio=float(args.test_ratio),
                seed=int(args.seed),
                strategy=str(args.strategy),
                keep_unknown=bool(args.keep_unknown),
            )

            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            for name, split_records_ in splits.items():
                write_jsonl(out_dir / f"{name}.jsonl", split_records_)

            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            print(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"Wrote: {out_dir}")
            return 0

        if args.dataset_command == "kfold":
            if args.index is None and args.root is None:
                raise SystemExit("Please provide either --index or --root.")

            if args.index is not None:
                records = list(read_jsonl(Path(args.index)))
            else:
                options = DatasetScanOptions(
                    root=Path(args.root),
                    with_size=bool(args.with_size),
                    limit=args.limit,
                )
                records = list(scan_dataset(options))

            fold_splits, meta = kfold_split_records_by_character(
                records,
                folds=int(args.folds),
                seed=int(args.seed),
                val_fold_offset=int(args.val_fold_offset),
                balance_folds=bool(args.balance_folds),
                keep_unknown=bool(args.keep_unknown),
            )

            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, splits in enumerate(fold_splits):
                fold_dir = out_dir / f"fold_{i:02d}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                for name in ["train", "val", "test"]:
                    write_jsonl(fold_dir / f"{name}.jsonl", splits[name])
                # unknown（若有）
                if "unknown" in splits:
                    write_jsonl(fold_dir / "unknown.jsonl", splits["unknown"])

                fold_meta = (meta.get("folds_meta") or [])[i] if isinstance(meta.get("folds_meta"), list) else None
                if fold_meta is not None:
                    (fold_dir / "meta.json").write_text(
                        json.dumps(fold_meta, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )

            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            print(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"Wrote: {out_dir}")
            return 0

    if args.command == "train":
        if args.train_command == "softmax":
            config = SoftmaxTrainConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                reg=float(args.reg),
                seed=int(args.seed),
                device=str(args.device),
                channels=str(args.channels),
                dtype=str(args.dtype),
                shape_mode=str(args.shape_mode),
                image_size=tuple(args.image_size) if args.image_size is not None else None,
                augment_hflip=bool(args.augment_hflip),
                max_train=args.max_train,
                max_val=args.max_val,
                max_test=args.max_test,
                verbose=not bool(args.quiet),
            )
            metrics = train_softmax_from_splits(
                splits_dir=Path(args.splits_dir),
                out_model_path=Path(args.out_model),
                config=config,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0

        if args.train_command == "mlp":
            config = MLPTrainConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                reg=float(args.reg),
                seed=int(args.seed),
                device=str(args.device),
                channels=str(args.channels),
                dtype=str(args.dtype),
                shape_mode=str(args.shape_mode),
                image_size=tuple(args.image_size) if args.image_size is not None else None,
                augment_hflip=bool(args.augment_hflip),
                hidden_sizes=tuple(int(x) for x in args.hidden_sizes),
                max_train=args.max_train,
                max_val=args.max_val,
                max_test=args.max_test,
                verbose=not bool(args.quiet),
            )
            metrics = train_mlp_from_splits(
                splits_dir=Path(args.splits_dir),
                out_model_path=Path(args.out_model),
                config=config,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0

        if args.train_command == "cnn":
            config = CNNTrainConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                sample_strategy=str(args.sample_strategy),
                lr=float(args.lr),
                reg=float(args.reg),
                optimizer=str(args.optimizer),
                momentum=float(args.momentum),
                weight_decay=float(args.weight_decay),
                lr_schedule=str(args.lr_schedule),
                lr_decay_factor=float(args.lr_decay_factor),
                lr_decay_epochs=int(args.lr_decay_epochs),
                label_smoothing=float(args.label_smoothing),
                save_best_val=bool(args.save_best_val),
                early_stopping_patience=int(args.early_stopping_patience),
                early_stopping_min_delta=float(args.early_stopping_min_delta),
                early_stopping_metric=str(args.early_stopping_metric),
                restore_best_val=not bool(args.no_restore_best_val),
                seed=int(args.seed),
                device=str(args.device),
                amp=bool(args.amp),
                max_grad_norm=float(args.max_grad_norm) if args.max_grad_norm is not None else None,
                channels=str(args.channels),
                dtype=str(args.dtype),
                shape_mode=str(args.shape_mode),
                image_size=tuple(args.image_size) if args.image_size is not None else None,
                cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
                augment_hflip=bool(args.augment_hflip),
                augment_random_resized_crop=bool(args.augment_rrc),
                augment_rrc_prob=float(args.augment_rrc_prob),
                augment_rrc_scale=tuple(float(x) for x in args.augment_rrc_scale),
                augment_brightness=float(args.augment_brightness),
                augment_contrast=float(args.augment_contrast),
                augment_cutout=bool(args.augment_cutout),
                augment_cutout_prob=float(args.augment_cutout_prob),
                augment_cutout_scale=tuple(float(x) for x in args.augment_cutout_scale),
                conv_out_channels=int(args.conv_out_channels),
                conv2_out_channels=int(args.conv2_out_channels),
                kernel_size=int(args.kernel_size),
                conv_stride=int(args.conv_stride),
                pool_size=int(args.pool_size),
                hidden_size=int(args.hidden_size),
                dropout=float(args.dropout),
                batch_norm=bool(args.batch_norm),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=int(args.prefetch_factor) if args.prefetch_factor is not None else None,
                max_train=args.max_train,
                max_val=args.max_val,
                max_test=args.max_test,
                verbose=not bool(args.quiet),
            )
            metrics = train_cnn_from_splits(
                splits_dir=Path(args.splits_dir),
                out_model_path=Path(args.out_model),
                config=config,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0

    if args.command == "eval":
        model = load_model(Path(args.model), device=str(args.device))
        if args.data is not None:
            records = iter_records_from_jsonl(Path(args.data), limit=args.limit)
        else:
            split_path = Path(args.splits_dir) / f"{args.split}.jsonl"
            records = iter_records_from_jsonl(split_path, limit=args.limit)

        result = evaluate(
            model,
            records=records,
            batch_size=int(args.batch_size),
            shape_mode=str(args.shape_mode),
            out_predictions=Path(args.out_predictions) if args.out_predictions is not None else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not bool(args.json_only):
            cm = np.array(result["confusion_matrix"], dtype=int)
            print()
            print(format_confusion_matrix(cm, result["class_names"]))
        return 0

    if args.command == "predict":
        model = load_model(Path(args.model), device=str(args.device))
        result = predict_image(model, image_path=Path(args.image), shape_mode=str(args.shape_mode))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "analyze":
        if args.analyze_command == "errors":
            model = load_model(Path(args.model), device=str(args.device))
            if args.data is not None:
                records = iter_records_from_jsonl(Path(args.data), limit=args.limit)
            else:
                split_path = Path(args.splits_dir) / f"{args.split}.jsonl"
                records = iter_records_from_jsonl(split_path, limit=args.limit)

            out_dir = Path(args.out_dir) if args.out_dir is not None else Path("data/reports") / time.strftime("errors_%Y%m%d_%H%M%S")
            summary = analyze_errors(
                model,
                records=records,
                out_dir=out_dir,
                config=ErrorAnalysisConfig(
                    batch_size=int(args.batch_size),
                    shape_mode=str(args.shape_mode),
                    max_per_pair=int(args.max_per_pair),
                    thumb_size=int(args.thumb_size),
                    seed=int(args.seed),
                ),
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"Wrote: {out_dir}")
            return 0

    if args.command == "experiment":
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path("data/experiments") / time.strftime("%Y%m%d_%H%M%S")
        should_analyze_errors = args.analyze_errors
        if should_analyze_errors is None:
            should_analyze_errors = bool(args.splits_dir is not None)
        suite = run_experiment_suite(
            splits_dir=Path(args.splits_dir) if args.splits_dir is not None else None,
            folds_dir=Path(args.folds_dir) if args.folds_dir is not None else None,
            out_dir=out_dir,
            presets=list(args.presets),
            device=str(args.device),
            io=ExperimentIOConfig(
                eval_batch_size=int(args.eval_batch_size),
                eval_limit=args.eval_limit,
                analyze_errors=bool(should_analyze_errors),
                errors_limit=args.errors_limit,
                errors_thumb_size=int(args.errors_thumb_size),
                errors_max_per_pair=int(args.errors_max_per_pair),
                seed=int(args.seed),
            ),
            max_folds=args.max_folds,
        )
        print(json.dumps(suite, ensure_ascii=False, indent=2))
        print(f"Wrote: {out_dir}")
        return 0

    parser.error("Unknown command")
    return 2
