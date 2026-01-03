from __future__ import annotations

import html
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dataset_index import ImageRecord
from .inference import LoadedModel, evaluate, summarize_confusion_matrix


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    batch_size: int = 64
    shape_mode: str = "auto"
    max_per_pair: int = 36
    thumb_size: int = 224
    seed: int = 0


def _safe_name(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("._")
    return name or "unknown"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _letterbox_rgb(img: Image.Image, *, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), (0, 0, 0))

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _annotate(img: Image.Image, *, lines: list[str]) -> Image.Image:
    if not lines:
        return img
    font = ImageFont.load_default()
    img_rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    line_height = font.getbbox("Ag")[3] + 2
    pad = 4
    box_h = pad * 2 + line_height * len(lines)
    draw.rectangle((0, 0, img_rgba.width, box_h), fill=(0, 0, 0, 160))

    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height

    out = Image.alpha_composite(img_rgba, overlay)
    return out.convert("RGB")


def _write_thumbnail(pred: dict, *, out_path: Path, size: int) -> None:
    src = pred["path"]
    with Image.open(src) as img:
        thumb = _letterbox_rgb(img, size=size)

    character = pred.get("character") or "unknown"
    true_name = pred.get("true_class_name")
    pred_name = pred.get("predicted_class_name")
    p_true = pred.get("true_probability")
    p_pred = pred.get("predicted_probability")
    margin = pred.get("margin")

    def _fmt(v) -> str:
        if v is None:
            return "NA"
        return f"{float(v):.3f}"

    lines = [
        f"{character}  {true_name}→{pred_name}",
        f"p_true={_fmt(p_true)} p_pred={_fmt(p_pred)} margin={_fmt(margin)}",
    ]
    thumb = _annotate(thumb, lines=lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(out_path)


def _write_grid(image_paths: list[Path], *, out_path: Path, size: int, cols: int = 8, pad: int = 4) -> None:
    if not image_paths:
        return

    cols = max(1, int(cols))
    rows = int(math.ceil(len(image_paths) / cols))
    grid_w = pad + cols * (size + pad)
    grid_h = pad + rows * (size + pad)
    canvas = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))

    for idx, path in enumerate(image_paths):
        with Image.open(path) as img:
            img = img.convert("RGB")
        r = idx // cols
        c = idx % cols
        x = pad + c * (size + pad)
        y = pad + r * (size + pad)
        canvas.paste(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _confusion_pairs(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    pairs: list[dict] = []
    n = int(cm.shape[0])
    for t in range(n):
        for p in range(n):
            if t == p:
                continue
            count = int(cm[t, p])
            if count <= 0:
                continue
            support = int(np.sum(cm[t, :]))
            rate = None if support == 0 else float(count / support)
            pairs.append(
                {
                    "true_class_id": t,
                    "true_class_name": class_names[t] if t < len(class_names) else str(t),
                    "predicted_class_id": p,
                    "predicted_class_name": class_names[p] if p < len(class_names) else str(p),
                    "count": count,
                    "rate": rate,
                    "support": support,
                }
            )
    pairs.sort(key=lambda d: (-int(d["count"]), str(d["true_class_name"]), str(d["predicted_class_name"])))
    return pairs


def _html_table_confusion(cm: np.ndarray, class_names: list[str]) -> str:
    esc = html.escape
    header = "".join(f"<th>{esc(name)}</th>" for name in class_names)
    rows = []
    for i, name in enumerate(class_names):
        cells = "".join(f"<td>{int(v)}</td>" for v in cm[i])
        rows.append(f"<tr><th>{esc(name)}</th>{cells}</tr>")
    return (
        "<table class='cm'>"
        "<thead><tr><th>true\\pred</th>"
        f"{header}</tr></thead>"
        "<tbody>"
        f"{''.join(rows)}"
        "</tbody></table>"
    )


def analyze_errors(
    model: LoadedModel,
    *,
    records: Iterable[ImageRecord],
    out_dir: Path,
    config: ErrorAnalysisConfig,
) -> dict:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    predictions_path = out_dir / "predictions.jsonl"
    overall = evaluate(
        model,
        records=records,
        batch_size=int(config.batch_size),
        shape_mode=str(config.shape_mode),
        return_predictions=True,
        out_predictions=predictions_path,
    )

    preds: list[dict] = list(overall.get("predictions") or [])
    overall_no_preds = {k: v for k, v in overall.items() if k != "predictions"}

    class_names = list(overall_no_preds.get("class_names") or [])
    num_classes = int(len(class_names))
    cm_global = np.array(overall_no_preds.get("confusion_matrix"), dtype=int)

    by_character_cm: dict[str, np.ndarray] = {}
    by_character_n: dict[str, int] = {}
    by_character_correct: dict[str, int] = {}

    for pred in preds:
        character = str(pred.get("character") or "unknown")
        true_id = int(pred.get("true_class_id"))
        pred_id = int(pred.get("predicted_class_id"))
        correct = bool(pred.get("correct"))

        if character not in by_character_cm:
            by_character_cm[character] = np.zeros((num_classes, num_classes), dtype=int)
            by_character_n[character] = 0
            by_character_correct[character] = 0

        by_character_cm[character][true_id, pred_id] += 1
        by_character_n[character] += 1
        if correct:
            by_character_correct[character] += 1

    per_character = []
    for character in sorted(by_character_cm.keys()):
        cm = by_character_cm[character]
        n = int(by_character_n[character])
        acc = None if n == 0 else float(by_character_correct[character] / n)
        per_character.append(
            {
                "character": character,
                "n": n,
                "accuracy": acc,
                "confusion_matrix": cm.tolist(),
                **summarize_confusion_matrix(cm),
                "top_confusions": _confusion_pairs(cm, class_names)[:10],
            }
        )

    top_confusions = _confusion_pairs(cm_global, class_names)[:10]

    name_to_id = {str(k): int(v) for k, v in model.class_map.items()}
    action_id = name_to_id.get("action")
    pose_id = name_to_id.get("pose")
    focus_pairs: list[tuple[int, int]] = []
    if action_id is not None and pose_id is not None:
        focus_pairs = [(int(action_id), int(pose_id)), (int(pose_id), int(action_id))]

    rng = random.Random(int(config.seed))
    thumbs_dir = out_dir / "thumbs"
    grids_dir = out_dir / "grids"
    _ensure_dir(thumbs_dir)
    _ensure_dir(grids_dir)

    artifacts: dict[str, str] = {
        "predictions_jsonl": str(predictions_path),
    }
    focus_artifacts: list[dict] = []

    def _sample(items: list[dict], k: int) -> list[dict]:
        if k <= 0 or not items:
            return []
        if len(items) <= k:
            return list(items)
        return rng.sample(items, k=k)

    # Global focus (action↔pose)
    for true_id, pred_id in focus_pairs:
        bucket = [p for p in preds if int(p.get("true_class_id")) == true_id and int(p.get("predicted_class_id")) == pred_id]
        bucket = [p for p in bucket if not bool(p.get("correct"))]
        bucket = _sample(bucket, int(config.max_per_pair))
        if not bucket:
            continue

        pair_name = f"{class_names[true_id]}_to_{class_names[pred_id]}"
        pair_dir = thumbs_dir / "global" / pair_name
        grid_out = grids_dir / "global" / f"{pair_name}.png"
        _ensure_dir(pair_dir)

        thumb_paths: list[Path] = []
        for i, p in enumerate(bucket):
            thumb_path = pair_dir / f"{i:03d}.png"
            _write_thumbnail(p, out_path=thumb_path, size=int(config.thumb_size))
            thumb_paths.append(thumb_path)

        _write_grid(thumb_paths, out_path=grid_out, size=int(config.thumb_size))
        focus_artifacts.append(
            {
                "scope": "global",
                "true_class_id": true_id,
                "predicted_class_id": pred_id,
                "true_class_name": class_names[true_id],
                "predicted_class_name": class_names[pred_id],
                "n_errors_visualized": len(bucket),
                "grid": str(grid_out.relative_to(out_dir)),
                "thumb_dir": str(pair_dir.relative_to(out_dir)),
            }
        )

    # Per-character focus (action↔pose)
    for character in sorted(by_character_cm.keys()):
        for true_id, pred_id in focus_pairs:
            bucket = [
                p
                for p in preds
                if str(p.get("character") or "unknown") == character
                and int(p.get("true_class_id")) == true_id
                and int(p.get("predicted_class_id")) == pred_id
                and not bool(p.get("correct"))
            ]
            bucket = _sample(bucket, int(config.max_per_pair))
            if not bucket:
                continue

            pair_name = f"{class_names[true_id]}_to_{class_names[pred_id]}"
            pair_dir = thumbs_dir / "by_character" / _safe_name(character) / pair_name
            grid_out = grids_dir / "by_character" / _safe_name(character) / f"{pair_name}.png"
            _ensure_dir(pair_dir)

            thumb_paths: list[Path] = []
            for i, p in enumerate(bucket):
                thumb_path = pair_dir / f"{i:03d}.png"
                _write_thumbnail(p, out_path=thumb_path, size=int(config.thumb_size))
                thumb_paths.append(thumb_path)
            _write_grid(thumb_paths, out_path=grid_out, size=int(config.thumb_size))

            focus_artifacts.append(
                {
                    "scope": "character",
                    "character": character,
                    "true_class_id": true_id,
                    "predicted_class_id": pred_id,
                    "true_class_name": class_names[true_id],
                    "predicted_class_name": class_names[pred_id],
                    "n_errors_visualized": len(bucket),
                    "grid": str(grid_out.relative_to(out_dir)),
                    "thumb_dir": str(pair_dir.relative_to(out_dir)),
                }
            )

    artifacts["focus_pairs"] = focus_artifacts

    report_html = out_dir / "report.html"
    per_character_html = []
    for item in per_character:
        character = str(item["character"])
        cm = np.array(item["confusion_matrix"], dtype=int)
        per_character_html.append(
            "<section>"
            f"<h2>Character: {html.escape(character)}</h2>"
            f"<p>n={int(item['n'])} accuracy={item['accuracy']}</p>"
            f"{_html_table_confusion(cm, class_names)}"
            "</section>"
        )

    focus_html = []
    for art in focus_artifacts:
        title_bits = []
        if art.get("scope") == "character":
            title_bits.append(f"character={art.get('character')}")
        title_bits.append(f"{art.get('true_class_name')}→{art.get('predicted_class_name')}")
        title = " / ".join(title_bits)
        grid_rel = art.get("grid")
        thumb_rel = art.get("thumb_dir")
        focus_html.append(
            "<section>"
            f"<h2>{html.escape(title)}</h2>"
            f"<p>visualized={int(art.get('n_errors_visualized', 0))} "
            f"thumbs=<code>{html.escape(str(thumb_rel))}</code></p>"
            f"<img class='grid' src='{html.escape(str(grid_rel))}' />"
            "</section>"
        )

    global_cm_table = _html_table_confusion(cm_global, class_names)
    top_confusions_html = "".join(
        f"<li>{html.escape(str(x['true_class_name']))}→{html.escape(str(x['predicted_class_name']))}: "
        f"{int(x['count'])} (rate={x['rate']})</li>"
        for x in top_confusions
    )
    report_html_text = "".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'/>",
            "<style>",
            "body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:24px;}",
            "code{background:#f2f2f2; padding:2px 4px; border-radius:4px;}",
            "table.cm{border-collapse:collapse; margin:12px 0;}",
            "table.cm th, table.cm td{border:1px solid #ddd; padding:6px 8px; text-align:right;}",
            "table.cm th:first-child{text-align:left;}",
            "img.grid{max-width:100%; height:auto; border:1px solid #ddd;}",
            "</style></head><body>",
            "<h1>VML Error Analysis</h1>",
            f"<p>predictions: <code>{html.escape(str(predictions_path.relative_to(out_dir)))}</code></p>",
            "<h2>Overall</h2>",
            f"<p>n={int(overall_no_preds.get('n') or 0)} "
            f"accuracy={overall_no_preds.get('accuracy')} loss={overall_no_preds.get('loss')}</p>",
            str(global_cm_table),
            "<h2>Top Confusions</h2>",
            "<ol>",
            top_confusions_html,
            "</ol>",
            "<h2>Focus: action↔pose</h2>",
            "".join(focus_html),
            "<h2>Per-character</h2>",
            "".join(per_character_html),
            "</body></html>",
        ]
    )
    report_html.write_text(report_html_text, encoding="utf-8")

    artifacts["report_html"] = str(report_html)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "overall": overall_no_preds,
                "per_character": per_character,
                "top_confusions": top_confusions,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "overall": overall_no_preds,
        "per_character": per_character,
        "top_confusions": top_confusions,
        "artifacts": artifacts,
    }
