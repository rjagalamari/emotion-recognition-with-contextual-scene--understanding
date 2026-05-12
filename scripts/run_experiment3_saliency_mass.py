#!/usr/bin/env python3
"""Quantify Experiment 3 Grad-CAM mass over face, body, and background.

For each sampled CAER-S test image:
- compute E3 Grad-CAM through the Places365 scene branch
- detect a face box with DeepFace/OpenCV
- detect person boxes with a COCO Faster R-CNN detector
- measure heatmap mass in face, body (person minus face), and background
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from deepface.modules import detection
from run_experiment3_saliency_gradcam import (
    CLASSES,
    CrossGatedTokenFusionClassifier,
    gradcam_e3,
    image_path_from_manifest_row,
    load_places365_resnet18,
    read_manifest,
)


def map_box_to_center_crop(
    box: tuple[float, float, float, float],
    original_size: tuple[int, int],
    output_size: int = 224,
    resize_short_side: int = 256,
) -> tuple[int, int, int, int] | None:
    width, height = original_size
    scale = resize_short_side / min(width, height)
    resized_w = width * scale
    resized_h = height * scale
    offset_x = (resized_w - output_size) / 2.0
    offset_y = (resized_h - output_size) / 2.0

    x0, y0, x1, y1 = box
    mapped = (
        x0 * scale - offset_x,
        y0 * scale - offset_y,
        x1 * scale - offset_x,
        y1 * scale - offset_y,
    )
    clipped = (
        max(0, min(output_size, int(round(mapped[0])))),
        max(0, min(output_size, int(round(mapped[1])))),
        max(0, min(output_size, int(round(mapped[2])))),
        max(0, min(output_size, int(round(mapped[3])))),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def mask_from_boxes(boxes: list[tuple[int, int, int, int]], size: int = 224) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = True
    return mask


def detect_face_boxes(image_path: Path, original_size: tuple[int, int], detector_backend: str) -> list[tuple[int, int, int, int]]:
    try:
        faces = detection.extract_faces(
            img_path=str(image_path),
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=False,
            align=False,
            expand_percentage=0,
            anti_spoofing=False,
            max_faces=3,
        )
    except Exception:
        return []

    boxes = []
    for face in faces:
        area = face.get("facial_area", {})
        x = float(area.get("x", 0))
        y = float(area.get("y", 0))
        w = float(area.get("w", 0))
        h = float(area.get("h", 0))
        mapped = map_box_to_center_crop((x, y, x + w, y + h), original_size)
        if mapped is not None:
            boxes.append(mapped)
    return boxes


def load_person_detector(device: torch.device):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.25).to(device)
    model.eval()
    return model, weights.transforms()


def detect_person_boxes(
    image: Image.Image,
    detector,
    detector_transform,
    device: torch.device,
    score_threshold: float,
) -> list[tuple[int, int, int, int]]:
    tensor = detector_transform(image.convert("RGB")).to(device)
    with torch.no_grad():
        out = detector([tensor])[0]

    boxes = []
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        if int(label.detach().cpu()) != 1 or float(score.detach().cpu()) < score_threshold:
            continue
        x0, y0, x1, y1 = [float(v) for v in box.detach().cpu().tolist()]
        mapped = map_box_to_center_crop((x0, y0, x1, y1), image.size)
        if mapped is not None:
            boxes.append(mapped)
    return boxes


def choose_sample(rows: list[dict[str, str]], sample_n: int, seed: int) -> list[dict[str, str]]:
    if sample_n <= 0 or sample_n >= len(rows):
        return list(rows)

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)

    rng = random.Random(seed)
    per_class = sample_n // len(CLASSES)
    remainder = sample_n % len(CLASSES)
    chosen = []
    for idx, label in enumerate(CLASSES):
        count = per_class + (1 if idx < remainder else 0)
        candidates = by_class[label]
        chosen.extend(rng.sample(candidates, min(count, len(candidates))))
    rng.shuffle(chosen)
    return chosen


def mass_values(cam: np.ndarray, face_mask: np.ndarray, person_mask: np.ndarray) -> dict[str, float]:
    total = float(cam.sum())
    if total <= 1e-12:
        return {
            "face_mass": 0.0,
            "body_mass": 0.0,
            "background_mass": 0.0,
            "total_mass": 0.0,
        }

    body_mask = np.logical_and(person_mask, np.logical_not(face_mask))
    foreground_mask = np.logical_or(person_mask, face_mask)
    background_mask = np.logical_not(foreground_mask)
    return {
        "face_mass": float(cam[face_mask].sum() / total),
        "body_mass": float(cam[body_mask].sum() / total),
        "background_mass": float(cam[background_mask].sum() / total),
        "total_mass": total,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    groups: dict[str, list[dict[str, object]]] = {"all": rows}
    for label in CLASSES:
        groups[label] = [row for row in rows if row["label"] == label]

    for label, group in groups.items():
        if not group:
            continue
        item: dict[str, object] = {"label": label, "n": len(group)}
        for key in ["face_mass", "body_mass", "background_mass"]:
            values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        item["face_detect_rate"] = float(np.mean([bool(row["face_detected"]) for row in group]))
        item["person_detect_rate"] = float(np.mean([bool(row["person_detected"]) for row in group]))
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, summary_rows: list[dict[str, object]]) -> None:
    rows = [row for row in summary_rows if row["label"] != "all"]
    labels = [str(row["label"]) for row in rows]
    face = [float(row["face_mass_mean"]) for row in rows]
    body = [float(row["body_mass_mean"]) for row in rows]
    background = [float(row["background_mass_mean"]) for row in rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x, face, label="face")
    ax.bar(x, body, bottom=face, label="body")
    ax.bar(x, background, bottom=np.asarray(face) + np.asarray(body), label="background")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Grad-CAM mass")
    ax.set_title("Experiment 3 Grad-CAM Mass by Region")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/experiment3_3_places365_cross_gated/manifest_experiment3_3_places365.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/experiment3_3_places365_cross_gated/best_model.pt"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("Zip_files/CAER-S"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/experiment3_3_places365_cross_gated/saliency"))
    parser.add_argument("--sample-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--person-threshold", type=float, default=0.5)
    parser.add_argument("--face-detector", type=str, default="opencv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = read_manifest(args.manifest)
    split_counts = Counter(row["split"] for row in all_rows)
    test_rows = [row for row in all_rows if row["split"] == "test"]
    sampled_rows = choose_sample(test_rows, args.sample_n, args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    fusion_head = CrossGatedTokenFusionClassifier(num_classes=len(CLASSES), dropout=0.3, num_heads=4).to(device)
    fusion_head.load_state_dict(checkpoint["model"])
    fusion_head.eval()

    scene_backbone = load_places365_resnet18(device)
    person_detector, person_transform = load_person_detector(device)

    per_image = []
    for idx, row in enumerate(sampled_rows, start=1):
        image_path = image_path_from_manifest_row(row, args.raw_root)
        image = Image.open(image_path).convert("RGB")
        label = row["label"]
        target_class = CLASSES.index(label)
        face_embedding = np.asarray(np.load(row["face_path"]), dtype=np.float32)

        cam = gradcam_e3(image, face_embedding, target_class, scene_backbone, fusion_head, device)
        face_boxes = detect_face_boxes(image_path, image.size, args.face_detector)
        person_boxes = detect_person_boxes(image, person_detector, person_transform, device, args.person_threshold)

        face_mask = mask_from_boxes(face_boxes)
        person_mask = mask_from_boxes(person_boxes)
        values = mass_values(cam, face_mask, person_mask)

        per_image.append(
            {
                "index": idx,
                "relative_path": row["relative_path"],
                "label": label,
                "face_mass": values["face_mass"],
                "body_mass": values["body_mass"],
                "background_mass": values["background_mass"],
                "total_mass": values["total_mass"],
                "face_detected": bool(face_boxes),
                "person_detected": bool(person_boxes),
                "num_face_boxes": len(face_boxes),
                "num_person_boxes": len(person_boxes),
            }
        )

        if idx % 25 == 0 or idx == len(sampled_rows):
            print(f"Processed {idx}/{len(sampled_rows)}")

    summary_rows = summarize(per_image)
    per_image_path = args.output_dir / "gradcam_e3_mass_per_image.csv"
    summary_path = args.output_dir / "gradcam_e3_mass_summary.csv"
    plot_path = args.output_dir / "gradcam_e3_mass_summary.png"
    metadata_path = args.output_dir / "gradcam_e3_mass_metadata.json"

    write_csv(per_image_path, per_image)
    write_csv(summary_path, summary_rows)
    plot_summary(plot_path, summary_rows)

    metadata = {
        "analysis": "Experiment 3 Grad-CAM face/body/background mass",
        "checkpoint": str(args.checkpoint),
        "data_used": "CAER-S in-domain test split",
        "manifest": str(args.manifest),
        "split_counts": dict(split_counts),
        "test_rows_available": len(test_rows),
        "sample_n": len(sampled_rows),
        "seed": args.seed,
        "person_detector": "torchvision Faster R-CNN ResNet50-FPN COCO person class",
        "person_threshold": args.person_threshold,
        "face_detector": f"DeepFace {args.face_detector}",
        "region_definition": "face = detected face boxes; body = detected person boxes excluding face; background = outside the union of detected person and face boxes",
        "per_image_csv": str(per_image_path),
        "summary_csv": str(summary_path),
        "summary_plot": str(plot_path),
        "summary": summary_rows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
