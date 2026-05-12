#!/usr/bin/env python3
"""Generate Experiment 1 Grad-CAM maps through the FaceNet512 backbone.

The saved E1 checkpoint is a PyTorch MLP over frozen FaceNet512 embeddings. To
make Grad-CAM possible, this script rebuilds the inference path in TensorFlow:
aligned face crop -> FaceNet512 Keras model -> E1 MLP head converted from the
PyTorch checkpoint. Gradients flow from the E1 class logit back to FaceNet's
last spatial convolution block.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from PIL import Image

from deepface.modules import detection, modeling, preprocessing


CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
CONV_LAYER_NAME = "Block8_6_Conv2d_1x1"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_best_epoch(history_path: Path) -> tuple[int | None, float | None]:
    if not history_path.exists():
        return None, None

    best_epoch = None
    best_value = None
    with history_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = float(row["val_macro_f1"])
            if best_value is None or value > best_value:
                best_value = value
                best_epoch = int(row["epoch"])
    return best_epoch, best_value


def image_path_from_manifest_row(row: dict[str, str], raw_root: Path) -> Path:
    relative = Path(row["relative_path"]).with_suffix(".png")
    image_path = raw_root / relative
    if image_path.exists():
        return image_path
    for suffix in [".jpg", ".jpeg"]:
        candidate = raw_root / relative.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for manifest row: {row}")


def prepare_face_tensor(image_path: Path, detector_backend: str, target_size: tuple[int, int]) -> np.ndarray:
    img_objs = detection.extract_faces(
        img_path=str(image_path),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=False,
        align=True,
        expand_percentage=0,
        anti_spoofing=False,
        max_faces=1,
    )
    if not img_objs:
        raise ValueError(f"No face extracted from {image_path}")

    img = img_objs[0]["face"]
    img = img[:, :, ::-1]
    img = preprocessing.resize_image(img=img, target_size=(target_size[1], target_size[0]))
    img = preprocessing.normalize_input(img=img, normalization="base")
    img = img.astype(np.float32)
    if img.ndim == 4:
        img = img[0]
    return img


class TensorFlowMLP:
    def __init__(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["model"]
        self.input_dim = int(ckpt.get("input_dim", 512))
        self.params = {key: tf.constant(value.detach().cpu().numpy(), dtype=tf.float32) for key, value in state.items()}

    @staticmethod
    def linear(x: tf.Tensor, weight: tf.Tensor, bias: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matmul(x, weight, transpose_b=True) + bias

    @staticmethod
    def batch_norm(
        x: tf.Tensor,
        gamma: tf.Tensor,
        beta: tf.Tensor,
        running_mean: tf.Tensor,
        running_var: tf.Tensor,
        eps: float = 1e-5,
    ) -> tf.Tensor:
        return (x - running_mean) / tf.sqrt(running_var + eps) * gamma + beta

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        p = self.params
        x = self.linear(x, p["net.0.weight"], p["net.0.bias"])
        x = self.batch_norm(x, p["net.1.weight"], p["net.1.bias"], p["net.1.running_mean"], p["net.1.running_var"])
        x = tf.nn.relu(x)
        x = self.linear(x, p["net.4.weight"], p["net.4.bias"])
        x = self.batch_norm(x, p["net.5.weight"], p["net.5.bias"], p["net.5.running_mean"], p["net.5.running_var"])
        x = tf.nn.relu(x)
        return self.linear(x, p["net.8.weight"], p["net.8.bias"])


def choose_examples(rows: list[dict[str, str]], per_class: int, seed: int) -> dict[str, list[dict[str, str]]]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)

    rng = random.Random(seed)
    examples = {}
    for label in CLASSES:
        candidates = by_class[label]
        if len(candidates) < per_class:
            raise ValueError(f"Not enough test rows for {label}: found {len(candidates)}")
        examples[label] = rng.sample(candidates, per_class)
    return examples


def gradcam(
    face_tensor: np.ndarray,
    grad_model: tf.keras.Model,
    mlp: TensorFlowMLP,
    target_class: int,
) -> np.ndarray:
    if face_tensor.ndim == 4:
        face_tensor = face_tensor[0]
    x = tf.convert_to_tensor(face_tensor[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, embedding = grad_model(x, training=False)
        tape.watch(conv_out)
        logits = mlp(embedding)
        score = logits[:, target_class]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_out, axis=-1)
    cam = tf.nn.relu(cam)[0]
    cam = tf.image.resize(cam[..., None], (face_tensor.shape[0], face_tensor.shape[1]), method="bilinear")[..., 0]
    cam_np = cam.numpy()
    cam_min = float(cam_np.min())
    cam_max = float(cam_np.max())
    if cam_max - cam_min <= 1e-8:
        return np.zeros_like(cam_np)
    return (cam_np - cam_min) / (cam_max - cam_min)


def overlay_heatmap(face_tensor: np.ndarray, cam: np.ndarray) -> np.ndarray:
    base = np.clip(face_tensor, 0, 1)
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)


def render_grid(
    examples: dict[str, list[dict[str, str]]],
    raw_root: Path,
    detector_backend: str,
    grad_model: tf.keras.Model,
    mlp: TensorFlowMLP,
    output_path: Path,
) -> None:
    rows = len(CLASSES)
    cols = len(next(iter(examples.values())))
    fig_width = max(cols * 2.6, 4.4)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, rows * 2.45))
    axes = np.asarray(axes).reshape(rows, cols)

    for row_idx, label in enumerate(CLASSES):
        target_class = CLASSES.index(label)
        for col_idx, sample in enumerate(examples[label]):
            ax = axes[row_idx, col_idx]
            image_path = image_path_from_manifest_row(sample, raw_root)
            face_tensor = prepare_face_tensor(image_path, detector_backend, target_size=(160, 160))
            cam = gradcam(face_tensor, grad_model, mlp, target_class)
            ax.imshow(overlay_heatmap(face_tensor, cam))
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle("Experiment 1 FaceNet Grad-CAM", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("experiments/experiment1_sujal_style/manifest_experiment1.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/experiment1_sujal_style/best_model.pt"))
    parser.add_argument("--history", type=Path, default=Path("experiments/experiment1_sujal_style/history.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("Zip_files/CAER-S"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/experiment1_sujal_style/saliency"))
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector-backend", type=str, default="opencv")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    split_counts = Counter(row["split"] for row in rows)
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise ValueError(f"No test rows found in {args.manifest}")

    facenet_client = modeling.build_model(task="facial_recognition", model_name="Facenet512")
    facenet = facenet_client.model
    grad_model = tf.keras.Model(
        inputs=facenet.inputs,
        outputs=[facenet.get_layer(CONV_LAYER_NAME).output, facenet.output],
    )
    mlp = TensorFlowMLP(args.checkpoint)

    examples = choose_examples(test_rows, args.per_class, args.seed)
    grid_path = args.output_dir / "gradcam_e1_facenet_grid.png"
    render_grid(examples, args.raw_root, args.detector_backend, grad_model, mlp, grid_path)

    best_epoch, best_val_macro_f1 = find_best_epoch(args.history)
    metadata = {
        "analysis": "Experiment 1 FaceNet512 Grad-CAM",
        "method_note": "Grad-CAM is computed through the FaceNet512 backbone, with the E1 PyTorch MLP head converted to TensorFlow for gradient flow.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_selection": "best validation macro-F1 checkpoint saved during E1 training",
        "best_epoch_from_history": best_epoch,
        "best_val_macro_f1_from_history": best_val_macro_f1,
        "data_used_for_saliency": "CAER-S in-domain test split",
        "manifest": str(args.manifest),
        "raw_root": str(args.raw_root),
        "split_counts": dict(split_counts),
        "test_rows_used_for_sampling": len(test_rows),
        "classes": CLASSES,
        "examples_per_class": args.per_class,
        "seed": args.seed,
        "detector_backend": args.detector_backend,
        "facenet_gradcam_layer": CONV_LAYER_NAME,
        "output_grid": str(grid_path),
    }
    metadata_path = args.output_dir / "gradcam_e1_facenet_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved FaceNet Grad-CAM grid: {grid_path}")
    print(f"Saved metadata: {metadata_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
