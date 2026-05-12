#!/usr/bin/env python3
"""Generate Experiment 3 Grad-CAM saliency maps.

Experiment 3 fuses frozen FaceNet512 face embeddings with Places365 ResNet-18
scene embeddings. Grad-CAM is computed through the scene backbone while the
cached face embedding is held fixed, then the trained cross-gated E3 fusion head
produces the target class logit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


class CrossGatedTokenFusionClassifier(nn.Module):
    def __init__(
        self,
        face_dim: int = 512,
        context_dim: int = 512,
        token_dim: int = 128,
        num_tokens: int = 4,
        num_classes: int = 7,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.face_proj = nn.Sequential(
            nn.Linear(face_dim, num_tokens * token_dim),
            nn.LayerNorm(num_tokens * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, num_tokens * token_dim),
            nn.LayerNorm(num_tokens * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.face_type = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)
        self.context_type = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)
        self.face_to_context = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.context_to_face = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.face_ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.context_ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.face_norm = nn.LayerNorm(token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.face_confidence = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 1), nn.Sigmoid())
        self.context_confidence = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 1), nn.Sigmoid()
        )
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 4, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(token_dim * 5, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, face: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size = face.shape[0]
        face_tokens = self.face_proj(face).view(batch_size, self.num_tokens, self.token_dim) + self.face_type
        context_tokens = self.context_proj(context).view(batch_size, self.num_tokens, self.token_dim) + self.context_type

        face_cross, _ = self.face_to_context(face_tokens, context_tokens, context_tokens, need_weights=False)
        context_cross, _ = self.context_to_face(context_tokens, face_tokens, face_tokens, need_weights=False)
        face_tokens = self.face_norm(face_tokens + face_cross)
        context_tokens = self.context_norm(context_tokens + context_cross)
        face_tokens = face_tokens + self.face_ffn(face_tokens)
        context_tokens = context_tokens + self.context_ffn(context_tokens)

        face_pool = face_tokens.mean(dim=1)
        context_pool = context_tokens.mean(dim=1)
        face_weight = self.face_confidence(face_pool)
        context_weight = self.context_confidence(context_pool)
        weight_sum = face_weight + context_weight + 1e-6
        face_weight = face_weight / weight_sum
        context_weight = context_weight / weight_sum

        gate = self.gate(torch.cat([face_pool, context_pool, face_pool * context_pool, torch.abs(face_pool - context_pool)], dim=1))
        gated = gate * (face_weight * face_pool) + (1.0 - gate) * (context_weight * context_pool)
        interaction = torch.cat(
            [gated, face_pool, context_pool, face_pool * context_pool, torch.abs(face_pool - context_pool)],
            dim=1,
        )
        return self.classifier(interaction)


PREPROC = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


def load_places365_resnet18(device: torch.device) -> nn.Module:
    model = models.resnet18(num_classes=365)
    url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
    ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
    state_dict = {key.replace("module.", ""): value for key, value in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.fc = nn.Identity()
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


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


def forward_with_conv_hook(
    backbone: nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    container: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        container["conv_act"] = out

    handle = backbone.layer4[-1].register_forward_hook(hook)
    with torch.enable_grad():
        feats = backbone(image_tensor.unsqueeze(0).to(device).requires_grad_(True))
    handle.remove()
    return feats, container["conv_act"]


def gradcam_from_conv_grad(conv_act: torch.Tensor) -> np.ndarray:
    grads = conv_act.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * conv_act).sum(dim=1))
    cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam_np = cam.detach().cpu().numpy()
    cam_min = float(cam_np.min())
    cam_max = float(cam_np.max())
    if cam_max - cam_min <= 1e-8:
        return np.zeros_like(cam_np)
    return (cam_np - cam_min) / (cam_max - cam_min)


def gradcam_e3(
    image: Image.Image,
    face_embedding: np.ndarray,
    target_class: int,
    backbone: nn.Module,
    fusion_head: nn.Module,
    device: torch.device,
) -> np.ndarray:
    image_tensor = PREPROC(image.convert("RGB"))
    scene_feats, conv_act = forward_with_conv_hook(backbone, image_tensor, device)
    face = torch.from_numpy(face_embedding.astype(np.float32)).unsqueeze(0).to(device)
    logits = fusion_head(face, scene_feats)
    logit = logits[0, target_class]
    backbone.zero_grad()
    fusion_head.zero_grad()
    if conv_act.grad is not None:
        conv_act.grad.zero_()
    logit.backward()
    return gradcam_from_conv_grad(conv_act)


def overlay_heatmap(image: Image.Image, cam: np.ndarray) -> np.ndarray:
    base = np.asarray(image.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)


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


def render_grid(
    examples: dict[str, list[dict[str, str]]],
    raw_root: Path,
    backbone: nn.Module,
    fusion_head: nn.Module,
    device: torch.device,
    output_path: Path,
) -> None:
    rows = len(CLASSES)
    cols = len(next(iter(examples.values())))
    fig, axes = plt.subplots(rows, cols, figsize=(max(cols * 2.6, 4.4), rows * 2.45))
    axes = np.asarray(axes).reshape(rows, cols)

    for row_idx, label in enumerate(CLASSES):
        target_class = CLASSES.index(label)
        for col_idx, sample in enumerate(examples[label]):
            ax = axes[row_idx, col_idx]
            image_path = image_path_from_manifest_row(sample, raw_root)
            image = Image.open(image_path)
            face_embedding = np.asarray(np.load(sample["face_path"]), dtype=np.float32)
            cam = gradcam_e3(image, face_embedding, target_class, backbone, fusion_head, device)
            ax.imshow(overlay_heatmap(image, cam))
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle("Experiment 3 Face + Scene Fusion Grad-CAM", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
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
    parser.add_argument("--history", type=Path, default=Path("experiments/experiment3_3_places365_cross_gated/history.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("Zip_files/CAER-S"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/experiment3_3_places365_cross_gated/saliency"))
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".torch_cache"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest(args.manifest)
    split_counts = Counter(row["split"] for row in rows)
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise ValueError(f"No test rows found in {args.manifest}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    fusion_head = CrossGatedTokenFusionClassifier(num_classes=len(CLASSES), dropout=0.3, num_heads=4).to(device)
    fusion_head.load_state_dict(checkpoint["model"])
    fusion_head.eval()

    backbone = load_places365_resnet18(device)
    examples = choose_examples(test_rows, args.per_class, args.seed)
    grid_path = args.output_dir / "gradcam_e3_grid.png"
    render_grid(examples, args.raw_root, backbone, fusion_head, device, grid_path)

    best_epoch, best_val_macro_f1 = find_best_epoch(args.history)
    metadata = {
        "analysis": "Experiment 3 face + scene fusion Grad-CAM",
        "method_note": "Grad-CAM is computed through the Places365 scene backbone while the cached FaceNet512 embedding is held fixed.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_selection": "best validation macro-F1 checkpoint saved during E3 training",
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
        "output_grid": str(grid_path),
    }
    metadata_path = args.output_dir / "gradcam_e3_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved Grad-CAM grid: {grid_path}")
    print(f"Saved metadata: {metadata_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
