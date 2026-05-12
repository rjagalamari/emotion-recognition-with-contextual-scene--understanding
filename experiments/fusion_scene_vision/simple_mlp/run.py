#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = {
    "seed": 42,
    "classes": ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"],
    "val_size": 0.15,
    "epochs": 30,
    "batch_size": 128,
    "scene_batch_size": 128,
    "num_workers": 0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "patience": 7,
    "use_class_weights": True,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_label(name: str) -> str:
    label = name.strip().lower().replace(" ", "_")
    aliases = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgusted",
        "disgusted": "disgusted",
        "fear": "fearful",
        "fearful": "fearful",
        "happiness": "happy",
        "happy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "sad": "sad",
        "surprise": "surprised",
        "surprised": "surprised",
    }
    return aliases.get(label, label)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def build_scene_manifest(image_root: Path, classes: list[str]) -> pd.DataFrame:
    rows = []
    class_set = set(classes)
    for image_path in sorted(image_root.rglob("*.png")):
        rel = image_path.relative_to(image_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        split = parts[0].lower()
        label = normalize_label(parts[1])
        if split not in {"train", "test"} or label not in class_set:
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "relative_path": rel.with_suffix(".npy").as_posix(),
                "label": label,
                "split": split,
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise FileNotFoundError(f"No CAER-S images found under {image_root}")
    return manifest.reset_index(drop=True)


def build_scene_model() -> tuple[nn.Module, str]:
    try:
        model = models.resnet18(num_classes=365)
        url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        source = "Places365"
    except Exception as exc:
        raise RuntimeError(
            "Unable to load Places365 ResNet-18 weights locally. "
            "Set TORCH_HOME to a writable cache and allow downloading the Places365 checkpoint."
        ) from exc
    model.fc = nn.Identity()
    model.eval()
    return model, source


def extract_scene_features(image_root: Path, scene_root: Path, dev: torch.device) -> None:
    manifest = build_scene_manifest(image_root, CONFIG["classes"])
    pending = []
    for row in manifest.itertuples(index=False):
        out_path = scene_root / row.relative_path
        if not out_path.exists():
            pending.append((Path(row.image_path), out_path))

    if not pending:
        print(f"Scene features already exist under {scene_root}")
        return

    scene_root.mkdir(parents=True, exist_ok=True)
    model, source = build_scene_model()
    model = model.to(dev)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImagePathDataset([image_path for image_path, _ in pending], transform)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["scene_batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    print(f"Extracting {len(pending)} scene embeddings with ResNet-18 ({source})")
    cursor = 0
    with torch.no_grad():
        for images, batch_paths in tqdm(loader, desc=f"Scene ResNet-18 ({source})"):
            embeddings = model(images.to(dev)).detach().cpu().numpy().astype(np.float32)
            for path_str, embedding in zip(batch_paths, embeddings):
                _, out_path = pending[cursor]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, embedding)
                cursor += 1


def build_fusion_manifest(
    face_root: Path,
    scene_root: Path,
    classes: list[str],
    val_size: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    class_set = set(classes)
    for face_path in sorted(face_root.rglob("*.npy")):
        rel = face_path.relative_to(face_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        split = parts[0].lower()
        label = normalize_label(parts[1])
        if split not in {"train", "test"} or label not in class_set:
            continue
        scene_path = scene_root / rel
        if not scene_path.exists():
            raise FileNotFoundError(f"Missing scene embedding for {rel}: {scene_path}")
        rows.append(
            {
                "relative_path": rel.as_posix(),
                "face_path": str(face_path),
                "scene_path": str(scene_path),
                "label": label,
                "split": split,
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise FileNotFoundError("No matched face/scene embedding pairs found.")

    train_df = manifest.loc[manifest["split"] == "train"].copy().reset_index(drop=True)
    test_df = manifest.loc[manifest["split"] == "test"].copy().reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("Expected non-empty train and test splits.")

    _, val_sub_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label"].to_numpy(),
    )
    train_df["split"] = "train"
    train_df.loc[val_sub_idx, "split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True).reset_index(drop=True)


class FusionDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, classes: list[str]):
        self.df = manifest.reset_index(drop=True)
        self.label_to_id = {label: i for i, label in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        face = np.asarray(np.load(row.face_path), dtype=np.float32)
        scene = np.asarray(np.load(row.scene_path), dtype=np.float32)
        features = np.concatenate([face, scene], axis=0)
        label = self.label_to_id[row.label]
        return torch.as_tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    top = logits.topk(k, dim=1).indices
    return top.eq(labels.view(-1, 1)).any(dim=1).sum().item()


def evaluate_loader(model, loader, classes, dev, return_predictions: bool = False) -> dict:
    model.eval()
    y_true, y_pred = [], []
    correct_top2 = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            correct_top2 += topk_accuracy(logits, yb, k=min(2, len(classes)))
            total += len(yb)

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "top2_accuracy": float(correct_top2 / max(total, 1)),
        "classification_report": report,
    }
    if return_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
    return metrics


def plot_confusion(y_true, y_pred, classes, path: Path) -> None:
    if plt is None or sns is None:
        print(f"Skipping confusion plot because plotting libraries are unavailable: {path}")
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main() -> None:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))
    set_seed(CONFIG["seed"])
    dev = device()

    image_root = PROJECT_ROOT / "Zip_files" / "CAER-S"
    face_root = PROJECT_ROOT / "Zip_files" / "facenetembeedings"
    scene_root = PROJECT_ROOT / "experiment3_data" / "scene_resnet18_places365"
    output_dir = PROJECT_ROOT / "experiments" / "experiment3_places365"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {dev}")
    print(f"CAER-S root: {image_root}")
    print(f"Face embeddings: {face_root}")
    print(f"Scene embeddings: {scene_root}")
    print(f"Outputs: {output_dir}")

    extract_scene_features(image_root=image_root, scene_root=scene_root, dev=dev)

    manifest = build_fusion_manifest(
        face_root=face_root,
        scene_root=scene_root,
        classes=CONFIG["classes"],
        val_size=CONFIG["val_size"],
        seed=CONFIG["seed"],
    )
    manifest.to_csv(output_dir / "manifest_experiment3_places365.csv", index=False)

    label_to_id = {label: i for i, label in enumerate(CONFIG["classes"])}
    y = manifest["label"].map(label_to_id).to_numpy()
    split = manifest["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]

    loaders = {
        "train": DataLoader(
            FusionDataset(manifest.iloc[train_idx].reset_index(drop=True), CONFIG["classes"]),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
        ),
        "val": DataLoader(
            FusionDataset(manifest.iloc[val_idx].reset_index(drop=True), CONFIG["classes"]),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        ),
        "test": DataLoader(
            FusionDataset(manifest.iloc[test_idx].reset_index(drop=True), CONFIG["classes"]),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        ),
    }

    sample_face = np.asarray(np.load(manifest.iloc[0].face_path), dtype=np.float32)
    sample_scene = np.asarray(np.load(manifest.iloc[0].scene_path), dtype=np.float32)
    input_dim = int(sample_face.shape[0] + sample_scene.shape[0])
    model = MLPClassifier(input_dim, len(CONFIG["classes"]), CONFIG["dropout"]).to(dev)

    if CONFIG["use_class_weights"]:
        counts = np.bincount(y[train_idx], minlength=len(CONFIG["classes"]))
        weights = len(train_idx) / np.maximum(counts, 1)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(dev))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_val = -1.0
    wait = 0
    history: list[dict[str, float | int]] = []
    best_model_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.csv"

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        train_bar = tqdm(loaders["train"], desc=f"experiment3 epoch {epoch}/{CONFIG['epochs']}", leave=False)
        for xb, yb in train_bar:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size_now = len(yb)
            total_loss += loss.item() * batch_size_now
            seen += batch_size_now
            train_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / seen:.4f}")

        val_metrics = evaluate_loader(model, loaders["val"], CONFIG["classes"], dev)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_idx),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_top2_accuracy": val_metrics["top2_accuracy"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(row)

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            wait = 0
            torch.save({"model": model.state_dict(), "input_dim": input_dim}, best_model_path)
        else:
            wait += 1
            if wait >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(best_model_path, map_location=dev)
    model.load_state_dict(checkpoint["model"])

    val_final = evaluate_loader(model, loaders["val"], CONFIG["classes"], dev, return_predictions=True)
    test_final = evaluate_loader(model, loaders["test"], CONFIG["classes"], dev, return_predictions=True)

    final_metrics = {
        "branch": "vision_fusion",
        "input_dim": input_dim,
        "best_val_macro_f1": float(best_val),
        "eval_source": "caer_s_validation_split",
        "accuracy": val_final["accuracy"],
        "macro_f1": val_final["macro_f1"],
        "weighted_f1": val_final["weighted_f1"],
        "top2_accuracy": val_final["top2_accuracy"],
        "test_accuracy": test_final["accuracy"],
        "test_macro_f1": test_final["macro_f1"],
        "test_weighted_f1": test_final["weighted_f1"],
        "test_top2_accuracy": test_final["top2_accuracy"],
        "test_source": "caer_s_in_domain_test",
        "classification_report": val_final["classification_report"],
        "test_classification_report": test_final["classification_report"],
    }

    write_json(output_dir / "metrics.json", final_metrics)
    pd.DataFrame(
        [
            {
                key: value
                for key, value in final_metrics.items()
                if key not in {"classification_report", "test_classification_report"}
            }
        ]
    ).to_csv(output_dir / "results.csv", index=False)

    plot_confusion(val_final["y_true"], val_final["y_pred"], CONFIG["classes"], output_dir / "confusion_validation.png")
    plot_confusion(test_final["y_true"], test_final["y_pred"], CONFIG["classes"], output_dir / "confusion_test.png")

    print("Final validation metrics:")
    print(
        {
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "weighted_f1": final_metrics["weighted_f1"],
            "top2_accuracy": final_metrics["top2_accuracy"],
        }
    )
    print("Final test metrics:")
    print(
        {
            "accuracy": final_metrics["test_accuracy"],
            "macro_f1": final_metrics["test_macro_f1"],
            "weighted_f1": final_metrics["test_weighted_f1"],
            "top2_accuracy": final_metrics["test_top2_accuracy"],
        }
    )


if __name__ == "__main__":
    main()
