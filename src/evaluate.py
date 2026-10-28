"""Evaluate the best checkpoint on the held-out Testing split.

This is the only script that reads the Testing folder. It writes:
  results/test_metrics.json       accuracy, macro-F1, AUROC, per-class recall
  results/classification_report.txt
  results/confusion_matrix.csv / .png
  results/gradcam_examples/       one Grad-CAM overlay per class

Usage:
    python -m src.evaluate
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from torch.utils.data import DataLoader

from src.common.seed import set_seed
from src.data_brain import (CLASS_NAMES, DISPLAY_NAMES, BrainDataset,
                            eval_transforms, list_images, load_rgb,
                            resolve_root)
from src.gradcam import GradCAM, overlay_heatmap
from src.models import NeuroNet

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CKPT = REPO_ROOT / "checkpoints" / "neurovision_resnet18_best.pt"


def main():
    set_seed(7)
    root = resolve_root()
    test_items = list_images(root, "Testing")
    ds = BrainDataset(test_items, eval_transforms(224))
    loader = DataLoader(ds, batch_size=32)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = NeuroNet(num_classes=4, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck.get('epoch')} val_acc {ck.get('val_acc')}")

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x), dim=1)
            y_prob.append(probs.numpy())
            y_pred += probs.argmax(1).tolist()
            y_true += y.tolist()
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    y_prob = np.concatenate(y_prob)

    acc = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    macro_f1 = float(report["macro avg"]["f1-score"])
    y_onehot = np.eye(len(CLASS_NAMES))[y_true]
    auroc = float(roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr"))
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        per_class[cls] = {
            "display": DISPLAY_NAMES[cls],
            "support": int((y_true == i).sum()),
            "recall_sensitivity": float(report[cls]["recall"]),
            "precision": float(report[cls]["precision"]),
            "f1": float(report[cls]["f1-score"]),
            "auroc_ovr": float(roc_auc_score(y_onehot[:, i], y_prob[:, i])),
        }

    metrics = {"test_images": int(len(y_true)), "accuracy": acc,
               "macro_f1": macro_f1, "auroc_macro_ovr": auroc,
               "per_class": per_class,
               "confusion_matrix": cm.tolist(),
               "class_order": CLASS_NAMES}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    with open(RESULTS_DIR / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + CLASS_NAMES)
        for cls, row in zip(CLASS_NAMES, cm):
            w.writerow([cls] + row.tolist())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"NeuroVision — confusion matrix (test acc {acc:.3f})")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # One Grad-CAM example per class: first correctly classified test image.
    cam_dir = RESULTS_DIR / "gradcam_examples"; cam_dir.mkdir(exist_ok=True)
    gradcam = GradCAM(model, model.layer4)
    tf = eval_transforms(224)
    seen = set()
    for path, y in test_items:
        if y in seen:
            continue
        img = load_rgb(path, 224)
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        if pred != y:
            continue
        heat = gradcam(x, pred)
        overlay = overlay_heatmap(np.asarray(img), heat)
        from PIL import Image
        Image.fromarray(overlay).save(cam_dir / f"{CLASS_NAMES[y]}_gradcam.png")
        seen.add(y)
        print(f"gradcam: {CLASS_NAMES[y]} <- {path.name}")
        if len(seen) == len(CLASS_NAMES):
            break

    print(json.dumps({k: metrics[k] for k in ("test_images", "accuracy", "macro_f1", "auroc_macro_ovr")}, indent=2))
    for cls, d in per_class.items():
        print(f"  {cls:12s} recall {d['recall_sensitivity']:.4f}  precision {d['precision']:.4f}  f1 {d['f1']:.4f}  auroc {d['auroc_ovr']:.4f}")


if __name__ == "__main__":
    main()
