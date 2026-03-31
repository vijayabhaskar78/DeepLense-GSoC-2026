"""Evaluation script for HybridFNOClassifier (Test III / Specific Test IV).

Uses the same 90:10 stratified split as train.py (seed=42) so the
validation set is identical across training and evaluation runs.

Generates plots/: roc_curves, confusion_matrix, training_curves,
                  confidence_analysis  (standard + 8x TTA variants).

Usage
-----
    python src/evaluate.py
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import HybridFNOClassifier, FNOClassifier
from train import get_split_paths, LensingDataset   # reuse split + dataset

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_HERE, "..")
DATASET_ROOT = os.path.normpath(
    os.path.join(_ROOT, "..", "Common_Test_I", "dataset", "dataset"))
WEIGHTS_PATH = os.path.join(_ROOT, "weights", "best_fno.pth")
PLOTS_DIR    = os.path.join(_ROOT, "plots")
HISTORY_PATH = os.path.join(_ROOT, "train_history.npy")

CLASS_NAMES  = ["No Substructure", "CDM Subhalo", "Axion Vortex"]


# ── Inference ──────────────────────────────────────────────────────────────────
def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, 1)
            all_labels.append(labels.numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))


def tta_predictions(model, loader, device):
    model.eval()

    def _transforms(x):
        return [x,
                torch.flip(x, [3]),
                torch.flip(x, [2]),
                torch.rot90(x, 1, [2, 3]),
                torch.rot90(x, 2, [2, 3]),
                torch.rot90(x, 3, [2, 3]),
                torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
                torch.flip(torch.rot90(x, 1, [2, 3]), [2])]

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            avg_p  = torch.zeros(images.size(0), 3, device=device)
            for t in _transforms(images):
                avg_p += torch.softmax(model(t), 1)
            avg_p /= 8.0
            all_labels.append(labels.numpy())
            all_probs.append(avg_p.cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    probs_arr  = np.concatenate(all_probs)
    return labels_arr, probs_arr.argmax(1), probs_arr


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_roc(labels, probs, tag=""):
    from sklearn.preprocessing import label_binarize
    lb = label_binarize(labels, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(lb[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(3)) / 3
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    fpr["micro"], tpr["micro"], _ = roc_curve(lb.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    colours = ["#4878CF", "#6ACC65", "#D65F5F"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for i, (name, c) in enumerate(zip(CLASS_NAMES, colours)):
        ax.plot(fpr[i], tpr[i], color=c, lw=1.8,
                label=f"{name}  (AUC={roc_auc[i]:.4f})")
    ax.plot(fpr["macro"], tpr["macro"], "k-",  lw=2.2,
            label=f"Macro avg  (AUC={roc_auc['macro']:.4f})")
    ax.plot(fpr["micro"], tpr["micro"], "k:",  lw=2.2,
            label=f"Micro avg  (AUC={roc_auc['micro']:.4f})")
    title = "HybridFNO — ROC Curves" + (f" ({tag})" if tag else "")
    ax.set(title=title, xlabel="False Positive Rate", ylabel="True Positive Rate",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=10); plt.tight_layout()
    suffix = "_" + tag.lower().replace(" ", "_") if tag else ""
    path   = os.path.join(PLOTS_DIR, f"roc_curves{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")
    return roc_auc


def plot_confusion(labels, preds, tag=""):
    cm = confusion_matrix(labels, preds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, sfx in zip(
            axes,
            [cm, cm.astype(float) / cm.sum(1, keepdims=True)],
            ["d", ".2f"], ["Counts", "Row-normalised"]):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=10)
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set(xlabel="Predicted", ylabel="True",
               title=f"Confusion Matrix — {sfx}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, format(data[r, c], fmt), ha="center", va="center",
                        color="white" if data[r, c] > data.max() / 2 else "black",
                        fontsize=10)
    title = "HybridFNO — Confusion Matrices" + (f" ({tag})" if tag else "")
    fig.suptitle(title, fontsize=13, y=1.02); plt.tight_layout()
    suffix = "_" + tag.lower().replace(" ", "_") if tag else ""
    path   = os.path.join(PLOTS_DIR, f"confusion_matrix{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_training_curves():
    if not os.path.exists(HISTORY_PATH):
        print("train_history.npy not found — skipping."); return
    h = np.load(HISTORY_PATH, allow_pickle=True).item()
    ep = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, h["train_loss"], label="Train")
    axes[0].plot(ep, h["val_loss"],   label="Val")
    axes[0].set(title="HybridFNO — Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(ep, [v*100 for v in h["train_acc"]], label="Train")
    axes[1].plot(ep, [v*100 for v in h["val_acc"]],   label="Val")
    axes[1].set(title="HybridFNO — Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_confidence(labels, preds, probs):
    conf = probs.max(1); correct = labels == preds
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf[correct],  bins=30, alpha=0.7, label="Correct",   color="#4878CF")
    ax.hist(conf[~correct], bins=30, alpha=0.7, label="Incorrect", color="#D65F5F")
    ax.set(xlabel="Max Softmax Confidence", ylabel="Count",
           title="HybridFNO — Confidence Distribution")
    ax.legend(fontsize=11); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confidence_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Same 90:10 split as training
    _, val_items = get_split_paths(DATASET_ROOT, val_frac=0.10)
    val_ds = LensingDataset(val_items, augment=False)
    loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds):,}")

    # Load model
    model = HybridFNOClassifier(
        in_channels=1, num_classes=3,
        fno_width=128, fno_modes=8, fno_depth=3,
        dropout=0.3, pretrained=False)
    model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.to(device)
    print(f"Loaded: {WEIGHTS_PATH}")
    print(f"Parameters: {model.count_all_parameters():,}")

    # Standard
    print("\n--- Standard inference ---")
    labels, preds, probs = get_predictions(model, loader, device)
    acc = (labels == preds).mean()
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    roc_std = plot_roc(labels, probs, "Standard")
    plot_confusion(labels, preds, "Standard")
    plot_confidence(labels, preds, probs)

    # 8× TTA
    print("\n--- 8× TTA ---")
    lbl_tta, prd_tta, prb_tta = tta_predictions(model, loader, device)
    acc_tta = (lbl_tta == prd_tta).mean()
    print(f"TTA Accuracy: {acc_tta*100:.2f}%")
    print(classification_report(lbl_tta, prd_tta, target_names=CLASS_NAMES, digits=4))
    roc_tta = plot_roc(lbl_tta, prb_tta, "TTA")
    plot_confusion(lbl_tta, prd_tta, "TTA")
    plot_training_curves()

    print("\n" + "="*58)
    print("RESULTS SUMMARY  (90:10 split, seed=42)")
    print("="*58)
    print(f"{'Metric':<28} {'Standard':>12} {'8x TTA':>12}")
    print("-"*58)
    print(f"{'Accuracy':<28} {acc*100:>11.2f}% {acc_tta*100:>11.2f}%")
    print(f"{'Macro AUC':<28} {roc_std['macro']:>12.4f} {roc_tta['macro']:>12.4f}")
    print(f"{'Micro AUC':<28} {roc_std['micro']:>12.4f} {roc_tta['micro']:>12.4f}")
    print("="*58)
    print(f"\nCommon Test I (ResNet-18): 95.64% std | 96.51% TTA")
    print(f"HybridFNO (v3):           {acc*100:.2f}% std | {acc_tta*100:.2f}% TTA")


if __name__ == "__main__":
    main()
