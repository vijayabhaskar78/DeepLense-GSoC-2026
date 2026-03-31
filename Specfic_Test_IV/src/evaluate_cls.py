"""Task IX.A — Evaluation: ROC curves, AUC, confusion matrix, classification report."""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(__file__))
from mae_model import MAE, ClassificationMAE
from data_utils import LensingClfDataset, CLASS_NAMES

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
WEIGHTS_PATH = os.path.join(_ROOT, "weights", "clf_best.pth")
PLOTS_DIR    = os.path.join(_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            all_labels.append(labels.numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))


def plot_roc(labels, probs, tag=""):
    labels_bin = label_binarize(labels, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(3)) / 3
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    colours = ["#4878CF", "#6ACC65", "#D65F5F"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for i, (name, c) in enumerate(zip(CLASS_NAMES, colours)):
        ax.plot(fpr[i], tpr[i], color=c, lw=1.8,
                label=f"{name}  (AUC={roc_auc[i]:.4f})")
    ax.plot(all_fpr, mean_tpr, "k-", lw=2.2,
            label=f"Macro avg  (AUC={roc_auc['macro']:.4f})")
    ax.plot(fpr["micro"], tpr["micro"], "k:", lw=2.2,
            label=f"Micro avg  (AUC={roc_auc['micro']:.4f})")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"MAE Classifier — ROC Curves ({tag})",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=10); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"clf_roc_{tag.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")
    return roc_auc


def plot_confusion(labels, preds, tag=""):
    cm = confusion_matrix(labels, preds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, sfx in zip(
        axes,
        [cm, cm.astype(float) / cm.sum(axis=1, keepdims=True)],
        ["d", ".2f"], ["Counts", "Row-normalised"]
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=10)
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {sfx}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = data.max() / 2
        for r in range(3):
            for c in range(3):
                ax.text(c, r, format(data[r, c], fmt), ha="center", va="center",
                        color="white" if data[r, c] > thresh else "black", fontsize=10)
    fig.suptitle(f"MAE Classifier — Confusion ({tag})", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"clf_confusion_{tag.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_training_curves():
    history_path = os.path.join(_ROOT, "clf_history.npy")
    if not os.path.exists(history_path):
        return
    h = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, h["train_loss"], label="Train")
    axes[0].plot(epochs, h["val_loss"],   label="Val")
    axes[0].set_title("MAE Classifier — Loss"); axes[0].legend()
    axes[1].plot(epochs, [v*100 for v in h["train_acc"]], label="Train")
    axes[1].plot(epochs, [v*100 for v in h["val_acc"]],   label="Val")
    axes[1].set_title("MAE Classifier — Accuracy (%)"); axes[1].legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "clf_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    mae = MAE(img_size=64, patch_size=8, encoder_embed_dim=384, encoder_depth=6,
              encoder_heads=6, decoder_embed_dim=192, decoder_depth=2,
              decoder_heads=6, mask_ratio=0.75)
    model = ClassificationMAE(mae.encoder, num_classes=3, dropout=0.3)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    print(f"Loaded: {WEIGHTS_PATH}  ({model.count_parameters():,} params)", flush=True)

    val_ds = LensingClfDataset(split="val", nosub_only=False, augment=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds):,}", flush=True)

    labels, preds, probs = get_predictions(model, val_loader, device)
    acc = (labels == preds).mean()
    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    roc_std = plot_roc(labels, probs, "Standard")
    plot_confusion(labels, preds, "Standard")
    plot_training_curves()

    print("\n" + "=" * 55)
    print(f"{'Metric':<30} {'Value':>22}")
    print("-" * 55)
    print(f"{'Accuracy':<30} {acc*100:>21.2f}%")
    print(f"{'Macro AUC':<30} {roc_std['macro']:>22.4f}")
    print(f"{'Micro AUC':<30} {roc_std['micro']:>22.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
