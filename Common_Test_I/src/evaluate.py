"""Evaluate the best checkpoint: ROC curves, AUC scores, confusion matrix, confidence analysis."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['No Substructure', 'Subhalo (Sphere)', 'Vortex']
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')


def _plots(fname: str) -> str:
    """Return path inside the plots/ directory, creating it if needed."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return os.path.join(PLOTS_DIR, fname)


# ─── Dataset ───────────────────────────────────────────────────────────────────

class LensingDataset(Dataset):
    def __init__(self, root_dir: str):
        self.paths, self.labels = [], []
        for lbl, cls in enumerate(['no', 'sphere', 'vort']):
            d = os.path.join(root_dir, cls)
            for f in sorted(os.listdir(d)):
                if f.endswith('.npy'):
                    self.paths.append(os.path.join(d, f))
                    self.labels.append(lbl)
        print(f'[Dataset] {len(self.paths)} validation samples loaded')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = np.load(self.paths[idx]).astype(np.float32)
        img = np.log1p(img * 10.0) / np.log1p(10.0)
        img = (img - 0.5) / 0.5
        img = np.repeat(img, 3, axis=0)
        return torch.from_numpy(img.astype(np.float32)), self.labels[idx]


# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model(weights_path: str = 'weights/best_model.pth') -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 3))
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


# ─── Inference ─────────────────────────────────────────────────────────────────

def get_predictions(model: nn.Module, loader: DataLoader):
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.append(probs)
            all_preds.extend(preds)
    return np.array(all_labels), np.vstack(all_probs), np.array(all_preds)


def tta_predictions(model: nn.Module, loader: DataLoader):
    """8-fold test-time augmentation: identity + flips + rotations."""
    def augment(imgs, k):
        if k == 0: return imgs
        if k == 1: return imgs.flip(-1)
        if k == 2: return imgs.flip(-2)
        if k == 3: return imgs.rot90(1, [-2, -1])
        if k == 4: return imgs.rot90(2, [-2, -1])
        if k == 5: return imgs.rot90(3, [-2, -1])
        if k == 6: return imgs.flip(-1).rot90(1, [-2, -1])
        if k == 7: return imgs.flip(-2).rot90(1, [-2, -1])

    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = sum(
                F.softmax(model(augment(imgs, k)), dim=1)
                for k in range(8)
            ) / 8.0
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    probs_arr = np.vstack(all_probs)
    return np.array(all_labels), probs_arr, probs_arr.argmax(axis=1)


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_roc(labels: np.ndarray, probs: np.ndarray, suffix: str = ''):
    labels_bin = label_binarize(labels, classes=[0, 1, 2])
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    fig, ax = plt.subplots(figsize=(8, 7))

    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(3):
        fpr_d[i], tpr_d[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc_d[i] = auc(fpr_d[i], tpr_d[i])
        ax.plot(fpr_d[i], tpr_d[i], color=colors[i], lw=2.5,
                label=f'{CLASS_NAMES[i]} (AUC = {auc_d[i]:.4f})')

    all_fpr   = np.unique(np.concatenate([fpr_d[i] for i in range(3)]))
    mean_tpr  = sum(np.interp(all_fpr, fpr_d[i], tpr_d[i]) for i in range(3)) / 3
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, '#E91E63', lw=2.5, ls='--',
            label=f'Macro-Avg (AUC = {macro_auc:.4f})')

    fpr_m, tpr_m, _ = roc_curve(labels_bin.ravel(), probs.ravel())
    micro_auc = auc(fpr_m, tpr_m)
    ax.plot(fpr_m, tpr_m, '#9C27B0', lw=2, ls=':',
            label=f'Micro-Avg (AUC = {micro_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
    title = 'ROC Curves — Multi-Class (One-vs-Rest)'
    if suffix:
        title += f' [{suffix}]'
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=title, xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'roc_curves{"_" + suffix.lower().replace(" ", "_") if suffix else ""}.png'
    plt.savefig(_plots(fname), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n=== AUC Scores {("(" + suffix + ")") if suffix else ""} ===')
    for i in range(3):
        print(f'  {CLASS_NAMES[i]:>20}: {auc_d[i]:.4f}')
    print(f'  {"Macro-Average":>20}: {macro_auc:.4f}')
    print(f'  {"Micro-Average":>20}: {micro_auc:.4f}')
    return auc_d, macro_auc, micro_auc


def plot_confusion(labels: np.ndarray, preds: np.ndarray):
    cm = confusion_matrix(labels, preds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title('Confusion Matrix (Counts)')
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_n, display_labels=CLASS_NAMES).plot(
        ax=ax2, cmap='Blues', values_format='.3f')
    ax2.set_title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(_plots('confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plots/confusion_matrix.png')


def plot_training_curves(history_path: str = 'train_history.npy'):
    if not os.path.exists(history_path):
        print(f'No training history found at {history_path}, skipping.')
        return
    h = np.load(history_path)
    epochs, tl, ta, vl, va = h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, tl, label='Train Loss', lw=2)
    ax1.plot(epochs, vl, label='Val Loss',   lw=2)
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss Curves')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, ta, label='Train Acc', lw=2)
    ax2.plot(epochs, va, label='Val Acc',   lw=2)
    ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy Curves')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_plots('training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved plots/training_curves.png')


def plot_confidence(probs: np.ndarray, labels: np.ndarray, preds: np.ndarray):
    """Plot confidence distribution split by correct vs incorrect predictions."""
    confidences = probs.max(axis=1)
    correct     = preds == labels
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(confidences[correct],  bins=bins, alpha=0.7, label=f'Correct ({correct.sum()})',  color='#4CAF50')
    ax.hist(confidences[~correct], bins=bins, alpha=0.7, label=f'Wrong ({(~correct).sum()})', color='#F44336')
    ax.set(xlabel='Max Softmax Confidence', ylabel='Count', title='Prediction Confidence Distribution')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_plots('confidence_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved plots/confidence_analysis.png')


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.pth')
    data_root    = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset', 'val')

    print('Loading validation dataset...')
    val_ds     = LensingDataset(data_root)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    print('Loading best model...')
    model = build_model(weights_path)

    print('\n--- Standard inference ---')
    labels, probs, preds = get_predictions(model, val_loader)
    acc = 100.0 * np.mean(preds == labels)
    print(f'Validation Accuracy: {acc:.2f}%')
    print('\n=== Classification Report ===')
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    plot_roc(labels, probs)
    plot_confusion(labels, preds)
    plot_confidence(probs, labels, preds)

    print('\n--- 8x Test Time Augmentation ---')
    labels_tta, probs_tta, preds_tta = tta_predictions(model, val_loader)
    acc_tta = 100.0 * np.mean(preds_tta == labels_tta)
    print(f'Validation Accuracy (TTA): {acc_tta:.2f}%')
    plot_roc(labels_tta, probs_tta, suffix='TTA')

    history_path = os.path.join(os.path.dirname(__file__), '..', 'train_history.npy')
    plot_training_curves(history_path)

    print('\nAll plots saved to plots/')


if __name__ == '__main__':
    main()
