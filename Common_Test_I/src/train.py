"""
DeepLense GSoC 2026 - Common Test I: Multi-Class Classification

Fine-tunes a pretrained ResNet-18 on single-channel gravitational lensing images.
Two-phase schedule: head-only warmup → full fine-tune with differential LRs.
Mixed precision (AMP) is used for ~2x speed on CUDA without accuracy loss.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = device.type == 'cuda'      # AMP only works on CUDA
if device.type == 'cuda':
    print(f'Device: {torch.cuda.get_device_name(0)}  |  AMP: {USE_AMP}')
else:
    print('Device: CPU')


# ─── Dataset ───────────────────────────────────────────────────────────────────

class LensingDataset(Dataset):
    """
    Loads .npy lensing images with log-stretch preprocessing and optional augmentation.

    The log-stretch transform log1p(x*10)/log1p(10) compresses the bright Einstein
    ring core and amplifies the faint outer region where substructure signatures live,
    giving roughly a 6x boost in sensitivity in the dim pixel regime.
    """
    CLASS_DIRS = ['no', 'sphere', 'vort']   # labels 0, 1, 2

    def __init__(self, root_dir: str, augment: bool = False):
        self.paths, self.labels = [], []
        self.augment = augment
        for lbl, cls in enumerate(self.CLASS_DIRS):
            d = os.path.join(root_dir, cls)
            for f in sorted(os.listdir(d)):
                if f.endswith('.npy'):
                    self.paths.append(os.path.join(d, f))
                    self.labels.append(lbl)
        counts = [self.labels.count(i) for i in range(3)]
        print(f'[Dataset] {root_dir}: {len(self.paths)} samples  '
              f'(no={counts[0]}, sphere={counts[1]}, vort={counts[2]})')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = np.load(self.paths[idx]).astype(np.float32)  # (1, H, W), values in [0, 1]

        # Log-stretch: amplifies faint substructure
        img = np.log1p(img * 10.0) / np.log1p(10.0)       # still [0, 1]

        # Shift to [-1, 1] to roughly match ImageNet normalisation range
        img = (img - 0.5) / 0.5

        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()                # horizontal flip
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()                # vertical flip
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k, axes=(1, 2)).copy()  # 90/180/270° rotation
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
                img = img + noise

        img = np.repeat(img, 3, axis=0)                    # (3, H, W) for ResNet
        return torch.from_numpy(img.astype(np.float32)), self.labels[idx]


# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> nn.Module:
    """ResNet-18 pretrained on ImageNet with a 3-class dropout head."""
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 3),
    )
    return model


# ─── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None,
              scaler=None, train: bool = True):
    """Single epoch of training or evaluation.

    Uses AMP (automatic mixed precision) when a GradScaler is provided.
    """
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                out  = model(imgs)
                loss = criterion(out, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, 100.0 * correct / total


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('weights', exist_ok=True)

    train_ds = LensingDataset('dataset/dataset/train', augment=True)
    val_ds   = LensingDataset('dataset/dataset/val',   augment=False)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0, pin_memory=pin)

    model = build_model(pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Model] ResNet-18 pretrained, {n_params:,} parameters')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = torch.cuda.amp.GradScaler() if USE_AMP else None

    best_val_acc   = 0.0
    patience_count = 0
    PATIENCE       = 8    # early stopping: stop if val_acc doesn't improve for 8 epochs
    history        = []

    PHASE1_EPOCHS = 10
    PHASE2_EPOCHS = 30
    header = f"{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>7} | {'VLoss':>7} | {'VAcc':>7} | {'LR':>10}"

    # ── Phase 1: head-only warmup ───────────────────────────────────────
    print(f'\n[Phase 1] Head-only warmup ({PHASE1_EPOCHS} epochs, backbone frozen)')
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(header)
    for epoch in range(1, PHASE1_EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, train=False)
        scheduler.step()
        lr     = optimizer.param_groups[0]['lr']
        marker = ''
        if va > best_val_acc:
            best_val_acc   = va
            patience_count = 0
            torch.save(model.state_dict(), 'weights/best_model.pth')
            marker = ' *'
        else:
            patience_count += 1
        history.append((epoch, tl, ta, vl, va))
        print(f'{epoch:>3} | {tl:>7.4f} | {ta:>6.2f}% | {vl:>7.4f} | {va:>6.2f}% | {lr:>10.7f}{marker}')

    # ── Phase 2: full fine-tune ─────────────────────────────────────────
    print(f'\n[Phase 2] Full fine-tune ({PHASE2_EPOCHS} epochs, differential LRs, early stop patience={PATIENCE})')
    for p in model.parameters():
        p.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    optimizer = torch.optim.AdamW(
        [
            {'params': model.fc.parameters(), 'lr': 1e-3},
            {'params': backbone_params,        'lr': 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-7)
    patience_count = 0   # reset patience for Phase 2

    print(header)
    for epoch in range(PHASE1_EPOCHS + 1, PHASE1_EPOCHS + PHASE2_EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, train=False)
        scheduler.step()
        lr     = optimizer.param_groups[0]['lr']
        marker = ''
        if va > best_val_acc:
            best_val_acc   = va
            patience_count = 0
            torch.save(model.state_dict(), 'weights/best_model.pth')
            marker = ' *'
        else:
            patience_count += 1
        history.append((epoch, tl, ta, vl, va))
        print(f'{epoch:>3} | {tl:>7.4f} | {ta:>6.2f}% | {vl:>7.4f} | {va:>6.2f}% | {lr:>10.7f}{marker}')
        if patience_count >= PATIENCE:
            print(f'\n[Early Stop] No improvement for {PATIENCE} consecutive epochs. Stopping at epoch {epoch}.')
            break

    print(f'\nBest Val Accuracy: {best_val_acc:.2f}%')
    np.save('train_history.npy', np.array(history))
    print('Saved: weights/best_model.pth, train_history.npy')


if __name__ == '__main__':
    main()
