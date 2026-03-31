"""Training script for PINNClassifier (Specific Test VII — Physics-Guided ML).

Dataset split
-------------
Merges the pre-split train/ (10 K/class) and val/ (2.5 K/class) directories
and re-splits 90:10 stratified per class (seed=42):
    33,750 train  /  3,750 val  (11,250 / 1,250 per class)

Two-phase training
------------------
Phase 1 (20 epochs, backbone frozen):
  Trains only physics head + residual encoder + classifier.
  High LR (1e-3) safe because pretrained backbone is protected.

Phase 2 (80 epochs, all unfrozen, differential LR):
  backbone: 1e-4  (small — preserves ImageNet features)
  rest:     5e-4

Loss
----
  L = CrossEntropy(logits, labels)  +  λ_phys * MSE(ring, blur(image))

Usage
-----
    python src/train.py

Outputs
-------
    weights/best_pinn.pth    -- Best validation checkpoint
    train_history.npy        -- {train_loss, val_loss, train_acc, val_acc}
"""

import math, os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import PINNClassifier

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

_HERE        = os.path.dirname(__file__)
_ROOT        = os.path.join(_HERE, "..")
DATASET_ROOT = os.path.normpath(
    os.path.join(_ROOT, "..", "Common_Test_I", "dataset", "dataset"))
WEIGHTS_DIR  = os.path.join(_ROOT, "weights")
HISTORY_PATH = os.path.join(_ROOT, "train_history.npy")
LABEL_MAP    = {"no": 0, "sphere": 1, "vort": 2}


# ── 90:10 split ────────────────────────────────────────────────────────────────
def get_split_paths(dataset_root, val_frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    train_items, val_items = [], []
    for sub, label in LABEL_MAP.items():
        paths = []
        for split_dir in ("train", "val"):
            folder = os.path.join(dataset_root, split_dir, sub)
            for fname in os.listdir(folder):
                if fname.endswith(".npy"):
                    paths.append(os.path.join(folder, fname))
        paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        paths = np.array(paths)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_frac))
        for p in paths[:-n_val]:
            train_items.append((p, label))
        for p in paths[-n_val:]:
            val_items.append((p, label))
    return train_items, val_items


# ── Dataset ────────────────────────────────────────────────────────────────────
def _log_stretch(x):
    return np.log1p(x * 10.0) / np.log1p(10.0)


class LensingDataset(Dataset):
    def __init__(self, items, augment=False):
        self.augment = augment
        print(f"  Preloading {len(items):,} images ...", flush=True)
        images, labels = [], []
        for path, label in items:
            img = np.load(path).astype(np.float32)
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            img = ((_log_stretch(img) - 0.5) / 0.5).astype(np.float32)
            images.append(torch.from_numpy(img))
            labels.append(label)
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.long)
        print(f"  Done. Shape: {self.images.shape}", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            img = self._augment(img)
        return img, self.labels[idx].item()

    @staticmethod
    def _augment(x):
        if random.random() > 0.5: x = torch.flip(x, [2])
        if random.random() > 0.5: x = torch.flip(x, [1])
        k = random.randint(0, 3)
        if k: x = torch.rot90(x, k, [1, 2])
        if random.random() > 0.4:
            angle = random.uniform(-30.0, 30.0)
            rad   = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            theta = torch.tensor([[cos_a, -sin_a, 0.0],
                                   [sin_a,  cos_a, 0.0]], dtype=torch.float32).unsqueeze(0)
            grid = F.affine_grid(theta, (1,) + x.shape, align_corners=False)
            x = F.grid_sample(x.unsqueeze(0).float(), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False).squeeze(0).to(x.dtype)
        if random.random() > 0.5:
            x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)).clamp(-1, 1)
        if random.random() > 0.5: x = x + 0.02 * torch.randn_like(x)
        if random.random() > 0.6:
            _, H, W = x.shape
            eh, ew = random.randint(H//8, H//4), random.randint(W//8, W//4)
            ri, rj = random.randint(0, H-eh), random.randint(0, W-ew)
            x = x.clone(); x[:, ri:ri+eh, rj:rj+ew] = x.mean()
        return x


# ── Training loop ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, cls_criterion, optimizer, scheduler,
              device, scaler, use_amp, phys_lambda):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast("cuda"):
                    logits, ring, blur_tgt = model(images)
                    l_cls  = cls_criterion(logits, labels)
                    l_phys = F.mse_loss(ring, blur_tgt)
                    loss   = l_cls + phys_lambda * l_phys
            else:
                logits, ring, blur_tgt = model(images)
                l_cls  = cls_criterion(logits, labels)
                l_phys = F.mse_loss(ring, blur_tgt)
                loss   = l_cls + phys_lambda * l_phys

            if training:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}", flush=True)

    print("Building 90:10 split ...", flush=True)
    train_items, val_items = get_split_paths(DATASET_ROOT, val_frac=0.10)
    print(f"  Train: {len(train_items):,}  |  Val: {len(val_items):,}", flush=True)

    train_ds = LensingDataset(train_items, augment=True)
    val_ds   = LensingDataset(val_items,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    model = PINNClassifier(
        in_channels=1, num_classes=3, img_size=150,
        ring_sigma=0.03, phys_lambda=0.1,
        dropout=0.3, pretrained=True,
    ).to(device)
    print(f"PINN parameters: {model.count_all_parameters():,}", flush=True)

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    PHYS_LAMBDA   = 0.1
    scaler        = torch.amp.GradScaler("cuda") if use_amp else None
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    history      = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_val_acc = 0.0

    # ── Phase 1: backbone frozen ──────────────────────────────────────────────
    PHASE1, PAT1 = 20, 10
    print(f"\n{'='*60}\nPHASE 1: backbone frozen, {PHASE1} epochs\n{'='*60}", flush=True)
    model.freeze_backbone()

    non_backbone = [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]
    opt1   = torch.optim.AdamW(non_backbone, lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1, max_lr=1e-3, epochs=PHASE1, steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3)

    pat_ctr = 0
    for epoch in range(1, PHASE1 + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, cls_criterion, opt1, sched1,
                                    device, scaler, use_amp, PHYS_LAMBDA)
        vl_loss, vl_acc = run_epoch(model, val_loader, cls_criterion, None, None,
                                    device, scaler, use_amp, PHYS_LAMBDA)
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)
        print(f"[P1] Ep {epoch:2d}/{PHASE1} | Train {tr_acc*100:.2f}% | "
              f"Val {vl_acc*100:.2f}% | LR {opt1.param_groups[0]['lr']:.2e}", flush=True)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc; pat_ctr = 0
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_pinn.pth"))
            print(f"  -> Saved ({best_val_acc*100:.2f}%)", flush=True)
        else:
            pat_ctr += 1
            if pat_ctr >= PAT1:
                print(f"  P1 early stop @ epoch {epoch}"); break

    print(f"Phase 1 done. Best: {best_val_acc*100:.2f}%", flush=True)

    # ── Phase 2: all unfrozen, differential LR ────────────────────────────────
    PHASE2, PAT2 = 80, 20
    print(f"\n{'='*60}\nPHASE 2: all unfrozen, {PHASE2} epochs\n{'='*60}", flush=True)
    model.unfreeze_backbone()

    opt2 = torch.optim.AdamW(
        [{"params": model.backbone.parameters(),  "lr": 1e-4},
         {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": 5e-4}],
        weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.OneCycleLR(
        opt2, max_lr=[1e-4, 5e-4], epochs=PHASE2, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e4)

    pat_ctr = 0
    for epoch in range(1, PHASE2 + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, cls_criterion, opt2, sched2,
                                    device, scaler, use_amp, PHYS_LAMBDA)
        vl_loss, vl_acc = run_epoch(model, val_loader, cls_criterion, None, None,
                                    device, scaler, use_amp, PHYS_LAMBDA)
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)
        print(f"[P2] Ep {epoch:2d}/{PHASE2} | Train {tr_acc*100:.2f}% | "
              f"Val {vl_acc*100:.2f}% | LR {opt2.param_groups[0]['lr']:.2e}", flush=True)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc; pat_ctr = 0
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_pinn.pth"))
            print(f"  -> Saved ({best_val_acc*100:.2f}%)", flush=True)
        else:
            pat_ctr += 1
            if pat_ctr >= PAT2:
                print(f"  P2 early stop @ epoch {epoch}"); break

    np.save(HISTORY_PATH, history)
    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
