"""Task IX.A — Classification fine-tuning on 3-class lensing dataset.

Two-phase fine-tuning strategy
-------------------------------
Phase 1 (10 epochs):  Freeze encoder, train only the classification head.
                      Head learns to classify from frozen no_sub-prior features.
Phase 2 (80 epochs):  Unfreeze encoder with layer-wise LR decay (LLRD).
                      Alternate MixUp + CutMix for maximally diverse augmentation.
                      CosineAnnealingWarmRestarts for thorough LR exploration.
                      Patience=20 — conservative early stopping.

v2 upgrades
-----------
* ViT-Small encoder (384/6/6): CLS + mean + max head (3×embed_dim=1152)
* CutMix (α=1.0) alternating with MixUp (α=0.4) every batch —
  CutMix cuts rectangular regions from one image into another, exposing
  the model to diverse partial-ring and local-perturbation patterns.
* Layer-wise LR decay (LLRD, rate=0.75): shallower encoder layers get
  lower LR (preserve general geometry prior), deeper layers get higher
  LR (adapt to classification). Standard for ViT fine-tuning.
* 80 epochs Phase 2: longer convergence for ViT-Small with LLRD.

Usage
-----
    python src/finetune_cls.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from mae_model import MAE, ClassificationMAE
from data_utils import LensingClfDataset, CLASS_NAMES

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
WEIGHTS_DIR   = os.path.join(_ROOT, "weights")
PRETRAIN_CKPT = os.path.join(WEIGHTS_DIR, "mae_pretrained.pth")


# ── MixUp ─────────────────────────────────────────────────────────────────────
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


# ── CutMix ────────────────────────────────────────────────────────────────────
def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """CutMix: paste a rectangular crop from a random image.

    Unlike MixUp (pixel-level blend), CutMix replaces a spatial region —
    forcing the model to classify from partial ring evidence only.
    This is particularly valuable for CDM subhalos (local perturbations)
    and axion patterns (spatially structured interference).
    """
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = x.shape
    idx = torch.randperm(B, device=x.device)

    # Sampled bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # Adjust lambda to actual area ratio
    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_mix, y, y[idx], lam_actual


def mixed_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ── Layer-wise LR decay optimizer ─────────────────────────────────────────────
def build_llrd_optimizer(model: ClassificationMAE,
                         base_encoder_lr: float = 5e-5,
                         head_lr: float = 5e-4,
                         decay: float = 0.75,
                         weight_decay: float = 1e-4):
    """Layer-wise LR decay for ViT fine-tuning.

    Shallower encoder layers receive smaller LR (preserve general geometry
    prior from no_sub pre-training). Deeper layers and head receive larger LR.

    Decay schedule for depth=6 encoder, rate=0.75:
      patch_embed       : base_lr × 0.75^6 ≈ 0.178 × base_lr
      block[0] (input)  : base_lr × 0.75^6
      block[1]          : base_lr × 0.75^5
      block[2]          : base_lr × 0.75^4
      block[3]          : base_lr × 0.75^3
      block[4]          : base_lr × 0.75^2
      block[5] (output) : base_lr × 0.75^1
      norm + cls_token  : base_lr × 1.0
      head              : head_lr
    """
    depth = model.encoder.depth
    param_groups = []

    # patch_embed and cls_token: deepest decay
    embed_params = (list(model.encoder.patch_embed.parameters()) +
                    [model.encoder.cls_token])
    param_groups.append({
        "params": embed_params,
        "lr": base_encoder_lr * (decay ** depth),
    })

    # Each transformer block: linearly increasing LR
    for i, block in enumerate(model.encoder.blocks):
        lr_i = base_encoder_lr * (decay ** (depth - i))
        param_groups.append({"params": list(block.parameters()), "lr": lr_i})

    # Encoder norm: no decay (full base LR)
    param_groups.append({
        "params": list(model.encoder.norm.parameters()),
        "lr": base_encoder_lr,
    })

    # Classification head: highest LR
    param_groups.append({"params": list(model.head.parameters()), "lr": head_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ── Training epoch ────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, scaler, use_amp,
              use_augmix=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            y_a = y_b = labels
            lam = 1.0
            if training and use_augmix:
                # Alternate MixUp and CutMix — each batch randomly picks one
                if np.random.rand() > 0.5:
                    imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
                else:
                    imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=1.0)

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(imgs)
                    loss = mixed_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(imgs)
                loss = mixed_criterion(criterion, logits, y_a, y_b, lam)

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

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == y_a).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main() -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}", flush=True)

    print("Loading classification data ...", flush=True)
    train_ds = LensingClfDataset(split="train", nosub_only=False, augment=True)
    val_ds   = LensingClfDataset(split="val",   nosub_only=False, augment=False)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

    # v2: ViT-Small encoder (384/6/6)
    mae = MAE(img_size=64, patch_size=8,
              encoder_embed_dim=384, encoder_depth=6, encoder_heads=6,
              decoder_embed_dim=192, decoder_depth=2, decoder_heads=6,
              mask_ratio=0.75)

    if os.path.exists(PRETRAIN_CKPT):
        state = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=True)
        mae.load_state_dict(state)
        print(f"Loaded pre-trained weights: {PRETRAIN_CKPT}", flush=True)
    else:
        print("WARNING: No pre-trained weights found — training from scratch.", flush=True)

    model = ClassificationMAE(mae.encoder, num_classes=3, dropout=0.3).to(device)
    print(f"ClassificationMAE parameters: {model.count_parameters():,}", flush=True)
    print(f"  Head: CLS + mean + max → 3×{mae.encoder.embed_dim} = "
          f"{mae.encoder.embed_dim*3} → 512 → 3", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None
    history   = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_val_acc = 0.0

    # ── Phase 1: Head-only warmup (10 epochs) ─────────────────────────────────
    print("\n[Phase 1] Head-only warmup (10 epochs) ...", flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-5
    )
    for epoch in range(1, 11):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device,
                           scaler, use_amp, use_augmix=False)
        vl, va = run_epoch(model, val_loader, criterion, None, device,
                           scaler, use_amp, use_augmix=False)
        scheduler.step()
        for k, v in zip(("train_loss","val_loss","train_acc","val_acc"), (tl,vl,ta,va)):
            history[k].append(v)
        print(f"  P1 Epoch {epoch:2d}/10 | Train {tl:.4f}/{ta*100:.2f}% | "
              f"Val {vl:.4f}/{va*100:.2f}%", flush=True)
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "clf_best.pth"))

    # ── Phase 2: Full fine-tuning (80 epochs, LLRD + MixUp/CutMix) ───────────
    print("\n[Phase 2] Full fine-tuning (80 epochs, LLRD + MixUp/CutMix) ...",
          flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = build_llrd_optimizer(
        model, base_encoder_lr=5e-5, head_lr=5e-4, decay=0.75, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=40, T_mult=1, eta_min=1e-6
    )
    patience, PATIENCE = 0, 20  # conservative — avoid premature stopping

    for epoch in range(1, 81):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device,
                           scaler, use_amp, use_augmix=True)
        vl, va = run_epoch(model, val_loader, criterion, None, device,
                           scaler, use_amp, use_augmix=False)
        scheduler.step()
        for k, v in zip(("train_loss","val_loss","train_acc","val_acc"), (tl,vl,ta,va)):
            history[k].append(v)
        print(f"  P2 Epoch {epoch:2d}/80 | Train {tl:.4f}/{ta*100:.2f}% | "
              f"Val {vl:.4f}/{va*100:.2f}%", flush=True)
        if va > best_val_acc:
            best_val_acc = va
            patience = 0
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "clf_best.pth"))
            print(f"  -> Checkpoint (val_acc={best_val_acc*100:.2f}%)", flush=True)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  Early stopping at P2 epoch {epoch} (patience={PATIENCE})",
                      flush=True)
                break

    np.save(os.path.join(_ROOT, "clf_history.npy"), history)
    print(f"\nClassification fine-tuning complete. Best val acc: {best_val_acc*100:.2f}%",
          flush=True)


if __name__ == "__main__":
    main()
