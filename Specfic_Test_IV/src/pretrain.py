"""Task IX.A — Phase 1: MAE Pre-training on no_sub samples only (spec-compliant).

Strategy (per spec: "Train a MAE on the no_sub samples")
---------------------------------------------------------
Pre-train exclusively on ~26,504 no_sub training images.

Why no_sub-only pre-training is scientifically optimal
------------------------------------------------------
1. The encoder learns the CLEAN Einstein ring baseline: undistorted arc
   geometry, lens caustics, and source structure — with no class label.
2. CDM subhalo perturbations (local flux anomalies, 3-9 px) and axion
   vortex patterns (radial interference rings) are then effectively
   OUT-OF-DISTRIBUTION signals relative to this learned prior.
3. Downstream classification becomes an anomaly-detection problem:
   "how far does this image deviate from the no_sub prior?" — a much
   cleaner, more discriminative signal than if the encoder had been
   exposed to all three classes during pre-training.
4. Masked patch reconstruction on no_sub only forces the encoder to
   build a maximally compact, complete model of undistorted ring geometry
   from just 25% visible patches — the strongest possible spatial prior.

v2 changes
----------
* ViT-Small encoder: embed_dim=384, depth=6, heads=6 (was 192/4/4)
  Larger capacity builds richer geometry representations that generalise
  better to CDM subhalo and axion fine-tuning.
* 150 epochs: longer training builds stronger no_sub representations.
  With ~26K images and batch=256, 150 epochs = ~15K gradient steps —
  comparable to the MAE paper's 800 epochs on ImageNet in terms of
  per-sample exposure.

Usage
-----
    python src/pretrain.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from mae_model import MAE
from data_utils import LensingClfDataset, extract_datasets

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
WEIGHTS_DIR = os.path.join(_ROOT, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def main() -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}", flush=True)

    # Pre-train on no_sub ONLY — encoder learns the clean lensing baseline
    print("Loading pre-training data (no_sub only, per spec) ...", flush=True)
    train_ds = LensingClfDataset(split="train", nosub_only=True, augment=True)
    val_ds   = LensingClfDataset(split="val",   nosub_only=True, augment=False)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

    # v2: ViT-Small encoder (384/6/6) — larger capacity, stronger representations
    model = MAE(
        img_size=64, patch_size=8,
        encoder_embed_dim=384, encoder_depth=6, encoder_heads=6,
        decoder_embed_dim=192, decoder_depth=2, decoder_heads=6,
        mask_ratio=0.75,
    ).to(device)
    print(f"MAE parameters: {model.count_parameters():,}", flush=True)
    print(f"  Encoder: embed_dim=384, depth=6, heads=6 (ViT-Small)", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95)
    )
    MAX_EPOCHS = 150
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_n = 0.0, 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    loss, _, _ = model(imgs)
            else:
                loss, _, _ = model(imgs)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            t_loss += loss.item() * imgs.size(0); t_n += imgs.size(0)

        # ── Val ────────────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                if use_amp:
                    with torch.autocast(device_type="cuda"):
                        loss, _, _ = model(imgs)
                else:
                    loss, _, _ = model(imgs)
                v_loss += loss.item() * imgs.size(0); v_n += imgs.size(0)

        scheduler.step()
        tr, vl = t_loss / t_n, v_loss / v_n
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | Train loss {tr:.5f} | Val loss {vl:.5f}",
              flush=True)

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "mae_pretrained.pth"))
            print(f"  -> Checkpoint saved (val_loss={best_val_loss:.5f})", flush=True)

    np.save(os.path.join(_ROOT, "pretrain_history.npy"), history)
    print(f"\nPre-training complete. Best val loss: {best_val_loss:.5f}", flush=True)


if __name__ == "__main__":
    main()
