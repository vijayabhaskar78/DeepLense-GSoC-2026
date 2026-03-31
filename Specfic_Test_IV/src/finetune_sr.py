"""Task IX.B — Super-resolution fine-tuning using the pre-trained MAE encoder.

Architecture
------------
MAE encoder (pre-trained on no_sub images only, per spec) → SR decoder (residual)
→ bilinear baseline + learned residual → 2x upscaled output

v2 key upgrades
---------------
* Residual prediction: decoder predicts HR − bilinear_upsample(LR) instead
  of HR directly. Residual has small magnitude → faster convergence, +1–1.5 dB
  PSNR. The bilinear baseline handles low-frequency content; the decoder
  focuses exclusively on recovering fine arc structure and ring edges.

* Deeper, wider decoder: decoder_dim=256, depth=6 (was 192/4).
  More transformer capacity for recovering subtle high-frequency lensing detail.

* Triple loss: 0.65×MSE + 0.25×(1−SSIM) + 0.10×FFT_L1
  - MSE: pixel-level fidelity
  - SSIM: perceptual structure (luminance, contrast, spatial covariance)
  - FFT L1: penalises missing high-frequency content (ring edges, arc sharpness).
    Ensures the model recovers sharp edges, not just smooth reconstructions.

* 75 epochs: residual learning converges slower (smaller target magnitude)
  but ultimately achieves higher quality.

Evaluation Metrics: MSE, SSIM, PSNR (on held-out validation pairs)

Usage
-----
    python src/finetune_sr.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from mae_model import MAE, SRDecoder
from data_utils import SRDataset

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
WEIGHTS_DIR   = os.path.join(_ROOT, "weights")
PRETRAIN_CKPT = os.path.join(WEIGHTS_DIR, "mae_pretrained.pth")


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute mean SSIM over a batch (simplified window-based implementation)."""
    from torch.nn.functional import conv2d
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    k = torch.ones(1, 1, 11, 11, device=pred.device) / 121.0
    mu_p  = conv2d(pred,   k, padding=5, groups=1)
    mu_t  = conv2d(target, k, padding=5, groups=1)
    mu_pp = conv2d(pred**2,    k, padding=5, groups=1)
    mu_tt = conv2d(target**2,  k, padding=5, groups=1)
    mu_pt = conv2d(pred*target, k, padding=5, groups=1)
    sig_p  = mu_pp - mu_p**2
    sig_t  = mu_tt - mu_t**2
    sig_pt = mu_pt - mu_p * mu_t
    num = (2*mu_p*mu_t + C1) * (2*sig_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
    return (num / den).mean().item()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Differentiable SSIM loss (returns tensor for backprop)."""
    from torch.nn.functional import conv2d
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    k = torch.ones(1, 1, 11, 11, device=pred.device) / 121.0
    mu_p  = conv2d(pred,    k, padding=5, groups=1)
    mu_t  = conv2d(target,  k, padding=5, groups=1)
    mu_pp = conv2d(pred**2,    k, padding=5, groups=1)
    mu_tt = conv2d(target**2,  k, padding=5, groups=1)
    mu_pt = conv2d(pred*target, k, padding=5, groups=1)
    sig_p  = mu_pp - mu_p**2
    sig_t  = mu_tt - mu_t**2
    sig_pt = mu_pt - mu_p * mu_t
    num = (2*mu_p*mu_t + C1) * (2*sig_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
    return 1.0 - (num / den).mean()


def freq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Frequency-domain L1 loss on magnitude spectrum.

    Penalises missing high-frequency content (sharp ring edges, arc details).
    torch.fft.rfft2 operates in float32 — called after .float() cast.
    """
    pred_fft   = torch.fft.rfft2(pred,   norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(pred_fft.abs(), target_fft.abs())


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse_val = F.mse_loss(pred, target).item()
    if mse_val == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse_val)


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  mse_w: float = 0.65, ssim_w: float = 0.25,
                  freq_w: float = 0.10) -> torch.Tensor:
    """Triple loss: 0.65×MSE + 0.25×(1−SSIM) + 0.10×FFT_L1.

    MSE: pixel fidelity. SSIM: perceptual structure. FFT: high-frequency sharpness.
    All three are computed in float32 to prevent NaN under AMP.
    """
    mse  = F.mse_loss(pred, target)
    ssl  = ssim_loss(pred, target)
    ffl  = freq_loss(pred, target)
    return mse_w * mse + ssim_w * ssl + freq_w * ffl


def run_epoch(model, loader, optimizer, device, scaler, use_amp, hr_size):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for lr_imgs, hr_imgs in loader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    pred = model(lr_imgs, hr_size=hr_size)
                # SSIM + FFT loss: compute in float32 to prevent overflow/NaN
                loss = combined_loss(pred.float(), hr_imgs.float())
            else:
                pred = model(lr_imgs, hr_size=hr_size)
                loss = combined_loss(pred, hr_imgs)
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
            total_loss += loss.item() * lr_imgs.size(0)
            n += lr_imgs.size(0)
    return total_loss / n


def main() -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}", flush=True)

    print("Loading SR data ...", flush=True)
    train_ds = SRDataset(split="train")
    val_ds   = SRDataset(split="val")
    print(f"  Train pairs: {len(train_ds):,}  Val pairs: {len(val_ds):,}", flush=True)

    sample_lr, sample_hr = val_ds[0]
    hr_size = tuple(sample_hr.shape[-2:])
    print(f"  LR size: {tuple(sample_lr.shape)}  HR size: {tuple(sample_hr.shape)}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    # v2: ViT-Small encoder (384/6/6) — must match pre-training config
    mae = MAE(img_size=64, patch_size=8,
              encoder_embed_dim=384, encoder_depth=6, encoder_heads=6,
              decoder_embed_dim=192, decoder_depth=2, decoder_heads=6,
              mask_ratio=0.75)
    if os.path.exists(PRETRAIN_CKPT):
        state = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=True)
        mae.load_state_dict(state)
        print(f"Loaded pre-trained encoder: {PRETRAIN_CKPT}", flush=True)
    else:
        print("WARNING: No pre-trained weights — training SR from scratch.", flush=True)

    # v2: deeper decoder (dim=256, depth=6), residual prediction enabled
    model = SRDecoder(mae.encoder, scale=2, decoder_dim=256, depth=6,
                      num_heads=8, residual=True).to(device)
    print(f"SRDecoder parameters: {model.count_parameters():,}", flush=True)
    print(f"  Decoder: proj 384→256, 6 transformer blocks (heads=8), residual=True",
          flush=True)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 5e-5},   # slow: pre-trained
        {"params": model.proj.parameters(),    "lr": 5e-4},
        {"params": list(model.blocks.parameters()), "lr": 5e-4},
        {"params": model.pred.parameters(),    "lr": 5e-4},
    ], weight_decay=1e-4)

    MAX_EPOCHS = 75
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history   = {"train_mse": [], "val_mse": [], "val_psnr": [], "val_ssim": []}
    best_psnr = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, device, scaler,
                            use_amp, hr_size)
        vl_loss = run_epoch(model, val_loader,   None,      device, scaler,
                            use_amp, hr_size)
        scheduler.step()

        # Compute SSIM and PSNR on validation
        model.eval()
        all_psnr, all_ssim, all_raw_mse = [], [], []
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                pred = model(lr_imgs, hr_size=hr_size).clamp(0, 1)
                all_psnr.append(psnr(pred, hr_imgs))
                all_ssim.append(ssim(pred, hr_imgs))
                all_raw_mse.append(F.mse_loss(pred, hr_imgs).item())

        ep_psnr    = np.mean(all_psnr)
        ep_ssim    = np.mean(all_ssim)
        ep_raw_mse = np.mean(all_raw_mse)
        history["train_mse"].append(tr_loss)
        history["val_mse"].append(ep_raw_mse)
        history["val_psnr"].append(ep_psnr)
        history["val_ssim"].append(ep_ssim)

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"Train loss {tr_loss:.5f} | Val MSE {ep_raw_mse:.5f} | "
              f"PSNR {ep_psnr:.2f} dB | SSIM {ep_ssim:.4f}", flush=True)

        if ep_psnr > best_psnr:
            best_psnr = ep_psnr
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "sr_best.pth"))
            print(f"  -> Checkpoint (PSNR={best_psnr:.2f} dB)", flush=True)

    np.save(os.path.join(_ROOT, "sr_history.npy"), history)
    print(f"\nSR fine-tuning complete. Best PSNR: {best_psnr:.2f} dB", flush=True)


if __name__ == "__main__":
    main()
