"""Task IX.B — Evaluation: MSE, SSIM, PSNR + visual comparison grid."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from mae_model import MAE, SRDecoder
from data_utils import SRDataset
from finetune_sr import ssim, psnr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
WEIGHTS_PATH = os.path.join(_ROOT, "weights", "sr_best.pth")
PLOTS_DIR    = os.path.join(_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Build model
    mae = MAE(img_size=64, patch_size=8, encoder_embed_dim=384, encoder_depth=6,
              encoder_heads=6, decoder_embed_dim=192, decoder_depth=2,
              decoder_heads=6, mask_ratio=0.75)
    model = SRDecoder(mae.encoder, scale=2, decoder_dim=256, depth=6,
                      num_heads=8, residual=True)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    print(f"Loaded: {WEIGHTS_PATH}  ({model.count_parameters():,} params)", flush=True)

    val_ds = SRDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Val pairs: {len(val_ds):,}", flush=True)

    sample_lr, sample_hr = val_ds[0]
    hr_size = tuple(sample_hr.shape[-2:])

    # Full evaluation
    model.eval()
    all_mse, all_psnr, all_ssim = [], [], []
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            pred = model(lr_imgs, hr_size=hr_size).clamp(0, 1)
            all_mse.append(F.mse_loss(pred, hr_imgs).item())
            all_psnr.append(psnr(pred, hr_imgs))
            all_ssim.append(ssim(pred, hr_imgs))

    mean_mse  = np.mean(all_mse)
    mean_psnr = np.mean(all_psnr)
    mean_ssim = np.mean(all_ssim)

    print(f"\nMSE  : {mean_mse:.6f}")
    print(f"PSNR : {mean_psnr:.2f} dB")
    print(f"SSIM : {mean_ssim:.4f}")

    # Visual comparison grid (4 examples: LR bilinear | SR prediction | HR GT)
    model.eval()
    n_show = 4
    samples = [val_ds[i] for i in range(n_show)]
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show))
    col_titles = ["LR Input (bilinear)", "SR Output (MAE)", "HR Ground Truth"]

    with torch.no_grad():
        for row, (lr, hr) in enumerate(samples):
            lr_t = lr.unsqueeze(0).to(device)
            sr   = model(lr_t, hr_size=hr_size).squeeze(0).clamp(0, 1).cpu()

            # Bilinear upsampled LR as baseline
            lr_up = F.interpolate(lr.unsqueeze(0), size=hr_size,
                                  mode="bilinear", align_corners=False).squeeze(0)

            imgs = [lr_up[0].numpy(), sr[0].numpy(), hr[0].numpy()]
            for col, (img, title) in enumerate(zip(imgs, col_titles)):
                axes[row, col].imshow(img, cmap="inferno", origin="lower",
                                      vmin=0, vmax=1)
                if row == 0:
                    axes[row, col].set_title(title, fontsize=11)
                axes[row, col].axis("off")

    fig.suptitle(f"Super-Resolution — PSNR {mean_psnr:.2f} dB | SSIM {mean_ssim:.4f}",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sr_comparison_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

    # Training curves
    history_path = os.path.join(_ROOT, "sr_history.npy")
    if os.path.exists(history_path):
        h = np.load(history_path, allow_pickle=True).item()
        epochs = range(1, len(h["val_mse"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(epochs, h["train_mse"], label="Train"); axes[0].plot(epochs, h["val_mse"], label="Val")
        axes[0].set_title("MSE Loss"); axes[0].legend()
        axes[1].plot(epochs, h["val_psnr"])
        axes[1].set_title("Val PSNR (dB)"); axes[1].set_xlabel("Epoch")
        axes[2].plot(epochs, h["val_ssim"])
        axes[2].set_title("Val SSIM"); axes[2].set_xlabel("Epoch")
        plt.tight_layout()
        path2 = os.path.join(PLOTS_DIR, "sr_training_curves.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved: {path2}")

    print("\n" + "=" * 45)
    print("SUPER-RESOLUTION RESULTS")
    print("=" * 45)
    print(f"{'MSE':<20} {mean_mse:>22.6f}")
    print(f"{'PSNR (dB)':<20} {mean_psnr:>22.2f}")
    print(f"{'SSIM':<20} {mean_ssim:>22.4f}")
    print("=" * 45)


if __name__ == "__main__":
    main()
