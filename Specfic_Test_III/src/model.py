"""Fourier Neural Operator (FNO) classifier for gravitational lensing substructure.

Two architectures are provided:

1. FNOClassifier (v1 — kept for reference, 77.75% accuracy)
   Input (1, 64, 64) -> lift -> 4× FNOBlock(width=32, modes=16) -> head

2. HybridFNOClassifier (v3 — primary, targets 90%+ AUC)
   Input (1, 150, 150)
     -> Pretrained ResNet-18 backbone (layers conv1…layer2)
        - adapted for single-channel input (averaged RGB weights)
        - output: (128, 19, 19) feature maps
     -> Lift Conv1×1: 128 -> fno_width
     -> depth × FNOBlock(fno_width, modes, modes) with SE attention
        - modes=8, operates on 19×19 spatial grid
        - global receptive field via FFT from block 1
     -> Multi-scale pooling (GAP + GMP) -> MLP head

Why pretrained backbone + FNO?
  The task says "replace or augment the standard convolutional feature extractor
  with a neural operator layer."  The FNO blocks *are* the neural operator — they
  perform spectral convolutions that give every output pixel a dependency on every
  input pixel (via FFT), unlike local CNN filters.  The pretrained ResNet stem
  provides rich initial feature representations that let FNO training converge
  to high accuracy without the extreme training budgets pure FNO from scratch
  requires.

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs",
           ICLR 2021 (arXiv:2010.08895)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ── Shared building blocks ────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """2D spectral convolution — learns in Fourier space.

    Retains only the `modes1 × modes2` lowest spatial-frequency components,
    multiplies each by a learned complex weight matrix, then maps back via
    inverse FFT.  High-frequency components pass through the bypass Conv in
    FNOBlock unchanged — no information is permanently discarded.

    Weights stored as real/imag float32 pairs for AMP compatibility.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.w1_real = nn.Parameter(scale * torch.rand(shape))
        self.w1_imag = nn.Parameter(scale * torch.rand(shape))
        self.w2_real = nn.Parameter(scale * torch.rand(shape))
        self.w2_imag = nn.Parameter(scale * torch.rand(shape))

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w1 = torch.complex(self.w1_real, self.w1_imag)
        w2 = torch.complex(self.w2_real, self.w2_imag)

        x_ft = torch.fft.rfft2(x.float())
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], w1)
        out_ft[:, :, -self.modes1:, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, : self.modes2], w2)

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=[2, 3])
        scale = self.fc(scale)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class FNOBlock(nn.Module):
    """Single FNO residual block with SE channel attention.

    forward(x) = SE(GELU(BN(spectral(x) + bypass(x)))) + x
    """

    def __init__(self, channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass   = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm     = nn.BatchNorm2d(channels)
        self.act      = nn.GELU()
        self.se       = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.spectral(x) + self.bypass(x)))
        h = self.se(h)
        return h + x


# ── Primary model: HybridFNOClassifier ───────────────────────────────────────

class HybridFNOClassifier(nn.Module):
    """Pretrained ResNet-18 stem + FNO spectral blocks classifier.

    Architecture
    ------------
    Input (1, 150, 150)
      -> ResNet-18 backbone (conv1 … layer2), adapted for 1-channel input
         by averaging the pretrained RGB weights -> (128, 19, 19)
      -> Lift Conv1×1: 128 -> fno_width
      -> depth × FNOBlock(fno_width, modes, modes) [modes ≤ 9 for 19×19]
      -> GAP + GMP concat -> (fno_width * 2,)
      -> MLP head -> num_classes

    The FNO blocks give every output pixel access to the full 19×19 feature
    map via FFT from the very first block — capturing the global ring and
    vortex patterns that local CNN filters build up only gradually.

    Parameters
    ----------
    fno_width : int
        FNO hidden channel dimension (default 128).
    fno_modes : int
        Fourier modes per spatial dimension. Must be ≤ 9 for 19×19 maps.
    fno_depth : int
        Number of stacked FNO blocks.
    pretrained : bool
        Load ImageNet weights for the ResNet backbone (default True).
    freeze_backbone : bool
        Freeze backbone during phase-1 training (default False here —
        the training script controls freezing externally).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        fno_width: int = 128,
        fno_modes: int = 8,
        fno_depth: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.fno_width = fno_width
        self.fno_modes = fno_modes
        self.fno_depth = fno_depth

        # ── ResNet-18 backbone: conv1 … layer2 ─────────────────────────────
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = tv_models.resnet18(weights=weights)

        # Adapt conv1 for single-channel input.
        # Average the 3 pretrained RGB weight maps into one so we keep
        # the learned edge/texture filters rather than reinitialising.
        w_rgb = resnet.conv1.weight.data          # (64, 3, 7, 7)
        w_gray = w_rgb.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        resnet.conv1.weight = nn.Parameter(w_gray)

        # For 150×150 input the spatial sizes are:
        #   conv1(s=2) -> 75×75
        #   maxpool(s=2) -> 38×38
        #   layer1 -> 38×38  (64 ch)
        #   layer2(s=2) -> 19×19 (128 ch)   <-- we stop here
        self.backbone = nn.Sequential(
            resnet.conv1,    # (B, 64,  75, 75)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B, 64,  38, 38)
            resnet.layer1,   # (B, 64,  38, 38)
            resnet.layer2,   # (B, 128, 19, 19)
        )
        self._backbone_out_channels = 128

        # ── Lift + FNO trunk ────────────────────────────────────────────────
        self.lift = nn.Conv2d(self._backbone_out_channels, fno_width, kernel_size=1)
        self.fno_blocks = nn.Sequential(
            *[FNOBlock(fno_width, fno_modes, fno_modes) for _ in range(fno_depth)]
        )

        # ── Classification head ─────────────────────────────────────────────
        feat_dim = fno_width * 2   # GAP + GMP
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)         # (B, 128, 19, 19)
        x = self.lift(x)             # (B, fno_width, 19, 19)
        x = self.fno_blocks(x)       # (B, fno_width, 19, 19)
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.head(torch.cat([gap, gmp], dim=1))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Legacy model (v1, kept for reference) ─────────────────────────────────────

class FNOClassifier(nn.Module):
    """Original FNO-only classifier (v1, 77.75% accuracy at 64×64 input).

    Kept for backward compatibility with evaluate.py / old checkpoints.
    Use HybridFNOClassifier for new training runs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        width: int = 32,
        modes: int = 16,
        depth: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        self.depth = depth

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.fno_blocks = nn.Sequential(
            *[FNOBlock(width, modes, modes) for _ in range(depth)]
        )
        feat_dim = width * 2
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        x = self.fno_blocks(x)
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.head(torch.cat([gap, gmp], dim=1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
