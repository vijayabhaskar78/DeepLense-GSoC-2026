"""Physics-Informed Neural Network (PINN) classifier for gravitational lensing.

Architecture
------------
The gravitational lensing equation (thin-lens approximation, SIS model):

    β⃗ = θ⃗ − θ_E · (θ⃗ / |θ⃗|)

where:
  β⃗    = source position
  θ⃗    = image position
  θ_E  = Einstein radius (angular scale of the Einstein ring)

For an on-axis source (β⃗ = 0) the image forms a perfect ring at |θ⃗| = θ_E.
Real lensing images deviate from this ideal template according to substructure
type — CDM subhalos create localised flux anomalies while axion vortices
imprint periodic interference patterns.

Network design
--------------
1. ResNet-18 backbone (pretrained, 1-ch adapted) → 512-dim features
2. Physics head : Linear(512 → 3) → (cx, cy, r_E)
   - cx, cy  : lens centroid in [−0.2, +0.2] (relative to image centre)
   - r_E     : Einstein radius  in [0.05, 0.40] (relative image units)
3. Ring template generator (differentiable SIS ring):
   T(cx, cy, r_E)[x,y] = exp(−½ · ((|(x,y)−(cx,cy)| − r_E) / σ)²)
   This is the physics prior — where the lensing equation predicts flux.
4. Residual encoder : small CNN on (image − ring_template) → 128-dim
   The residual captures everything the smooth SIS ring cannot explain:
   subhalo-induced flux anomalies and axion vortex fringes.
5. Classifier : Linear(512 + 128 → 256 → 3)

Loss
----
    L = L_cls  +  λ · L_phys
    L_cls  = CrossEntropy(logits, labels)
    L_phys = MSE(ring_template, Gaussian_blur(image_01))
             where image_01 = (image + 1) / 2  ∈ [0, 1]

L_phys penalises ring parameters that are inconsistent with the smooth
(blurred) ring structure actually present in the image.  The Gaussian blur
removes fine substructure so the ring template only needs to match the
dominant ring, not the substructure noise.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class PINNClassifier(nn.Module):
    """Physics-Informed Neural Network for gravitational lensing classification.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for single-channel lensing images).
    num_classes : int
        Output classes (3).
    img_size : int
        Spatial size of the input images (150 for this dataset).
    ring_sigma : float
        Width (std) of the Gaussian ring template in relative image units.
    phys_lambda : float
        Weight of the physics loss (stored here for reference; applied in
        training script).
    pretrained : bool
        Load ImageNet weights for the ResNet backbone.
    """

    def __init__(
        self,
        in_channels: int   = 1,
        num_classes: int   = 3,
        img_size: int      = 150,
        ring_sigma: float  = 0.03,
        phys_lambda: float = 0.1,
        dropout: float     = 0.3,
        pretrained: bool   = True,
    ) -> None:
        super().__init__()
        self.img_size    = img_size
        self.ring_sigma  = ring_sigma
        self.phys_lambda = phys_lambda

        # ── ResNet-18 backbone (conv1 … avgpool → 512-dim) ──────────────────
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = tv_models.resnet18(weights=weights)

        # Average pretrained RGB weights → single-channel kernel
        w_gray = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        resnet.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        resnet.conv1.weight = nn.Parameter(w_gray)

        # Remove final FC; keep up to avgpool  → (B, 512)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool, nn.Flatten(),
        )

        # ── Physics head: predict (cx, cy, r_E) ─────────────────────────────
        self.physics_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 3),
        )

        # ── Residual encoder: small CNN on (image − ring) → 128-dim ─────────
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),   # 75×75
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),  # 38×38
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), # 19×19
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                   # 128-dim
        )

        # ── Classifier head: backbone (512) + residual (128) → classes ──────
        feat_dim = 512 + 128
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Pre-build coordinate grids (not parameters — registered as buffers)
        y = torch.linspace(-0.5, 0.5, img_size)
        x = torch.linspace(-0.5, 0.5, img_size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")     # (H, W)
        self.register_buffer("grid_yy", yy.unsqueeze(0))  # (1, H, W)
        self.register_buffer("grid_xx", xx.unsqueeze(0))  # (1, H, W)

        # Pre-build Gaussian blur kernel (fixed, not learned)
        self._build_blur_kernel(kernel_size=15, sigma=3.0)

    # ── Differentiable SIS ring template ──────────────────────────────────────
    def ring_template(
        self, cx: torch.Tensor, cy: torch.Tensor, rE: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a per-sample soft Einstein ring.

        Parameters
        ----------
        cx, cy : (B,)   Lens centroid in [−0.2, +0.2].
        rE     : (B,)   Einstein radius in [0.05, 0.40].

        Returns
        -------
        (B, 1, H, W) ring template in [0, 1].
        """
        cx_ = cx.view(-1, 1, 1)
        cy_ = cy.view(-1, 1, 1)
        rE_ = rE.view(-1, 1, 1)

        dist = torch.sqrt(
            (self.grid_xx - cx_) ** 2 + (self.grid_yy - cy_) ** 2 + 1e-8
        )  # (B, H, W)

        ring = torch.exp(-0.5 * ((dist - rE_) / self.ring_sigma) ** 2)
        return ring.unsqueeze(1)   # (B, 1, H, W)

    # ── Fixed Gaussian blur (physics target smoothing) ─────────────────────
    def _build_blur_kernel(self, kernel_size: int = 15, sigma: float = 3.0) -> None:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g      = torch.exp(-0.5 * (coords / sigma) ** 2)
        g      = g / g.sum()
        k2d    = g.outer(g)                          # (k, k)
        k2d    = k2d.view(1, 1, kernel_size, kernel_size)
        self.register_buffer("blur_kernel", k2d)
        self._blur_pad = kernel_size // 2

    def gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fixed Gaussian blur channel-wise."""
        C = x.size(1)
        k = self.blur_kernel.expand(C, 1, -1, -1)
        return F.conv2d(x, k, padding=self._blur_pad, groups=C)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        logits      : (B, num_classes)
        ring        : (B, 1, H, W)   ring template (for physics loss)
        blur_target : (B, 1, H, W)   Gaussian-blurred image_01 (physics target)
        """
        # Backbone
        feat = self.backbone(x)          # (B, 512)

        # Physics params
        raw     = self.physics_head(feat)                      # (B, 3)
        cx      = torch.tanh(raw[:, 0]) * 0.2                  # [−0.2, +0.2]
        cy      = torch.tanh(raw[:, 1]) * 0.2
        rE      = torch.sigmoid(raw[:, 2]) * 0.35 + 0.05       # [0.05, 0.40]

        # Ring template  (differentiable w.r.t. cx, cy, rE)
        ring = self.ring_template(cx, cy, rE)                  # (B, 1, H, W)

        # Residual encoding  (image − ring_template carries substructure info)
        image_01  = (x + 1.0) / 2.0                           # [−1,1] → [0,1]
        residual  = image_01 - ring.detach()                   # stop-gradient on ring
        res_feat  = self.residual_encoder(residual)            # (B, 128)

        # Classification
        combined = torch.cat([feat, res_feat], dim=1)          # (B, 640)
        logits   = self.classifier(combined)

        # Physics target
        blur_target = self.gaussian_blur(image_01.detach())    # (B, 1, H, W)

        return logits, ring, blur_target

    # ── Convenience ───────────────────────────────────────────────────────────
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
