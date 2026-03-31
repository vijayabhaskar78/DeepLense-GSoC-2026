# Specific Test III — Neural Operators for Gravitational Lensing Classification

**ML4SCI / DeepLense — GSoC 2026**

Classify strong gravitational lensing images into three categories using a **Fourier Neural Operator (FNO)** backbone — an architecture that operates in function space via spectral convolutions, giving it a global receptive field from the first layer.

---

## Task

Build a neural operator classifier (FNO or DeepONet) replacing the standard convolutional backbone from Common Test I. Compare against the ResNet-18 baseline.

**Dataset**: Same 3-class gravitational lensing dataset as Common Test I
- 30,000 training images (10,000 per class)
- 7,500 validation images (2,500 per class)
- Single-channel 150×150 `.npy` images, min-max normalised to [0, 1]

**Classes**: No Substructure | CDM Subhalo | Axion Vortex

---

## Architecture: Fourier Neural Operator (FNO)

```
Input  (1, 150, 150)  [raw .npy]
  ↓  Bilinear resize → (1, 64, 64)        [src/train.py preprocessing]
  ↓  Conv1×1  (1 → 32 channels)           [Lifting]
  ↓  FNOBlock × 4                          [FNO Trunk]
       SpectralConv2d(32→32, modes=16)
     + Conv3×3 bypass(32→32)
     → BatchNorm2d → GELU → skip(+ x)
  ↓  GAP + GMP → concat → (B, 64)         [Multi-Scale Pooling]
  ↓  BN → Linear(64→512) → BN → GELU → Dropout(0.25)
  ↓  Linear(512→256) → BN → GELU → Dropout(0.125)
  ↓  Linear(256→3)                         [Logits]

Total parameters: ~4.4M  (vs ~11.2M for ResNet-18)
Pretrained backbone: None — trained from scratch
```

### Why FNO for Gravitational Lensing?

Standard CNNs build their receptive field gradually through stacked local convolutions. For gravitational lensing this is a suboptimal inductive bias:

- **CDM subhalos** create correlated flux anomalies across opposite sides of the Einstein ring — a pair-wise perturbation that spans the full image
- **Axion vortices** produce radially symmetric interference patterns that are inherently global

A Fourier Neural Operator applies the FFT on every forward pass, giving every output pixel a dependency on every input pixel from the very first layer. This is physically motivated: gravitational lensing follows an integral equation (the lens equation), and neural operators are universal approximators for such equations (Li et al., ICLR 2021).

### Key Design Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| `modes = 16` | 16 Fourier modes per dim | CDM perturbations appear at 5–15 cycles/image; wider spectral coverage than modes=12 |
| `width = 32` | 32 hidden channels | Lightweight, trains from scratch without overfitting |
| `depth = 4` | 4 FNO blocks | Sufficient hierarchical spectral representation |
| `BatchNorm2d` | (not InstanceNorm) | InstanceNorm zeros per-sample spatial mean, making GAP identical across classes |
| `Conv3×3 bypass` | (not Conv1×1) | Spatial receptive field grows 3→5→9→17 px with depth, capturing CDM subhalo scales |
| FocalLoss (γ=2.0, smoothing=0.05) | (not CrossEntropy) | Down-weights easy no_sub examples, focuses training on hard CDM/axion boundaries |

---

## Results

| Metric | ResNet-18 (Test I) | FNO (Test III) |
|--------|-------------------|----------------|
| Accuracy (Standard) | 95.64% | **77.75%** |
| Accuracy (8× TTA) | 96.51% | **79.87%** |
| Macro AUC (Standard) | 0.9941 | **0.9155** |
| Micro AUC (Standard) | 0.9951 | **0.9258** |
| Macro AUC (TTA) | 0.9956 | **0.9254** |
| Micro AUC (TTA) | 0.9965 | **0.9355** |
| Parameters | ~11.2M (pretrained) | ~4.4M (scratch) |

> Results columns update automatically when you run the notebook — the comparison cell reads `HISTORY` and live metrics directly.

---

## Preprocessing

Identical to Common Test I for a fair comparison:

**Log-stretch**: `log1p(x × 10) / log1p(10)` maps [0,1] → [0,1] with ~6× amplification of faint substructure relative to the bright arc, then shifted to [−1, 1].

**Training augmentation** exploiting rotational symmetry of lensing:
- Random horizontal/vertical flips
- Random 90°/180°/270° rotations
- Gaussian noise (σ = 0.02)
- Random patch erasing (prob=0.4, patch H/8–H/4) — simulates partial occlusion of substructure

---

## Training

- **Optimiser**: AdamW (weight_decay=1e-4)
- **Schedule**: OneCycleLR (max_lr=2e-3, pct_start=0.10, epochs=100) — warm up to 2e-3 then cosine anneal
- **Loss**: FocalLoss (gamma=2.0, label_smoothing=0.05)
- **Batch size**: 128
- **Max epochs**: 100
- **Early stopping**: patience=20 (conservative — spectral learning needs time)
- **Mixed precision**: AMP on CUDA (~2× speedup)
- **Gradient clipping**: max_norm=1.0 (stabilises complex-valued spectral weight training)

---

## Repository Structure

```
Specfic_Test_III/
├── Test3_Neural_Operators.ipynb  ← main notebook
├── src/
│   ├── model.py       ← SpectralConv2d, FNOBlock, FNOClassifier
│   ├── train.py       ← training script
│   └── evaluate.py    ← evaluation + plots
├── weights/
│   └── best_fno.pth   ← best validation checkpoint
├── plots/             ← generated figures
├── requirements.txt
└── README.md
```

---

## Reproducing Results

```bash
pip install -r requirements.txt

# Train
python src/train.py

# Evaluate (generates all plots)
python src/evaluate.py

# Or run the full notebook end-to-end
jupyter notebook Test3_Neural_Operators.ipynb
```

**Dataset**: Shared with Common Test I at `../Common_Test_I/dataset/dataset/`

---

## Reference

Li, Z. et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
