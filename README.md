# DeepLense GSoC 2026 — Test Submissions

**Applicant:** Vijaya Bhaskar Santhuluri<br>
**Email:** vijayabhaskar.santhuluri@gmail.com<br>
**Organization:** ML4SCI / DeepLense<br>
**Program:** Google Summer of Code 2026

---

## Overview

This repository contains solutions for the DeepLense GSoC 2026 evaluation tests and the accompanying project proposals.

```
Common_Test_I/        — Multi-class gravitational lensing classification (ResNet-18)
Specific_Test_II/     — Agentic AI workflow for autonomous DeepLenseSim orchestration
Specfic_Test_III/     — Neural Operator classifier (HybridFNO)
Specfic_Test_IV/      — Foundation Model for lensing (Masked Autoencoder)
Specfic_Test_VII/     — Physics-Informed Neural Network (PINN)
```

---

## Test I — Multi-Class Gravitational Lensing Classification

**Task:** Classify strong lensing images into three dark matter substructure categories — no substructure, CDM subhalo, and axion vortex — using a deep learning classifier trained on 150×150 single-channel images.

**Approach:** Fine-tuned a pretrained ResNet-18 on single-channel lensing images. The three classes are visually nearly identical at the pixel level (mean-image correlation >0.999), making standard pipelines insufficient. Two key ingredients: a log-stretch preprocessing step (`img = log1p(10*img)/log1p(10)`) that amplifies faint substructure signal approximately 6x, and transfer learning from ImageNet pretrained weights whose edge detectors transfer well to lensing morphology. Training uses a two-phase schedule — backbone frozen for warmup, then full fine-tuning with differential learning rates. Test-time augmentation (8x TTA) applied at evaluation.

**Results:**

| Metric | Standard | 8x TTA |
|--------|----------|--------|
| Accuracy | 95.64% | **96.51%** |
| Macro AUC | 0.9941 | **0.9956** |
| Micro AUC | 0.9951 | **0.9965** |

No overfitting — validation accuracy exceeds training accuracy throughout (augmentation makes training harder by design).

**Files:**
- `Common_Test_I/Test1_MultiClass_Classification.ipynb` — fully executed notebook with all outputs
- `Common_Test_I/src/train.py` — training script
- `Common_Test_I/src/evaluate.py` — evaluation and plot generation
- `Common_Test_I/weights/` — trained model weights

**Run:**
```bash
cd Common_Test_I
pip install -r requirements.txt
python src/train.py        # ~45 min on GPU
python src/evaluate.py     # regenerate all plots
```

---

## Test II — Agentic AI for DeepLenseSim Orchestration

**Task:** Build an agentic workflow that lets a researcher generate gravitational lensing images through natural language interaction with the DeepLenseSim simulation pipeline.

**Approach:** Built with Pydantic AI for typed tool definitions and native Pydantic model integration. The agent exposes three tools in sequence: a dry-run parameter validator (the user sees what will happen before anything runs), the simulation executor, and a visualization generator. Human-in-the-loop confirmation is enforced in the system prompt — the agent always summarizes parameters and requests approval before executing. Physics constraints (z_source > z_lens, mass bounds, axion mass auto-defaults for vortex simulations) are validated at the Pydantic model level before the simulation pipeline is touched, making invalid configurations impossible to execute.

The LLM runs locally via llama.cpp (Qwen3.5-9B) through an OpenAI-compatible server — no API cost dependency. Multi-turn conversation history is propagated via `result.all_messages()` so the agent retains confirmed parameters across clarification exchanges.

**Results:** All 3 instrument models x 3 substructure types = **9/9 configurations verified working.**

| Model | Resolution | Instrument | No Sub | CDM | Vortex |
|-------|-----------|------------|--------|-----|--------|
| Model I | 150x150 | Generic PSF | OK | OK | OK |
| Model II | 64x64 | Euclid | OK | OK | OK |
| Model III | 64x64 | HST | OK | OK | OK |

**Real failure modes discovered and handled during development:**
- Vortex simulations with `res > 200` cause OOM on 16GB GPUs — agent retries with reduced resolution
- Very low halo mass + high source redshift produces images with no visible Einstein ring (unusable for classifiers) — validation check before accepting output
- `z_source <= z_lens` is physically impossible — blocked by Pydantic validator at parse time, before any simulation runs

**Files:**
- `Specific_Test_II/Test2_Agentic_AI.ipynb` — fully executed notebook
- `Specific_Test_II/agent.py` — main agent implementation
- `Specific_Test_II/models.py` — Pydantic models with physics constraints
- `Specific_Test_II/tools/simulator.py` — DeepLenseSim tool wrapper

**Run:**
```bash
cd Specific_Test_II
pip install -r requirements.txt
# Start local LLM server (requires Qwen3.5-9B GGUF model):
llama-server -m Qwen3.5-9B-Q5_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99
python agent.py
```

---

## Test III — Neural Operator Classifier (HybridFNO)

**Task:** Apply a neural operator architecture to the lensing classification task, demonstrating the advantage of Fourier-based global receptive fields over standard CNNs for dark matter substructure detection.

**Approach:** Designed a hybrid architecture combining a pretrained ResNet-18 encoder (layers 1-2, outputting 128 channels at 19x19 feature maps) with three Fourier Neural Operator blocks and a classification head. Each FNO block contains a SpectralConv2d (modes=8, width=128), a parallel Conv3x3 bypass for local features, BatchNorm, GELU activation, and Squeeze-and-Excitation channel attention that learns which Fourier modes are discriminative for each substructure class. A residual connection wraps each block.

The FNO blocks provide a global receptive field from the first layer via FFT — CDM subhalos create correlated flux anomalies on opposite sides of the Einstein ring that CNNs need many layers to capture, but FNO sees the full image spectrally at every block. The pretrained ResNet backbone provides strong low-level features; a pure FNO from scratch (without backbone) stuck at 33% accuracy. The SE attention acts as a learnable spectral filter, suppressing irrelevant frequency modes and focusing capacity on the substructure signal.

**Results:**

| Metric | Standard | 8x TTA |
|--------|----------|--------|
| Accuracy | 96.80% | **97.12%** |
| Macro AUC | 0.9960 | **0.9975** |
| Micro AUC | 0.9965 | **0.9977** |

Outperforms the ResNet-18 baseline (96.51% / 0.9956) across all metrics. Largest gain on CDM subhalos — the hardest class.

**Architecture:**
```
Input (150x150, 1-channel)
  -> ResNet-18 backbone (pretrained, 1-ch adapted, layers 1-2)
     -> 128 channels @ 19x19
  -> Lift Conv1x1 (128 -> 128)
  -> 3x FNO Block:
       SpectralConv2d(modes=8, width=128)
       + Conv3x3 bypass
       + BatchNorm + GELU + SE attention
       + Residual connection
  -> Global Average Pool + Global Max Pool -> concat
  -> MLP head -> 3-class output
```

**Files:**
- `Specfic_Test_III/Test3_Neural_Operators.ipynb` — fully executed notebook
- `Specfic_Test_III/src/model.py` — HybridFNO architecture
- `Specfic_Test_III/src/train.py` — training script
- `Specfic_Test_III/src/evaluate.py` — evaluation script
- `Specfic_Test_III/weights/` — trained model weights (best_fno.pth, 14M params)

**Run:**
```bash
cd Specfic_Test_III
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

---

## Test IV — Foundation Model for Gravitational Lensing (MAE)

**Task:** Apply a Masked Autoencoder (MAE) self-supervised pretraining strategy to gravitational lensing images, then fine-tune for (A) substructure classification and (B) super-resolution.

**Approach:** Implemented a Vision Transformer (ViT-Small) based MAE with 75% random patch masking during pretraining. The encoder learns latent representations of lensing morphology from unlabeled images; the decoder reconstructs masked patches. After pretraining, the encoder is fine-tuned on two downstream tasks: classification (same 3-class task as Test I) and super-resolution (upscaling low-resolution lensing images). The MAE framework is well suited to lensing because the high spatial redundancy of the Einstein ring structure makes reconstruction a well-posed pretraining task, while the rare substructure perturbations represent the hard discriminative signal the encoder must learn to preserve.

**Results:** Both notebooks fully executed — pretraining converged, classification fine-tuning and super-resolution fine-tuning both completed with results and plots.

**Files:**
- `Specfic_Test_IV/Test4A_MAE_Classification.ipynb` — MAE pretraining + classification fine-tuning
- `Specfic_Test_IV/Test4B_SuperResolution.ipynb` — MAE fine-tuned for super-resolution
- `Specfic_Test_IV/weights/` — pretrained MAE, classification, and SR model weights

---

## Test VII — Physics-Informed Neural Network (PINN)

**Task:** Incorporate the thin-lens (SIS) equation as a physics constraint directly into the neural network training objective for lensing image classification.

**Approach:** ResNet backbone combined with a physics head that predicts lens parameters (center coordinates, Einstein radius), a differentiable ring template generator, and a residual encoder that processes the difference between the predicted ring template and the actual image. The loss combines standard cross-entropy with an MSE term between the predicted ring template and the blurred image, enforcing physical consistency with the SIS model throughout training.

**Files:**
- `Specfic_Test_VII/Test7_PINN_Classification.ipynb`

---

## Results Summary

| Test | Task | Key Result |
|------|------|------------|
| Common Test I | ResNet-18 classifier | 96.51% accuracy, 0.9956 AUC (8x TTA) |
| Specific Test II | Agentic AI (Pydantic AI) | 9/9 DeepLenseSim configurations verified |
| Specific Test III | HybridFNO neural operator | 97.12% accuracy, 0.9975 AUC (8x TTA) |
| Specific Test IV | MAE foundation model | Both notebooks executed |
| Specific Test VII | PINN physics-guided | Notebook prepared |

---

## Contact

**Vijaya Bhaskar Santhuluri**
vijayabhaskar.santhuluri@gmail.com
