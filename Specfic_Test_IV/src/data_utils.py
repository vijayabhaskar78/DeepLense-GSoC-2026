"""Dataset utilities for Test IV — extracts zips and provides PyTorch Dataset classes.

Dataset 1 (Classification): no_sub / cdm / axion — 64x64 float64 .npy images
Dataset 2 (Super-resolution): HR (1,150,150) and LR (1,75,75) float64 .npy pairs
"""

import os
import io
import zipfile
import random
import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
DATA_DIR = os.path.join(_ROOT, "data")
CLF_DIR  = os.path.join(DATA_DIR, "classification")
SR_DIR   = os.path.join(DATA_DIR, "superres")
ZIP1     = os.path.join(_ROOT, "Dataset 1.zip")
ZIP2     = os.path.join(_ROOT, "Dataset 2.zip")

CLF_CLASSES = {"no_sub": 0, "cdm": 1, "axion": 2}
CLASS_NAMES  = ["No Substructure", "CDM Subhalo", "Axion"]

IMG_SIZE = 64   # Classification images are 64x64


# ── Extraction ────────────────────────────────────────────────────────────────
def extract_datasets(force: bool = False) -> None:
    """Extract Dataset 1 and Dataset 2 zips to data/ directories.

    Uses a sentinel file to skip extraction if already done.
    """
    sentinel1 = os.path.join(CLF_DIR, ".extracted")
    sentinel2 = os.path.join(SR_DIR,  ".extracted")

    if not os.path.exists(sentinel1) or force:
        print("Extracting Dataset 1 (classification) ...", flush=True)
        os.makedirs(CLF_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP1) as z:
            members = [m for m in z.namelist()
                       if m.endswith(".npy") and "__MACOSX" not in m]
            for i, name in enumerate(members):
                # Determine class from path: Dataset/no_sub/xxx.npy
                parts = name.replace("\\", "/").split("/")
                cls = next((p for p in parts if p in CLF_CLASSES), None)
                if cls is None:
                    continue
                fname = parts[-1]
                out_dir = os.path.join(CLF_DIR, cls)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, fname)
                if not os.path.exists(out_path):
                    with z.open(name) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                if (i + 1) % 10000 == 0:
                    print(f"  {i+1:,}/{len(members):,} extracted", flush=True)
        open(sentinel1, "w").close()
        print("Dataset 1 extracted.", flush=True)
    else:
        print("Dataset 1 already extracted.", flush=True)

    if not os.path.exists(sentinel2) or force:
        print("Extracting Dataset 2 (super-resolution) ...", flush=True)
        os.makedirs(SR_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP2) as z:
            members = [m for m in z.namelist()
                       if m.endswith(".npy") and "__MACOSX" not in m]
            for name in members:
                parts = name.replace("\\", "/").split("/")
                sub = next((p for p in parts if p in ("HR", "LR")), None)
                if sub is None:
                    continue
                fname = parts[-1]
                out_dir = os.path.join(SR_DIR, sub)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, fname)
                if not os.path.exists(out_path):
                    with z.open(name) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
        open(sentinel2, "w").close()
        print("Dataset 2 extracted.", flush=True)
    else:
        print("Dataset 2 already extracted.", flush=True)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _log_stretch_norm(img: np.ndarray) -> np.ndarray:
    """Normalise to [0,1] then apply log-stretch to amplify faint substructure.

    Images in Dataset 1 are NOT min-max normalised (values up to ~4.0).
    We first clip and min-max normalise, then apply log-stretch.
    """
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)       # [0, 1]
    img = np.log1p(img * 10.0) / np.log1p(10.0)  # log-stretch
    img = (img - 0.5) / 0.5                # [-1, 1]
    return img


def _minmax_norm(img: np.ndarray) -> np.ndarray:
    """Simple [0,1] min-max for super-resolution (preserves pixel ratios)."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img


# ── Classification dataset ────────────────────────────────────────────────────
class LensingClfDataset(Dataset):
    """Loads 64x64 gravitational lensing images for 3-class classification.

    Preloads ALL images into CPU RAM tensors during __init__ for O(1) access.
    At 64×64 float32, 89K images ≈ 1.46 GB; 26K no_sub ≈ 434 MB — fits in RAM
    and eliminates per-epoch disk I/O bottleneck (especially important on
    network/cloud paths like OneDrive).

    Applies per-image min-max normalisation + log-stretch (identical to
    Common Test I preprocessing for a fair cross-test comparison).

    Parameters
    ----------
    split      : 'train' or 'val' (90/10 split, fixed seed for reproducibility)
    nosub_only : If True, only load no_sub class (for MAE pre-training)
    augment    : Apply random flips and rotations (training only)
    """

    def __init__(
        self,
        split: str = "train",
        nosub_only: bool = False,
        augment: bool = False,
        seed: int = 42,
    ):
        self.augment = augment
        paths_and_labels: list[tuple[str, int]] = []

        classes = ["no_sub"] if nosub_only else list(CLF_CLASSES.keys())
        for cls in classes:
            cls_dir = os.path.join(CLF_DIR, cls)
            files = sorted([
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.endswith(".npy")
            ])
            rng = random.Random(seed)
            rng.shuffle(files)
            n = len(files)
            n_train = int(n * 0.9)
            files = files[:n_train] if split == "train" else files[n_train:]
            paths_and_labels.extend((f, CLF_CLASSES[cls]) for f in files)

        print(f"  Preloading {len(paths_and_labels):,} images into RAM ...", flush=True)
        images, labels = [], []
        for path, label in paths_and_labels:
            img = np.load(path, allow_pickle=True)
            if img.dtype == object:
                img = np.array(img.flat[0])
            img = img.astype(np.float32)
            if img.ndim == 2:
                img = img[np.newaxis, ...]   # (1, H, W)
            img = _log_stretch_norm(img)
            images.append(torch.from_numpy(img.astype(np.float32)))
            labels.append(label)

        self.images = torch.stack(images)           # (N, 1, 64, 64)
        self.labels = torch.tensor(labels, dtype=torch.long)
        print(f"  Preload complete. Shape: {self.images.shape}", flush=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.augment:
            img = _augment(img)
        return img, self.labels[idx].item()


# ── Super-resolution dataset ──────────────────────────────────────────────────
class SRDataset(Dataset):
    """Paired LR (75x75) / HR (150x150) lensing images for super-resolution.

    Preloads all image pairs into RAM for O(1) access during training.
    LR (1,75,75) + HR (1,150,150) per pair ≈ 112 KB; 5K pairs ≈ 563 MB total.

    Parameters
    ----------
    split   : 'train' or 'val' (90/10 split)
    """

    def __init__(self, split: str = "train", seed: int = 42):
        hr_dir = os.path.join(SR_DIR, "HR")
        lr_dir = os.path.join(SR_DIR, "LR")

        hr_files = sorted(os.listdir(hr_dir))
        lr_files = sorted(os.listdir(lr_dir))

        # Match by sample name
        hr_names = {os.path.splitext(f)[0]: f for f in hr_files}
        lr_names = {os.path.splitext(f)[0]: f for f in lr_files}
        common = sorted(set(hr_names) & set(lr_names))

        rng = random.Random(seed)
        rng.shuffle(common)
        n = len(common)
        n_train = int(n * 0.9)
        common = common[:n_train] if split == "train" else common[n_train:]

        pairs = [
            (os.path.join(hr_dir, hr_names[k]),
             os.path.join(lr_dir, lr_names[k]))
            for k in common
        ]

        print(f"  Preloading {len(pairs):,} SR pairs into RAM ...", flush=True)
        lr_list, hr_list = [], []
        for hr_path, lr_path in pairs:
            hr = np.load(hr_path, allow_pickle=True)
            lr = np.load(lr_path, allow_pickle=True)
            if hr.dtype == object:
                hr = np.array(hr.flat[0])
            if lr.dtype == object:
                lr = np.array(lr.flat[0])
            hr = _minmax_norm(hr.astype(np.float32))
            lr = _minmax_norm(lr.astype(np.float32))
            if hr.ndim == 2:
                hr = hr[np.newaxis, ...]
            if lr.ndim == 2:
                lr = lr[np.newaxis, ...]
            lr_list.append(torch.from_numpy(lr.astype(np.float32)))
            hr_list.append(torch.from_numpy(hr.astype(np.float32)))

        self.lr_images = torch.stack(lr_list)   # (N, 1, 75, 75)
        self.hr_images = torch.stack(hr_list)   # (N, 1, 150, 150)
        print(f"  SR preload complete. LR: {self.lr_images.shape}, HR: {self.hr_images.shape}",
              flush=True)

    def __len__(self) -> int:
        return len(self.lr_images)

    def __getitem__(self, idx: int):
        return self.lr_images[idx], self.hr_images[idx]


# ── Augmentation ──────────────────────────────────────────────────────────────
def _augment(x: torch.Tensor) -> torch.Tensor:
    """Lensing-appropriate augmentation: flips + rotations only (no colour jitter)."""
    if random.random() > 0.5:
        x = torch.flip(x, dims=[2])
    if random.random() > 0.5:
        x = torch.flip(x, dims=[1])
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k=k, dims=[1, 2])
    return x
