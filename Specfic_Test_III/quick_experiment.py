"""Quick experiment: compare 4 configs, 15 epochs each on 3K samples.

A: Original  — FNO width=32, depth=4, 64x64 input (downscaled)
B: Wider     — FNO width=64, depth=6, 64x64 input (more capacity)
C: Stem64    — CNN stem 150->64 + FNO width=64, depth=6 (full-res local + fast FNO)
D: Stem75    — CNN stem 150->75 + FNO width=64, depth=6 (full-res local + larger FNO)

Goal: find which gives best val_acc in 15 epochs without sacrificing too much speed.
"""
import math, os, sys, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import SpectralConv2d, SEBlock  # reuse building blocks

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda")
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Common_Test_I", "dataset", "dataset"))
LABEL_MAP = {"no": 0, "sphere": 1, "vort": 2}
EPOCHS = 15
N_TRAIN = 3000
N_VAL = 750


# ── Shared FNOBlock ─────────────────────────────────────────────────────────────
class FNOBlock(nn.Module):
    def __init__(self, ch, modes):
        super().__init__()
        self.spec   = SpectralConv2d(ch, ch, modes, modes)
        self.bypass = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm   = nn.BatchNorm2d(ch)
        self.act    = nn.GELU()
        self.se     = SEBlock(ch)
    def forward(self, x):
        h = self.act(self.norm(self.spec(x) + self.bypass(x)))
        return self.se(h) + x


# ── Config A: Original FNO (baseline) ─────────────────────────────────────────
class ConfigA(nn.Module):
    """width=32, depth=4, 64x64 — mirrors the existing best_fno_v1_77pct.pth."""
    def __init__(self):
        super().__init__()
        self.lift   = nn.Conv2d(1, 32, 1)
        self.blocks = nn.Sequential(*[FNOBlock(32, 16) for _ in range(4)])
        feat = 64
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat), nn.Linear(feat, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(0.25), nn.Linear(256, 3))
    def forward(self, x):
        x = self.lift(x); x = self.blocks(x)
        return self.head(torch.cat([F.adaptive_avg_pool2d(x,1).flatten(1),
                                     F.adaptive_max_pool2d(x,1).flatten(1)], 1))


# ── Config B: Wider FNO at 64x64 ──────────────────────────────────────────────
class ConfigB(nn.Module):
    """width=64, depth=6, 64x64 — more capacity, same speed."""
    def __init__(self):
        super().__init__()
        self.lift   = nn.Conv2d(1, 64, 1)
        self.blocks = nn.Sequential(*[FNOBlock(64, 16) for _ in range(6)])
        feat = 128
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat), nn.Linear(feat, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(256, 3))
    def forward(self, x):
        x = self.lift(x); x = self.blocks(x)
        return self.head(torch.cat([F.adaptive_avg_pool2d(x,1).flatten(1),
                                     F.adaptive_max_pool2d(x,1).flatten(1)], 1))


# ── Config C: CNN stem 150→64 + FNO at 64x64 ──────────────────────────────────
class ConfigC(nn.Module):
    """CNN extracts features from 150x150, adaptive-pools to 64x64 for FNO."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(64),   # 75 → 64 cleanly
        )
        self.lift   = nn.Conv2d(64, 64, 1)
        self.blocks = nn.Sequential(*[FNOBlock(64, 16) for _ in range(6)])
        feat = 128
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat), nn.Linear(feat, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(256, 3))
    def forward(self, x):
        x = self.stem(x); x = self.lift(x); x = self.blocks(x)
        return self.head(torch.cat([F.adaptive_avg_pool2d(x,1).flatten(1),
                                     F.adaptive_max_pool2d(x,1).flatten(1)], 1))


# ── Config D: CNN stem 150→75 + FNO at 75x75 ──────────────────────────────────
class ConfigD(nn.Module):
    """Same as C but FNO at 75x75 (stride-2 gives exact 75x75 from 150x150)."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.lift   = nn.Conv2d(64, 64, 1)
        self.blocks = nn.Sequential(*[FNOBlock(64, 16) for _ in range(6)])
        feat = 128
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat), nn.Linear(feat, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(256, 3))
    def forward(self, x):
        x = self.stem(x); x = self.lift(x); x = self.blocks(x)
        return self.head(torch.cat([F.adaptive_avg_pool2d(x,1).flatten(1),
                                     F.adaptive_max_pool2d(x,1).flatten(1)], 1))


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_split(split, n, target_size=None):
    """Load n samples per class from split, optionally resize."""
    imgs, labs = [], []
    per_class = n // 3
    for sub, lbl in LABEL_MAP.items():
        folder = os.path.join(ROOT, split, sub)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))[:per_class]
        for f in files:
            img = np.load(os.path.join(folder, f)).astype(np.float32)
            if img.ndim == 2: img = img[None]
            img = (np.log1p(img * 10) / np.log1p(10) - 0.5) / 0.5
            t = torch.from_numpy(img)
            if target_size:
                t = F.interpolate(t.unsqueeze(0), size=(target_size, target_size),
                                   mode="bilinear", align_corners=False).squeeze(0)
            imgs.append(t); labs.append(lbl)
    return torch.stack(imgs).float(), torch.tensor(labs, dtype=torch.long)


def make_loader(images, labels, batch_size, shuffle):
    ds = list(zip(images, labels))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: (torch.stack([x for x,_ in b]),
                                            torch.stack([y for _,y in b])))


def run(name, model, tr_img, tr_lbl, va_img, va_lbl, lr=3e-3):
    model = model.to(DEVICE)
    tl = make_loader(tr_img, tr_lbl, 128, True)
    vl = make_loader(va_img, va_lbl, 256, False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=EPOCHS, steps_per_epoch=len(tl),
        pct_start=0.2, anneal_strategy="cos")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")
    best = 0.0; t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in tl:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

        model.eval(); c = n = 0
        with torch.no_grad():
            for x, y in vl:
                x = x.to(DEVICE); y = y.to(DEVICE)
                c += model(x).argmax(1).eq(y).sum().item(); n += y.size(0)
        acc = c / n; best = max(best, acc)
        if ep % 5 == 0 or ep == EPOCHS:
            print(f"  [{name}] ep{ep:2d} val={acc*100:.1f}% best={best*100:.1f}%", flush=True)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elapsed = time.time() - t0
    return best, elapsed, params


# ── Main ───────────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}\nLoading data...", flush=True)

tr64, trl   = load_split("train", N_TRAIN, target_size=64)
va64, val   = load_split("val",   N_VAL,   target_size=64)
tr150, _    = load_split("train", N_TRAIN, target_size=None)
va150, _    = load_split("val",   N_VAL,   target_size=None)
print(f"  Train 64x64: {tr64.shape}  |  Train 150x150: {tr150.shape}", flush=True)

results = {}

print(f"\n{'='*55}", flush=True)
print("A: Original FNO (width=32, depth=4, 64x64)", flush=True)
results["A"] = run("A", ConfigA(), tr64, trl, va64, val, lr=2e-3)

print(f"\n{'='*55}", flush=True)
print("B: Wider FNO (width=64, depth=6, 64x64)", flush=True)
results["B"] = run("B", ConfigB(), tr64, trl, va64, val, lr=3e-3)

print(f"\n{'='*55}", flush=True)
print("C: CNN-stem→64 + FNO (width=64, depth=6)", flush=True)
results["C"] = run("C", ConfigC(), tr150, trl, va150, val, lr=3e-3)

print(f"\n{'='*55}", flush=True)
print("D: CNN-stem→75 + FNO (width=64, depth=6)", flush=True)
results["D"] = run("D", ConfigD(), tr150, trl, va150, val, lr=3e-3)

print(f"\n{'='*55}")
print(f"{'Config':<6} {'Best val%':>10} {'Time(s)':>9} {'Params':>10}")
print("-" * 40)
for k, (best, t, p) in results.items():
    print(f"  {k}    {best*100:>8.2f}%  {t:>8.0f}s  {p:>9,}")
print()
winner = max(results, key=lambda k: results[k][0])
print(f"Winner: Config {winner}  ({results[winner][0]*100:.2f}%)")
if results[winner][0] > results["A"][0] + 0.05:
    print("-> Improvement confirmed. Proceed with full training on this config.")
else:
    print("-> No clear improvement. Revisit architecture.")
