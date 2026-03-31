"""End-to-end runner for Specific Test IV — MAE Foundation Model.

Runs in sequence:
  1. Extract datasets from zip files
  2. Phase 1: MAE pre-training on no_sub images
  3. Phase 2: Classification fine-tuning (Task IX.A)
  4. Phase 3: Super-resolution fine-tuning (Task IX.B)
  5. Evaluate classification (ROC/AUC)
  6. Evaluate super-resolution (MSE/SSIM/PSNR)

Usage
-----
    python run_test4.py
"""

import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC   = os.path.join(_HERE, "src")


def run(script_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-u",
         os.path.join(SRC, script_name)],
        cwd=_HERE,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)


def extract() -> None:
    """Extract datasets if not already done."""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-u", "-c",
         "import sys; sys.path.insert(0,'src'); "
         "from data_utils import extract_datasets; extract_datasets()"],
        cwd=_HERE,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("=" * 65)
    print("SPECIFIC TEST IV — MAE Foundation Model (Tasks IX.A + IX.B)")
    print("=" * 65)

    print("\n[SETUP] Extracting datasets from zip files ...")
    extract()

    print("\n[1/4] MAE Pre-training on no_sub images ...")
    run("pretrain.py")

    print("\n[2/4] Classification fine-tuning (Task IX.A) ...")
    run("finetune_cls.py")

    print("\n[3/4] Super-resolution fine-tuning (Task IX.B) ...")
    run("finetune_sr.py")

    print("\n[4/4] Evaluation ...")
    run("evaluate_cls.py")
    run("evaluate_sr.py")

    print("\n" + "=" * 65)
    print("ALL DONE. Outputs:")
    print("  weights/mae_pretrained.pth  — pre-trained MAE encoder")
    print("  weights/clf_best.pth        — classification model (IX.A)")
    print("  weights/sr_best.pth         — super-resolution model (IX.B)")
    print("  plots/clf_roc_standard.png")
    print("  plots/clf_confusion_standard.png")
    print("  plots/clf_training_curves.png")
    print("  plots/sr_comparison_grid.png")
    print("  plots/sr_training_curves.png")
