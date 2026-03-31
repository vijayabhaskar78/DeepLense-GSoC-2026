"""End-to-end runner for Specific Test III — FNO Gravitational Lensing Classifier.

Usage
-----
    python run_test3.py

Runs training, then evaluation, then prints the full comparison table
against Common Test I (ResNet-18).  All outputs saved to weights/ and plots/.
"""

import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def run(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-u", script],
        cwd=_HERE,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("=" * 60)
    print("SPECIFIC TEST III — FNO Lensing Classifier")
    print("=" * 60)
    print("\n[1/2] Training FNO ...\n")
    run(os.path.join(_HERE, "src", "train.py"))

    print("\n[2/2] Evaluating + generating plots ...\n")
    run(os.path.join(_HERE, "src", "evaluate.py"))

    print("\nDone. Outputs:")
    print(f"  weights/best_fno.pth")
    print(f"  plots/roc_curves_standard.png")
    print(f"  plots/roc_curves_tta.png")
    print(f"  plots/confusion_matrix_standard.png")
    print(f"  plots/confusion_matrix_tta.png")
    print(f"  plots/confidence_analysis.png")
    print(f"  plots/training_curves.png")
