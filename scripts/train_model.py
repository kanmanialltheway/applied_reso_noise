"""Build the bundled classifier: generate synthetic clips, extract features, train and save.

Usage:  python scripts/train_model.py [--train 64] [--val 16] [--seed 7]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonoise import config  # noqa: E402
from resonoise.classifier import train_classifier  # noqa: E402
from resonoise.synthetic import build_feature_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=64, help="synthetic training clips per class")
    ap.add_argument("--val", type=int, default=16, help="held-out validation clips per class")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()

    def progress(done, total):
        if done % 20 == 0 or done == total:
            print(f"  features {done}/{total}", flush=True)

    X, y, split = build_feature_dataset(args.train, args.val, args.seed, progress)
    np.savez_compressed(config.SYNTH_DATASET_PATH, X=X, y=y, split=split)
    clf = train_classifier(X, y, split)
    clf.save(config.DEFAULT_MODEL_PATH)
    print(f"Saved {config.DEFAULT_MODEL_PATH.name}  validation accuracy "
          f"{clf.meta['validation_accuracy']:.3f} on {clf.meta['validation_n']} held-out clips "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
