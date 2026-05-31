#!/usr/bin/env python3
"""
Build the processed dataset from raw GeoTIFFs.

Usage
-----
    python scripts/preprocess.py
    python scripts/preprocess.py --samples 50000 --seed 0
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import build_dataset, save_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Build crop-classification dataset")
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Max pixels sampled per class (default: 100 000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = build_dataset(n_samples_per_class=args.samples, random_seed=args.seed)
    save_dataset(df)
    print(f"Done. Dataset shape: {df.shape}")


if __name__ == "__main__":
    main()
