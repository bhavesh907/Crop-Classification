#!/usr/bin/env python3
"""
Train and evaluate a crop-classification model.

Usage
-----
    python scripts/train.py                        # default: bilstm
    python scripts/train.py --model cnn --epochs 30
    python scripts/train.py --model mlp --batch-size 512
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import evaluate, plot_training_curve
from src.preprocess import load_dataset
from src.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Train a crop-classification model")
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "lstm"],
        default="lstm",
        help="Model architecture (default: lstm = Bidirectional LSTM)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--test-size", type=float, default=0.15, help="Test-set fraction"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15, help="Validation-set fraction"
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    df = load_dataset()
    model, scaler, encoder, history, (X_test, y_test) = train(
        df,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        dropout_rate=args.dropout,
    )

    evaluate(model, X_test, y_test, encoder, model_type=args.model)
    plot_training_curve(history, model_type=args.model)


if __name__ == "__main__":
    main()
