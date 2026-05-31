"""Evaluation utilities: classification report, confusion matrix, training curve."""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .config import CLASS_NAMES, RESULTS_DIR

logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test_cat, encoder, model_type: str = "model") -> dict:
    """
    Print and save:
    - Per-class precision / recall / F1 report
    - Confusion matrix PNG
    - Training curve PNG (if *history* passed separately)

    Returns a dict with ``report`` and ``confusion_matrix``.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    class_labels = [CLASS_NAMES.get(int(c), str(c)) for c in encoder.classes_]

    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(f"\n{'='*60}\n{model_type.upper()} — Classification Report\n{'='*60}")
    print(report)
    (RESULTS_DIR / f"{model_type}_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_labels, model_type)

    return {"report": report, "confusion_matrix": cm}


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], model_type: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_type.upper()} — Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = RESULTS_DIR / f"{model_type}_confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", out)


def plot_training_curve(history, model_type: str = "model") -> None:
    """Save loss and accuracy curves from a Keras History object."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle(f"{model_type.upper()} — Training curves")
    plt.tight_layout()
    out = RESULTS_DIR / f"{model_type}_training_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Training curve saved → %s", out)
