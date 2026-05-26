from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_regression_training_curves(
    history: dict,
    save_path: str | Path,
    title: str | None = None,
) -> None:
    """
    Save one figure with train/validation curves for regression training.

    Expected keys in history:
        train_loss, val_loss
        train_nrmse, val_nrmse
        train_mae, val_mae
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_nrmse"], label="train")
    axes[1].plot(epochs, history["val_nrmse"], label="val")
    axes[1].axhline(1.0, linestyle="--", linewidth=1, label="baseline nRMSE=1")
    axes[1].set_title("nRMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("nRMSE")
    axes[1].legend()

    axes[2].plot(epochs, history["train_mae"], label="train")
    axes[2].plot(epochs, history["val_mae"], label="val")
    axes[2].set_title("MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE")
    axes[2].legend()

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)