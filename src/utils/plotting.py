from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_fold_training_history(
    history_csv: str | Path,
    output_path: str | Path,
) -> None:
    history_csv = Path(history_csv)
    output_path = Path(output_path)

    df = pd.read_csv(history_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(df["epoch"], df["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["val_epoch_loss"], marker="o")
    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)