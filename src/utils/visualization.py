from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_fold_curves(
    history: dict[str, list[float]],
    save_path: str | Path,
    title: str,
) -> None:
    """
    Save one plot per fold with loss curves and validation accuracy.

    Expected history keys:
        train_loss
        val_loss
        val_acc
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax_loss = plt.subplots(figsize=(8, 5))

    ax_loss.plot(epochs, history["train_loss"], label="train loss")
    ax_loss.plot(epochs, history["val_loss"], label="val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    ax_acc = ax_loss.twinx()
    ax_acc.plot(epochs, history["val_acc"], linestyle="--", label="val acc")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_ylim(0.0, 1.0)

    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()

    ax_loss.legend(
        lines_loss + lines_acc,
        labels_loss + labels_acc,
        loc="best",
    )

    ax_loss.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Global confusion matrix",
    normalize: bool = False,
) -> None:
    """
    Save a global confusion matrix.

    Args:
        y_true: ground-truth labels.
        y_pred: predicted labels.
        class_names: names displayed on axes.
        save_path: output image path.
        title: plot title.
        normalize: if True, normalize rows by support.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)

    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1.0

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(
            cm,
            row_sum,
            out=np.zeros_like(cm),
            where=row_sum != 0,
        )
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot)

    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = cm_plot.max() / 2.0 if cm_plot.size and cm_plot.max() > 0 else 0.0

    for i in range(n_classes):
        for j in range(n_classes):
            if normalize:
                text = f"{cm_plot[i, j]:.2f}\n({int(cm[i, j])})"
            else:
                text = str(int(cm[i, j]))

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)