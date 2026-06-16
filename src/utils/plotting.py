from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(
    fold_id: int,
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: list[float] | None = None,
    val_metrics: list[float] | None = None,
    metric_name: str = "Accuracy",
    save_path: str | Path | None = None,
) -> plt.Figure:
    epochs = range(1, len(train_losses) + 1)
    best_epoch = int(np.argmin(val_losses)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Fold {fold_id:02d} — Training History", fontsize=13, fontweight="bold")

    sns.lineplot(ax=ax1, x=epochs, y=train_losses, label="Train", marker=".")
    sns.lineplot(ax=ax1, x=epochs, y=val_losses, label="Val", marker=".")
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax1.scatter(best_epoch, val_losses[best_epoch - 1], color="gray", zorder=5, s=60,
                marker="*", label=f"Best val @ epoch {best_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if train_metrics is not None and val_metrics is not None:
        sns.lineplot(ax=ax2, x=epochs, y=train_metrics, label="Train", marker=".")
        sns.lineplot(ax=ax2, x=epochs, y=val_metrics, label="Val", marker=".")
        ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax2.scatter(best_epoch, val_metrics[best_epoch - 1], color="gray", zorder=5,
                    s=60, marker="*", label=f"Best val @ epoch {best_epoch}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(metric_name)
        ax2.set_title(f"{metric_name} Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No metrics recorded", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11, color="gray")
        ax2.set_title("Metrics")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str | Path | None = None,
    normalize: bool = True,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100 if normalize else cm

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"
            else:
                annot[i, j] = f"{cm[i, j]}"

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm_perc if normalize else cm, annot=annot, fmt="",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", cbar=True, ax=ax, square=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_fold_figures(
    fold_id: int,
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: list[float] | None,
    val_metrics: list[float] | None,
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_dir: str | Path,
    metric_name: str = "Accuracy",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        fold_id=fold_id,
        train_losses=train_losses,
        val_losses=val_losses,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        metric_name=metric_name,
        save_path=output_dir / f"fold_{fold_id:02d}_training_curves.png",
    )

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=output_dir / f"fold_{fold_id:02d}_confusion_matrix.png",
    )

    plt.close("all")
