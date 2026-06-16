from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _plot_cm_on_ax(
    ax, y_true, y_pred, class_names, title, cmap="Blues", vmin=0, vmax=100,
):
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    cm_perc = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"

    sns.heatmap(
        cm_perc, annot=annot, fmt="", vmin=vmin, vmax=vmax,
        xticklabels=class_names, yticklabels=class_names,
        cmap=cmap, cbar=False, ax=ax, square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")


def plot_dual_confusion_matrix(
    y_true_val: list[int],
    y_pred_val: list[int],
    y_true_test: list[int],
    y_pred_test: list[int],
    class_names: list[str],
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    all_perc = []
    for yt, yp in [(y_true_val, y_pred_val), (y_true_test, y_pred_test)]:
        cm = confusion_matrix(yt, yp)
        cm_p = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        all_perc.append(cm_p)

    vmin = 0
    vmax = max(p.max() for p in all_perc) if all_perc else 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    _plot_cm_on_ax(ax1, y_true_val, y_pred_val, class_names,
                   "Validation", vmin=vmin, vmax=vmax)
    _plot_cm_on_ax(ax2, y_true_test, y_pred_test, class_names,
                   "Test", vmin=vmin, vmax=vmax)

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="%")

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show(block=True)

    return fig


def save_fold_figures(
    fold_id: int,
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: list[float] | None,
    val_metrics: list[float] | None,
    y_true_val: list[int],
    y_pred_val: list[int],
    y_true_test: list[int],
    y_pred_test: list[int],
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

    plot_dual_confusion_matrix(
        y_true_val=y_true_val,
        y_pred_val=y_pred_val,
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        class_names=class_names,
        save_path=output_dir / f"fold_{fold_id:02d}_confusion_matrices.png",
        show=False,
    )

    plt.close("all")


def _generate_from_predictions(
    results_dir: Path,
    class_names: list[str],
    save: bool,
) -> list[plt.Figure]:
    pred_path = results_dir / "predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {results_dir}")

    df = pd.read_csv(pred_path)
    plots_dir = results_dir / "plots"
    n_classes = df[["true_label", "pred_label"]].max().max() + 1

    if class_names is None:
        class_names = [str(i) for i in range(int(n_classes))]

    if len(class_names) != n_classes:
        class_names = [str(i) for i in range(int(n_classes))]

    figs = []

    for fold_id in sorted(df["fold"].unique()):
        df_fold = df[df["fold"] == fold_id]

        val = df_fold[df_fold["split"] == "val"]
        test = df_fold[df_fold["split"] == "test"]

        if len(val) == 0 or len(test) == 0:
            missing = "val" if len(val) == 0 else "test"
            print(f"  Fold {fold_id:02d}: no {missing} predictions found, skipping")
            continue

        y_true_val = val["true_label"].tolist()
        y_pred_val = val["pred_label"].tolist()
        y_true_test = test["true_label"].tolist()
        y_pred_test = test["pred_label"].tolist()

        save_path = None
        show = not save

        if save:
            save_path = plots_dir / f"fold_{fold_id:02d}_confusion_matrices.png"

        fig = plot_dual_confusion_matrix(
            y_true_val=y_true_val,
            y_pred_val=y_pred_val,
            y_true_test=y_true_test,
            y_pred_test=y_pred_test,
            class_names=class_names,
            save_path=save_path,
            show=show,
        )
        figs.append(fig)

    return figs


def main():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices from benchmark results",
    )

    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to benchmark output directory (containing predictions.csv)",
    )
    parser.add_argument(
        "--save-img",
        action="store_true",
        help="Save plots to file instead of displaying on screen",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Class names (e.g. HC MDD)",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    mode = "saving" if args.save_img else "displaying"
    print(f"Reading predictions from: {results_dir}")
    print(f"Mode: {mode}")

    figs = _generate_from_predictions(
        results_dir=results_dir,
        class_names=args.class_names,
        save=args.save_img,
    )

    print(f"Generated {len(figs)} confusion matrix plots")

    if not args.save_img:
        print("Close all plot windows to exit.")


if __name__ == "__main__":
    main()
