from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_true - y_pred

    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    y_std = float(np.std(y_true, ddof=0))
    nrmse = float(rmse / y_std) if y_std > 0 else float("nan")

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "r2": r2,
        "corr": corr,
        "y_std": y_std,
    }


def apply_filters(
    df: pd.DataFrame,
    condition: str | None,
    split_seed: int | None,
    init_seed: int | None,
    fold: int | None,
) -> pd.DataFrame:
    out = df.copy()

    if condition is not None and "condition" in out.columns:
        out = out[out["condition"] == condition]

    if split_seed is not None and "split_seed" in out.columns:
        out = out[out["split_seed"] == split_seed]

    if init_seed is not None and "init_seed" in out.columns:
        out = out[out["init_seed"] == init_seed]

    if fold is not None and "fold" in out.columns:
        out = out[out["fold"] == fold]

    return out.reset_index(drop=True)


def default_save_path(predictions_path: Path, condition: str | None, fold: int | None) -> Path:
    stem = "scatter_ytrue_vs_ypred"

    if condition is not None:
        stem += f"_{condition}"

    if fold is not None:
        stem += f"_fold-{fold:02d}"

    return predictions_path.parent / "plots" / "scatter" / f"{stem}.png"


def plot_scatter(
    df: pd.DataFrame,
    save_path: Path,
    title: str | None = None,
) -> None:
    y_true = df["y_true"].to_numpy(dtype=np.float64)
    y_pred = df["y_pred"].to_numpy(dtype=np.float64)

    metrics = regression_metrics(y_true, y_pred)

    lim_min = float(min(y_true.min(), y_pred.min()))
    lim_max = float(max(y_true.max(), y_pred.max()))

    margin = 0.05 * (lim_max - lim_min + 1e-8)
    lim_min -= margin
    lim_max += margin

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.75)

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        linewidth=1,
        label=r"$\hat{y}=y$",
    )

    ax.axhline(
        y_true.mean(),
        linestyle=":",
        linewidth=1,
        label=r"$\hat{y}=\bar{y}$",
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_xlabel(r"True $y$")
    ax.set_ylabel(r"Predicted $\hat{y}$")

    metric_text = (
        f"N = {len(df)}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"nRMSE = {metrics['nrmse']:.4f}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"R² = {metrics['r2']:.4f}\n"
        f"r = {metrics['corr']:.4f}"
    )

    ax.text(
        0.05,
        0.95,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="lower right")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved scatter:")
    print(f"  {save_path}")

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.csv",
    )

    parser.add_argument("--condition", type=str, default=None, choices=["EO", "EC"])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional output PNG path.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    predictions_path = Path(args.predictions)

    df = pd.read_csv(predictions_path)

    required_cols = {"y_true", "y_pred"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {predictions_path}: {sorted(missing)}"
        )

    df = apply_filters(
        df=df,
        condition=args.condition,
        split_seed=args.split_seed,
        init_seed=args.init_seed,
        fold=args.fold,
    )

    if len(df) == 0:
        raise ValueError("No rows left after applying filters.")

    if args.save_path is None:
        save_path = default_save_path(
            predictions_path=predictions_path,
            condition=args.condition,
            fold=args.fold,
        )
    else:
        save_path = Path(args.save_path)

    if args.title is None:
        pieces = []

        if args.condition is not None:
            pieces.append(f"condition={args.condition}")

        if args.split_seed is not None:
            pieces.append(f"split={args.split_seed}")

        if args.init_seed is not None:
            pieces.append(f"init={args.init_seed}")

        if args.fold is not None:
            pieces.append(f"fold={args.fold:02d}")

        title = " | ".join(pieces) if pieces else "True vs predicted"
    else:
        title = args.title

    print("\nLoaded predictions:")
    print(f"  path: {predictions_path}")
    print(f"  rows: {len(df)}")

    plot_scatter(
        df=df,
        save_path=save_path,
        title=title,
    )


if __name__ == "__main__":
    main()