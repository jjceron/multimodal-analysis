from __future__ import annotations

import copy
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.datasets.hbn_db import HBNRestingStateDataset, create_k_folders
from src.models.eegnet import EEGNet
from src.utils.regression_plots import plot_regression_training_curves


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_SPLIT_SEEDS = [3407]
DEFAULT_INIT_SEEDS = [3001]


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_save_dir(save_dir: str) -> Path:
    save_dir = Path(save_dir)

    if save_dir.is_absolute():
        return save_dir

    return OUTPUT_ROOT / save_dir


def save_config(args, save_dir: Path) -> None:
    config = vars(args).copy()
    config["save_dir"] = str(save_dir)
    config["project_root"] = str(PROJECT_ROOT)

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, sort_keys=True)


def to_device(X, y, device):
    y = y.to(device)

    if isinstance(X, torch.Tensor):
        X = X.to(device)
    else:
        X = [x.to(device) for x in X]

    return X, y


def standardize_eeg(X, eps: float = 1e-6):
    """
    Subject-wise, channel-wise standardization.

    List mode:
        X: list[Tensor[C, T_i]]

    This does not use train/val/test population statistics.
    """
    if isinstance(X, torch.Tensor):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (X - mean) / std

    return [standardize_eeg(x, eps=eps) for x in X]


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

    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "r2": r2,
        "y_std": y_std,
    }


def get_criterion(args):
    if args.loss == "mse":
        return nn.MSELoss()

    if args.loss == "huber":
        return nn.HuberLoss(delta=args.huber_delta)

    raise ValueError(f"Unknown loss: {args.loss}")


def model_predict(model, X):
    pred, pred_time = model(X)

    if pred.ndim == 2 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    return pred, pred_time


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip: float | None,
    standardize: bool,
):
    model.train()

    total_loss = 0.0
    total_n = 0

    y_true_all = []
    y_pred_all = []

    for _, X, y in loader:
        X, y = to_device(X, y, device)

        if standardize:
            X = standardize_eeg(X)

        optimizer.zero_grad(set_to_none=True)

        pred, _ = model_predict(model, X)

        loss = criterion(pred, y.float())

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_n = y.shape[0]

        total_loss += loss.item() * batch_n
        total_n += batch_n

        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    metrics = regression_metrics(y_true_all, y_pred_all)
    metrics["loss"] = total_loss / max(total_n, 1)

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    standardize: bool,
    return_predictions: bool = False,
):
    model.eval()

    total_loss = 0.0
    total_n = 0

    names_all = []
    y_true_all = []
    y_pred_all = []

    for names, X, y in loader:
        X, y = to_device(X, y, device)

        if standardize:
            X = standardize_eeg(X)

        pred, _ = model_predict(model, X)

        loss = criterion(pred, y.float())

        batch_n = y.shape[0]

        total_loss += loss.item() * batch_n
        total_n += batch_n

        names_all.extend(list(names))
        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_pred_all.extend(pred.detach().cpu().numpy().tolist())

    metrics = regression_metrics(y_true_all, y_pred_all)
    metrics["loss"] = total_loss / max(total_n, 1)

    if return_predictions:
        return metrics, names_all, y_true_all, y_pred_all

    return metrics


def train_one_fold(
    args,
    train_loader,
    val_loader,
    test_loader,
    split_seed: int,
    init_seed: int,
    fold_id: int,
    n_channels: int,
    device,
    save_dir: Path,
):
    set_seed(init_seed)

    model = EEGNet(
        n_channels=n_channels,
        n_classes=1,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        temporal_kern=args.temporal_kern,
        separable_kern=args.separable_kern,
        pool1=args.pool1,
        pool2=args.pool2,
        dropout=args.dropout,
        meanmax_alpha=args.meanmax_alpha,
        pp_as="list",
        aggregate=True,
        norm=args.norm,
    ).to(device)

    criterion = get_criterion(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = defaultdict(list)

    best_val_nrmse = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    standardize = not args.no_standardize

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            standardize=standardize,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            standardize=standardize,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_nrmse"].append(train_metrics["nrmse"])
        history["val_nrmse"].append(val_metrics["nrmse"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"    epoch={epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_nrmse={val_metrics['nrmse']:.4f} | "
                f"val_r2={val_metrics['r2']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f}"
            )

        improved = (
            val_metrics["nrmse"] < best_val_nrmse
            or (
                val_metrics["nrmse"] == best_val_nrmse
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_nrmse = val_metrics["nrmse"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    plot_path = (
        save_dir
        / "plots"
        / "fold_curves"
        / f"condition-{args.condition}_split-{split_seed}_init-{init_seed}_fold-{fold_id:02d}.png"
    )

    plot_regression_training_curves(
        history=history,
        save_path=plot_path,
        title=(
            f"{save_dir.name} | "
            f"condition={args.condition} | "
            f"split={split_seed} | "
            f"init={init_seed} | "
            f"fold={fold_id:02d}"
        ),
    )

    checkpoint_path = (
        save_dir
        / "checkpoints"
        / f"condition-{args.condition}_split-{split_seed}_init-{init_seed}_fold-{fold_id:02d}.pt"
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "condition": args.condition,
            "target": args.target,
            "split_seed": split_seed,
            "init_seed": init_seed,
            "fold": fold_id,
            "best_epoch": best_epoch,
            "best_val_nrmse": best_val_nrmse,
            "best_val_loss": best_val_loss,
            "n_channels": n_channels,
            "n_outputs": 1,
            "pp_as": "list",
            "aggregate": True,
            "standardize": standardize,
            "model_params": {
                "F1": args.F1,
                "D": args.D,
                "F2": args.F2,
                "temporal_kern": args.temporal_kern,
                "separable_kern": args.separable_kern,
                "pool1": args.pool1,
                "pool2": args.pool2,
                "dropout": args.dropout,
                "meanmax_alpha": args.meanmax_alpha,
                "norm": args.norm,
            },
            "training_args": vars(args),
        },
        checkpoint_path,
    )

    test_metrics, names, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        standardize=standardize,
        return_predictions=True,
    )

    metric_row = {
        "experiment": save_dir.name,
        "condition": args.condition,
        "target": args.target,
        "loss": args.loss,
        "meanmax_alpha": args.meanmax_alpha,
        "standardize": standardize,
        "split_seed": split_seed,
        "init_seed": init_seed,
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_nrmse": best_val_nrmse,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_nrmse": test_metrics["nrmse"],
        "test_y_std": test_metrics["y_std"],
        "test_r2": test_metrics["r2"],
        "checkpoint_path": str(checkpoint_path),
        "plot_path": str(plot_path),
    }

    prediction_rows = []

    for subject_id, yt, yp in zip(names, y_true, y_pred):
        prediction_rows.append(
            {
                "experiment": save_dir.name,
                "condition": args.condition,
                "target": args.target,
                "split_seed": split_seed,
                "init_seed": init_seed,
                "fold": fold_id,
                "subject_id": subject_id,
                "y_true": yt,
                "y_pred": yp,
                "error": yt - yp,
                "abs_error": abs(yt - yp),
            }
        )

    return metric_row, prediction_rows


def inspect_loaded_dataset(dataset):
    print("\nDataset:")
    print(f"condition={dataset.condition}")
    print(f"target={dataset.target}")
    print(f"subjects={len(dataset)}")
    print("pp_as=list")

    y = np.asarray([float(sample["y"]) for sample in dataset.samples])
    print(
        f"y mean={y.mean():.4f} | "
        f"std={y.std():.4f} | "
        f"min={y.min():.4f} | "
        f"max={y.max():.4f}"
    )

def print_overall_summary(overall_metrics: pd.DataFrame) -> None:
    print("\nOverall summary:")

    if len(overall_metrics) == 1:
        print(overall_metrics.iloc[0].to_string())
    else:
        for _, row in overall_metrics.iterrows():
            print("\n" + "-" * 60)
            print(row.to_string())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/hbn_db/R1_L100_bdf",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["EO", "EC"],
        required=True,
    )
    parser.add_argument("--target", type=str, default="externalizing")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=5)

    parser.add_argument("--split-seeds", type=int, nargs="+", default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--init-seeds", type=int, nargs="+", default=DEFAULT_INIT_SEEDS)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=parse_optional_float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.0)

    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--temporal-kern", type=int, default=63)
    parser.add_argument("--separable-kern", type=int, default=15)
    parser.add_argument("--pool1", type=int, default=8)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--meanmax-alpha", type=float, default=0.0)
    parser.add_argument("--norm", type=str, default="auto", choices=["auto", "batch", "group"])

    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/processed/hbn_db",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-every", type=int, default=5)

    parser.add_argument("--save-dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_dir is None:
        args.save_dir = f"hbn_{args.condition.lower()}_{args.target}_eegnet_regression"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = resolve_save_dir(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, save_dir)

    print(f"\nDevice: {device}")
    print(f"Experiment: {save_dir.name}")
    print(f"Save dir: {save_dir}")
    print(f"condition: {args.condition}")
    print(f"target: {args.target}")
    print(f"pp_as: list")
    print(f"aggregate: True")
    print(f"meanmax_alpha: {args.meanmax_alpha}")
    print(f"standardize: {not args.no_standardize}")
    print(f"loss: {args.loss}")
    print(f"cache: {args.cache}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"refresh_cache: {args.refresh_cache}")

    dataset = HBNRestingStateDataset(
        root=args.root,
        condition=args.condition,
        target=args.target,
        preload=args.preload,
        cache=args.cache,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )

    inspect_loaded_dataset(dataset)

    first_name, first_eeg, first_y = dataset[0]
    n_channels = first_eeg.shape[0]

    print(
        f"\nFirst sample: {first_name} | "
        f"X={tuple(first_eeg.shape)} | "
        f"y={float(first_y):.4f}"
    )

    print(f"\nModel channels: {n_channels}")

    if args.validate_only:
        return

    selected_folds = args.folds

    if selected_folds is None:
        selected_folds = list(range(1, args.k + 1))

    invalid_folds = [f for f in selected_folds if f < 1 or f > args.k]

    if invalid_folds:
        raise ValueError(
            f"Invalid folds {invalid_folds}. Valid range is 1..{args.k}."
        )

    all_metric_rows = []
    all_prediction_rows = []

    for split_seed in args.split_seeds:
        folds = create_k_folders(
            dataset=dataset,
            k_folder=args.k,
            batch_size=args.batch_size,
            shuffle=True,
            split_seed=split_seed,
            inner_split=args.inner_splits,
            n_bins=args.n_bins,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"\nsplit_seed={split_seed} | folds={selected_folds}")

        for init_seed in args.init_seeds:
            for fold_id, (train_loader, val_loader, test_loader) in enumerate(
                folds,
                start=1,
            ):
                if fold_id not in selected_folds:
                    continue

                print(
                    f"\n  fold={fold_id:02d} | "
                    f"init_seed={init_seed} | "
                    f"condition={args.condition}"
                )

                metric_row, prediction_rows = train_one_fold(
                    args=args,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    split_seed=split_seed,
                    init_seed=init_seed,
                    fold_id=fold_id,
                    n_channels=n_channels,
                    device=device,
                    save_dir=save_dir,
                )

                all_metric_rows.append(metric_row)
                all_prediction_rows.extend(prediction_rows)

                print(
                    f"    test_rmse={metric_row['test_rmse']:.4f} | "
                    f"test_nrmse={metric_row['test_nrmse']:.4f} | "
                    f"test_mae={metric_row['test_mae']:.4f} | "
                    f"test_r2={metric_row['test_r2']:.4f}"
                )

    fold_metrics = pd.DataFrame(all_metric_rows)
    predictions = pd.DataFrame(all_prediction_rows)

    fold_metrics_path = save_dir / "fold_metrics.csv"
    predictions_path = save_dir / "predictions.csv"

    fold_metrics.to_csv(fold_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    overall = (
        fold_metrics
        .groupby(["experiment", "condition", "target", "loss", "meanmax_alpha", "standardize"])
        .agg(
            test_nrmse_mean=("test_nrmse", "mean"),
            test_nrmse_std=("test_nrmse", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            best_val_nrmse_mean=("best_val_nrmse", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .reset_index()
    )

    overall_path = save_dir / "overall_metrics.csv"
    overall.to_csv(overall_path, index=False)

    print_overall_summary(overall)

    print("\nSaved:")
    print(f"  {fold_metrics_path}")
    print(f"  {overall_path}")
    print(f"  {predictions_path}")


if __name__ == "__main__":
    main()