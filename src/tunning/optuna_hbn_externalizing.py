from __future__ import annotations 

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn

from src.datasets.hbn_db import HBNRestingStateDataset, create_k_folders
from src.models.eegnet import EEGNet
from src.training.train_hbn import set_seed, train_one_epoch, evaluate
from src.utils.regression_plots import plot_regression_training_curves


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def resolve_save_dir(save_dir: str) -> Path:
    save_dir = Path(save_dir)
    if save_dir.is_absolute():
        return save_dir
    return OUTPUT_ROOT / save_dir


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def make_criterion(loss: str, huber_delta: float):
    if loss == "mse":
        return nn.MSELoss()
    if loss == "huber":
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError(f"Unknown loss: {loss}")


def build_model(params: dict, n_channels: int, device):
    return EEGNet(
        n_channels=n_channels,
        n_classes=1,
        F1=int(params["F1"]),
        D=int(params["D"]),
        F2=int(params["F2"]),
        temporal_kern=int(params["temporal_kern"]),
        separable_kern=int(params["separable_kern"]),
        pool1=int(params["pool1"]),
        pool2=int(params["pool2"]),
        dropout=float(params["dropout"]),
        meanmax_alpha=float(params["meanmax_alpha"]),
        pp_as="list",
        aggregate=True,
        norm=str(params["norm"]),
    ).to(device)


def create_folds(args, dataset, batch_size: int):
    return create_k_folders(
        dataset=dataset,
        k_folder=args.k,
        batch_size=batch_size,
        shuffle=True,
        split_seed=args.split_seed,
        inner_split=args.inner_splits,
        n_bins=args.n_bins,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_selected_folds(args) -> list[int]:
    selected = list(range(1, args.k + 1)) if args.folds is None else args.folds
    invalid = [f for f in selected if f < 1 or f > args.k]
    if invalid:
        raise ValueError(f"Invalid folds {invalid}. Valid range is 1..{args.k}.")
    return selected


def get_objective_folds(args, selected_folds: list[int]) -> list[int]:
    if args.objective_folds is None:
        return selected_folds
    invalid = [f for f in args.objective_folds if f not in selected_folds]
    if invalid:
        raise ValueError(
            f"objective_folds {invalid} are not contained in folds={selected_folds}."
        )
    return args.objective_folds


def sample_params(trial: optuna.Trial, args) -> dict:
    loss = trial.suggest_categorical("loss", args.loss_choices)
    huber_delta = (
        trial.suggest_float("huber_delta", args.huber_delta_min, args.huber_delta_max)
        if loss == "huber"
        else 1.0
    )
    standardize = (
        trial.suggest_categorical("standardize", [True, False])
        if args.allow_no_standardize
        else True
    )

    return {
        "F1": trial.suggest_categorical("F1", args.F1_choices),
        "D": trial.suggest_categorical("D", args.D_choices),
        "F2": trial.suggest_categorical("F2", args.F2_choices),
        "temporal_kern": trial.suggest_categorical("temporal_kern", args.temporal_kern_choices),
        "separable_kern": trial.suggest_categorical("separable_kern", args.separable_kern_choices),
        "pool1": trial.suggest_categorical("pool1", args.pool1_choices),
        "pool2": trial.suggest_categorical("pool2", args.pool2_choices),
        "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
        "meanmax_alpha": trial.suggest_categorical("meanmax_alpha", args.meanmax_alpha_choices),
        "norm": trial.suggest_categorical("norm", args.norm_choices),
        "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", args.weight_decay_choices),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_size_choices),
        "grad_clip": trial.suggest_categorical("grad_clip", args.grad_clip_choices),
        "loss": loss,
        "huber_delta": huber_delta,
        "standardize": standardize,
    }


def train_until_stop(
    params: dict,
    train_loader,
    val_loader,
    n_channels: int,
    device,
    epochs: int,
    patience: int,
    seed: int,
    print_every: int,
    keep_best_state: bool,
):
    set_seed(seed)

    model = build_model(params=params, n_channels=n_channels, device=device)
    criterion = make_criterion(
        loss=params["loss"],
        huber_delta=float(params["huber_delta"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    history = defaultdict(list)

    best_val_nrmse = float("inf")
    best_val_loss = float("inf")
    best_val_metrics = None
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=float(params["grad_clip"]),
            standardize=bool(params["standardize"]),
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            standardize=bool(params["standardize"]),
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_nrmse"].append(train_metrics["nrmse"])
        history["val_nrmse"].append(val_metrics["nrmse"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])

        if print_every > 0 and (epoch == 1 or epoch % print_every == 0):
            print(
                f"      epoch={epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
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
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch
            patience_counter = 0
            if keep_best_state:
                best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if keep_best_state and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "history": history,
    }


def make_objective(
    args,
    dataset,
    n_channels: int,
    device,
    objective_folds: list[int],
    save_dir: Path,
):
    trial_fold_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, args)
        folds = create_folds(
            args=args,
            dataset=dataset,
            batch_size=int(params["batch_size"]),
        )

        print("\n" + "=" * 80)
        print(f"trial={trial.number}")
        print(json.dumps(params, indent=2, sort_keys=True))

        fold_scores = []
        fold_details = []

        for fold_id, (train_loader, val_loader, _) in enumerate(folds, start=1):
            if fold_id not in objective_folds:
                continue

            print(f"\n    trial={trial.number} | fold={fold_id:02d}")

            result = train_until_stop(
                params=params,
                train_loader=train_loader,
                val_loader=val_loader,
                n_channels=n_channels,
                device=device,
                epochs=args.epochs_search,
                patience=args.patience_search,
                seed=args.search_init_seed + fold_id,
                print_every=args.print_every,
                keep_best_state=False,
            )

            val_metrics = result["best_val_metrics"]
            fold_score = float(val_metrics["nrmse"])
            fold_scores.append(fold_score)

            detail = {
                "trial": trial.number,
                "fold": fold_id,
                "best_epoch": result["best_epoch"],
                "best_val_loss": val_metrics["loss"],
                "best_val_rmse": val_metrics["rmse"],
                "best_val_nrmse": val_metrics["nrmse"],
                "best_val_mae": val_metrics["mae"],
                "best_val_r2": val_metrics["r2"],
            }

            fold_details.append(detail)
            trial_fold_rows.append({**detail, **params})

            running_score = float(np.mean(fold_scores))

            print(
                f"    trial={trial.number} | fold={fold_id:02d} | "
                f"best_epoch={result['best_epoch']} | "
                f"best_val_nrmse={fold_score:.4f} | "
                f"running_mean_val_nrmse={running_score:.4f}"
            )

            trial.report(running_score, step=fold_id)

            pd.DataFrame(trial_fold_rows).to_csv(
                save_dir / "trial_fold_metrics_partial.csv",
                index=False,
            )

            if trial.should_prune():
                del result["model"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            del result["model"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

        trial.set_user_attr("fold_details", fold_details)
        trial.set_user_attr("mean_best_val_nrmse", mean_score)
        trial.set_user_attr("std_best_val_nrmse", std_score)

        print(f"\ntrial={trial.number} finished | mean_best_val_nrmse={mean_score:.4f}")

        return mean_score

    return objective


def finalize_best_params(best_params: dict) -> dict:
    out = best_params.copy()
    if out.get("loss") == "mse":
        out["huber_delta"] = 1.0
    if "standardize" not in out:
        out["standardize"] = True
    return out


def final_training_and_test(
    args,
    dataset,
    n_channels: int,
    device,
    selected_folds: list[int],
    best_params: dict,
    save_dir: Path,
):
    folds = create_folds(
        args=args,
        dataset=dataset,
        batch_size=int(best_params["batch_size"]),
    )

    metric_rows = []
    prediction_rows = []

    for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
        if fold_id not in selected_folds:
            continue

        print("\n" + "-" * 80)
        print(f"Final training | fold={fold_id:02d}")

        result = train_until_stop(
            params=best_params,
            train_loader=train_loader,
            val_loader=val_loader,
            n_channels=n_channels,
            device=device,
            epochs=args.epochs_final,
            patience=args.patience_final,
            seed=args.final_init_seed + fold_id,
            print_every=args.print_every,
            keep_best_state=True,
        )

        model = result["model"]
        criterion = make_criterion(
            loss=best_params["loss"],
            huber_delta=float(best_params["huber_delta"]),
        )

        test_metrics, names, y_true, y_pred = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            standardize=bool(best_params["standardize"]),
            return_predictions=True,
        )

        checkpoint_path = (
            save_dir
            / "checkpoints"
            / f"condition-{args.condition}_split-{args.split_seed}_init-{args.final_init_seed}_fold-{fold_id:02d}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": result["best_state"],
                "condition": args.condition,
                "target": args.target,
                "split_seed": args.split_seed,
                "init_seed": args.final_init_seed,
                "fold": fold_id,
                "best_epoch": result["best_epoch"],
                "best_val_metrics": result["best_val_metrics"],
                "test_metrics": test_metrics,
                "n_channels": n_channels,
                "n_outputs": 1,
                "pp_as": "list",
                "aggregate": True,
                "best_params": best_params,
                "training_args": vars(args),
            },
            checkpoint_path,
        )

        plot_path = (
            save_dir
            / "plots"
            / "fold_curves"
            / f"condition-{args.condition}_split-{args.split_seed}_init-{args.final_init_seed}_fold-{fold_id:02d}.png"
        )

        plot_regression_training_curves(
            history=result["history"],
            save_path=plot_path,
            title=(
                f"{save_dir.name} | condition={args.condition} | "
                f"split={args.split_seed} | init={args.final_init_seed} | fold={fold_id:02d}"
            ),
        )

        val_metrics = result["best_val_metrics"]

        metric_row = {
            "experiment": save_dir.name,
            "condition": args.condition,
            "target": args.target,
            "split_seed": args.split_seed,
            "init_seed": args.final_init_seed,
            "fold": fold_id,
            "best_epoch": result["best_epoch"],
            "best_val_loss": val_metrics["loss"],
            "best_val_rmse": val_metrics["rmse"],
            "best_val_nrmse": val_metrics["nrmse"],
            "best_val_mae": val_metrics["mae"],
            "best_val_r2": val_metrics["r2"],
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

        for key, value in best_params.items():
            metric_row[f"param_{key}"] = value

        metric_rows.append(metric_row)

        for subject_id, yt, yp in zip(names, y_true, y_pred):
            prediction_rows.append(
                {
                    "experiment": save_dir.name,
                    "condition": args.condition,
                    "target": args.target,
                    "split_seed": args.split_seed,
                    "init_seed": args.final_init_seed,
                    "fold": fold_id,
                    "subject_id": subject_id,
                    "y_true": yt,
                    "y_pred": yp,
                    "error": yt - yp,
                    "abs_error": abs(yt - yp),
                }
            )

        print(
            f"    test_rmse={test_metrics['rmse']:.4f} | "
            f"test_nrmse={test_metrics['nrmse']:.4f} | "
            f"test_r2={test_metrics['r2']:.4f} | "
            f"test_mae={test_metrics['mae']:.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)

    fold_metrics_path = save_dir / "fold_metrics.csv"
    predictions_path = save_dir / "predictions.csv"

    fold_metrics.to_csv(fold_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    overall = (
        fold_metrics
        .groupby(["experiment", "condition", "target"])
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

    print("\nOverall final:")
    print(overall.to_string(index=False))

    print("\nSaved:")
    print(f"  {save_dir / 'best_params.json'}")
    print(f"  {save_dir / 'optuna_trials.csv'}")
    print(f"  {fold_metrics_path}")
    print(f"  {overall_path}")
    print(f"  {predictions_path}")


def inspect_dataset(dataset):
    y = np.asarray([float(sample["y"]) for sample in dataset.samples])

    print("\nDataset:")
    print(f"condition={dataset.condition}")
    print(f"target={dataset.target}")
    print(f"subjects={len(dataset)}")
    print(
        f"y mean={y.mean():.4f} | "
        f"std={y.std():.4f} | "
        f"min={y.min():.4f} | "
        f"max={y.max():.4f}"
    )

    first_name, first_eeg, first_y = dataset[0]

    print(
        f"First sample: {first_name} | "
        f"X={tuple(first_eeg.shape)} | "
        f"y={float(first_y):.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="data/raw/hbn_db/R1_L100_bdf")
    parser.add_argument("--condition", type=str, required=True, choices=["EO", "EC"])
    parser.add_argument("--target", type=str, default="externalizing")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--objective-folds", type=int, nargs="+", default=None)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=3407)

    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="data/processed/hbn_db")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs-search", type=int, default=40)
    parser.add_argument("--patience-search", type=int, default=8)
    parser.add_argument("--epochs-final", type=int, default=80)
    parser.add_argument("--patience-final", type=int, default=15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-init-seed", type=int, default=3001)
    parser.add_argument("--final-init-seed", type=int, default=3001)

    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pruner", action="store_true")

    parser.add_argument("--batch-size-choices", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--loss-choices",
        type=str,
        nargs="+",
        default=["mse", "huber"],
        choices=["mse", "huber"],
    )
    parser.add_argument("--huber-delta-min", type=float, default=0.2)
    parser.add_argument("--huber-delta-max", type=float, default=1.5)

    parser.add_argument("--F1-choices", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--D-choices", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--F2-choices", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    parser.add_argument(
        "--temporal-kern-choices",
        type=int,
        nargs="+",
        default=[31, 63, 95, 127],
    )
    parser.add_argument(
        "--separable-kern-choices",
        type=int,
        nargs="+",
        default=[7, 15, 31, 63],
    )
    parser.add_argument("--pool1-choices", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--pool2-choices", type=int, nargs="+", default=[4, 8, 16])

    parser.add_argument("--dropout-min", type=float, default=0.2)
    parser.add_argument("--dropout-max", type=float, default=0.7)
    parser.add_argument(
        "--meanmax-alpha-choices",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5],
    )
    parser.add_argument(
        "--norm-choices",
        type=str,
        nargs="+",
        default=["group"],
        choices=["auto", "batch", "group"],
    )

    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=3e-3)
    parser.add_argument(
        "--weight-decay-choices",
        type=float,
        nargs="+",
        default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    )
    parser.add_argument(
        "--grad-clip-choices",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0],
    )

    parser.add_argument("--allow-no-standardize", action="store_true")
    parser.add_argument("--print-every", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.study_name is None:
        args.study_name = f"hbn_{args.condition.lower()}_{args.target}_eegnet_optuna"

    if args.save_dir is None:
        args.save_dir = f"hbn_db/optuna/{args.study_name}"

    save_dir = resolve_save_dir(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), save_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Study: {args.study_name}")
    print(f"Save dir: {save_dir}")
    print(f"condition: {args.condition}")
    print(f"target: {args.target}")
    print(f"n_trials: {args.n_trials}")

    dataset = HBNRestingStateDataset(
        root=args.root,
        condition=args.condition,
        target=args.target,
        preload=args.preload,
        cache=args.cache,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )

    inspect_dataset(dataset)

    _, first_eeg, _ = dataset[0]
    n_channels = int(first_eeg.shape[0])

    selected_folds = get_selected_folds(args)
    objective_folds = get_objective_folds(args, selected_folds)

    print(f"\nselected_folds: {selected_folds}")
    print(f"objective_folds: {objective_folds}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    if args.no_pruner:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, args.n_trials // 10),
            n_warmup_steps=1,
        )

    study_db = save_dir / "study.sqlite3"
    storage = f"sqlite:///{study_db.as_posix()}"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=args.resume,
    )

    objective = make_objective(
        args=args,
        dataset=dataset,
        n_channels=n_channels,
        device=device,
        objective_folds=objective_folds,
        save_dir=save_dir,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    trials_path = save_dir / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    best_params = finalize_best_params(study.best_params)

    save_json(best_params, save_dir / "best_params.json")

    best_summary = {
        "best_trial": study.best_trial.number,
        "best_value_mean_val_nrmse": study.best_value,
        "best_params": best_params,
        "study_db": str(study_db),
        "trials_csv": str(trials_path),
    }

    save_json(best_summary, save_dir / "best_summary.json")

    print("\nBest trial:")
    print(json.dumps(best_summary, indent=2, sort_keys=True))

    final_training_and_test(
        args=args,
        dataset=dataset,
        n_channels=n_channels,
        device=device,
        selected_folds=selected_folds,
        best_params=best_params,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
