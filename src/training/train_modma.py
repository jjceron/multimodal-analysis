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

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.datasets.modma_db import MODMADataset, create_dataloaders
from src.models.eegnet import EEGNet
from src.utils.visualization import plot_fold_curves, plot_confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_SPLIT_SEEDS = [3407]
DEFAULT_INIT_SEEDS = [3001]
RANDOM_SEED_COUNT = 3
RANDOM_SEED_MAX = 10000


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def parse_bool(value: str) -> bool:
    value = value.lower()

    if value in {"true", "1", "yes"}:
        return True

    if value in {"false", "0", "no"}:
        return False

    raise argparse.ArgumentTypeError("Expected true/false.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_random_seeds(n: int, seed_max: int) -> list[int]:
    return random.sample(range(seed_max), k=n)


def resolve_save_dir(save_dir: str) -> Path:
    save_dir = Path(save_dir)

    if save_dir.is_absolute():
        return save_dir

    return OUTPUT_ROOT / save_dir


def save_config(args, save_dir: Path) -> None:
    config = vars(args).copy()
    config["save_dir"] = str(save_dir)
    config["project_root"] = str(PROJECT_ROOT)

    config_path = save_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, sort_keys=True)


def to_device(X, y, device):
    y = y.to(device)

    if isinstance(X, torch.Tensor):
        X = X.to(device)
    else:
        X = [x.to(device) for x in X]

    return X, y


def standardize_eeg(X, eps: float = 1e-6):
    """Subject-wise, channel-wise standardization.

    Tensor mode:
        X: Tensor[B, C, T]

    List mode:
        X: list[Tensor[C, T_i]]
    """
    if isinstance(X, torch.Tensor):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (X - mean) / std

    return [standardize_eeg(x, eps=eps) for x in X]


def get_class_weights(labels: list[int], n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    weights = counts.sum() / np.clip(counts, a_min=1, a_max=None)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def temporal_cross_entropy(logits, y, criterion):
    """Cross entropy for aggregate=False.

    Tensor mode:
        logits: Tensor[B, T', L]
        y:      Tensor[B]

    List mode:
        logits: list[Tensor[T'_i, L]]
        y:      Tensor[B]
    """
    if isinstance(logits, torch.Tensor):
        B, T, L = logits.shape
        y_time = y.unsqueeze(1).expand(B, T)

        return criterion(
            logits.reshape(B * T, L),
            y_time.reshape(B * T),
        )

    weight = getattr(criterion, "weight", None)
    losses = []

    for logits_i, y_i in zip(logits, y):
        y_time_i = y_i.repeat(logits_i.shape[0])

        loss_i = torch.nn.functional.cross_entropy(
            logits_i,
            y_time_i,
            reduction="mean",
            weight=None,
        )

        losses.append(loss_i)

    losses = torch.stack(losses)

    if weight is None:
        return losses.mean()

    subject_weights = weight[y]
    return (losses * subject_weights).sum() / subject_weights.sum().clamp_min(1e-8)


def compute_loss(logits, y, aggregate: bool, criterion):
    """CrossEntropyLoss receives raw logits."""
    if aggregate:
        return criterion(logits, y)

    return temporal_cross_entropy(logits, y, criterion)


@torch.no_grad()
def majority_vote_from_temporal_logits(
    logits_i: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    pred_time = logits_i.argmax(dim=-1)
    counts = torch.bincount(pred_time, minlength=n_classes).float()

    max_count = counts.max()
    tied = torch.where(counts == max_count)[0]

    if tied.numel() > 1:
        probs = torch.softmax(logits_i, dim=-1)
        mean_probs = probs.mean(dim=0)

        tied_probs = mean_probs[tied]
        tied = tied[torch.where(tied_probs == tied_probs.max())[0]]

    return tied.min().long()


def predict_from_logits(logits, aggregate: bool, n_classes: int) -> torch.Tensor:
    if aggregate:
        return logits.argmax(dim=-1)

    if isinstance(logits, torch.Tensor):
        return torch.stack(
            [
                majority_vote_from_temporal_logits(
                    logits_i=logits_i,
                    n_classes=n_classes,
                )
                for logits_i in logits
            ]
        )

    return torch.stack(
        [
            majority_vote_from_temporal_logits(
                logits_i=logits_i,
                n_classes=n_classes,
            )
            for logits_i in logits
        ]
    )


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    aggregate: bool,
    n_classes: int,
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

        logits, _ = model(X)
        loss = compute_loss(
            logits=logits,
            y=y,
            aggregate=aggregate,
            criterion=criterion,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        with torch.no_grad():
            pred = predict_from_logits(
                logits=logits,
                aggregate=aggregate,
                n_classes=n_classes,
            )

        batch_n = y.shape[0]
        total_loss += loss.item() * batch_n
        total_n += batch_n

        y_true_all.extend(y.detach().cpu().tolist())
        y_pred_all.extend(pred.detach().cpu().tolist())

    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics["loss"] = total_loss / total_n

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    aggregate: bool,
    n_classes: int,
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

        logits, _ = model(X)
        loss = compute_loss(
            logits=logits,
            y=y,
            aggregate=aggregate,
            criterion=criterion,
        )

        pred = predict_from_logits(
            logits=logits,
            aggregate=aggregate,
            n_classes=n_classes,
        )

        batch_n = y.shape[0]
        total_loss += loss.item() * batch_n
        total_n += batch_n

        names_all.extend(list(names))
        y_true_all.extend(y.detach().cpu().tolist())
        y_pred_all.extend(pred.detach().cpu().tolist())

    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics["loss"] = total_loss / total_n

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
    n_classes: int,
    device,
    save_dir: Path,
):
    set_seed(init_seed)

    model = EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        temporal_kern=args.temporal_kern,
        separable_kern=args.separable_kern,
        pool1=args.pool1,
        pool2=args.pool2,
        dropout=args.dropout,
        meanmax_alpha=args.meanmax_alpha,
        pp_as=args.pp_as,
        aggregate=args.aggregate,
        norm=args.norm,
    ).to(device)

    class_weights = None

    if not args.no_class_weights:
        train_labels = train_loader.dataset.y.detach().cpu().tolist()
        class_weights = get_class_weights(
            labels=train_labels,
            n_classes=n_classes,
            device=device,
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = defaultdict(list)

    best_val_balanced_acc = -1.0
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
            aggregate=args.aggregate,
            n_classes=n_classes,
            grad_clip=args.grad_clip,
            standardize=standardize,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            aggregate=args.aggregate,
            n_classes=n_classes,
            standardize=standardize,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"    epoch={epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} | "
                f"val_bacc={val_metrics['balanced_acc']:.4f}"
            )

        improved = (
            val_metrics["balanced_acc"] > best_val_balanced_acc
            or (
                val_metrics["balanced_acc"] == best_val_balanced_acc
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_balanced_acc = val_metrics["balanced_acc"]
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

    checkpoint_path = (
        save_dir
        / "checkpoints"
        / f"modality-{args.modality}_split-{split_seed}_init-{init_seed}_fold-{fold_id:02d}.pt"
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "dataset": "MODMA",
            "modality": args.modality,
            "target": "HC=0 vs MDD=1",
            "pp_as": args.pp_as,
            "aggregate": args.aggregate,
            "standardize": not args.no_standardize,
            "class_weights": not args.no_class_weights,
            "split_seed": split_seed,
            "init_seed": init_seed,
            "fold": fold_id,
            "best_epoch": best_epoch,
            "best_val_balanced_acc": best_val_balanced_acc,
            "best_val_loss": best_val_loss,
            "n_channels": n_channels,
            "n_classes": n_classes,
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
        aggregate=args.aggregate,
        n_classes=n_classes,
        standardize=standardize,
        return_predictions=True,
    )

    plot_path = (
        save_dir
        / "plots"
        / "fold_curves"
        / f"modality-{args.modality}_split-{split_seed}_init-{init_seed}_fold-{fold_id:02d}.png"
    )

    plot_fold_curves(
        history=history,
        save_path=plot_path,
        title=(
            f"{save_dir.name} | modality={args.modality} | "
            f"split={split_seed} | init={init_seed} | fold={fold_id:02d}"
        ),
    )

    metric_row = {
        "experiment": save_dir.name,
        "dataset": "MODMA",
        "modality": args.modality,
        "pp_as": args.pp_as,
        "aggregate": args.aggregate,
        "norm": args.norm,
        "standardize": standardize,
        "class_weights": not args.no_class_weights,
        "split_seed": split_seed,
        "init_seed": init_seed,
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_val_balanced_acc,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "test_f1_macro": test_metrics["f1_macro"],
        "plot_path": str(plot_path),
        "checkpoint_path": str(checkpoint_path),
    }

    prediction_rows = []

    for subject_id, yt, yp in zip(names, y_true, y_pred):
        prediction_rows.append(
            {
                "experiment": save_dir.name,
                "dataset": "MODMA",
                "modality": args.modality,
                "pp_as": args.pp_as,
                "aggregate": args.aggregate,
                "norm": args.norm,
                "standardize": standardize,
                "class_weights": not args.no_class_weights,
                "split_seed": split_seed,
                "init_seed": init_seed,
                "fold": fold_id,
                "subject_id": subject_id,
                "y_true": yt,
                "y_pred": yp,
            }
        )

    return metric_row, prediction_rows


def inspect_dataset(dataset: MODMADataset) -> None:
    labels = []
    shapes = []

    for _, eeg, label in dataset:
        labels.append(label.item())
        shapes.append(tuple(eeg.shape))

    label_count = dict(pd.Series(labels).value_counts().sort_index())
    channels = sorted(set(shape[0] for shape in shapes))
    lengths = [shape[1] for shape in shapes]

    print("\nDataset inspection:")
    print("Dataset: MODMA 128-channel resting-state EEG")
    print(f"Root: {dataset.root}")
    print(f"EEG dir: {dataset.data_root}")
    print(f"Subjects/files: {len(dataset)}")
    print(f"Unique subjects: {len(set(sample['subject'] for sample in dataset.samples))}")
    print(f"Labels: {label_count}")
    print(f"Channels kept: {dataset.n_channels}")
    print("Shapes:")

    if len(channels) == 1:
        C = channels[0]

        if min(lengths) == max(lengths):
            print(f"  ({C}, {min(lengths)}): {len(shapes)}")
        else:
            print(f"  ({C}, T:{min(lengths)}/{max(lengths)})")
    else:
        print(f"  C={channels} | T:{min(lengths)}/{max(lengths)}")


def print_overall_summary(overall_metrics: pd.DataFrame) -> None:
    print("\nOverall:")

    if len(overall_metrics) == 1:
        print(overall_metrics.iloc[0].to_string())
    else:
        for _, row in overall_metrics.iterrows():
            print("\n" + "-" * 60)
            print(row.to_string())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "data/raw/modma_db"))
    parser.add_argument("--eeg-dir", type=str, default="EEG_128channels_resting_lanzhou_2015")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--subject-col", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--modality", type=str, default="resting128")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--inner-splits", type=int, default=5)

    parser.add_argument("--split-seeds", type=int, nargs="+", default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--init-seeds", type=int, nargs="+", default=DEFAULT_INIT_SEEDS)

    parser.add_argument("--rand-split-seed", action="store_true")
    parser.add_argument("--rand-init-seed", action="store_true")
    parser.add_argument("--n-rand-split-seeds", type=int, default=RANDOM_SEED_COUNT)
    parser.add_argument("--n-rand-init-seeds", type=int, default=RANDOM_SEED_COUNT)
    parser.add_argument("--seed-max", type=int, default=RANDOM_SEED_MAX)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=parse_optional_float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--default-fs", type=float, default=250.0)
    parser.add_argument("--target-fs", type=parse_optional_float, default=None)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)
    parser.add_argument("--expected-channels", type=int, default=128)
    parser.add_argument("--pp-as", type=str, default="tensor", choices=["tensor", "list"])
    parser.add_argument("--channel-strategy", type=str, default="common", choices=["common", "all"])

    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--temporal-kern", type=int, default=63)
    parser.add_argument("--separable-kern", type=int, default=15)
    parser.add_argument("--pool1", type=int, default=8)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--aggregate", type=parse_bool, default=True)
    parser.add_argument("--norm", type=str, default="auto", choices=["auto", "batch", "group"])
    parser.add_argument("--meanmax-alpha", type=float, default=0.0)

    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")

    parser.add_argument("--inspect-shapes", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-every", type=int, default=5)

    parser.add_argument("--save-dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rand_split_seed:
        args.split_seeds = sample_random_seeds(args.n_rand_split_seeds, args.seed_max)

    if args.rand_init_seed:
        args.init_seeds = sample_random_seeds(args.n_rand_init_seeds, args.seed_max)

    if args.save_dir is None:
        agg_name = "agg" if args.aggregate else "temporal"
        args.save_dir = f"modma_db/eegnet_{args.modality}_{args.pp_as}_{agg_name}_{args.norm}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = resolve_save_dir(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, save_dir)

    print(f"\nDevice: {device}")
    print(f"Experiment: {save_dir.name}")
    print(f"Save dir: {save_dir}")
    print("dataset: MODMA")
    print(f"modality: {args.modality}")
    print("target: HC/control=0 vs MDD=1")
    print(f"pp_as: {args.pp_as}")
    print(f"aggregate: {args.aggregate}")
    print(f"norm: {args.norm}")
    print(f"standardize: {not args.no_standardize}")
    print(f"class_weights: {not args.no_class_weights}")
    print(f"split_seeds: {args.split_seeds}")
    print(f"init_seeds:  {args.init_seeds}")

    dataset = MODMADataset(
        root=args.root,
        eeg_dir=args.eeg_dir,
        metadata_csv=args.metadata_csv,
        subject_col=args.subject_col,
        label_col=args.label_col,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        default_fs=args.default_fs,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as=args.pp_as,
        channel_strategy=args.channel_strategy,
        expected_channels=args.expected_channels,
    )

    if args.inspect_shapes:
        inspect_dataset(dataset)

    if args.validate_only:
        return

    first_eeg = dataset[0][1]
    n_channels = first_eeg.shape[0]

    labels = [dataset[i][2].item() for i in range(len(dataset))]
    n_classes = len(set(labels))
    class_names = ["HC", "MDD"]

    print(
        f"\nDataset: MODMA | modality={args.modality} | "
        f"subjects/files={len(dataset)} | "
        f"unique_subjects={len(set(sample['subject'] for sample in dataset.samples))} | "
        f"C={n_channels} | classes={n_classes}"
    )

    print(f"Class balance: {dict(pd.Series(labels).value_counts().sort_index())}")

    if n_classes != 2:
        raise ValueError(
            f"Expected exactly 2 classes for MDD vs HC, but found {n_classes}: {sorted(set(labels))}"
        )

    all_metric_rows = []
    all_prediction_rows = []

    for split_seed in args.split_seeds:
        folds = create_dataloaders(
            dataset=dataset,
            k_folder=args.k,
            batch_size=args.batch_size,
            shuffle=True,
            split_seed=split_seed,
            inner_split=args.inner_splits,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        selected_folds = args.folds
        if selected_folds is None:
            selected_folds = list(range(1, len(folds) + 1))

        print(f"\nsplit_seed={split_seed} | folds={selected_folds}")

        for init_seed in args.init_seeds:
            for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
                if fold_id not in selected_folds:
                    continue

                print(
                    f"  fold={fold_id:02d} | "
                    f"init_seed={init_seed} | "
                    f"modality={args.modality} | "
                    f"pp_as={args.pp_as} | "
                    f"aggregate={args.aggregate}"
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
                    n_classes=n_classes,
                    device=device,
                    save_dir=save_dir,
                )

                all_metric_rows.append(metric_row)
                all_prediction_rows.extend(prediction_rows)

                print(
                    f"    test_acc={metric_row['test_acc']:.4f} | "
                    f"test_bal_acc={metric_row['test_balanced_acc']:.4f} | "
                    f"test_f1={metric_row['test_f1_macro']:.4f}"
                )

    fold_metrics = pd.DataFrame(all_metric_rows)
    predictions = pd.DataFrame(all_prediction_rows)

    fold_metrics_path = save_dir / "fold_metrics.csv"
    predictions_path = save_dir / "predictions.csv"

    fold_metrics.to_csv(fold_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    overall_metrics = (
        fold_metrics
        .groupby(
            [
                "experiment",
                "dataset",
                "modality",
                "pp_as",
                "aggregate",
                "norm",
                "standardize",
                "class_weights",
            ]
        )
        .agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_balanced_acc_mean=("test_balanced_acc", "mean"),
            test_balanced_acc_std=("test_balanced_acc", "std"),
            test_f1_macro_mean=("test_f1_macro", "mean"),
            test_f1_macro_std=("test_f1_macro", "std"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            best_val_balanced_acc_mean=("best_val_balanced_acc", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .reset_index()
    )

    overall_metrics_path = save_dir / "overall_metrics.csv"
    overall_metrics.to_csv(overall_metrics_path, index=False)

    cm_path = save_dir / "plots" / "confusion_matrix_global.png"

    plot_confusion_matrix(
        y_true=predictions["y_true"].tolist(),
        y_pred=predictions["y_pred"].tolist(),
        class_names=class_names,
        save_path=cm_path,
        title=f"{save_dir.name} global confusion matrix",
        normalize=False,
    )

    print_overall_summary(overall_metrics)

    print("\nSaved:")
    print(f"  {fold_metrics_path}")
    print(f"  {overall_metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {cm_path}")


if __name__ == "__main__":
    main()
