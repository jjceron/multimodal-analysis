from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.datasets.modma_db import MODMADataset, create_dataloaders, create_windowed_dataloaders, DEFAULT_ROOT
from src.models import (
    BandPowerSVM, CNNLSTM, CSPLDA, EEGConformer, EEGFormer, EEGNet,
    RiemannianMDM, ShallowConvNet,
)
from src.models.augmentations import GaussianNoise, ChannelDropout, TimeMasking, Mixup, mixup_criterion
from src.utils.plotting import save_fold_figures, plot_dual_confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "modma_db"


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None

    if isinstance(value, str) and value.lower() in {"none", "null", "nan"}:
        return None

    return float(value)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--condition", type=str, default="EC")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--target-fs", type=parse_optional_float, default=None)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=120.0)

    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--pool1", type=int, default=8)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--meanmax-alpha", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation (GaussianNoise, ChannelDropout, TimeMasking)")

    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", action="store_true",
                        help="Enable ReduceLROnPlateau scheduler")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="Patience for LR scheduler")
    parser.add_argument("--lr-factor", type=float, default=0.5,
                        help="Factor to reduce LR on plateau")

    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--init-seed", type=int, default=3001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="eegnet",
                        choices=["eegnet", "eegformer", "eegconformer",
                                 "shallowconvnet", "cnn_lstm",
                                 "csp_lda", "riemann_mdm", "bandpower_svm"],
                        help="Model architecture")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override directory name (default: capitalized --model)")
    parser.add_argument("--version-name", type=str, default=None,
                        help="Version/tag for this run (e.g., agg, meanpool)")

    parser.add_argument("--window-sec", type=float, default=0.0,
                        help="Window size in seconds (0 = no windowing, use full recording)")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Overlap fraction (0.0 = no overlap, 0.5 = 50%%)")

    return parser.parse_args()


def auto_adjust_pooling(
    n_samples: int, pool1: int, pool2: int, model_name: str,
) -> tuple[int, int]:
    model_name = model_name.lower()
    if model_name == "eegnet":
        target_min, target_max = 30, 500
    else:
        target_min, target_max = 20, 300

    p1, p2 = pool1, pool2
    t_prime = n_samples // p1 // p2

    if t_prime < target_min:
        while t_prime < target_min and (p1 > 1 or p2 > 1):
            if p2 >= p1 and p2 > 1:
                p2 //= 2
            elif p1 > 1:
                p1 //= 2
            else:
                break
            t_prime = n_samples // p1 // p2
        print(f"  [Auto-pool] Reduced pooling: ({pool1},{pool2}) -> ({p1},{p2}), T'={t_prime}")
    elif t_prime > target_max:
        while t_prime > target_max:
            if p1 <= p2 and p1 < 64:
                p1 *= 2
            elif p2 < 64:
                p2 *= 2
            else:
                break
            t_prime = n_samples // p1 // p2
        print(f"  [Auto-pool] Increased pooling: ({pool1},{pool2}) -> ({p1},{p2}), T'={t_prime}")

    return p1, p2


class _ModelAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.model(x), None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    augment: bool = False,
    mixup_fn: Mixup | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    aug_transforms = nn.Sequential(
        GaussianNoise(snr=20.0),
        ChannelDropout(p=0.15),
        TimeMasking(max_mask_ratio=0.15),
    ) if augment else None

    for batch in loader:
        _, X, y = batch
        X, y = X.to(device), y.to(device)

        if aug_transforms is not None:
            X = aug_transforms(X)

        optimizer.zero_grad()

        if mixup_fn is not None:
            X, y_a, y_b = mixup_fn(X, y)
            logits, _ = model(X)
            loss = mixup_criterion(criterion, logits, y_a, y_b, mixup_fn.lam)
        else:
            logits, _ = model(X)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)

        if mixup_fn is not None and mixup_fn.shuffled_idxs is not None:
            all_labels.extend(y.cpu().tolist())
        else:
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        _, X, y = batch
        X, y = X.to(device), y.to(device)

        logits, _ = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], dict[str, float]]:
    model.eval()
    all_preds, all_labels, all_names = [], [], []

    for batch in loader:
        names, X, y = batch
        X = X.to(device)

        logits, _ = model(X)
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        all_names.extend(names)
        all_preds.extend(preds)
        all_labels.extend(y.tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }

    return all_labels, all_preds, metrics


@torch.no_grad()
def evaluate_subject_level(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], dict[str, float], list[str]]:
    model.eval()
    subject_votes: dict[str, list[int]] = {}
    subject_trues: dict[str, int] = {}

    for batch in loader:
        names, X, y = batch
        X = X.to(device)
        logits, _ = model(X)
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        for name, pred, true in zip(names, preds, y.tolist()):
            if name not in subject_votes:
                subject_votes[name] = []
                subject_trues[name] = true
            subject_votes[name].append(pred)

    all_subjects: list[str] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    for name, votes in subject_votes.items():
        majority = max(set(votes), key=votes.count)
        all_subjects.append(name)
        all_preds.append(majority)
        all_labels.append(subject_trues[name])

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }

    return all_labels, all_preds, metrics, all_subjects


def build_model(args, n_channels: int, n_classes: int) -> nn.Module:
    model_key = args.model.lower()

    if model_key == "csp_lda":
        return CSPLDA(n_channels=n_channels, n_classes=n_classes)
    elif model_key == "riemann_mdm":
        return RiemannianMDM(n_channels=n_channels, n_classes=n_classes)
    elif model_key == "bandpower_svm":
        return BandPowerSVM(n_channels=n_channels, n_classes=n_classes,
                            window_sec=max(args.window_sec, 2.0))

    sfreq = 250.0
    use_win = getattr(args, 'window_sec', 0.0) > 0
    n_samples = int(round((args.window_sec if use_win else args.duration_sec) * sfreq))

    pool1, pool2 = auto_adjust_pooling(
        n_samples, args.pool1, args.pool2, args.model,
    )

    if model_key == "eegnet":
        return EEGNet(
            n_channels=n_channels, n_classes=n_classes,
            F1=args.F1, D=args.D, F2=args.F2,
            pool1=pool1, pool2=pool2,
            dropout=args.dropout, meanmax_alpha=args.meanmax_alpha,
        )
    elif model_key == "eegformer":
        base = EEGFormer(
            n_channels=n_channels, n_samples=n_samples, num_classes=n_classes,
            F1=args.F1, D=args.D, F2=args.F2,
            pool1=pool1, pool2=pool2, dropout_eeg=args.dropout,
        )
        return _ModelAdapter(base)
    elif model_key == "eegconformer":
        base = EEGConformer(
            num_channels=n_channels, n_samples=n_samples, num_classes=n_classes,
            dropout=args.dropout,
        )
        return _ModelAdapter(base)
    elif model_key == "shallowconvnet":
        return _ModelAdapter(ShallowConvNet(
            n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
            dropout=args.dropout,
        ))
    elif model_key == "cnn_lstm":
        return _ModelAdapter(CNNLSTM(
            n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
            dropout=args.dropout,
        ))

    raise ValueError(f"Unknown model: {args.model}")


def build_out_dir(args) -> Path:
    NAME_MAP = {
        "eegnet": "EEGNet", "eegformer": "EEGFormer", "eegconformer": "EEGConformer",
        "shallowconvnet": "ShallowConvNet", "cnn_lstm": "CNNLSTM",
        "csp_lda": "CSPLDA", "riemann_mdm": "RiemannianMDM", "bandpower_svm": "BandPowerSVM",
    }
    model_name = args.model_name or NAME_MAP.get(args.model.lower(), args.model.capitalize())
    version = args.version_name or args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / model_name / version
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    print(f"\n>>> Running: {args.model} / {args.version_name} <<<\n")
    print(f"Device: {device}")
    print(f"Loading MODMA dataset...")

    dataset = MODMADataset(
        root=args.root or DEFAULT_ROOT,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
    )

    n_channels = dataset.samples[0]["eeg"].shape[0]
    n_classes = 2
    class_names = ["HC", "MDD"]

    print(f"  Subjects: {len(dataset)}")
    print(f"  Channels: {n_channels}")
    print(f"  Tensor shape: {dataset.samples[0]['eeg'].shape}")

    labels = [int(s["label"].item()) for s in dataset.samples]
    label_counts = Counter(labels)
    print(f"  HC: {label_counts.get(0, 0)}, MDD: {label_counts.get(1, 0)}")

    use_windowing = args.window_sec > 0
    sfreq = 250.0

    if use_windowing:
        window_samples = int(round(args.window_sec * sfreq))
        stride = int(round(window_samples * (1.0 - args.overlap)))
        print(f"Windowing: {args.window_sec}s windows ({window_samples} samples), "
              f"stride={stride}, overlap={args.overlap:.0%}")

    DEEP_MODELS = {"eegnet", "eegformer", "eegconformer", "shallowconvnet", "cnn_lstm"}
    if args.model.lower() in DEEP_MODELS and args.window_sec == 0:
        safe_bs = min(args.batch_size, 4)
        if safe_bs < args.batch_size:
            print(f"  [Auto-batch] Reducing batch_size {args.batch_size} -> {safe_bs} for {args.model} on full signal")
            args.batch_size = safe_bs

    print(f"Creating {args.k}-fold cross-validation...")

    if use_windowing:
        folds = create_windowed_dataloaders(
            dataset=dataset,
            k_folder=args.k,
            batch_size=args.batch_size,
            shuffle=True,
            split_seed=args.split_seed,
            inner_split=args.inner_splits,
            window_samples=window_samples,
            stride=stride,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
        )
    else:
        folds = create_dataloaders(
            dataset=dataset,
            k_folder=args.k,
            batch_size=args.batch_size,
            shuffle=True,
            split_seed=args.split_seed,
            inner_split=args.inner_splits,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
        )

    out_dir = build_out_dir(args)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["device"] = str(device)
    config["n_channels"] = n_channels
    config["n_classes"] = n_classes
    config["tensor_shape"] = list(dataset.samples[0]["eeg"].shape)
    if use_windowing:
        config["windowing"] = {
            "window_sec": args.window_sec,
            "overlap": args.overlap,
            "window_samples": window_samples,
            "stride": stride,
        }
    save_json(out_dir / "config.json", config)

    all_fold_metrics: list[dict] = []
    all_predictions: list[dict] = []
    fold_data_list: list[dict] = []
    overall_val_true: list[int] = []
    overall_val_pred: list[int] = []
    overall_test_true: list[int] = []
    overall_test_pred: list[int] = []

    for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_id:02d}/{args.k - 1:02d}")
        if use_windowing:
            print(f"  Train: {len(train_loader.dataset)} windows "
                  f"(~{len(train_loader.dataset) // max(len(train_loader.dataset.names), 1)}/subj)")
            print(f"  Val:   {len(val_loader.dataset)} windows "
                  f"(~{len(val_loader.dataset) // max(len(val_loader.dataset.names), 1)}/subj)")
            print(f"  Test:  {len(test_loader.dataset)} windows "
                  f"(~{len(test_loader.dataset) // max(len(test_loader.dataset.names), 1)}/subj)")
        else:
            print(f"  Train: {len(train_loader.dataset)} subjects")
            print(f"  Val:   {len(val_loader.dataset)} subjects")
            print(f"  Test:  {len(test_loader.dataset)} subjects")

        set_seed(args.init_seed + fold_id)
        model = build_model(args, n_channels=n_channels, n_classes=n_classes)
        model = model.to(device)

        is_sklearn = getattr(model, '_is_sklearn', False)

        if is_sklearn:
            print("  Fitting sklearn model...")
            model.fit(train_loader, device)
            train_losses: list[float] = []
            val_losses: list[float] = []
            train_accs: list[float] = []
            val_accs: list[float] = []
            best_epoch = 0
            n_epochs = 1
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()

            scheduler = None
            if args.lr_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=args.lr_factor,
                    patience=args.lr_patience, min_lr=1e-6,
                )

            best_val_loss = float("inf")
            best_state_dict = None
            patience_counter = 0
            current_lr = args.lr

            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            lr_log: list[float] = []

            mixup_fn = Mixup(alpha=0.2) if args.augment else None

            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    augment=args.augment, mixup_fn=mixup_fn,
                )
                val_loss, val_acc, _, _ = validate(
                    model, val_loader, criterion, device,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                if scheduler is not None:
                    scheduler.step(val_loss)
                    current_lr = optimizer.param_groups[0]["lr"]
                    lr_log.append(current_lr)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch == 1 or epoch % 10 == 0 or patience_counter == 0:
                    lr_str = f" lr={current_lr:.2e}" if scheduler is not None else ""
                    print(
                        f"  Epoch {epoch:3d}/{args.epochs} | "
                        f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                        f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} |"
                        f"{lr_str}"
                        f" Patience: {patience_counter:2d}/{args.patience}"
                    )

                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            best_epoch = len(train_losses) - patience_counter
            n_epochs = len(train_losses)

        if use_windowing:
            y_true_val, y_pred_val, val_metrics, val_subjects = evaluate_subject_level(
                model, val_loader, device)
            y_true_test, y_pred_test, test_metrics, test_subjects = evaluate_subject_level(
                model, test_loader, device)
            print(f"  Val (subject-level, majority vote):")
            for k, v in val_metrics.items():
                print(f"    {k}: {v:.4f}")
            print(f"  Test (subject-level, majority vote):")
        else:
            y_true_val, y_pred_val, _ = evaluate(model, val_loader, device)
            y_true_test, y_pred_test, test_metrics = evaluate(model, test_loader, device)

        print(f"  Test results:")
        for metric_name, value in test_metrics.items():
            print(f"    {metric_name}: {value:.4f}")

        fold_metrics = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "n_epochs": n_epochs,
            "val_accuracy": accuracy_score(y_true_val, y_pred_val),
            "val_balanced_accuracy": balanced_accuracy_score(y_true_val, y_pred_val),
            "val_f1_macro": f1_score(y_true_val, y_pred_val, average="macro"),
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        all_fold_metrics.append(fold_metrics)

        pred_names_val = val_subjects if use_windowing else val_loader.dataset.names
        pred_names_test = test_subjects if use_windowing else test_loader.dataset.names

        for name, true_label, pred_label in zip(
            pred_names_val, y_true_val, y_pred_val
        ):
            all_predictions.append({
                "fold": fold_id,
                "split": "val",
                "subject": name,
                "true_label": true_label,
                "pred_label": pred_label,
            })

        for name, true_label, pred_label in zip(
            pred_names_test, y_true_test, y_pred_test
        ):
            all_predictions.append({
                "fold": fold_id,
                "split": "test",
                "subject": name,
                "true_label": true_label,
                "pred_label": pred_label,
            })

        overall_val_true.extend(y_true_val)
        overall_val_pred.extend(y_pred_val)
        overall_test_true.extend(y_true_test)
        overall_test_pred.extend(y_pred_test)

        if not is_sklearn:
            save_fold_figures(
                fold_id=fold_id,
                train_losses=train_losses,
                val_losses=val_losses,
                train_metrics=train_accs,
                val_metrics=val_accs,
                y_true_val=y_true_val,
                y_pred_val=y_pred_val,
                y_true_test=y_true_test,
                y_pred_test=y_pred_test,
                class_names=class_names,
                output_dir=plots_dir,
                metric_name="Accuracy",
            )

        fold_data_list.append({
            "fold_id": fold_id,
            "best_epoch": fold_metrics["best_epoch"],
            "train_losses": train_losses or [0.0],
            "val_losses": val_losses or [0.0],
            "train_accs": train_accs or [0.0],
            "val_accs": val_accs or [0.0],
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_fold = pd.DataFrame(all_fold_metrics)
    df_fold.to_csv(out_dir / "fold_metrics.csv", index=False)

    df_pred = pd.DataFrame(all_predictions)
    df_pred.to_csv(out_dir / "predictions.csv", index=False)

    overall = {
        "mean_accuracy": float(df_fold["test_accuracy"].mean()),
        "std_accuracy": float(df_fold["test_accuracy"].std()),
        "mean_balanced_accuracy": float(df_fold["test_balanced_accuracy"].mean()),
        "std_balanced_accuracy": float(df_fold["test_balanced_accuracy"].std()),
        "mean_f1_macro": float(df_fold["test_f1_macro"].mean()),
        "std_f1_macro": float(df_fold["test_f1_macro"].std()),
    }

    df_overall = pd.DataFrame([overall])
    df_overall.to_csv(out_dir / "overall_metrics.csv", index=False)

    plot_dual_confusion_matrix(
        y_true_val=overall_val_true,
        y_pred_val=overall_val_pred,
        y_true_test=overall_test_true,
        y_pred_test=overall_test_pred,
        class_names=class_names,
        save_path=plots_dir / "overall_confusion_matrices.png",
        show=False,
    )

    final_report = {
        "config": config,
        "fold_metrics": all_fold_metrics,
        "overall": overall,
        "fold_data": fold_data_list,
    }
    save_json(out_dir / "results.json", final_report)

    print(f"\n{'='*50}")
    print(f"Benchmark complete!")
    print(f"Output directory: {out_dir}")
    print(f"\nOverall results:")
    for metric_name, value in overall.items():
        print(f"  {metric_name}: {value:.4f}")

    print("\nFold details:")
    print(df_fold.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[WARNING] Benchmark failed for this model: {e}")
        print("Continuing to next model...")
        sys.exit(0)
