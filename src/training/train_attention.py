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

from src.datasets.modma_db import MODMADataset, create_windowed_dataloaders, DEFAULT_ROOT
from src.models import (
    CNNLSTM, EEGConformer, EEGFormer, EEGNet, ShallowConvNet,
)
from src.models.attention_pooling import SubjectGroupedDataset, collate_subjects, SubjectAttentionModel
from src.models.augmentations import GaussianNoise, ChannelDropout, TimeMasking
from src.utils.plotting import save_fold_figures, plot_dual_confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "modma_db"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--condition", type=str, default="EC")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=60.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--duration-sec", type=float, default=120.0)

    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)

    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", action="store_true", default=True)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true", default=True)

    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--init-seed", type=int, default=3001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--model", type=str, default="eegnet",
                        choices=["eegnet", "eegformer", "eegconformer",
                                 "shallowconvnet", "cnn_lstm"])
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--version-name", type=str, default="attention_v1")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_backbone(args, n_channels: int, n_classes: int) -> nn.Module:
    sfreq = 250.0
    n_samples = int(round(args.window_sec * sfreq))

    if args.model == "eegnet":
        return EEGNet(
            n_channels=n_channels, n_classes=n_classes,
            F1=8, D=2, F2=16, pool1=8, pool2=8,
            dropout=args.dropout, meanmax_alpha=0.0,
        )
    elif args.model == "eegformer":
        return EEGFormer(
            n_channels=n_channels, n_samples=n_samples, num_classes=n_classes,
            F1=8, D=2, F2=16, pool1=4, pool2=4, dropout_eeg=args.dropout,
        )
    elif args.model == "eegconformer":
        return EEGConformer(
            num_channels=n_channels, n_samples=n_samples, num_classes=n_classes,
            dropout=args.dropout,
        )
    elif args.model == "shallowconvnet":
        return ShallowConvNet(
            n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
            dropout=args.dropout,
        )
    elif args.model == "cnn_lstm":
        return CNNLSTM(
            n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {args.model}")


def build_out_dir(args) -> Path:
    NAME_MAP = {
        "eegnet": "EEGNet", "eegformer": "EEGFormer", "eegconformer": "EEGConformer",
        "shallowconvnet": "ShallowConvNet", "cnn_lstm": "CNNLSTM",
    }
    model_name = args.model_name or NAME_MAP.get(args.model.lower(), args.model.capitalize())
    version = args.version_name or time.strftime("%Y%m%d_%H%M%S")
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

    print(f"\n>>> Attention: {args.model} / {args.version_name} <<<\n")
    print(f"Device: {device}")

    dataset = MODMADataset(
        root=args.root or DEFAULT_ROOT,
        lowcut=args.lowcut, highcut=args.highcut,
        notch=args.notch, target_fs=args.target_fs, duration_sec=args.duration_sec,
    )

    n_channels = dataset.samples[0]["eeg"].shape[0]
    n_classes = 2
    class_names = ["HC", "MDD"]

    print(f"  Subjects: {len(dataset)}, Channels: {n_channels}")
    labels = [int(s["label"].item()) for s in dataset.samples]
    print(f"  HC: {Counter(labels).get(0, 0)}, MDD: {Counter(labels).get(1, 0)}")

    window_samples = int(round(args.window_sec * 250.0))
    stride = int(round(window_samples * (1.0 - args.overlap)))

    folds = create_windowed_dataloaders(
        dataset=dataset, k_folder=args.k, batch_size=args.batch_size,
        shuffle=True, split_seed=args.split_seed, inner_split=5,
        window_samples=window_samples, stride=stride,
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
    save_json(out_dir / "config.json", config)

    aug_transforms = nn.Sequential(
        GaussianNoise(snr=20.0),
        ChannelDropout(p=0.15),
        TimeMasking(max_mask_ratio=0.15),
    )

    all_fold_metrics: list[dict] = []
    overall_val_true: list[int] = []
    overall_val_pred: list[int] = []
    overall_test_true: list[int] = []
    overall_test_pred: list[int] = []

    for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_id:02d}/{args.k - 1:02d}")

        subject_train = SubjectGroupedDataset(train_loader.dataset)
        subject_val = SubjectGroupedDataset(val_loader.dataset)
        subject_test = SubjectGroupedDataset(test_loader.dataset)

        train_loader = torch.utils.data.DataLoader(
            subject_train, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_subjects,
        )
        val_loader = torch.utils.data.DataLoader(
            subject_val, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_subjects,
        )
        test_loader = torch.utils.data.DataLoader(
            subject_test, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_subjects,
        )

        print(f"  Train subjects: {len(subject_train)}")
        print(f"  Val subjects:   {len(subject_val)}")
        print(f"  Test subjects:  {len(subject_test)}")

        set_seed(args.init_seed + fold_id)
        backbone = build_backbone(args, n_channels=n_channels, n_classes=n_classes)
        model = SubjectAttentionModel(backbone, n_classes=n_classes)
        model = model.to(device)

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

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            train_preds, train_labels = [], []

            for _, X, y, mask in train_loader:
                X, y, mask = X.to(device), y.to(device), mask.to(device)

                if args.augment:
                    B, W, C, T = X.shape
                    aug_X = X.view(B * W, C, T)
                    aug_X = aug_transforms(aug_X)
                    X = aug_X.view(B, W, C, T)

                optimizer.zero_grad()
                logits = model(X, mask)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X.size(0)
                train_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                train_labels.extend(y.cpu().tolist())

            train_loss = epoch_loss / len(subject_train)
            train_acc = accuracy_score(train_labels, train_preds)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for _, X, y, mask in val_loader:
                    X, y, mask = X.to(device), y.to(device), mask.to(device)
                    logits = model(X, mask)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * X.size(0)
                    val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                    val_labels.extend(y.cpu().tolist())

            val_loss = val_loss / len(subject_val)
            val_acc = accuracy_score(val_labels, val_preds)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch == 1 or epoch % 20 == 0 or patience_counter == 0:
                print(f"  Epoch {epoch:3d}/{args.epochs} | "
                      f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                      f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
                      f"Patience: {patience_counter:2d}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        best_epoch = len(train_losses) - patience_counter
        n_epochs = len(train_losses)

        model.eval()
        y_true_test, y_pred_test = [], []

        with torch.no_grad():
            for _, X, y, mask in test_loader:
                X, y, mask = X.to(device), y.to(device), mask.to(device)
                logits = model(X, mask)
                y_pred_test.extend(torch.argmax(logits, dim=1).cpu().tolist())
                y_true_test.extend(y.cpu().tolist())

        test_metrics = {
            "accuracy": accuracy_score(y_true_test, y_pred_test),
            "balanced_accuracy": balanced_accuracy_score(y_true_test, y_pred_test),
            "f1_macro": f1_score(y_true_test, y_pred_test, average="macro"),
        }

        print(f"  Test results:")
        for k, v in test_metrics.items():
            print(f"    {k}: {v:.4f}")

        fold_metrics = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "n_epochs": n_epochs,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        all_fold_metrics.append(fold_metrics)

        overall_test_true.extend(y_true_test)
        overall_test_pred.extend(y_pred_test)

        save_fold_figures(
            fold_id=fold_id,
            train_losses=train_losses, val_losses=val_losses,
            train_metrics=train_accs, val_metrics=val_accs,
            y_true_val=[], y_pred_val=[],
            y_true_test=y_true_test, y_pred_test=y_pred_test,
            class_names=class_names,
            output_dir=plots_dir, metric_name="Accuracy",
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_fold = pd.DataFrame(all_fold_metrics)
    df_fold.to_csv(out_dir / "fold_metrics.csv", index=False)

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
        y_true_val=[], y_pred_val=[],
        y_true_test=overall_test_true, y_pred_test=overall_test_pred,
        class_names=class_names,
        save_path=plots_dir / "overall_confusion_matrices.png",
        show=False,
    )

    final_report = {
        "config": config,
        "fold_metrics": all_fold_metrics,
        "overall": overall,
    }
    save_json(out_dir / "results.json", final_report)

    print(f"\n{'='*50}")
    print(f"Attention benchmark complete!")
    print(f"Output directory: {out_dir}")
    for metric_name, value in overall.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[WARNING] Attention benchmark failed: {e}")
        import traceback
        traceback.print_exc()
