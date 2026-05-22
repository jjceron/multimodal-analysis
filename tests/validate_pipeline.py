from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from src.datasets import EEGDataset, create_kfold_dataloaders
from src.models.eegnet import EEGNet
from src.utils.eeg_utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_out_dir(out_root: Path, run_name: str | None) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)

    if run_name:
        out_dir = out_root / run_name
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"validation_{stamp}"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_model(n_channels: int, n_classes: int, alpha: float) -> EEGNet:
    return EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=8,
        D=2,
        F2=16,
        temporal_kern=63,
        separable_kern=15,
        pool1=8,
        pool2=8,
        dropout=0.5,
        meanmax_alpha=alpha,
    )


def get_k_for_dataset(labels: list[int], requested_k: int) -> int:
    counts = Counter(labels)
    min_count = min(counts.values()) if counts else 0

    k = min(requested_k, min_count)

    if k < 2:
        raise ValueError("Not enough subjects per class to create folds.")

    return k


def summarize_subject_dataset(
    dataset: build_dataset.EEGDataset,
    out_dir: Path,
) -> dict:
    summary_df = dataset.get_summary_dataframe()
    summary_path = out_dir / "subject_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    labels = dataset.get_labels()
    label_counts = dict(Counter(labels))

    report = {
        "n_subjects": len(dataset),
        "label_counts": label_counts,
        "unique_sfreq": summary_df["sfreq"].unique().tolist(),
        "unique_n_channels": summary_df["n_channels"].unique().tolist(),
        "selected_condition": dataset.condition,
        "selected_duration_sec": summary_df["selected_duration_sec"].describe().to_dict(),
        "summary_csv": str(summary_path),
    }

    return report


def summarize_epoch_dataset(
    dataset: build_eeg.EEGDataset,
    out_dir: Path,
) -> dict:
    summary_df = dataset.get_summary_dataframe()
    summary_path = out_dir / "epoch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    labels = dataset.get_labels()
    label_counts = dict(Counter(labels))

    report = {
        "n_subjects": len(dataset),
        "label_counts": label_counts,
        "unique_sfreq": summary_df["sfreq"].unique().tolist(),
        "unique_n_channels": summary_df["n_channels"].unique().tolist(),
        "unique_n_samples": summary_df["n_samples"].unique().tolist(),
        "selected_condition": dataset.condition,
        "selected_shape": summary_df["selected_shape"].head(5).tolist(),
        "summary_csv": str(summary_path),
    }

    return report


def check_subject_pipeline(
    args,
    config: dict,
    out_dir: Path,
) -> dict:
    report: dict[str, object] = {"mode": "subject"}

    dataset = build_dataset.EEGDataset(
        config_path=args.config_path,
        condition=args.condition,
    )

    if args.max_subjects is not None:
        dataset.samples = dataset.samples[: args.max_subjects]

    report["dataset"] = summarize_subject_dataset(dataset, out_dir)

    labels = dataset.get_labels()
    n_classes = 3
    k = get_k_for_dataset(labels, args.k)

    folds = build_dataset.create_kfold_dataloaders(
        dataset,
        k=k,
        batch_size=args.batch_size,
        shuffle=False,
        split_seed=args.split_seed,
        num_workers=0,
        pin_memory=False,
    )

    train_loader, _, _ = folds[0]
    batch = next(iter(train_loader))

    x = batch["X"]
    y = batch["y"]
    mask = batch["mask"]

    model = build_model(n_channels=x.shape[1], n_classes=n_classes, alpha=args.meanmax_alpha)

    logits_subj, logits_time = model(x, mask=mask)
    loss = nn.CrossEntropyLoss()(logits_subj, y)

    report["model_check"] = {
        "batch_shape": list(x.shape),
        "mask_shape": list(mask.shape),
        "logits_time_shape": list(logits_time.shape),
        "logits_subj_shape": list(logits_subj.shape),
        "loss": float(loss.detach().cpu().item()),
    }

    return report


def check_epoch_pipeline(
    args,
    config: dict,
    out_dir: Path,
) -> dict:
    report: dict[str, object] = {"mode": "epoch"}

    dataset = build_eeg.EEGDataset(
        config_path=args.config_path,
        condition=args.condition,
    )

    if args.max_subjects is not None:
        dataset.samples = dataset.samples[: args.max_subjects]

    report["dataset"] = summarize_epoch_dataset(dataset, out_dir)

    labels = dataset.get_labels()
    n_classes = 3
    k = get_k_for_dataset(labels, args.k)

    folds = build_eeg.create_kfold_dataloaders(
        dataset,
        k=k,
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_loader, _, _ = folds[0]
    x, y, _ = next(iter(train_loader))

    model = build_model(n_channels=x.shape[1], n_classes=n_classes, alpha=args.meanmax_alpha)

    logits_subj, logits_time = model(x, mask=None)
    loss = nn.CrossEntropyLoss()(logits_subj, y)

    report["model_check"] = {
        "batch_shape": list(x.shape),
        "logits_time_shape": list(logits_time.shape),
        "logits_subj_shape": list(logits_subj.shape),
        "loss": float(loss.detach().cpu().item()),
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--condition", type=str, default="closed")
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-root", type=str, default="outputs/validation")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--check-subject", action="store_true")
    parser.add_argument("--check-epoch", action="store_true")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--meanmax-alpha", type=float, default=0.5)
    parser.add_argument("--max-subjects", type=int, default=None)

    args = parser.parse_args()

    if not args.check_subject and not args.check_epoch:
        args.check_subject = True
        args.check_epoch = True

    config_path = Path(args.config_path)
    config = load_yaml(config_path)

    out_dir = build_out_dir(Path(args.out_root), args.run_name)

    report = {
        "condition": args.condition,
        "config_path": str(config_path),
        "eeg_config": config.get("eeg", {}),
    }

    if args.check_subject:
        report["subject"] = check_subject_pipeline(args, config, out_dir)

    if args.check_epoch:
        report["epoch"] = check_epoch_pipeline(args, config, out_dir)

    report_path = out_dir / "validation_report.json"
    save_json(report_path, report)

    print("Saved validation outputs to:", out_dir)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
