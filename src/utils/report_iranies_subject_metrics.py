from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from src.data.build_iranies import EEGDataset_ADHD, create_kfold_dataloaders_
from src.models.eegnet import EEGNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PREFIX = "eegnet_iranies_tmin_temporal_ce_clean"

FINAL_CONFIG = {
    "name": MODEL_PREFIX,
    "n_classes": 2,
    "F1": 8,
    "D": 2,
    "F2": 16,
    "temporal_kern": 63,
    "separable_kern": 15,
    "pool1": 8,
    "pool2": 8,
    "dropout": 0.5,
}

DEFAULT_SPLIT_SEEDS = [3407]
DEFAULT_INIT_SEEDS = [2025]


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def label_to_int(value):
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def expected_t_prime(T: int, pool1: int, pool2: int) -> int:
    return (T // pool1) // pool2


def make_config_name(n_samples: int, t_prime: int) -> str:
    return f"{MODEL_PREFIX}_t{int(n_samples)}_tp{int(t_prime)}"


def compact_subjects(subjects: list[str], max_items: int | None = None) -> str:
    subjects = [str(s) for s in subjects]
    if max_items is None or len(subjects) <= max_items:
        return "[" + ", ".join(subjects) + "]"
    head = subjects[:max_items]
    return "[" + ", ".join(head) + f", ... +{len(subjects) - max_items} more]"


def standardize_eeg(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def build_dataset(args):
    return EEGDataset_ADHD(
        adhd_dir=args.adhd_dir,
        control_dir=args.control_dir,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        default_fs=args.default_fs,
        duration_sec=args.duration_sec,
        crop_from=args.crop_from,
        max_channels=args.max_channels,
        scale=args.scale,
        apply_reference=not args.no_reference,
        apply_notch=not args.no_notch,
        apply_filter=not args.no_filter,
        apply_resample=not args.no_resample,
    )


def build_model_from_checkpoint(args, n_channels: int, n_classes: int, device: torch.device):
    cfg = getattr(args, "checkpoint_config", None) or {}
    model = EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=int(cfg.get("F1", FINAL_CONFIG["F1"])),
        D=int(cfg.get("D", FINAL_CONFIG["D"])),
        F2=int(cfg.get("F2", FINAL_CONFIG["F2"])),
        temporal_kern=int(cfg.get("temporal_kern", FINAL_CONFIG["temporal_kern"])),
        separable_kern=int(cfg.get("separable_kern", FINAL_CONFIG["separable_kern"])),
        pool1=int(cfg.get("pool1", FINAL_CONFIG["pool1"])),
        pool2=int(cfg.get("pool2", FINAL_CONFIG["pool2"])),
        dropout=float(cfg.get("dropout", FINAL_CONFIG["dropout"])),
        meanmax_alpha=0.0,
    )
    return model.to(device)


def phi_to_temporal_logits(model: EEGNet, x: torch.Tensor) -> dict:
    """Forward explícito: phi(X_s)=A_s, sin agregación y sin máscaras."""
    if x.ndim != 3:
        raise ValueError(f"Expected x=(B,C,T), got {tuple(x.shape)}")
    x4 = x.unsqueeze(1)
    z_temporal = model.temporal_block(x4)
    z_spatial = model.spatial_block(z_temporal)
    z_separable = model.separable_block(z_spatial)
    H = z_separable.squeeze(2).permute(0, 2, 1).contiguous()
    logits = model.classifier(z_separable)
    A = logits.squeeze(2).permute(0, 2, 1).contiguous()
    P = torch.softmax(A, dim=-1)
    return {"H": H, "A": A, "P": P}


def make_temporal_targets(y: torch.Tensor, T_prime: int) -> torch.Tensor:
    return y.unsqueeze(1).expand(-1, T_prime).contiguous()


@torch.no_grad()
def subject_predictions_from_A(A: torch.Tensor, n_classes: int):
    B, T_prime, L = A.shape
    if L != n_classes:
        raise ValueError(f"Expected L={n_classes}, got {L}")

    P = torch.softmax(A, dim=-1)
    pred_time = A.argmax(dim=-1)

    preds = []
    mean_probs_rows = []
    vote_counts_rows = []
    vote_props_rows = []

    for i in range(B):
        counts = torch.bincount(pred_time[i], minlength=n_classes).float()
        props = counts / counts.sum().clamp_min(1.0)
        mean_probs = P[i].mean(dim=0)

        max_count = counts.max()
        tied = torch.where(counts == max_count)[0]
        if tied.numel() > 1:
            tied_probs = mean_probs[tied]
            tied = tied[torch.where(tied_probs == tied_probs.max())[0]]
        pred_subject = int(tied.min().item())

        preds.append(pred_subject)
        mean_probs_rows.append(mean_probs)
        vote_counts_rows.append(counts)
        vote_props_rows.append(props)

    return (
        torch.tensor(preds, dtype=torch.long, device=A.device),
        torch.stack(mean_probs_rows, dim=0),
        torch.stack(vote_counts_rows, dim=0),
        torch.stack(vote_props_rows, dim=0),
        pred_time,
    )


def compute_subject_metrics(targets, preds, n_classes: int):
    labels = list(range(n_classes))
    return {
        "acc": accuracy_score(targets, preds),
        "balanced_acc": balanced_accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, labels=labels, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),
        "target_counts": dict(Counter(int(v) for v in targets)),
        "pred_counts": dict(Counter(int(v) for v in preds)),
    }


@torch.no_grad()
def evaluate_test_loader(model, loader, device: torch.device, n_classes: int, standardize: bool, config_name: str, split_seed: int, init_seed: int, fold_id: int, inspect_shapes: bool = False):
    model.eval()
    rows = []
    all_targets = []
    all_preds = []
    temporal_correct = 0
    temporal_count = 0

    printed_shape = False

    for batch_idx, batch in enumerate(loader):
        x = batch["X"].to(device)
        y = batch["y"].to(device)
        if standardize:
            x = standardize_eeg(x)

        outputs = phi_to_temporal_logits(model, x)
        A = outputs["A"]
        P = outputs["P"]
        B, T_prime, L = A.shape
        y_time = make_temporal_targets(y, T_prime)

        preds, mean_probs, vote_counts, vote_props, pred_time = subject_predictions_from_A(A, n_classes=n_classes)

        temporal_correct += int((pred_time == y_time).sum().item())
        temporal_count += int(B * T_prime)

        if inspect_shapes and not printed_shape:
            print("  shape: X_s={} | H_s={} | A_s={} | softmax(A_s)={} | y_time={}".format(
                tuple(x.shape), tuple(outputs["H"].shape), tuple(A.shape), tuple(P.shape), tuple(y_time.shape)
            ))
            printed_shape = True

        y_cpu = y.detach().cpu().tolist()
        pred_cpu = preds.detach().cpu().tolist()
        mean_probs_cpu = mean_probs.detach().cpu().numpy()
        vote_counts_cpu = vote_counts.detach().cpu().numpy()
        vote_props_cpu = vote_props.detach().cpu().numpy()

        all_targets.extend(y_cpu)
        all_preds.extend(pred_cpu)

        for i, subject_id in enumerate(batch["subject_id"]):
            row = {
                "config": config_name,
                "split_seed": split_seed,
                "init_seed": init_seed,
                "fold": fold_id,
                "split": "test",
                "subject_id": str(subject_id),
                "y_true": int(y_cpu[i]),
                "y_pred": int(pred_cpu[i]),
                "correct": int(y_cpu[i] == pred_cpu[i]),
                "length": int(batch["lengths"][i].item()),
                "T_prime": int(T_prime),
                "decision_rule": "majority_vote_over_argmax_A_s",
                "aggregation": "none",
                "mask_used": False,
            }
            for class_idx in range(n_classes):
                row[f"meanprob_class_{class_idx}"] = float(mean_probs_cpu[i, class_idx])
                row[f"votes_class_{class_idx}"] = int(vote_counts_cpu[i, class_idx])
                row[f"vote_prop_class_{class_idx}"] = float(vote_props_cpu[i, class_idx])
            rows.append(row)

    metrics = compute_subject_metrics(all_targets, all_preds, n_classes=n_classes)
    metrics["temporal_acc"] = temporal_correct / max(temporal_count, 1)
    metrics["temporal_count"] = temporal_count
    metrics["n_test_subjects"] = len(all_targets)
    return metrics, rows


def get_split_subjects_and_labels(dataset: EEGDataset_ADHD, loader):
    indices = loader.dataset.indices
    subjects = [str(dataset.samples[i]["subject_id"]) for i in indices]
    labels = [label_to_int(dataset.samples[i]["label"]) for i in indices]
    return subjects, labels


def assert_no_leakage(train_subjects, val_subjects, test_subjects):
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Subject leakage detected between train/val/test.")


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def print_subject_votes(rows: list[dict], n_classes: int):
    print("  subject votes")
    header = "    subject  y_true  y_pred  correct  " + "  ".join([f"votes_{c}" for c in range(n_classes)]) + "  " + "  ".join([f"prop_{c}" for c in range(n_classes)])
    print(header)
    for row in sorted(rows, key=lambda r: str(r["subject_id"])):
        votes = "  ".join([f"{int(row[f'votes_class_{c}']):7d}" for c in range(n_classes)])
        props = "  ".join([f"{float(row[f'vote_prop_class_{c}']):6.3f}" for c in range(n_classes)])
        print(
            f"    {row['subject_id']:>7s}  "
            f"{int(row['y_true']):6d}  "
            f"{int(row['y_pred']):6d}  "
            f"{int(row['correct']):7d}  "
            f"{votes}  {props}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adhd-dir", type=str, default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "ADHD"))
    parser.add_argument("--control-dir", type=str, default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "Control"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--init-seeds", type=int, nargs="+", default=DEFAULT_INIT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--target-fs", type=float, default=128.0)
    parser.add_argument("--default-fs", type=float, default=128.0)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)
    parser.add_argument("--crop-from", type=str, default="start", choices=["start", "center"])
    parser.add_argument("--max-channels", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)

    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-notch", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--inspect-shapes", action="store_true")
    parser.add_argument("--print-subject-votes", action="store_true", help="Imprime votos por sujeto en consola. Siempre se guardan en CSV.")
    parser.add_argument("--max-subject-ids-in-log", type=int, default=12)
    parser.add_argument("--full-subject-ids", action="store_true")

    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "iraniesdataset"))
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Carpeta del experimento. Si se omite, se infiere por T y T'.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Project root: {PROJECT_ROOT}")

    dataset = build_dataset(args)
    n_classes = FINAL_CONFIG["n_classes"]
    n_channels = int(dataset.samples[0]["eeg"].shape[0])
    n_samples = int(dataset.samples[0]["eeg"].shape[1])
    t_prime = expected_t_prime(n_samples, FINAL_CONFIG["pool1"], FINAL_CONFIG["pool2"])
    config_name = make_config_name(n_samples=n_samples, t_prime=t_prime)

    if args.checkpoint_root is None:
        experiment_dir = Path(args.save_dir) / config_name
    else:
        experiment_dir = Path(args.checkpoint_root)
        config_name = experiment_dir.name

    reports_dir = experiment_dir / "reports_subject_majority_clean"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset.get_summary_dataframe().to_csv(reports_dir / "dataset_summary_used_for_report.csv", index=False)

    print("\nReport configuration")
    print(f"  dataset: N={len(dataset)} | C={n_channels} | T={n_samples} | T_prime={t_prime} | classes={n_classes}")
    print(f"  experiment_dir={experiment_dir}")
    print("  decision=majority vote over argmax_L A_s | aggregation=none | mask=none")

    all_rows = []
    summary_rows = []

    for split_seed in args.split_seeds:
        print(f"\nRecreating folds with split_seed={split_seed}")
        folds = create_kfold_dataloaders_(
            dataset,
            k=args.k,
            batch_size=args.batch_size,
            shuffle=False,
            split_seed=split_seed,
            inner_splits=args.inner_splits,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        selected_folds = args.folds if args.folds is not None else list(range(len(folds)))
        seen_test_subjects = []

        for fold_idx in selected_folds:
            if fold_idx < 0 or fold_idx >= len(folds):
                raise ValueError(f"Fold index must be in [0,{len(folds)-1}], got {fold_idx}")

            fold_id = fold_idx + 1
            train_loader, val_loader, test_loader = folds[fold_idx]
            train_subjects, train_labels = get_split_subjects_and_labels(dataset, train_loader)
            val_subjects, val_labels = get_split_subjects_and_labels(dataset, val_loader)
            test_subjects, test_labels = get_split_subjects_and_labels(dataset, test_loader)
            assert_no_leakage(train_subjects, val_subjects, test_subjects)
            seen_test_subjects.extend(test_subjects)

            max_ids = None if args.full_subject_ids else args.max_subject_ids_in_log
            print(f"\nFold {fold_id}")
            print(f"  train: n={len(train_subjects):3d} | labels={dict(Counter(train_labels))} | subjects={compact_subjects(sorted(train_subjects), max_ids)}")
            print(f"  val:   n={len(val_subjects):3d} | labels={dict(Counter(val_labels))} | subjects={compact_subjects(sorted(val_subjects), max_ids)}")
            print(f"  test:  n={len(test_subjects):3d} | labels={dict(Counter(test_labels))} | subjects={compact_subjects(sorted(test_subjects), max_ids)}")

            for init_seed in args.init_seeds:
                checkpoint_path = experiment_dir / f"splitseed_{split_seed}" / f"initseed_{init_seed}" / f"fold_{fold_id}" / "best_model.pt"
                checkpoint = load_checkpoint(checkpoint_path, device=device)
                args.checkpoint_config = checkpoint.get("config", {})
                model = build_model_from_checkpoint(args, n_channels=n_channels, n_classes=n_classes, device=device)
                model.load_state_dict(checkpoint["model_state_dict"])

                metrics, rows = evaluate_test_loader(
                    model=model,
                    loader=test_loader,
                    device=device,
                    n_classes=n_classes,
                    standardize=not args.no_standardize,
                    config_name=config_name,
                    split_seed=split_seed,
                    init_seed=init_seed,
                    fold_id=fold_id,
                    inspect_shapes=args.inspect_shapes,
                )

                print(
                    f"  metrics: acc={metrics['acc']:.4f} | "
                    f"bacc={metrics['balanced_acc']:.4f} | "
                    f"macro_f1={metrics['macro_f1']:.4f} | "
                    f"temporal_acc={metrics['temporal_acc']:.4f} | "
                    f"pred={metrics['pred_counts']} | cm={metrics['confusion_matrix']}"
                )

                if args.print_subject_votes:
                    print_subject_votes(rows, n_classes=n_classes)

                summary_rows.append({
                    "config": config_name,
                    "split_seed": split_seed,
                    "init_seed": init_seed,
                    "fold": fold_id,
                    "n_test_subjects": metrics["n_test_subjects"],
                    "majority_acc": metrics["acc"],
                    "majority_balanced_acc": metrics["balanced_acc"],
                    "majority_macro_f1": metrics["macro_f1"],
                    "temporal_acc": metrics["temporal_acc"],
                    "temporal_count": metrics["temporal_count"],
                    "target_counts": json.dumps(metrics["target_counts"]),
                    "pred_counts": json.dumps(metrics["pred_counts"]),
                    "majority_confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                    "checkpoint_path": str(checkpoint_path),
                    "decision_rule": "majority_vote_over_argmax_A_s",
                    "aggregation": "none",
                    "mask_used": False,
                })
                all_rows.extend(rows)

        if len(selected_folds) == args.k:
            counts = Counter(seen_test_subjects)
            bad = sorted([s for s, c in counts.items() if c != 1])
            if bad:
                raise RuntimeError(f"Test subject duplication/missing issue for split_seed={split_seed}: {bad}")
            if len(counts) != len(dataset):
                raise RuntimeError(f"Expected every subject exactly once in test across folds. Got {len(counts)} for dataset size {len(dataset)}.")

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.DataFrame(all_rows)

    summary_by_fold_path = reports_dir / "summary_subject_majority_by_fold.csv"
    predictions_path = reports_dir / "predictions_subject_votes_all.csv"
    summary_overall_path = reports_dir / "summary_subject_majority_overall.csv"

    summary_df.to_csv(summary_by_fold_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    overall = pd.DataFrame([
        {
            "config": config_name,
            "n_runs": len(summary_df),
            "total_test_subject_predictions": len(predictions_df),
            "mean_majority_acc": summary_df["majority_acc"].mean(),
            "std_majority_acc": summary_df["majority_acc"].std(),
            "mean_majority_balanced_acc": summary_df["majority_balanced_acc"].mean(),
            "std_majority_balanced_acc": summary_df["majority_balanced_acc"].std(),
            "mean_majority_macro_f1": summary_df["majority_macro_f1"].mean(),
            "std_majority_macro_f1": summary_df["majority_macro_f1"].std(),
            "mean_temporal_acc": summary_df["temporal_acc"].mean(),
            "std_temporal_acc": summary_df["temporal_acc"].std(),
            "decision_rule": "majority_vote_over_argmax_A_s",
            "aggregation": "none",
            "mask_used": False,
        }
    ])
    overall.to_csv(summary_overall_path, index=False)

    print("\nSummary by fold")
    cols = ["fold", "n_test_subjects", "majority_acc", "majority_balanced_acc", "majority_macro_f1", "temporal_acc", "pred_counts"]
    print(summary_df[cols].to_string(index=False))

    print("\nSummary overall")
    print(overall.to_string(index=False))

    print("\nSaved report files")
    print(f"  by_fold:      {summary_by_fold_path}")
    print(f"  votes:        {predictions_path}")
    print(f"  overall:      {summary_overall_path}")


if __name__ == "__main__":
    main()
