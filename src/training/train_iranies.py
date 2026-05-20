from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.data.build_iranies import EEGDataset_ADHD, create_kfold_dataloaders_
from src.models.eegnet import EEGNet

try:
    from src.utils.visualization import plot_fold_training_history
except Exception:
    plot_fold_training_history = None


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
    "lr": 1e-4,
    "weight_decay": 1e-4,
}

DEFAULT_SPLIT_SEEDS = [3407]
DEFAULT_INIT_SEEDS = [2025]
RANDOM_SEED_COUNT = 10
RANDOM_SEED_MAX = 10000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_random_seeds(count: int = RANDOM_SEED_COUNT, max_value: int = RANDOM_SEED_MAX):
    if count < 1:
        raise ValueError("count must be >= 1")
    return sorted(random.sample(range(0, max_value + 1), k=count))


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


def standardize_eeg(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def compact_subjects(subjects: list[str], max_items: int | None = None) -> str:
    subjects = [str(s) for s in subjects]
    if max_items is None or len(subjects) <= max_items:
        return "[" + ", ".join(subjects) + "]"
    head = subjects[:max_items]
    return "[" + ", ".join(head) + f", ... +{len(subjects) - max_items} more]"


def get_split_subjects_and_labels(dataset: EEGDataset_ADHD, loader):
    indices = loader.dataset.indices
    subjects = [str(dataset.samples[i]["subject_id"]) for i in indices]
    labels = [label_to_int(dataset.samples[i]["label"]) for i in indices]
    return subjects, labels


def get_class_weights(dataset: EEGDataset_ADHD, train_loader, n_classes: int, device: torch.device):
    _, train_labels = get_split_subjects_and_labels(dataset, train_loader)
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = []
    for class_idx in range(n_classes):
        class_count = counts.get(class_idx, 0)
        weights.append(0.0 if class_count == 0 else total / (n_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(targets, preds, n_classes: int, total_loss: float | None = None, loss_denom: float | None = None, extra: dict | None = None):
    labels = list(range(n_classes))
    n = max(len(targets), 1)
    metrics = {
        "acc": accuracy_score(targets, preds),
        "balanced_acc": balanced_accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, labels=labels, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),
        "target_counts": dict(Counter(int(v) for v in targets)),
        "pred_counts": dict(Counter(int(v) for v in preds)),
    }
    if total_loss is not None:
        denom = float(loss_denom if loss_denom is not None else n)
        metrics["loss"] = total_loss / max(denom, 1.0)
    if extra:
        metrics.update(extra)
    return metrics


def print_metrics(prefix: str, metrics: dict):
    loss = metrics.get("loss", None)
    loss_text = "" if loss is None else f"loss={loss:.4f} | "
    temporal_text = "" if "temporal_acc" not in metrics else f" | temp_acc={metrics['temporal_acc']:.4f}"
    print(
        f"{prefix}: {loss_text}"
        f"acc={metrics['acc']:.4f} | "
        f"bacc={metrics['balanced_acc']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"pred={metrics['pred_counts']}"
        f"{temporal_text}"
    )


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


def build_model(args, n_channels: int, n_classes: int, device: torch.device):
    model = EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=FINAL_CONFIG["F1"],
        D=FINAL_CONFIG["D"],
        F2=FINAL_CONFIG["F2"],
        temporal_kern=FINAL_CONFIG["temporal_kern"],
        separable_kern=FINAL_CONFIG["separable_kern"],
        pool1=FINAL_CONFIG["pool1"],
        pool2=FINAL_CONFIG["pool2"],
        dropout=args.dropout,
        meanmax_alpha=0.0,  # no se usa; phi se evalúa explícitamente hasta A_s.
    )
    return model.to(device)


def phi_to_temporal_logits(model: EEGNet, x: torch.Tensor) -> dict:
    """
    Forward explícito sin máscaras y sin agregación.

        phi(X_s) = A_s

    X_s:       (B, C, T)
    H_s:       (B, T', F2)  diagnóstico interno
    A_s:       (B, T', L)   logits temporales
    P_s:       (B, T', L)   softmax sobre L
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x=(B,C,T), got {tuple(x.shape)}")
    if x.shape[1] != model.n_channels:
        raise ValueError(f"Expected {model.n_channels} channels, got {x.shape[1]}")

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


def temporal_cross_entropy_loss(A: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
    if A.ndim != 3:
        raise ValueError(f"Expected A=(B,T',L), got {tuple(A.shape)}")
    B, T_prime, L = A.shape
    y_time = make_temporal_targets(y, T_prime)
    loss = criterion(A.reshape(B * T_prime, L), y_time.reshape(B * T_prime))
    with torch.no_grad():
        pred_time = A.argmax(dim=-1)
        temporal_correct = int((pred_time == y_time).sum().item())
        temporal_count = int(B * T_prime)
    return loss, {"y_time": y_time, "temporal_correct": temporal_correct, "temporal_count": temporal_count}


@torch.no_grad()
def subject_predictions_from_A(A: torch.Tensor, n_classes: int):
    """
    A=(B,T',L) -> decisión por sujeto.

    Regla:
      1. argmax_L para cada t'
      2. voto mayoritario sobre T'
      3. empate: mayor probabilidad media temporal; si persiste, clase menor.
    """
    if A.ndim != 3:
        raise ValueError(f"Expected A=(B,T',L), got {tuple(A.shape)}")
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
    )


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device, n_classes: int, standardize: bool, grad_clip: float | None):
    model.train()
    total_loss = 0.0
    loss_denom = 0.0
    temporal_correct = 0
    temporal_count = 0
    all_targets = []
    all_preds = []

    for batch in loader:
        x = batch["X"].to(device)
        y = batch["y"].to(device)
        if standardize:
            x = standardize_eeg(x)

        outputs = phi_to_temporal_logits(model, x)
        A = outputs["A"]
        loss, loss_info = temporal_cross_entropy_loss(A, y, criterion)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        preds, _, _, _ = subject_predictions_from_A(A.detach(), n_classes=n_classes)

        total_loss += float(loss.item()) * max(loss_info["temporal_count"], 1)
        loss_denom += max(loss_info["temporal_count"], 1)
        temporal_correct += int(loss_info["temporal_correct"])
        temporal_count += int(loss_info["temporal_count"])
        all_targets.extend(y.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return compute_metrics(
        all_targets,
        all_preds,
        n_classes=n_classes,
        total_loss=total_loss,
        loss_denom=loss_denom,
        extra={"temporal_acc": temporal_correct / max(temporal_count, 1)},
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, n_classes: int, standardize: bool, collect_predictions: bool = False, split_name: str | None = None, split_seed: int | None = None, init_seed: int | None = None, fold_id: int | None = None, config_name: str | None = None):
    model.eval()
    total_loss = 0.0
    loss_denom = 0.0
    temporal_correct = 0
    temporal_count = 0
    all_targets = []
    all_preds = []
    prediction_rows = []

    for batch in loader:
        x = batch["X"].to(device)
        y = batch["y"].to(device)
        if standardize:
            x = standardize_eeg(x)

        outputs = phi_to_temporal_logits(model, x)
        A = outputs["A"]
        loss, loss_info = temporal_cross_entropy_loss(A, y, criterion)
        preds, mean_probs, vote_counts, vote_props = subject_predictions_from_A(A, n_classes=n_classes)

        total_loss += float(loss.item()) * max(loss_info["temporal_count"], 1)
        loss_denom += max(loss_info["temporal_count"], 1)
        temporal_correct += int(loss_info["temporal_correct"])
        temporal_count += int(loss_info["temporal_count"])

        y_cpu = y.detach().cpu().tolist()
        pred_cpu = preds.detach().cpu().tolist()
        mean_probs_cpu = mean_probs.detach().cpu().numpy()
        vote_counts_cpu = vote_counts.detach().cpu().numpy()
        vote_props_cpu = vote_props.detach().cpu().numpy()

        all_targets.extend(y_cpu)
        all_preds.extend(pred_cpu)

        if collect_predictions:
            for i, subject_id in enumerate(batch["subject_id"]):
                row = {
                    "config": config_name,
                    "split_seed": split_seed,
                    "init_seed": init_seed,
                    "fold": fold_id,
                    "split": split_name,
                    "subject_id": str(subject_id),
                    "y_true": int(y_cpu[i]),
                    "y_pred": int(pred_cpu[i]),
                    "correct": int(y_cpu[i] == pred_cpu[i]),
                    "length": int(batch["lengths"][i].item()),
                    "T_prime": int(A.shape[1]),
                    "decision_rule": "majority_vote_over_argmax_A_s",
                    "aggregation": "none",
                    "mask_used": False,
                }
                for class_idx in range(n_classes):
                    row[f"meanprob_class_{class_idx}"] = float(mean_probs_cpu[i, class_idx])
                    row[f"votes_class_{class_idx}"] = int(vote_counts_cpu[i, class_idx])
                    row[f"vote_prop_class_{class_idx}"] = float(vote_props_cpu[i, class_idx])
                prediction_rows.append(row)

    metrics = compute_metrics(
        all_targets,
        all_preds,
        n_classes=n_classes,
        total_loss=total_loss,
        loss_denom=loss_denom,
        extra={"temporal_acc": temporal_correct / max(temporal_count, 1)},
    )
    return metrics, prediction_rows


def inspect_first_batch(model: EEGNet, loader, device: torch.device, standardize: bool):
    batch = next(iter(loader))
    x = batch["X"].to(device)
    y = batch["y"].to(device)
    if standardize:
        x = standardize_eeg(x)

    model.eval()
    with torch.no_grad():
        outputs = phi_to_temporal_logits(model, x)
        A = outputs["A"]
        y_time = make_temporal_targets(y, A.shape[1])
        loss, _ = temporal_cross_entropy_loss(A, y, nn.CrossEntropyLoss())

    B, C, T = x.shape
    T_prime_expected = expected_t_prime(T, model.pool1, model.pool2)

    print("\nShape check")
    print(f"  X_s            : {tuple(x.shape)}")
    print(f"  H_s            : {tuple(outputs['H'].shape)}")
    print(f"  A_s logits     : {tuple(outputs['A'].shape)}")
    print(f"  softmax(A_s)   : {tuple(outputs['P'].shape)}")
    print(f"  y_time         : {tuple(y_time.shape)}")
    print(f"  T_prime        : {T_prime_expected}")
    print(f"  CE temporal    : {float(loss):.6f}")

    assert tuple(outputs["H"].shape) == (B, T_prime_expected, model.F2)
    assert tuple(outputs["A"].shape) == (B, T_prime_expected, model.n_classes)
    assert tuple(outputs["P"].shape) == (B, T_prime_expected, model.n_classes)
    assert tuple(y_time.shape) == (B, T_prime_expected)


def print_fold_split(fold_id: int, train_subjects, train_labels, val_subjects, val_labels, test_subjects, test_labels, max_ids: int | None):
    print(f"\nFold {fold_id} split")
    print(f"  train: n={len(train_subjects):3d} | labels={dict(Counter(train_labels))} | subjects={compact_subjects(sorted(train_subjects), max_ids)}")
    print(f"  val:   n={len(val_subjects):3d} | labels={dict(Counter(val_labels))} | subjects={compact_subjects(sorted(val_subjects), max_ids)}")
    print(f"  test:  n={len(test_subjects):3d} | labels={dict(Counter(test_labels))} | subjects={compact_subjects(sorted(test_subjects), max_ids)}")


def run_one_training(dataset, train_loader, val_loader, test_loader, args, device: torch.device, n_channels: int, n_classes: int, split_seed: int, init_seed: int, fold_idx: int, config_name: str):
    set_seed(init_seed)
    fold_id = fold_idx + 1

    train_subjects, train_labels = get_split_subjects_and_labels(dataset, train_loader)
    val_subjects, val_labels = get_split_subjects_and_labels(dataset, val_loader)
    test_subjects, test_labels = get_split_subjects_and_labels(dataset, test_loader)

    if set(train_subjects) & set(val_subjects) or set(train_subjects) & set(test_subjects) or set(val_subjects) & set(test_subjects):
        raise RuntimeError("Subject leakage detected inside this fold.")

    print_fold_split(
        fold_id,
        train_subjects,
        train_labels,
        val_subjects,
        val_labels,
        test_subjects,
        test_labels,
        None if args.full_subject_ids else args.max_subject_ids_in_log,
    )

    standardize = not args.no_standardize
    model = build_model(args, n_channels=n_channels, n_classes=n_classes, device=device)
    class_weights = None if args.no_class_weights else get_class_weights(dataset, train_loader, n_classes=n_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("  config:", config_name)
    print(f"  loss: temporal CE on A_s | decision: majority vote | aggregation: none | mask: none")
    print(f"  lr={args.lr} | wd={args.weight_decay} | dropout={args.dropout} | standardize={standardize} | class_weights={None if class_weights is None else class_weights.detach().cpu().tolist()}")

    if args.inspect_shapes and fold_idx == (args.folds[0] if args.folds else 0):
        inspect_first_batch(model=model, loader=train_loader, device=device, standardize=standardize)

    best_state_dict = None
    best_epoch = 0
    best_val_bacc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, n_classes, standardize, args.grad_clip)
        val_metrics, _ = evaluate(model, val_loader, criterion, device, n_classes, standardize)

        history_rows.append({
            "config": config_name,
            "split_seed": split_seed,
            "init_seed": init_seed,
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_balanced_acc": train_metrics["balanced_acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_temporal_acc": train_metrics.get("temporal_acc"),
            "train_pred_counts": json.dumps(train_metrics["pred_counts"]),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_temporal_acc": val_metrics.get("temporal_acc"),
            "val_pred_counts": json.dumps(val_metrics["pred_counts"]),
        })

        should_print_epoch = (epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0)
        improved = (
            val_metrics["balanced_acc"] > best_val_bacc
            or (val_metrics["balanced_acc"] == best_val_bacc and val_metrics["loss"] < best_val_loss)
        )

        if improved:
            best_val_bacc = val_metrics["balanced_acc"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if should_print_epoch or improved:
            mark = "*" if improved else " "
            print(
                f"  {mark}epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_bacc={train_metrics['balanced_acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_bacc={val_metrics['balanced_acc']:.4f} | "
                f"val_pred={val_metrics['pred_counts']}"
            )

        if args.patience is not None and patience_counter >= args.patience:
            print(f"  early stopping at epoch {epoch}")
            break

    if best_state_dict is None:
        raise RuntimeError("No best model state was stored.")

    model.load_state_dict(best_state_dict)
    train_best_metrics, train_pred_rows = evaluate(model, train_loader, criterion, device, n_classes, standardize, True, "train", split_seed, init_seed, fold_id, config_name)
    val_best_metrics, val_pred_rows = evaluate(model, val_loader, criterion, device, n_classes, standardize, True, "val", split_seed, init_seed, fold_id, config_name)
    test_metrics, test_pred_rows = evaluate(model, test_loader, criterion, device, n_classes, standardize, True, "test", split_seed, init_seed, fold_id, config_name)

    print(
        f"  best_epoch={best_epoch} | "
        f"train_bacc={train_best_metrics['balanced_acc']:.4f} | "
        f"val_bacc={val_best_metrics['balanced_acc']:.4f} | "
        f"test_bacc={test_metrics['balanced_acc']:.4f} | "
        f"test_acc={test_metrics['acc']:.4f} | "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} | "
        f"test_pred={test_metrics['pred_counts']}"
    )

    run_dir = Path(args.save_dir) / config_name / f"splitseed_{split_seed}" / f"initseed_{init_seed}" / f"fold_{fold_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.csv"
    predictions_path = run_dir / "predictions_train_val_test.csv"
    checkpoint_path = run_dir / "best_model.pt"

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    pd.DataFrame(train_pred_rows + val_pred_rows + test_pred_rows).to_csv(predictions_path, index=False)

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "config": copy.deepcopy(FINAL_CONFIG),
            "args": vars(args),
            "split_seed": split_seed,
            "init_seed": init_seed,
            "fold": fold_id,
            "best_epoch": best_epoch,
            "best_val_bacc": best_val_bacc,
            "best_val_loss": best_val_loss,
            "n_channels": n_channels,
            "n_classes": n_classes,
            "train_subjects": sorted(train_subjects),
            "val_subjects": sorted(val_subjects),
            "test_subjects": sorted(test_subjects),
            "decision_rule": "majority_vote_over_argmax_A_s",
            "aggregation": "none",
            "mask_used": False,
        },
        checkpoint_path,
    )

    summary_row = {
        "config": config_name,
        "split_seed": split_seed,
        "init_seed": init_seed,
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_bacc": best_val_bacc,
        "best_val_loss": best_val_loss,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "n_samples": int(dataset.samples[0]["eeg"].shape[1]),
        "T_prime": int(expected_t_prime(int(dataset.samples[0]["eeg"].shape[1]), FINAL_CONFIG["pool1"], FINAL_CONFIG["pool2"])),
        "F1": FINAL_CONFIG["F1"],
        "D": FINAL_CONFIG["D"],
        "F2": FINAL_CONFIG["F2"],
        "pool1": FINAL_CONFIG["pool1"],
        "pool2": FINAL_CONFIG["pool2"],
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "standardize": standardize,
        "class_weights": None if class_weights is None else json.dumps(class_weights.detach().cpu().tolist()),
        "train_subject_count": len(train_subjects),
        "val_subject_count": len(val_subjects),
        "test_subject_count": len(test_subjects),
        "train_subjects": json.dumps(sorted(train_subjects)),
        "val_subjects": json.dumps(sorted(val_subjects)),
        "test_subjects": json.dumps(sorted(test_subjects)),
        "train_acc": train_best_metrics["acc"],
        "train_bacc": train_best_metrics["balanced_acc"],
        "train_macro_f1": train_best_metrics["macro_f1"],
        "train_temporal_acc": train_best_metrics.get("temporal_acc"),
        "val_acc": val_best_metrics["acc"],
        "val_bacc": val_best_metrics["balanced_acc"],
        "val_macro_f1": val_best_metrics["macro_f1"],
        "val_temporal_acc": val_best_metrics.get("temporal_acc"),
        "test_acc": test_metrics["acc"],
        "test_bacc": test_metrics["balanced_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "test_temporal_acc": test_metrics.get("temporal_acc"),
        "test_pred_counts": json.dumps(test_metrics["pred_counts"]),
        "test_target_counts": json.dumps(test_metrics["target_counts"]),
        "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        "history_path": str(history_path),
        "predictions_path": str(predictions_path),
        "checkpoint_path": str(checkpoint_path),
        "decision_rule": "majority_vote_over_argmax_A_s",
        "aggregation": "none",
        "mask_used": False,
    }
    return summary_row, history_rows


def summarize_results(summary_df: pd.DataFrame, save_dir: Path, config_name: str):
    summary_path = save_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grouped_overall = pd.DataFrame([
        {
            "config": config_name,
            "n_runs": len(summary_df),
            "mean_test_bacc": summary_df["test_bacc"].mean(),
            "std_test_bacc": summary_df["test_bacc"].std(),
            "median_test_bacc": summary_df["test_bacc"].median(),
            "min_test_bacc": summary_df["test_bacc"].min(),
            "max_test_bacc": summary_df["test_bacc"].max(),
            "mean_test_acc": summary_df["test_acc"].mean(),
            "std_test_acc": summary_df["test_acc"].std(),
            "mean_test_macro_f1": summary_df["test_macro_f1"].mean(),
            "std_test_macro_f1": summary_df["test_macro_f1"].std(),
            "mean_test_temporal_acc": summary_df["test_temporal_acc"].mean(),
            "std_test_temporal_acc": summary_df["test_temporal_acc"].std(),
            "mean_val_bacc": summary_df["val_bacc"].mean(),
            "std_val_bacc": summary_df["val_bacc"].std(),
            "mean_best_epoch": summary_df["best_epoch"].mean(),
            "decision_rule": "majority_vote_over_argmax_A_s",
            "aggregation": "none",
            "mask_used": False,
        }
    ])
    grouped_overall_path = save_dir / "summary_overall.csv"
    grouped_overall.to_csv(grouped_overall_path, index=False)

    grouped_fold = (
        summary_df.groupby(["split_seed", "init_seed", "fold"])
        .agg(
            n_runs=("test_bacc", "count"),
            mean_test_bacc=("test_bacc", "mean"),
            mean_test_acc=("test_acc", "mean"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            mean_test_temporal_acc=("test_temporal_acc", "mean"),
        )
        .reset_index()
    )
    grouped_fold_path = save_dir / "summary_by_fold.csv"
    grouped_fold.to_csv(grouped_fold_path, index=False)

    print("\nSummary by fold")
    cols = ["fold", "best_epoch", "val_bacc", "test_bacc", "test_acc", "test_macro_f1", "test_temporal_acc", "test_pred_counts"]
    print(summary_df[cols].to_string(index=False))

    print("\nSummary overall")
    print(grouped_overall.to_string(index=False))

    print("\nSaved files")
    print(f"  summary:          {summary_path}")
    print(f"  summary_overall:  {grouped_overall_path}")
    print(f"  summary_by_fold:  {grouped_fold_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adhd-dir", type=str, default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "ADHD"))
    parser.add_argument("--control-dir", type=str, default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "Control"))
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

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=FINAL_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=FINAL_CONFIG["weight_decay"])
    parser.add_argument("--dropout", type=float, default=FINAL_CONFIG["dropout"])
    parser.add_argument("--grad-clip", type=parse_optional_float, default=1.0)
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
    parser.add_argument("--no-class-weights", action="store_true")

    parser.add_argument("--inspect-shapes", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--max-subject-ids-in-log", type=int, default=12)
    parser.add_argument("--full-subject-ids", action="store_true")
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "iraniesdataset"))

    args = parser.parse_args()

    if args.rand_split_seed:
        args.split_seeds = generate_random_seeds(args.n_rand_split_seeds, args.seed_max)
    if args.rand_init_seed:
        args.init_seeds = generate_random_seeds(args.n_rand_init_seeds, args.seed_max)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Project root: {PROJECT_ROOT}")

    dataset = build_dataset(args)
    labels = [label_to_int(v) for v in dataset.get_labels()]
    n_classes = FINAL_CONFIG["n_classes"]
    n_channels = int(dataset.samples[0]["eeg"].shape[0])
    n_samples = int(dataset.samples[0]["eeg"].shape[1])
    t_prime = expected_t_prime(n_samples, FINAL_CONFIG["pool1"], FINAL_CONFIG["pool2"])
    config_name = make_config_name(n_samples=n_samples, t_prime=t_prime)

    save_dir = Path(args.save_dir) / config_name
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset.get_summary_dataframe().to_csv(save_dir / "dataset_summary.csv", index=False)

    print("\nDataset")
    print(f"  subjects={len(dataset)} | labels={dict(Counter(labels))} | C={n_channels} | T={n_samples} | T_prime={t_prime}")
    print(f"  X_s: R^{{{n_channels} x {n_samples}}} | A_s: R^{{{t_prime} x {n_classes}}}")
    print(f"  crop_policy={'dataset_min' if args.duration_sec is None else 'requested_duration'} | target_fs={args.target_fs}")
    print(f"  save_dir={save_dir}")
    print("  aggregation=none | mask=none")

    print("\nExecution")
    print(f"  split_seeds={args.split_seeds} | init_seeds={args.init_seeds} | folds={args.folds if args.folds is not None else list(range(args.k))}")

    all_summary_rows = []
    all_history_rows = []

    for split_seed in args.split_seeds:
        print(f"\nCreating folds with split_seed={split_seed}")
        set_seed(split_seed)
        folds = create_kfold_dataloaders_(
            dataset,
            k=args.k,
            batch_size=args.batch_size,
            shuffle=True,
            split_seed=split_seed,
            inner_splits=args.inner_splits,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        selected_folds = args.folds if args.folds is not None else list(range(len(folds)))
        for fold_idx in selected_folds:
            if fold_idx < 0 or fold_idx >= len(folds):
                raise ValueError(f"Fold index must be in [0,{len(folds)-1}], got {fold_idx}")

        if args.validate_only:
            train_loader, val_loader, test_loader = folds[selected_folds[0]]
            train_subjects, train_labels = get_split_subjects_and_labels(dataset, train_loader)
            val_subjects, val_labels = get_split_subjects_and_labels(dataset, val_loader)
            test_subjects, test_labels = get_split_subjects_and_labels(dataset, test_loader)
            print_fold_split(1 + selected_folds[0], train_subjects, train_labels, val_subjects, val_labels, test_subjects, test_labels, None if args.full_subject_ids else args.max_subject_ids_in_log)
            model = build_model(args, n_channels, n_classes, device)
            inspect_first_batch(model, train_loader, device, standardize=not args.no_standardize)
            print("\nvalidate-only finished successfully.")
            return

        for fold_idx in selected_folds:
            train_loader, val_loader, test_loader = folds[fold_idx]
            for init_seed in args.init_seeds:
                summary_row, history_rows = run_one_training(
                    dataset=dataset,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    args=args,
                    device=device,
                    n_channels=n_channels,
                    n_classes=n_classes,
                    split_seed=split_seed,
                    init_seed=init_seed,
                    fold_idx=fold_idx,
                    config_name=config_name,
                )
                all_summary_rows.append(summary_row)
                all_history_rows.extend(history_rows)
                pd.DataFrame(all_summary_rows).to_csv(save_dir / "summary_partial.csv", index=False)
                pd.DataFrame(all_history_rows).to_csv(save_dir / "history_all_runs_partial.csv", index=False)

    summary_df = pd.DataFrame(all_summary_rows)
    history_all_df = pd.DataFrame(all_history_rows)
    history_all_df.to_csv(save_dir / "history_all_runs.csv", index=False)
    summarize_results(summary_df, save_dir, config_name)

    if plot_fold_training_history is not None:
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_fold_training_history(history_csv=save_dir / "history_all_runs.csv", output_dir=plots_dir)
        except Exception as exc:
            print(f"WARNING: no pude generar plots: {exc}")


if __name__ == "__main__":
    main()
