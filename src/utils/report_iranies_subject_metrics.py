from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

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

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_training_summary(experiment_dir: Path) -> pd.DataFrame:
    summary_path = experiment_dir / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            "No encontré summary.csv para elegir el mejor init_seed. "
            f"Ruta esperada: {summary_path}"
        )

    summary_df = pd.read_csv(summary_path)

    required = {
        "split_seed",
        "init_seed",
        "fold",
        "best_val_bacc",
        "best_val_loss",
        "checkpoint_path",
    }
    missing = sorted(required - set(summary_df.columns))

    if missing:
        raise ValueError(
            f"summary.csv no tiene columnas requeridas para selección de init: {missing}"
        )

    return summary_df


def select_best_init_rows(
    summary_df: pd.DataFrame,
    split_seed: int,
    selected_folds: list[int],
    candidate_init_seeds: list[int] | None = None,
) -> dict[int, dict]:
    """
    Elige un checkpoint por fold usando solo validación.

    Regla:
      1) mayor best_val_bacc
      2) menor best_val_loss
      3) menor best_epoch
      4) menor init_seed, para desempate reproducible

    selected_folds usa índices 0-based; el summary usa fold 1-based.
    """

    df = summary_df[summary_df["split_seed"].astype(int) == int(split_seed)].copy()

    if candidate_init_seeds is not None:
        df = df[df["init_seed"].astype(int).isin([int(v) for v in candidate_init_seeds])]

    if df.empty:
        raise ValueError(
            f"No hay filas en summary.csv para split_seed={split_seed} "
            f"e init_seeds={candidate_init_seeds}"
        )

    best_by_fold = {}

    for fold_idx in selected_folds:
        fold_id = int(fold_idx) + 1
        fold_df = df[df["fold"].astype(int) == fold_id].copy()

        if fold_df.empty:
            raise ValueError(
                f"No hay filas en summary.csv para split_seed={split_seed}, fold={fold_id}."
            )

        sort_cols = ["best_val_bacc", "best_val_loss"]
        ascending = [False, True]

        if "best_epoch" in fold_df.columns:
            sort_cols.append("best_epoch")
            ascending.append(True)

        sort_cols.append("init_seed")
        ascending.append(True)

        fold_df = fold_df.sort_values(sort_cols, ascending=ascending)
        best_by_fold[fold_id] = fold_df.iloc[0].to_dict()

    return best_by_fold


def print_init_selection_table(
    split_seed: int,
    best_by_fold: dict[int, dict],
    candidate_init_seeds: list[int] | None = None,
):
    """Imprime en consola qué init_seed se usará por fold.

    No guarda una tabla aparte; esta información queda solo en el log.
    La selección usa únicamente validación.
    """
    candidates = "all" if candidate_init_seeds is None else ",".join(str(int(v)) for v in candidate_init_seeds)
    print("\nSeed selection")
    print("  split_seed_mode: evaluate requested split_seed")
    print("  init_seed_mode:  best init per fold by validation")
    print(f"  split_seed:      {int(split_seed)}")
    print(f"  init_candidates: [{candidates}]")
    print("  rule:            max best_val_bacc, tie min best_val_loss, tie min best_epoch, tie min init_seed")
    print("  selected:")
    print("    split_seed  fold  init_seed  best_epoch  best_val_bacc  best_val_loss")
    for fold_id in sorted(best_by_fold):
        row = best_by_fold[fold_id]
        best_epoch = row.get("best_epoch", "NA")
        print(
            f"    {int(split_seed):10d}  "
            f"{int(fold_id):4d}  "
            f"{int(row['init_seed']):9d}  "
            f"{best_epoch!s:>10s}  "
            f"{float(row['best_val_bacc']):13.4f}  "
            f"{float(row['best_val_loss']):13.6f}"
        )



def select_global_init_rows(
    summary_df: pd.DataFrame,
    split_seed: int,
    selected_folds: list[int],
    candidate_init_seeds: list[int] | None = None,
) -> tuple[int, dict[int, dict], pd.DataFrame]:
    """Elige una sola init_seed global para un split_seed usando solo validación.

    La selección se hace promediando el desempeño de validación de cada init_seed
    sobre los folds seleccionados. Después se evalúan todos esos folds usando
    esa misma init_seed.

    Regla:
      1) mayor mean(best_val_bacc)
      2) menor mean(best_val_loss)
      3) menor mean(best_epoch)
      4) menor init_seed, para desempate reproducible

    selected_folds usa índices 0-based; el summary usa fold 1-based.
    """

    fold_ids = [int(f) + 1 for f in selected_folds]
    df = summary_df[summary_df["split_seed"].astype(int) == int(split_seed)].copy()

    if candidate_init_seeds is not None:
        df = df[df["init_seed"].astype(int).isin([int(v) for v in candidate_init_seeds])]

    df = df[df["fold"].astype(int).isin(fold_ids)]

    if df.empty:
        raise ValueError(
            f"No hay filas en summary.csv para split_seed={split_seed}, "
            f"folds={fold_ids}, init_seeds={candidate_init_seeds}"
        )

    # Exigir que cada init_seed tenga todos los folds seleccionados.
    counts = df.groupby("init_seed")["fold"].nunique()
    complete_init_seeds = counts[counts == len(fold_ids)].index.tolist()
    df_complete = df[df["init_seed"].isin(complete_init_seeds)].copy()

    if df_complete.empty:
        raise ValueError(
            "Ningún init_seed tiene checkpoints/filas para todos los folds seleccionados. "
            f"split_seed={split_seed}, folds={fold_ids}, init_seeds={candidate_init_seeds}"
        )

    agg_kwargs = {
        "n_folds": ("fold", "nunique"),
        "mean_best_val_bacc": ("best_val_bacc", "mean"),
        "mean_best_val_loss": ("best_val_loss", "mean"),
    }
    if "best_epoch" in df_complete.columns:
        agg_kwargs["mean_best_epoch"] = ("best_epoch", "mean")
    else:
        df_complete["best_epoch"] = np.nan
        agg_kwargs["mean_best_epoch"] = ("best_epoch", "mean")

    ranking = (
        df_complete
        .groupby("init_seed")
        .agg(**agg_kwargs)
        .reset_index()
    )

    ranking = ranking.sort_values(
        ["mean_best_val_bacc", "mean_best_val_loss", "mean_best_epoch", "init_seed"],
        ascending=[False, True, True, True],
    )

    selected_init_seed = int(ranking.iloc[0]["init_seed"])

    selected_df = df_complete[df_complete["init_seed"].astype(int) == selected_init_seed].copy()
    rows_by_fold = {}
    for fold_id in fold_ids:
        fold_df = selected_df[selected_df["fold"].astype(int) == int(fold_id)].copy()
        if fold_df.empty:
            raise ValueError(
                f"Falta fila para split_seed={split_seed}, init_seed={selected_init_seed}, fold={fold_id}."
            )
        rows_by_fold[int(fold_id)] = fold_df.iloc[0].to_dict()

    return selected_init_seed, rows_by_fold, ranking


def print_global_init_selection_table(
    split_seed: int,
    selected_init_seed: int,
    rows_by_fold: dict[int, dict],
    ranking: pd.DataFrame,
    candidate_init_seeds: list[int] | None = None,
):
    """Imprime en consola la init_seed global elegida para un split_seed."""
    candidates = "all" if candidate_init_seeds is None else ",".join(str(int(v)) for v in candidate_init_seeds)
    print("\nSeed selection")
    print("  split_seed_mode: evaluate requested split_seed")
    print("  init_seed_mode:  single global init by mean validation across folds")
    print(f"  split_seed:      {int(split_seed)}")
    print(f"  init_candidates: [{candidates}]")
    print("  rule:            max mean(best_val_bacc), tie min mean(best_val_loss), tie min mean(best_epoch), tie min init_seed")
    print("  ranking:")
    print("    init_seed  n_folds  mean_val_bacc  mean_val_loss  mean_best_epoch")
    for _, row in ranking.iterrows():
        print(
            f"    {int(row['init_seed']):9d}  "
            f"{int(row['n_folds']):7d}  "
            f"{float(row['mean_best_val_bacc']):13.4f}  "
            f"{float(row['mean_best_val_loss']):13.6f}  "
            f"{float(row['mean_best_epoch']):15.2f}"
        )
    print(f"  selected_global_init_seed: {int(selected_init_seed)}")
    print("  folds evaluated with this same init_seed:")
    print("    split_seed  fold  init_seed  best_epoch  best_val_bacc  best_val_loss")
    for fold_id in sorted(rows_by_fold):
        row = rows_by_fold[fold_id]
        best_epoch = row.get("best_epoch", "NA")
        print(
            f"    {int(split_seed):10d}  "
            f"{int(fold_id):4d}  "
            f"{int(row['init_seed']):9d}  "
            f"{best_epoch!s:>10s}  "
            f"{float(row['best_val_bacc']):13.4f}  "
            f"{float(row['best_val_loss']):13.6f}"
        )

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

def _format_cm_cell(value: int, row_total: int) -> str:
    if row_total <= 0:
        return str(int(value))
    return f"{int(value)}\n{value / row_total:.1%}"


def _draw_confusion_matrix(ax, cm: np.ndarray, title: str, class_names: list[str]):
    cm = np.asarray(cm, dtype=int)
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        row_total = int(cm[i].sum())
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                _format_cm_cell(int(cm[i, j]), row_total),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    return image


def save_confusion_matrix_plots(
    summary_df: pd.DataFrame,
    plots_dir: Path,
    config_name: str,
    n_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, Path]:
    """Guarda una figura global con CM por fold + CM acumulada.

    También guarda una imagen individual por fold para inspección rápida.
    """
    if summary_df.empty:
        return {}

    plots_dir.mkdir(parents=True, exist_ok=True)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    cm_rows = []
    for _, row in summary_df.sort_values(["split_seed", "fold", "init_seed"]).iterrows():
        cm = np.asarray(json.loads(row["majority_confusion_matrix"]), dtype=int)
        cm_rows.append((int(row["fold"]), int(row["init_seed"]), cm, row))

    global_cm = np.zeros((n_classes, n_classes), dtype=int)
    for _, _, cm, _ in cm_rows:
        global_cm += cm

    saved = {}

    # Imagen individual por fold.
    for fold_id, init_seed, cm, row in cm_rows:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        title = (
            f"Fold {fold_id} | init {init_seed}\n"
            f"BAcc={float(row['majority_balanced_acc']):.3f}, "
            f"Acc={float(row['majority_acc']):.3f}"
        )
        image = _draw_confusion_matrix(ax, cm, title=title, class_names=class_names)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = plots_dir / f"cm_fold_{fold_id:02d}_init_{init_seed}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved[f"fold_{fold_id}"] = path

    # Imagen resumen: folds + global acumulada.
    n_panels = len(cm_rows) + 1
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.1 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    last_image = None
    for ax, (fold_id, init_seed, cm, row) in zip(axes, cm_rows):
        title = (
            f"Fold {fold_id} | init {init_seed}\n"
            f"BAcc={float(row['majority_balanced_acc']):.3f}"
        )
        last_image = _draw_confusion_matrix(ax, cm, title=title, class_names=class_names)

    global_ax = axes[len(cm_rows)]
    total = int(global_cm.sum())
    global_acc = float(np.trace(global_cm) / max(total, 1))
    recalls = []
    for i in range(n_classes):
        denom = int(global_cm[i].sum())
        recalls.append(float(global_cm[i, i] / denom) if denom > 0 else 0.0)
    global_bacc = float(np.mean(recalls)) if recalls else 0.0
    last_image = _draw_confusion_matrix(
        global_ax,
        global_cm,
        title=f"Global pooled\nBAcc={global_bacc:.3f}, Acc={global_acc:.3f}",
        class_names=class_names,
    )

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"Confusion matrices | {config_name}", y=1.02, fontsize=14)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes[:n_panels].tolist(), fraction=0.025, pad=0.02)
    fig.tight_layout()
    summary_path = plots_dir / "confusion_matrices_by_fold_and_global.png"
    fig.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved["summary"] = summary_path

    return saved



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adhd-dir", type=str, default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "ADHD"))
    parser.add_argument("--control-dir", type=str, default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "Control"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--init-seeds", type=int, nargs="+", default=DEFAULT_INIT_SEEDS)
    parser.add_argument(
        "--select-best-init",
        action="store_true",
        help=(
            "Elige automáticamente un init_seed por fold leyendo summary.csv. "
            "La selección usa solo validación: max best_val_bacc, tie min best_val_loss. "
            "Luego evalúa test una sola vez por fold."
        ),
    )
    parser.add_argument(
        "--select-global-init",
        action="store_true",
        help=(
            "Elige una sola init_seed global por split_seed usando el promedio de validación "
            "sobre los folds seleccionados. Luego evalúa todos los folds con esa misma init_seed."
        ),
    )
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
    parser.add_argument("--print-checkpoint-paths", action="store_true", help="Imprime la ruta del checkpoint usado por fold. Por defecto solo imprime las seeds seleccionadas.")
    parser.add_argument("--max-subject-ids-in-log", type=int, default=12)
    parser.add_argument("--full-subject-ids", action="store_true")

    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "iraniesdataset"))
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Carpeta del experimento. Si se omite, se infiere por T y T'.")

    args = parser.parse_args()

    if args.select_best_init and args.select_global_init:
        raise ValueError("Usa solo una opción: --select-best-init o --select-global-init.")

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

    if args.select_global_init:
        report_suffix = "globalinit"
    elif args.select_best_init:
        report_suffix = "bestinit"
    else:
        report_suffix = "all_inits"
    reports_dir = experiment_dir / f"reports_subject_majority_clean_{report_suffix}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset.get_summary_dataframe().to_csv(reports_dir / "dataset_summary_used_for_report.csv", index=False)

    training_summary_df = load_training_summary(experiment_dir) if (args.select_best_init or args.select_global_init) else None

    print("\nReport configuration")
    print(f"  dataset: N={len(dataset)} | C={n_channels} | T={n_samples} | T_prime={t_prime} | classes={n_classes}")
    print(f"  experiment_dir={experiment_dir}")
    print("  decision=majority vote over argmax_L A_s | aggregation=none | mask=none")
    print(f"  split_seed_mode=evaluate_requested | split_seeds={args.split_seeds}")
    if args.select_global_init:
        init_mode_text = "single_global_init_by_mean_validation"
    elif args.select_best_init:
        init_mode_text = "best_by_validation_per_fold"
    else:
        init_mode_text = "evaluate_all_requested"
    print(f"  init_seed_mode={init_mode_text} | init_seeds={args.init_seeds}")

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

        best_init_by_fold = None
        global_init_by_fold = None
        selected_global_init_seed = None
        if args.select_global_init:
            selected_global_init_seed, global_init_by_fold, global_init_ranking = select_global_init_rows(
                summary_df=training_summary_df,
                split_seed=split_seed,
                selected_folds=selected_folds,
                candidate_init_seeds=args.init_seeds,
            )
            print_global_init_selection_table(
                split_seed=split_seed,
                selected_init_seed=selected_global_init_seed,
                rows_by_fold=global_init_by_fold,
                ranking=global_init_ranking,
                candidate_init_seeds=args.init_seeds,
            )
        elif args.select_best_init:
            best_init_by_fold = select_best_init_rows(
                summary_df=training_summary_df,
                split_seed=split_seed,
                selected_folds=selected_folds,
                candidate_init_seeds=args.init_seeds,
            )
            print_init_selection_table(
                split_seed=split_seed,
                best_by_fold=best_init_by_fold,
                candidate_init_seeds=args.init_seeds,
            )

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

            if args.select_global_init:
                selected_init_rows = [global_init_by_fold[fold_id]]
            elif args.select_best_init:
                selected_init_rows = [best_init_by_fold[fold_id]]
            else:
                selected_init_rows = [
                    {
                        "init_seed": int(init_seed),
                        "checkpoint_path": str(
                            experiment_dir
                            / f"splitseed_{split_seed}"
                            / f"initseed_{int(init_seed)}"
                            / f"fold_{fold_id}"
                            / "best_model.pt"
                        ),
                        "best_val_bacc": np.nan,
                        "best_val_loss": np.nan,
                        "best_epoch": np.nan,
                    }
                    for init_seed in args.init_seeds
                ]

            for selected_init_row in selected_init_rows:
                init_seed = int(selected_init_row["init_seed"])
                checkpoint_path = Path(str(selected_init_row["checkpoint_path"]))

                if not checkpoint_path.exists():
                    checkpoint_path = (
                        experiment_dir
                        / f"splitseed_{split_seed}"
                        / f"initseed_{init_seed}"
                        / f"fold_{fold_id}"
                        / "best_model.pt"
                    )

                val_bacc = selected_init_row.get("best_val_bacc", np.nan)
                val_loss = selected_init_row.get("best_val_loss", np.nan)
                best_epoch = selected_init_row.get("best_epoch", np.nan)
                print(
                    f"  using seeds: split_seed={split_seed} | fold={fold_id} | init_seed={init_seed} | "
                    f"best_epoch={best_epoch} | val_bacc={float(val_bacc):.4f} | val_loss={float(val_loss):.6f}"
                )
                if args.print_checkpoint_paths:
                    print(f"  checkpoint: {checkpoint_path}")

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
                    "selection_mode": "global_init_by_mean_validation" if args.select_global_init else ("best_init_by_validation" if args.select_best_init else "all_requested_inits"),
                    "selected_by_best_val_bacc": bool(args.select_best_init or args.select_global_init),
                    "selected_best_val_bacc": selected_init_row.get("best_val_bacc", np.nan),
                    "selected_best_val_loss": selected_init_row.get("best_val_loss", np.nan),
                    "selected_best_epoch": selected_init_row.get("best_epoch", np.nan),
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
            "selection_mode": "global_init_by_mean_validation" if args.select_global_init else ("best_init_by_validation" if args.select_best_init else "all_requested_inits"),
            "decision_rule": "majority_vote_over_argmax_A_s",
            "aggregation": "none",
            "mask_used": False,
        }
    ])
    overall.to_csv(summary_overall_path, index=False)

    plots_dir = experiment_dir / "plots"
    cm_plot_paths = save_confusion_matrix_plots(
        summary_df=summary_df,
        plots_dir=plots_dir,
        config_name=config_name,
        n_classes=n_classes,
        class_names=["Control", "ADHD"],
    )

    print("\nSummary by fold")
    cols = ["fold", "n_test_subjects", "majority_acc", "majority_balanced_acc", "majority_macro_f1", "temporal_acc", "pred_counts"]
    print(summary_df[cols].to_string(index=False))

    print("\nSummary overall")
    print(overall.to_string(index=False))

    print("\nSaved report files")
    print(f"  by_fold:      {summary_by_fold_path}")
    print(f"  votes:        {predictions_path}")
    print(f"  overall:      {summary_overall_path}")
    if cm_plot_paths:
        print("  cm plots:")
        for key, path in cm_plot_paths.items():
            print(f"    {key}: {path}")


if __name__ == "__main__":
    main()
