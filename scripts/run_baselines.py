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
from scipy import signal, linalg
from pyriemann.classification import MDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.modma_db import MODMADataset, DEFAULT_ROOT

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "modma_db"

FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}


def window_subjects(subjects, labels, eeg_list, window_len, stride):
    win_subjects: list[str] = []
    win_labels: list[int] = []
    win_data: list[np.ndarray] = []
    for subj_id, lbl, eeg in zip(subjects, labels, eeg_list):
        T = eeg.shape[-1]
        if isinstance(eeg, torch.Tensor):
            eeg = eeg.numpy()
        for start in range(0, T - window_len + 1, stride):
            win_subjects.append(subj_id)
            win_labels.append(lbl)
            win_data.append(eeg[:, start:start + window_len].copy())
    return win_subjects, win_labels, win_data


def majority_vote(subject_names, all_preds):
    subject_votes: dict[str, list[int]] = {}
    for name, pred in zip(subject_names, all_preds):
        subject_votes.setdefault(name, []).append(pred)
    preds, labels = [], []
    for name, votes in subject_votes.items():
        preds.append(max(set(votes), key=votes.count))
    return preds


def compute_csp(epochs, labels, n_components=4):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    classes = np.unique(labels)
    covs = {}
    for cls in classes:
        X_cls = epochs[labels == cls]
        covs[cls] = np.mean([np.cov(x.reshape(x.shape[0], -1)) for x in X_cls], axis=0)
    eigvals, eigvecs = linalg.eigh(covs[classes[0]], covs[classes[1]])
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx]
    selected = np.concatenate([W[:, :n_components], W[:, -n_components:]], axis=1)
    return selected.T


def csp_features(W, epoch):
    projected = W @ epoch.reshape(epoch.shape[0], -1)
    var = np.var(projected, axis=1)
    return np.log(var / np.sum(var) + 1e-10)


# ---------------------------------------------------------------------------
# CSP + LDA
# ---------------------------------------------------------------------------
def run_csp_lda(dataset, args):
    print("\n" + "=" * 60)
    print("CSP + LDA")
    print("=" * 60)

    subjects = [s["participant_id"] for s in dataset.samples]
    labels_arr = [int(s["label"].item()) for s in dataset.samples]
    eeg_list = [s["eeg"].numpy() for s in dataset.samples]
    n_channels = eeg_list[0].shape[0]

    sfreq = 250.0
    window_len = int(round(args.window_sec * sfreq))
    stride = int(round(window_len * (1.0 - args.overlap)))

    all_fold_metrics = []
    all_predictions = []

    gkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=3407)
    nested = StratifiedGroupKFold(n_splits=args.inner_splits, shuffle=True, random_state=3407)

    for fold_id, (train_val_idx, test_idx) in enumerate(gkf.split(eeg_list, labels_arr, groups=subjects)):
        train_idx, val_idx = next(nested.split(
            [eeg_list[i] for i in train_val_idx],
            [labels_arr[i] for i in train_val_idx],
            groups=[subjects[i] for i in train_val_idx],
        ))
        train_subs = [train_val_idx[i] for i in train_idx]
        val_subs = [train_val_idx[i] for i in val_idx]

        ws_train, wl_train, wd_train = window_subjects(
            [subjects[i] for i in train_subs],
            [labels_arr[i] for i in train_subs],
            [eeg_list[i] for i in train_subs],
            window_len, stride,
        )
        ws_val, wl_val, wd_val = window_subjects(
            [subjects[i] for i in val_subs],
            [labels_arr[i] for i in val_subs],
            [eeg_list[i] for i in val_subs],
            window_len, stride,
        )
        ws_test, wl_test, wd_test = window_subjects(
            [subjects[i] for i in test_idx],
            [labels_arr[i] for i in test_idx],
            [eeg_list[i] for i in test_idx],
            window_len, stride,
        )

        train_data = np.stack([w.reshape(-1) for w in wd_train])
        val_data = np.stack([w.reshape(-1) for w in wd_val])
        test_data = np.stack([w.reshape(-1) for w in wd_test])
        wl_train_np = np.array(wl_train)

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis()),
        ])

        clf.fit(train_data, wl_train_np)

        val_preds = clf.predict(val_data)
        val_subj_preds = majority_vote(ws_val, val_preds)
        val_subj_true = list(dict.fromkeys(ws_val))
        val_gt = [wl_val[ws_val.index(s)] for s in val_subj_true]

        test_preds = clf.predict(test_data)
        test_subj_preds = majority_vote(ws_test, test_preds)
        test_subj_true = list(dict.fromkeys(ws_test))
        test_gt = [wl_test[ws_test.index(s)] for s in test_subj_true]

        val_metrics = {
            "accuracy": accuracy_score(val_gt, val_subj_preds),
            "balanced_accuracy": balanced_accuracy_score(val_gt, val_subj_preds),
            "f1_macro": f1_score(val_gt, val_subj_preds, average="macro"),
        }
        test_metrics = {
            "accuracy": accuracy_score(test_gt, test_subj_preds),
            "balanced_accuracy": balanced_accuracy_score(test_gt, test_subj_preds),
            "f1_macro": f1_score(test_gt, test_subj_preds, average="macro"),
        }

        fold_metrics = {
            "fold": fold_id,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        all_fold_metrics.append(fold_metrics)

        for name, true_label, pred_label in zip(val_subj_true, val_gt, val_subj_preds):
            all_predictions.append({
                "fold": fold_id, "split": "val", "subject": name,
                "true_label": true_label, "pred_label": pred_label,
            })
        for name, true_label, pred_label in zip(test_subj_true, test_gt, test_subj_preds):
            all_predictions.append({
                "fold": fold_id, "split": "test", "subject": name,
                "true_label": true_label, "pred_label": pred_label,
            })

        print(f"  Fold {fold_id:02d}: val_acc={val_metrics['accuracy']:.2%}, test_acc={test_metrics['accuracy']:.2%}")

    save_results(args, "csp_lda", all_fold_metrics, all_predictions)


# ---------------------------------------------------------------------------
# Band Power + SVM
# ---------------------------------------------------------------------------
def run_bandpower_svm(dataset, args):
    print("\n" + "=" * 60)
    print("Band Power + SVM")
    print("=" * 60)

    subjects = [s["participant_id"] for s in dataset.samples]
    labels_arr = [int(s["label"].item()) for s in dataset.samples]
    eeg_list = [s["eeg"].numpy() for s in dataset.samples]

    sfreq = 250.0
    window_len = int(round(args.window_sec * sfreq))
    stride = int(round(window_len * (1.0 - args.overlap)))
    nperseg = min(256, window_len)

    def extract_band_power(eeg):
        f, psd = signal.welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1)
        features = []
        for lo, hi in FREQ_BANDS.values():
            mask = (f >= lo) & (f <= hi)
            bp = np.trapz(psd[:, mask], f[mask], axis=1)
            features.append(bp)
        return np.concatenate(features)

    gkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=3407)
    nested = StratifiedGroupKFold(n_splits=args.inner_splits, shuffle=True, random_state=3407)

    all_fold_metrics = []
    all_predictions = []

    for fold_id, (train_val_idx, test_idx) in enumerate(gkf.split(eeg_list, labels_arr, groups=subjects)):
        train_idx, val_idx = next(nested.split(
            [eeg_list[i] for i in train_val_idx],
            [labels_arr[i] for i in train_val_idx],
            groups=[subjects[i] for i in train_val_idx],
        ))
        train_subs = [train_val_idx[i] for i in train_idx]
        val_subs = [train_val_idx[i] for i in val_idx]

        ws_train, wl_train, wd_train = window_subjects(
            [subjects[i] for i in train_subs],
            [labels_arr[i] for i in train_subs],
            [eeg_list[i] for i in train_subs],
            window_len, stride,
        )
        ws_val, wl_val, wd_val = window_subjects(
            [subjects[i] for i in val_subs],
            [labels_arr[i] for i in val_subs],
            [eeg_list[i] for i in val_subs],
            window_len, stride,
        )
        ws_test, wl_test, wd_test = window_subjects(
            [subjects[i] for i in test_idx],
            [labels_arr[i] for i in test_idx],
            [eeg_list[i] for i in test_idx],
            window_len, stride,
        )

        X_train = np.array([extract_band_power(w) for w in wd_train])
        X_val = np.array([extract_band_power(w) for w in wd_val])
        X_test = np.array([extract_band_power(w) for w in wd_test])
        y_train = np.array(wl_train)

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced")),
        ])
        clf.fit(X_train, y_train)

        val_preds = clf.predict(X_val)
        val_subj_true = list(dict.fromkeys(ws_val))
        val_gt = [wl_val[ws_val.index(s)] for s in val_subj_true]
        val_subj_preds = majority_vote(ws_val, val_preds)

        test_preds = clf.predict(X_test)
        test_subj_true = list(dict.fromkeys(ws_test))
        test_gt = [wl_test[ws_test.index(s)] for s in test_subj_true]
        test_subj_preds = majority_vote(ws_test, test_preds)

        val_metrics = {
            "accuracy": accuracy_score(val_gt, val_subj_preds),
            "balanced_accuracy": balanced_accuracy_score(val_gt, val_subj_preds),
            "f1_macro": f1_score(val_gt, val_subj_preds, average="macro"),
        }
        test_metrics = {
            "accuracy": accuracy_score(test_gt, test_subj_preds),
            "balanced_accuracy": balanced_accuracy_score(test_gt, test_subj_preds),
            "f1_macro": f1_score(test_gt, test_subj_preds, average="macro"),
        }

        fold_metrics = {
            "fold": fold_id,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        all_fold_metrics.append(fold_metrics)

        for name, true_label, pred_label in zip(val_subj_true, val_gt, val_subj_preds):
            all_predictions.append({
                "fold": fold_id, "split": "val", "subject": name,
                "true_label": true_label, "pred_label": pred_label,
            })
        for name, true_label, pred_label in zip(test_subj_true, test_gt, test_subj_preds):
            all_predictions.append({
                "fold": fold_id, "split": "test", "subject": name,
                "true_label": true_label, "pred_label": pred_label,
            })

        print(f"  Fold {fold_id:02d}: val_acc={val_metrics['accuracy']:.2%}, test_acc={test_metrics['accuracy']:.2%}")

    save_results(args, "bandpower_svm", all_fold_metrics, all_predictions)


# ---------------------------------------------------------------------------
# Riemannian MDM
# ---------------------------------------------------------------------------
def run_riemann_mdm(dataset, args):
    print("\n" + "=" * 60)
    print("Riemannian MDM")
    print("=" * 60)

    subjects = [s["participant_id"] for s in dataset.samples]
    labels_arr = [int(s["label"].item()) for s in dataset.samples]
    eeg_list = [s["eeg"].numpy() for s in dataset.samples]

    sfreq = 250.0
    window_len = int(round(args.window_sec * sfreq))
    stride = int(round(window_len * (1.0 - args.overlap)))

    gkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=3407)
    nested = StratifiedGroupKFold(n_splits=args.inner_splits, shuffle=True, random_state=3407)

    all_fold_metrics = []
    all_predictions = []

    for fold_id, (train_val_idx, test_idx) in enumerate(gkf.split(eeg_list, labels_arr, groups=subjects)):
        train_idx, val_idx = next(nested.split(
            [eeg_list[i] for i in train_val_idx],
            [labels_arr[i] for i in train_val_idx],
            groups=[subjects[i] for i in train_val_idx],
        ))
        train_subs = [train_val_idx[i] for i in train_idx]
        val_subs = [train_val_idx[i] for i in val_idx]

        ws_train, wl_train, wd_train = window_subjects(
            [subjects[i] for i in train_subs],
            [labels_arr[i] for i in train_subs],
            [eeg_list[i] for i in train_subs],
            window_len, stride,
        )
        ws_val, wl_val, wd_val = window_subjects(
            [subjects[i] for i in val_subs],
            [labels_arr[i] for i in val_subs],
            [eeg_list[i] for i in val_subs],
            window_len, stride,
        )
        ws_test, wl_test, wd_test = window_subjects(
            [subjects[i] for i in test_idx],
            [labels_arr[i] for i in test_idx],
            [eeg_list[i] for i in test_idx],
            window_len, stride,
        )

        def _cov(x):
            return np.cov(x.reshape(x.shape[0], -1))

        cov_train = np.stack([_cov(w) for w in wd_train])
        cov_val = np.stack([_cov(w) for w in wd_val])
        cov_test = np.stack([_cov(w) for w in wd_test])
        y_train = np.array(wl_train)

        mdm = MDM()
        mdm.fit(cov_train, y_train)

        val_preds = mdm.predict(cov_val)
        val_subj_true = list(dict.fromkeys(ws_val))
        val_gt = [wl_val[ws_val.index(s)] for s in val_subj_true]
        val_subj_preds = majority_vote(ws_val, val_preds)

        test_preds = mdm.predict(cov_test)
        test_subj_true = list(dict.fromkeys(ws_test))
        test_gt = [wl_test[ws_test.index(s)] for s in test_subj_true]
        test_subj_preds = majority_vote(ws_test, test_preds)

        val_metrics = {
            "accuracy": accuracy_score(val_gt, val_subj_preds),
            "balanced_accuracy": balanced_accuracy_score(val_gt, val_subj_preds),
            "f1_macro": f1_score(val_gt, val_subj_preds, average="macro"),
        }
        test_metrics = {
            "accuracy": accuracy_score(test_gt, test_subj_preds),
            "balanced_accuracy": balanced_accuracy_score(test_gt, test_subj_preds),
            "f1_macro": f1_score(test_gt, test_subj_preds, average="macro"),
        }

        fold_metrics = {
            "fold": fold_id,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        all_fold_metrics.append(fold_metrics)

        for name, true_label, pred_label in zip(val_subj_true, val_gt, val_subj_preds):
            all_predictions.append({
                "fold": fold_id, "split": "val", "subject": name,
                "true_label": true_label, "pred_label": pred_label,
            })
        for name, true_label, pred_label in zip(test_subj_true, test_gt, test_subj_preds):
            all_predictions.append({
                "fold": fold_id, "split": "test", "subject": name,
                "true_label": true_label, "pred_label": pred_label,
            })

        print(f"  Fold {fold_id:02d}: val_acc={val_metrics['accuracy']:.2%}, test_acc={test_metrics['accuracy']:.2%}")

    save_results(args, "riemann_mdm", all_fold_metrics, all_predictions)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def save_results(args, model_name, all_fold_metrics, all_predictions):
    df_fold = pd.DataFrame(all_fold_metrics)
    overall = {
        "mean_accuracy": float(df_fold["test_accuracy"].mean()),
        "std_accuracy": float(df_fold["test_accuracy"].std()),
        "mean_balanced_accuracy": float(df_fold["test_balanced_accuracy"].mean()),
        "std_balanced_accuracy": float(df_fold["test_balanced_accuracy"].std()),
        "mean_f1_macro": float(df_fold["test_f1_macro"].mean()),
        "std_f1_macro": float(df_fold["test_f1_macro"].std()),
    }

    base = OUTPUT_ROOT / model_name
    base.mkdir(parents=True, exist_ok=True)
    n_existing = len([d for d in base.iterdir() if d.is_dir() and d.name.startswith(args.version_prefix)])
    version = f"{args.version_prefix}"
    out_dir = base / version
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": model_name,
        "version": version,
        "window_sec": args.window_sec,
        "overlap": args.overlap,
        "k": args.k,
        "n_channels": len(all_fold_metrics),
        "n_classes": 2,
        "duration_sec": 120.0,
        "dataset": "modma_db",
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    df_fold.to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([overall]).to_csv(out_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n  Results saved to: {out_dir}")
    print(f"  Overall test_acc: {overall['mean_accuracy']:.2%} +/- {overall['std_accuracy']:.2%}")
    print(f"  Balanced acc:    {overall['mean_balanced_accuracy']:.2%} +/- {overall['std_balanced_accuracy']:.2%}")
    print(f"  F1-macro:        {overall['mean_f1_macro']:.4f} +/- {overall['std_f1_macro']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["csp_lda", "bandpower_svm", "riemann_mdm", "all"],
                        help="Baseline model to run")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--version-prefix", type=str, default="v1",
                        help="Version label for output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading MODMA dataset...")
    dataset = MODMADataset(
        root=DEFAULT_ROOT, lowcut=0.5, highcut=60.0, notch=50.0, duration_sec=120.0,
    )
    print(f"  Subjects: {len(dataset)}")
    print(f"  Channels: {dataset.samples[0]['eeg'].shape[0]}")

    models = ["csp_lda", "bandpower_svm", "riemann_mdm"] if args.model == "all" else [args.model]

    for model_name in models:
        if model_name == "csp_lda":
            run_csp_lda(dataset, args)
        elif model_name == "bandpower_svm":
            run_bandpower_svm(dataset, args)
        elif model_name == "riemann_mdm":
            run_riemann_mdm(dataset, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
