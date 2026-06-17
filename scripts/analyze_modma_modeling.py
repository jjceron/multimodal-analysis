from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.modma_db import MODMADataset, DEFAULT_ROOT, split_into_windows

warnings.filterwarnings("ignore", category=UserWarning)

OUT = Path("outputs/exploratory/modma")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FS = 250.0
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 60)}


def bandpower_features(eeg, fs=250):
    from scipy.signal import periodogram
    freqs, psd = periodogram(eeg, fs=fs, axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        feats.append(np.trapezoid(psd[..., idx], freqs[idx], axis=-1))
    return np.concatenate(feats, axis=-1)


def torch_t(arr):
    import torch
    return torch.tensor(arr, dtype=torch.float32)


def main():
    print("Loading MODMA dataset...")
    ds = MODMADataset(root=DEFAULT_ROOT, lowcut=0.5, highcut=60.0, notch=50.0, target_fs=FS)

    subjects = [s["participant_id"] for s in ds.samples]
    labels = np.array([int(s["label"].item()) for s in ds.samples])
    eeg_list = [s["eeg"].numpy() for s in ds.samples]
    n_ch = len(ds.channel_names)
    print(f"  Subjects: {len(ds)}, Channels: {n_ch}")

    print("  Precomputing full-signal bandpower features...")
    X_full = np.vstack([bandpower_features(eeg[np.newaxis, :]) for eeg in eeg_list])
    print(f"  Feature matrix: {X_full.shape}")

    print("\n[1/4] Window benchmark (LDA, subject-level CV)...")
    window_rows = []
    for win_sec in [1, 2]:
        win_samp = int(win_sec * FS)
        print(f"  {win_sec}s windows...")
        X_win, y_win, subj_win = [], [], []
        for i in range(len(ds)):
            for w in split_into_windows(torch_t(eeg_list[i]), win_samp, win_samp):
                X_win.append(bandpower_features(w.numpy()[np.newaxis, :]))
                y_win.append(labels[i])
                subj_win.append(subjects[i])
        X_win = np.vstack(X_win)
        y_win = np.array(y_win)

        preds_list, true_list = [], []
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=3407)
        for train_idx, test_idx in gkf.split(eeg_list, labels, groups=subjects):
            train_set = set(subjects[i] for i in train_idx)
            mask = np.array([s in train_set for s in subj_win])
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_win[mask], y_win[mask])
            preds = clf.predict(X_full[test_idx])
            preds_list.extend(preds)
            true_list.extend(labels[test_idx])

        true_arr, pred_arr = np.array(true_list), np.array(preds_list)
        acc = accuracy_score(true_arr, pred_arr)
        bal_acc = np.mean([accuracy_score(true_arr[true_arr == c], pred_arr[true_arr == c]) for c in [0, 1]])
        window_rows.append({"window_sec": win_sec, "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4)})
        print(f"    acc={acc:.2%}, bal_acc={bal_acc:.2%}")

    pd.DataFrame(window_rows).to_csv(OUT / "window_benchmark.csv", index=False)
    fig, ax = plt.subplots(figsize=(4, 3))
    xs = [r["window_sec"] for r in window_rows]
    ax.plot(xs, [r["accuracy"] for r in window_rows], "o-", color="#4A90D9", label="Accuracy")
    ax.plot(xs, [r["balanced_accuracy"] for r in window_rows], "s--", color="#E74C3C", label="Balanced Acc")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Subject-level accuracy")
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "window_vs_accuracy.png", dpi=120, pil_kwargs={"optimize": True})
    plt.close(fig)

    print(f"\n[2/4] Feature importance (RF) on {X_full.shape}...")
    print(f"  Feature range: [{X_full.min():.4f}, {X_full.max():.4f}], "
          f"label balance: HC={(labels==0).sum()}, MDD={(labels==1).sum()}")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=3407, n_jobs=-1)
    rf.fit(X_full, labels)
    imp = pd.DataFrame({"feature_idx": np.arange(X_full.shape[1]), "importance": rf.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    print(f"  Top importance: {imp['importance'].iloc[0]:.4f}")

    band_names = list(BANDS.keys())
    n_bands = len(band_names)
    top20 = imp.head(20)
    labels_fi = [f"ch{f//n_bands}_{band_names[f%n_bands]}" for f in top20["feature_idx"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(20), top20["importance"].values[::-1], color="#2ECC71")
    ax.set_yticks(range(20))
    ax.set_yticklabels(labels_fi[::-1], fontsize=6)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=120, pil_kwargs={"optimize": True})
    plt.close(fig)

    print("\n[3/4] t-SNE...")
    tsne = TSNE(n_components=2, perplexity=5, max_iter=500, random_state=3407)
    emb = tsne.fit_transform(X_full)
    df_tsne = pd.DataFrame({
        "subject": subjects, "group": np.where(labels == 0, "HC", "MDD"),
        "tsne_1": emb[:, 0], "tsne_2": emb[:, 1],
    })
    df_tsne.to_csv(OUT / "tsne_embedding.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    for g, c in [("HC", "#4A90D9"), ("MDD", "#E74C3C")]:
        s = df_tsne[df_tsne["group"] == g]
        ax.scatter(s["tsne_1"], s["tsne_2"], c=c, label=g, s=30, alpha=0.8)
    ax.set_title("t-SNE: full-signal bandpower")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "tsne_comparison.png", dpi=120, pil_kwargs={"optimize": True})
    plt.close(fig)

    print(f"\nDone. Files saved to {OUT}")


if __name__ == "__main__":
    main()
