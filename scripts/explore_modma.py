from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.modma_db import MODMADataset, split_into_windows, DEFAULT_ROOT

sns.set_theme(style="whitegrid")
OUT = Path(__file__).resolve().parent.parent / "outputs" / "exploratory"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
FS = 250.0
BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 48),
}


def load_data():
    print("Loading MODMA dataset (full, no duration limit)...")
    ds = MODMADataset(root=DEFAULT_ROOT, lowcut=0.5, highcut=60.0, notch=50.0)
    n = len(ds)
    hc_count = sum(1 for i in range(n) if ds[i][2].item() == 0)
    mdd_count = sum(1 for i in range(n) if ds[i][2].item() == 1)
    ch_names = ds.channel_names
    n_ch = len(ch_names)
    T = ds[0][1].shape[1]
    print(f"  Subjects: {n} (HC={hc_count}, MDD={mdd_count})")
    print(f"  Channels: {n_ch}")
    print(f"  Samples per subject: {T} ({T/FS:.1f}s @ {FS}Hz)")
    return ds, hc_count, mdd_count, ch_names, n_ch, T


def analyze_temporal_distribution(ds, n_ch, T):
    print("\n[1/7] Temporal distribution analysis...")
    window_sec = 2.0
    win_len = int(window_sec * FS)
    stride = win_len
    n_windows_per_subj = (T - win_len) // stride + 1
    all_power = np.zeros((n_windows_per_subj, 2))
    all_count = np.zeros(n_windows_per_subj, dtype=int)

    for i in range(len(ds)):
        _, eeg, label = ds[i]
        eeg_np = eeg.cpu().numpy()
        lbl = label.item()
        for w_idx, start in enumerate(range(0, T - win_len + 1, stride)):
            win = eeg_np[:, start:start + win_len]
            power = np.mean(win ** 2)
            all_power[w_idx, lbl] += power
            all_count[w_idx] += 1

    for lbl_idx in range(2):
        all_power[:, lbl_idx] = np.divide(
            all_power[:, lbl_idx],
            np.maximum(all_count / 2, 1),
            where=np.maximum(all_count / 2, 1) > 0,
        )

    fig, ax = plt.subplots(figsize=(12, 5))
    time_axis = np.arange(n_windows_per_subj) * window_sec
    ax.plot(time_axis, all_power[:, 0], label="HC", color="#4A90D9", linewidth=2)
    ax.plot(time_axis, all_power[:, 1], label="MDD", color="#E74C3C", linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mean Signal Power", fontsize=12)
    ax.set_title("Temporal Distribution of EEG Power (2s windows)", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_temporal_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 01_temporal_distribution.png ({n_windows_per_subj} windows)")


def analyze_spectral_profiles(ds, n_ch, ch_names):
    print("\n[2/7] Spectral profile analysis...")
    n_fft = int(2 * FS)
    all_psd = {"HC": [], "MDD": []}
    all_freqs = None

    for i in range(len(ds)):
        _, eeg, label = ds[i]
        eeg_np = eeg.cpu().numpy()
        freqs, psd = signal.welch(eeg_np, fs=FS, nperseg=n_fft, axis=-1)
        lbl_name = "HC" if label.item() == 0 else "MDD"
        all_psd[lbl_name].append(psd)
        if all_freqs is None:
            all_freqs = freqs

    per_ch = {}
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, color in [("HC", "#4A90D9"), ("MDD", "#E74C3C")]:
        stacked = np.stack(all_psd[name], axis=0)
        mean_psd = stacked.mean(axis=(0, 1))
        ax.plot(all_freqs, 10 * np.log10(mean_psd + 1e-12), label=name, color=color, linewidth=2)
        per_ch[name] = stacked.mean(axis=0)
    for band_name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.08, color="gray")
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95, band_name, ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density (dB)", fontsize=12)
    ax.set_title("Average Power Spectral Density — HC vs MDD", fontsize=14)
    ax.legend()
    ax.set_xlim(0.5, 50)
    fig.tight_layout()
    fig.savefig(OUT / "02_spectral_profiles.png", dpi=150)
    plt.close(fig)

    ch_top = 8
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    band_list = list(BANDS.items())
    for idx, (band_name, (lo, hi)) in enumerate(band_list):
        mask = (all_freqs >= lo) & (all_freqs <= hi)
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        hc_band = per_ch["HC"][:, mask].mean(axis=1)
        mdd_band = per_ch["MDD"][:, mask].mean(axis=1)
        top_ch = np.argsort(np.abs(hc_band - mdd_band))[::-1][:ch_top]
        x = np.arange(ch_top)
        width = 0.35
        ax.bar(x - width / 2, hc_band[top_ch], width, label="HC", color="#4A90D9", alpha=0.8)
        ax.bar(x + width / 2, mdd_band[top_ch], width, label="MDD", color="#E74C3C", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([ch_names[c].replace("EEG ", "") for c in top_ch], rotation=45, fontsize=7)
        ax.set_title(f"{band_name} ({lo}-{hi} Hz)", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Top Channels by Band Power Difference (HC vs MDD)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "02b_band_top_channels.png", dpi=150)
    plt.close(fig)
    print("  Saved 02_spectral_profiles.png + 02b_band_top_channels.png")


def analyze_channel_correlation(ds, n_ch):
    print("\n[3/7] Channel correlation analysis...")
    corr_mats = {}
    for lbl, name in [(0, "HC"), (1, "MDD")]:
        all_eeg = []
        for i in range(len(ds)):
            _, eeg, label = ds[i]
            if label.item() == lbl:
                all_eeg.append(eeg.cpu().numpy())
        data = np.concatenate(all_eeg, axis=1)
        corr_mats[name] = np.corrcoef(data)

    diff = corr_mats["MDD"] - corr_mats["HC"]
    diff_flat = diff[np.triu_indices_from(diff, k=1)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = max(abs(corr_mats["HC"]).max(), abs(corr_mats["MDD"]).max())
    for idx, (name, cmap) in enumerate([("HC", "Blues"), ("MDD", "Reds")]):
        im = axes[idx].imshow(corr_mats[name], cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        axes[idx].set_title(f"{name} — Channel Correlation", fontsize=12)
        axes[idx].set_xlabel("Channel")
        axes[idx].set_ylabel("Channel")
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axes[2].set_title("MDD - HC Difference", fontsize=12)
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Channel")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("Functional Connectivity (Pearson Correlation)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "03_channel_correlation.png", dpi=150)
    plt.close(fig)

    stat, p_value = stats.mannwhitneyu(diff_flat, np.zeros_like(diff_flat), alternative="two-sided")
    print(f"  MDD-HC correlation difference vs zero: U={stat:.0f}, p={p_value:.4f}")
    print(f"  Mean diff: {diff_flat.mean():.4f} ± {diff_flat.std():.4f}")
    print("  Saved 03_channel_correlation.png")


def analyze_bandpower_boxplot(ds):
    print("\n[4/7] Band power boxplot analysis...")
    n_fft = int(2 * FS)
    records = {band: {"HC": [], "MDD": []} for band in BANDS}

    for i in range(len(ds)):
        _, eeg, label = ds[i]
        eeg_np = eeg.cpu().numpy()
        freqs, psd = signal.welch(eeg_np, fs=FS, nperseg=n_fft, axis=-1)
        lbl_name = "HC" if label.item() == 0 else "MDD"
        for band_name, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            records[band_name][lbl_name].append(psd[:, mask].mean())

    band_stats = {}
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for idx, (band_name, _) in enumerate(BANDS.items()):
        ax = axes[idx]
        hc_vals = np.log10(np.array(records[band_name]["HC"]) + 1e-12)
        mdd_vals = np.log10(np.array(records[band_name]["MDD"]) + 1e-12)
        data = [hc_vals, mdd_vals]
        bp = ax.boxplot(data, labels=["HC", "MDD"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#4A90D9")
        bp["boxes"][1].set_facecolor("#E74C3C")
        stat, p = stats.mannwhitneyu(hc_vals, mdd_vals, alternative="two-sided")
        band_stats[band_name] = {"p_value": float(p), "hc_mean": float(hc_vals.mean()), "mdd_mean": float(mdd_vals.mean())}
        y_max = max(hc_vals.max(), mdd_vals.max())
        ax.text(0.5, y_max * 1.05, f"p={p:.4f}", ha="center", fontsize=9,
                fontweight="bold" if p < 0.05 else "normal")
        ax.set_title(f"{band_name}\n({BANDS[band_name][0]}-{BANDS[band_name][1]} Hz)", fontsize=10)
        ax.set_ylabel("Log Power")
    fig.suptitle("Band Power Comparison: HC vs MDD", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "04_bandpower_boxplot.png", dpi=150)
    plt.close(fig)

    sig_bands = [b for b, s in band_stats.items() if s["p_value"] < 0.05]
    print(f"  Significant bands (p<0.05): {sig_bands if sig_bands else 'None'}")
    print("  Saved 04_bandpower_boxplot.png")
    return band_stats


def analyze_window_sizes(ds):
    print("\n[5/7] Window size comparison...")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    window_sizes = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    results = []
    n_fft = int(2 * FS)

    for ws in window_sizes:
        win_len = int(ws * FS)
        stride = win_len // 2
        features_list = []
        labels_list = []
        for i in range(len(ds)):
            _, eeg, label = ds[i]
            eeg_np = eeg.cpu().numpy()
            windows = [eeg_np[:, s:s + win_len] for s in range(0, eeg_np.shape[1] - win_len + 1, stride)]
            for w in windows:
                freqs, psd = signal.welch(w, fs=FS, nperseg=min(n_fft, win_len), axis=-1)
                band_powers = []
                for _, (lo, hi) in BANDS.items():
                    mask = (freqs >= lo) & (freqs <= hi)
                    band_powers.append(psd[:, mask].mean(axis=1))
                feats = np.concatenate(band_powers)
                features_list.append(feats)
                labels_list.append(label.item())
        X = np.array(features_list)
        y = np.array(labels_list)
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy")
        results.append({
            "window_sec": ws,
            "n_windows": len(X),
            "balanced_acc_mean": scores.mean(),
            "balanced_acc_std": scores.std(),
        })
        print(f"  Window {ws}s: BA={scores.mean():.2%} ± {scores.std():.2%} ({len(X)} windows)")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "05_window_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(df["window_sec"], df["balanced_acc_mean"], yerr=df["balanced_acc_std"],
                fmt="-o", capsize=5, color="#2E86AB", linewidth=2, markersize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Window Size (seconds)", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (LDA)", fontsize=12)
    ax.set_title("Classification Performance vs Window Size", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_window_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved 05_window_comparison.csv + 05_window_comparison.png")
    return df


def analyze_feature_importance(ds):
    print("\n[6/7] Feature importance analysis...")
    win_len = int(2.0 * FS)
    stride = win_len // 2
    n_fft = int(2 * FS)
    features_list = []
    labels_list = []

    for i in range(len(ds)):
        _, eeg, label = ds[i]
        eeg_np = eeg.cpu().numpy()
        windows = [eeg_np[:, s:s + win_len] for s in range(0, eeg_np.shape[1] - win_len + 1, stride)]
        for w in windows:
            feats = []
            for ch_idx in range(w.shape[0]):
                freqs, psd = signal.welch(w[ch_idx], fs=FS, nperseg=n_fft)
                for _, (lo, hi) in BANDS.items():
                    mask = (freqs >= lo) & (freqs <= hi)
                    feats.append(psd[mask].mean())
                feats.append(w[ch_idx].std())
                feats.append(np.percentile(w[ch_idx], 75) - np.percentile(w[ch_idx], 25))
                feats.append(stats.skew(w[ch_idx]))
                feats.append(stats.kurtosis(w[ch_idx]))
            features_list.append(feats)
            labels_list.append(label.item())

    X = np.array(features_list)
    y = np.array(labels_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_

    n_ch = len(ds.channel_names)
    n_feats_per_ch = len(BANDS) + 4
    ch_names = ds.channel_names
    feat_names = []
    for ch in ch_names:
        for band in BANDS:
            feat_names.append(f"{ch}_{band}")
        feat_names.append(f"{ch}_std")
        feat_names.append(f"{ch}_IQR")
        feat_names.append(f"{ch}_skew")
        feat_names.append(f"{ch}_kurtosis")

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#4A90D9" if "MDD" not in f else "#E74C3C" for f in imp_df["feature"]]
    ax.barh(range(len(imp_df)), imp_df["importance"].values, color="#2E86AB")
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"].values, fontsize=8)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top 30 Features (Random Forest)", fontsize=14)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "06_feature_importance.png", dpi=150)
    plt.close(fig)

    ch_importance = {}
    for ch in ch_names:
        mask = [f.startswith(ch) for f in feat_names]
        ch_importance[ch] = importances[mask].sum()
    top_chs = sorted(ch_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"  Top 10 channels by feature importance:")
    for ch, imp in top_chs[:10]:
        print(f"    {ch}: {imp:.4f}")
    print(f"  RF CV accuracy: {rf.score(X_scaled, y):.2%}")  
    print("  Saved 06_feature_importance.png")
    return imp_df, top_chs


def analyze_tsne_embeddings(ds):
    print("\n[7/7] t-SNE embedding visualization...")
    win_len = int(2.0 * FS)
    stride = win_len
    n_fft = int(2 * FS)
    max_windows_per_subj = 30
    features_list = []
    labels_list = []

    for i in range(len(ds)):
        _, eeg, label = ds[i]
        eeg_np = eeg.cpu().numpy()
        windows = [
            eeg_np[:, s:s + win_len]
            for s in range(0, eeg_np.shape[1] - win_len + 1, stride)
        ][:max_windows_per_subj]
        for w in windows:
            feats = []
            for _, (lo, hi) in BANDS.items():
                freqs, psd = signal.welch(w, fs=FS, nperseg=n_fft, axis=-1)
                mask = (freqs >= lo) & (freqs <= hi)
                feats.append(psd[:, mask].mean())
            features_list.append(np.concatenate(feats))
            labels_list.append(label.item())

    X = np.array(features_list)
    y = np.array(labels_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = min(3000, len(X))
    if len(X) > n_samples:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X), n_samples, replace=False)
        X_sample = X_scaled[idx]
        y_sample = y[idx]
    else:
        X_sample = X_scaled
        y_sample = y

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, n_jobs=-1)
    emb = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl, color, marker, name in [(0, "#4A90D9", "o", "HC"), (1, "#E74C3C", "x", "MDD")]:
        mask = y_sample == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, marker=marker, label=name,
                   alpha=0.6, s=15, edgecolors="none")
    ax.set_title("t-SNE: Window Embeddings (Band Power Features)", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "07_tsne_embeddings.png", dpi=150)
    plt.close(fig)
    print(f"  t-SNE on {len(X_sample)} windows ({len(BANDS)} bands)")
    print("  Saved 07_tsne_embeddings.png")


def main():
    print("=" * 60)
    print("MODMA EXPLORATORY ANALYSIS")
    print("=" * 60)
    ds, hc_count, mdd_count, ch_names, n_ch, T = load_data()

    analyze_temporal_distribution(ds, n_ch, T)
    analyze_spectral_profiles(ds, n_ch, ch_names)
    analyze_channel_correlation(ds, n_ch)
    band_stats = analyze_bandpower_boxplot(ds)
    df_win = analyze_window_sizes(ds)
    imp_df, top_chs = analyze_feature_importance(ds)
    analyze_tsne_embeddings(ds)

    summary = {
        "dataset": {
            "subjects": len(ds),
            "hc": int(hc_count),
            "mdd": int(mdd_count),
            "channels": n_ch,
            "samples_per_subject": int(T),
            "duration_sec": T / FS,
            "fs": FS,
        },
        "spectral": {
            band: {"p_value": s["p_value"], "hc_mean_log": s["hc_mean"], "mdd_mean_log": s["mdd_mean"]}
            for band, s in band_stats.items()
        },
        "window_optimization": {
            "best_window_sec": float(df_win.loc[df_win["balanced_acc_mean"].idxmax(), "window_sec"]),
            "best_balanced_accuracy": float(df_win["balanced_acc_mean"].max()),
        },
        "top_channels": [{"channel": ch, "importance": float(imp)} for ch, imp in top_chs[:10]],
    }

    with open(OUT / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {OUT / 'summary_report.json'}")
    print("=" * 60)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
