from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sp_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.modma_db import MODMADataset, DEFAULT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exploratory" / "modma"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_PATH = OUTPUT_DIR / "summary_report.txt"

FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 60),
}
CLASS_NAMES = {0: "HC", 1: "MDD"}
CLASS_COLORS = {0: "#4A90D9", 1: "#E74C3C"}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
})


def log(msg: str) -> None:
    print(msg)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------
def analyze_dataset():
    print("Loading MODMA dataset...")
    ds = MODMADataset(
        root=DEFAULT_ROOT,
        lowcut=0.5,
        highcut=60.0,
        notch=50.0,
        duration_sec=120.0,
    )

    labels = [int(s["label"].item()) for s in ds.samples]
    cnt = Counter(labels)
    n_subjects = len(ds)
    n_channels = ds.samples[0]["eeg"].shape[0]
    n_timepoints = ds.samples[0]["eeg"].shape[1]
    sfreq = 250.0

    log("=" * 60)
    log("DATASET OVERVIEW")
    log("=" * 60)
    log(f"Total subjects: {n_subjects}")
    log(f"  HC  (class 0): {cnt.get(0, 0)} ({cnt.get(0, 0)/n_subjects*100:.1f}%)")
    log(f"  MDD (class 1): {cnt.get(1, 0)} ({cnt.get(1, 0)/n_subjects*100:.1f}%)")
    log(f"Channels: {n_channels}")
    log(f"Timepoints per sample: {n_timepoints}")
    log(f"Sampling rate: {sfreq} Hz")
    log(f"Duration: {n_timepoints/sfreq:.0f} s")
    log(f"Condition: Eyes Closed (EC)")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [CLASS_COLORS[0], CLASS_COLORS[1]]
    bars = ax.bar(["HC", "MDD"], [cnt.get(0, 0), cnt.get(1, 0)],
                  color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, [cnt.get(0, 0), cnt.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Class Distribution in MODMA Dataset")
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close(fig)
    log(f"Plot saved: class_distribution.png")

    info = {
        "n_subjects": n_subjects,
        "n_hc": cnt.get(0, 0),
        "n_mdd": cnt.get(1, 0),
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "sfreq": sfreq,
        "condition": "EC",
    }
    (OUTPUT_DIR / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return ds, n_channels, n_timepoints, sfreq


# ---------------------------------------------------------------------------
# 2. Per-subject prediction analysis from benchmark
# ---------------------------------------------------------------------------
def analyze_predictions(benchmark_dir: Path, label: str):
    pred_csv = benchmark_dir / "predictions.csv"
    if not pred_csv.exists():
        log(f"  No predictions.csv found at {benchmark_dir}, skipping per-subject analysis")
        return

    df = pd.read_csv(pred_csv)
    log("\n" + "=" * 60)
    log(f"PER-SUBJECT ANALYSIS — {label}")
    log("=" * 60)

    subject_stats = (
        df.groupby("subject")
        .agg(
            total=("true_label", "count"),
            correct=("pred_label", lambda x: (x == df.loc[x.index, "true_label"]).sum()),
            true_class=("true_label", "first"),
        )
        .reset_index()
    )
    subject_stats["accuracy"] = subject_stats["correct"] / subject_stats["total"]

    log(f"  Total predictions: {len(df)}")
    log(f"  Subjects: {subject_stats['subject'].nunique()}")
    worst = subject_stats.nsmallest(5, "accuracy")
    log(f"\n  5 worst classified subjects:")
    for _, row in worst.iterrows():
        label_str = CLASS_NAMES.get(int(row["true_class"]), str(row["true_class"]))
        log(f"    {row['subject']} ({label_str}): {row['correct']}/{row['total']} = {row['accuracy']:.0%}")

    best = subject_stats.nlargest(5, "accuracy")
    log(f"\n  5 best classified subjects:")
    for _, row in best.iterrows():
        label_str = CLASS_NAMES.get(int(row["true_class"]), str(row["true_class"]))
        log(f"    {row['subject']} ({label_str}): {row['correct']}/{row['total']} = {row['accuracy']:.0%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cls, ax in [(0, axes[0]), (1, axes[1])]:
        sub = subject_stats[subject_stats["true_class"] == cls]
        ax.bar(range(len(sub)), sub["accuracy"].values, color=CLASS_COLORS[cls],
               edgecolor="black", linewidth=0.5)
        ax.axhline(sub["accuracy"].mean(), color="gray", linestyle="--",
                   label=f"Mean: {sub['accuracy'].mean():.0%}")
        ax.set_title(f"{CLASS_NAMES[cls]} Subjects")
        ax.set_xlabel("Subject index")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        sns.despine(ax=ax)

    fig.suptitle(f"Per-Subject Classification Accuracy — {label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"per_subject_accuracy_{label.replace('/', '_')}.png")
    plt.close(fig)

    subject_stats.to_csv(OUTPUT_DIR / f"per_subject_stats_{label.replace('/', '_')}.csv", index=False)
    log(f"  Saved: per_subject_stats_{label.replace('/', '_')}.csv")


# ---------------------------------------------------------------------------
# 3. Power Spectral Density analysis
# ---------------------------------------------------------------------------
def analyze_psd(ds, sfreq: float):
    log("\n" + "=" * 60)
    log("POWER SPECTRAL DENSITY ANALYSIS")
    log("=" * 60)

    labels = [int(s["label"].item()) for s in ds.samples]
    class_indices = {0: [], 1: []}
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    n_channels = ds.samples[0]["eeg"].shape[0]
    nperseg = int(4 * sfreq)

    psd_all = {}
    for cls in (0, 1):
        idx = class_indices[cls]
        freqs = None
        psd_sum = None
        for i in idx:
            eeg = ds.samples[i]["eeg"].cpu().numpy().astype(np.float64)
            f, Pxx = sp_signal.welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1)
            if freqs is None:
                freqs = f
                psd_sum = np.zeros((n_channels, len(f)))
            psd_sum += Pxx
        psd_all[cls] = {
            "freqs": freqs,
            "mean_psd": psd_sum / len(idx),
            "n_subjects": len(idx),
        }
        log(f"  {CLASS_NAMES[cls]}: {len(idx)} subjects, PSD computed")

    ch_names = getattr(ds, 'ch_names', None) or [str(i) for i in range(n_channels)]

    # --- Full spectrum plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for cls in (0, 1):
        p = psd_all[cls]
        mean_p = np.mean(p["mean_psd"], axis=0)
        ax.semilogy(p["freqs"], mean_p, color=CLASS_COLORS[cls],
                    label=f"{CLASS_NAMES[cls]} (n={p['n_subjects']})", linewidth=2)
    for name, (lo, hi) in FREQ_BANDS.items():
        ax.axvspan(lo, hi, alpha=0.08, color="gray")
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95, name,
                ha="center", fontsize=9, fontweight="bold", alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax.set_title("Average PSD — HC vs MDD (all channels)")
    ax.legend()
    ax.set_xlim(0.5, 60)
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "psd_full_spectrum.png")
    plt.close(fig)
    log("  Plot saved: psd_full_spectrum.png")

    # --- Band power per class ---
    log("\n  Band power comparison (mean over channels):")
    band_power_rows = []
    for band_name, (lo, hi) in FREQ_BANDS.items():
        row = {"band": band_name}
        for cls in (0, 1):
            p = psd_all[cls]
            mask = (p["freqs"] >= lo) & (p["freqs"] <= hi)
            bp = np.trapz(p["mean_psd"][:, mask], p["freqs"][mask], axis=1)
            row[f"{CLASS_NAMES[cls]}_mean"] = np.mean(bp)
            row[f"{CLASS_NAMES[cls]}_std"] = np.std(bp)
        band_power_rows.append(row)
        log(f"    {band_name:6s}: HC={row['HC_mean']:.2e}±{row['HC_std']:.2e}  "
            f"MDD={row['MDD_mean']:.2e}±{row['MDD_std']:.2e}")

    df_bp = pd.DataFrame(band_power_rows)
    df_bp.to_csv(OUTPUT_DIR / "band_power_summary.csv", index=False)

    # --- Band power bar plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FREQ_BANDS))
    w = 0.35
    hc_means = df_bp["HC_mean"].values
    mdd_means = df_bp["MDD_mean"].values
    ax.bar(x - w / 2, hc_means, w, label="HC", color=CLASS_COLORS[0], edgecolor="black")
    ax.bar(x + w / 2, mdd_means, w, label="MDD", color=CLASS_COLORS[1], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(list(FREQ_BANDS.keys()))
    ax.set_ylabel("Mean Band Power (µV²)")
    ax.set_title("Band Power Comparison: HC vs MDD")
    ax.legend()
    sns.despine()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "band_power_comparison.png")
    plt.close(fig)
    log("  Plot saved: band_power_comparison.png")

    # --- Channel-wise discriminability ---
    log("\n  Top-10 most discriminative channels (by Alpha band power ratio):")
    lo, hi = FREQ_BANDS["Alpha"]
    freqs = psd_all[0]["freqs"]
    mask = (freqs >= lo) & (freqs <= hi)

    channel_ratios = []
    for ch in range(n_channels):
        bp_hc = np.trapz(psd_all[0]["mean_psd"][ch, mask], freqs[mask])
        bp_mdd = np.trapz(psd_all[1]["mean_psd"][ch, mask], freqs[mask])
        ratio = bp_mdd / max(bp_hc, 1e-12)
        channel_ratios.append({"channel": ch_names[ch], "hc_power": bp_hc,
                               "mdd_power": bp_mdd, "ratio_mdd_hc": ratio})

    df_ch = pd.DataFrame(channel_ratios).sort_values("ratio_mdd_hc", ascending=False)
    df_ch.to_csv(OUTPUT_DIR / "channel_discriminability_alpha.csv", index=False)

    for _, row in df_ch.head(10).iterrows():
        log(f"    {row['channel']:10s}: MDD/HC ratio = {row['ratio_mdd_hc']:.3f}  "
            f"(HC={row['hc_power']:.2e}, MDD={row['mdd_power']:.2e})")

    # --- Topographic-like plot of alpha power per class ---
    n_cols = 8
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))
    axes = axes.flatten()
    for ch in range(n_channels):
        ax = axes[ch]
        for cls, style in [(0, {"color": CLASS_COLORS[0], "linestyle": "-"}),
                           (1, {"color": CLASS_COLORS[1], "linestyle": "--"})]:
            p = psd_all[cls]
            ax.plot(p["freqs"], p["mean_psd"][ch, :], **style, alpha=0.7, linewidth=0.8,
                    label=CLASS_NAMES[cls] if ch == 0 else "")
        ax.set_xlim(0.5, 60)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ch_names[ch], fontsize=7)
    for ch in range(n_channels, len(axes)):
        axes[ch].axis("off")
    handles = [plt.Line2D([], [], color=CLASS_COLORS[0], label="HC"),
               plt.Line2D([], [], color=CLASS_COLORS[1], linestyle="--", label="MDD")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("PSD per Channel — HC vs MDD", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "psd_per_channel_grid.png", bbox_inches="tight")
    plt.close(fig)
    log("  Plot saved: psd_per_channel_grid.png")

    return df_ch, df_bp


# ---------------------------------------------------------------------------
# 4. Demographics analysis
# ---------------------------------------------------------------------------
def analyze_demographics():
    tsv_path = (
        PROJECT_ROOT / "data" / "raw" / "modma" / "MODMA_EEG_BIDS_format"
        / "EEG_LZU_2015_2_resting state" / "participants.tsv"
    )
    if not tsv_path.exists():
        log("  participants.tsv not found, skipping demographics")
        return

    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [c.replace("\uff08", "(").replace("\uff09", ")") for c in df.columns]
    log("  Columns: %s" % str(list(df.columns)))

    num_cols_all = ["age", "education(years)", "PHQ-9", "GAD-7", "PSQI"]
    for col in num_cols_all:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log("\n" + "=" * 60)
    log("DEMOGRAPHICS")
    log("=" * 60)

    if "group" in df.columns:
        log("  group:")
        for val, cnt in df["group"].value_counts().items():
            log(f"    {val}: {cnt}")

    for col in df.columns:
        if col in ("participant_id", "group"):
            continue
        series = df[col].dropna()
        if series.dtype.kind in "iuf":
            log(f"  {col}: mean={series.mean():.1f} +/- {series.std():.1f}, "
                f"range=[{series.min()}, {series.max()}]")

    df.to_csv(OUTPUT_DIR / "demographics.csv", index=False)

    if "age" in df.columns and "group" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        for grp_name, color in [("HC", CLASS_COLORS[0]), ("MDD", CLASS_COLORS[1])]:
            sub = df[df["group"] == grp_name]["age"].dropna()
            if not sub.empty:
                ax.hist(sub, bins=10, alpha=0.6, color=color, label=grp_name, edgecolor="black")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution by Group")
        ax.legend()
        sns.despine()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "age_distribution.png")
        plt.close(fig)
        log("  Plot saved: age_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    ds, n_channels, n_timepoints, sfreq = analyze_dataset()
    analyze_demographics()

    benchmark_models = [
        ("EEGNet/Aggregation_b16",
         PROJECT_ROOT / "outputs" / "models" / "modma_db" / "EEGNet" / "Aggregation_b16"),
        ("EEGNet/Aggregation",
         PROJECT_ROOT / "outputs" / "models" / "modma_db" / "EEGNet" / "Aggregation"),
    ]
    for label, path in benchmark_models:
        if path.exists():
            analyze_predictions(path, label)

    df_ch, df_bp = analyze_psd(ds, sfreq)

    log("\n" + "=" * 60)
    log("SUMMARY & RECOMMENDATIONS")
    log("=" * 60)
    log("1. Class balance: moderate imbalance (29 HC vs 24 MDD) — use balanced accuracy")
    log("2. 128-channel data at 250 Hz, 120s resting-state EC")
    log("3. Key frequency bands for discrimination: review alpha band channel map")
    log("4. High inter-subject variability — consider subject-specific normalization")
    log("5. Current best test accuracy: ~58% (EEGNet, batch=16)")
    log("6. Next steps: try CSP+LDA, Riemannian MDM, BandPower+SVM baselines")
    log("\nFull outputs saved to: %s" % OUTPUT_DIR)
    print("\nDone. Report saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()
