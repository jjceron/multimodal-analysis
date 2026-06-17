from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.modma_db import MODMADataset, DEFAULT_ROOT

OUT = Path("outputs/exploratory/modma")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}
FS = 250.0


def compute_bandpower(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[:, idx], freqs[idx], axis=1)


def main():
    print("Loading MODMA dataset...")
    ds = MODMADataset(root=DEFAULT_ROOT, lowcut=0.5, highcut=60.0, notch=50.0, target_fs=FS)

    n_subj = len(ds)
    n_ch = len(ds.channel_names)
    print(f"  Subjects: {n_subj}, Channels: {n_ch}, Samples: {ds.samples[0]['eeg'].shape[1]}")

    labels = np.array([int(s["label"].item()) for s in ds.samples])
    group_names = np.where(labels == 0, "HC", "MDD")

    stats_rows = []
    bp_rows = []
    corr_hc = np.zeros((n_ch, n_ch))
    corr_mdd = np.zeros((n_ch, n_ch))
    count_hc, count_mdd = 0, 0

    for i, s in enumerate(ds.samples):
        eeg = s["eeg"].numpy()
        label = int(s["label"].item())
        pid = s["participant_id"]
        gname = "HC" if label == 0 else "MDD"

        for ch_idx in range(n_ch):
            sig = eeg[ch_idx]
            stats_rows.append({
                "subject": pid, "group": gname, "channel": ch_idx,
                "mean": float(np.mean(sig)),
                "std": float(np.std(sig)),
                "energy": float(np.sum(sig ** 2)),
                "skewness": float(skew(sig)),
                "kurtosis": float(kurtosis(sig)),
            })

        freqs, psd = welch(eeg, fs=FS, nperseg=min(256, eeg.shape[1]), axis=1)
        total_power = np.trapezoid(psd, freqs, axis=1)
        for bname, band in FREQ_BANDS.items():
            bp = compute_bandpower(psd, freqs, band)
            for ch_idx in range(n_ch):
                bp_rows.append({
                    "subject": pid, "group": gname, "channel": ch_idx,
                    "band": bname,
                    "absolute": float(bp[ch_idx]),
                    "relative": float(bp[ch_idx] / total_power[ch_idx]) if total_power[ch_idx] > 0 else 0.0,
                })

        corr = np.corrcoef(eeg)
        if label == 0:
            corr_hc += corr
            count_hc += 1
        else:
            corr_mdd += corr
            count_mdd += 1

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(OUT / "signal_stats.csv", index=False)
    print(f"  signal_stats.csv: {len(df_stats)} rows")

    df_bp = pd.DataFrame(bp_rows)
    df_bp.to_csv(OUT / "bandpower.csv", index=False)
    print(f"  bandpower.csv: {len(df_bp)} rows")

    corr_hc /= count_hc
    corr_mdd /= count_mdd
    corr_diff = corr_hc - corr_mdd
    triu_idx = np.triu_indices(n_ch, k=1)
    corr_diff_flat = corr_diff[triu_idx]
    df_corr = pd.DataFrame({
        "channel_i": triu_idx[0],
        "channel_j": triu_idx[1],
        "corr_hc": corr_hc[triu_idx],
        "corr_mdd": corr_mdd[triu_idx],
        "diff_hc_mdd": corr_diff_flat,
    })
    df_corr.to_csv(OUT / "corr_diff.csv", index=False)
    print(f"  corr_diff.csv: {len(df_corr)} rows")

    plt.rcParams.update({"font.size": 8, "figure.dpi": 120})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in zip(axes, [corr_hc, corr_mdd], ["HC correlation", "MDD correlation"]):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    fig.savefig(FIG / "correlation_matrices.png", dpi=120, pil_kwargs={"optimize": True})
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(corr_diff, vmin=-0.3, vmax=0.3, cmap="RdBu_r", aspect="auto")
    ax.set_title("Correlation difference (HC - MDD)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIG / "corr_diff_heatmap.png", dpi=120, pil_kwargs={"optimize": True})
    plt.close(fig)

    bp_means = df_bp.groupby(["group", "channel", "band"])["absolute"].mean().reset_index()
    bp_diff = bp_means.pivot_table(index=["channel", "band"], columns="group", values="absolute").reset_index()
    bp_diff.columns.name = None
    bp_diff["diff"] = (bp_diff.get("HC", 0) - bp_diff.get("MDD", 0)).abs()
    top_channels = {}
    for band in FREQ_BANDS:
        sub = bp_diff[bp_diff["band"] == band].nlargest(5, "diff")
        top_channels[band] = sub["channel"].values[:5]

    all_top = sorted(set(np.concatenate(list(top_channels.values()))))
    ch_subset = df_bp[df_bp["channel"].isin(all_top)]
    bp_plot = ch_subset.groupby(["group", "channel", "band"])["absolute"].mean().reset_index()
    colors = {"HC": "#4A90D9", "MDD": "#E74C3C"}
    n_bands = len(FREQ_BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(n_bands * 2.5, 3.5), sharey=True)
    for ax, (bname, _) in zip(axes, FREQ_BANDS.items()):
        sub = bp_plot[bp_plot["band"] == bname]
        x = np.arange(len(all_top))
        w = 0.3
        for gi, gname in enumerate(["HC", "MDD"]):
            vals = sub[sub["group"] == gname].set_index("channel")["absolute"].reindex(all_top).fillna(0)
            ax.bar(x + gi * w - w / 2, vals.values, w, color=colors[gname], label=gname if bname == "delta" else "")
        ax.set_xticks(x, [str(c) for c in all_top], fontsize=6)
        ax.set_title(bname, fontsize=9)
        ax.set_xlabel("Channel")
    axes[0].set_ylabel("Abs. power")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bandpower_top_channels.png", dpi=120, pil_kwargs={"optimize": True})
    plt.close(fig)

    print(f"Figures saved to {FIG}")
    print("Done.")


if __name__ == "__main__":
    main()
