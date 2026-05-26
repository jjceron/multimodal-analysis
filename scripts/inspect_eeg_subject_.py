from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.build_eeg import EEGDataset


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug_eeg"


def compute_channel_stats(channel_epochs: np.ndarray) -> dict:
    peak_to_peak = np.ptp(channel_epochs, axis=1)
    rms = np.sqrt(np.mean(channel_epochs**2, axis=1))

    return {
        "mean": float(np.mean(channel_epochs)),
        "std": float(np.std(channel_epochs)),
        "min": float(np.min(channel_epochs)),
        "max": float(np.max(channel_epochs)),
        "abs_max": float(np.max(np.abs(channel_epochs))),
        "p2p_median": float(np.median(peak_to_peak)),
        "p2p_max": float(np.max(peak_to_peak)),
        "rms_mean": float(np.mean(rms)),
        "flat_epochs": int(np.sum(peak_to_peak < 1.0)),
        "high_amp_epochs": int(np.sum(peak_to_peak > 200.0)),
    }


def compute_psd_features(
    channel_epochs: np.ndarray,
    sfreq: float,
    lowcut: float,
    highcut: float,
) -> dict:
    nperseg = min(channel_epochs.shape[1], int(2 * sfreq))

    freqs, psd = welch(
        channel_epochs,
        fs=sfreq,
        nperseg=nperseg,
        axis=1,
    )

    mean_psd = psd.mean(axis=0)

    valid_mask = (freqs >= lowcut) & (freqs <= highcut)

    if np.any(valid_mask):
        valid_freqs = freqs[valid_mask]
        valid_psd = mean_psd[valid_mask]
        dominant_freq = float(valid_freqs[np.argmax(valid_psd)])
    else:
        dominant_freq = float(freqs[np.argmax(mean_psd)])

    bands = []

    if highcut > 1.0:
        bands.append(("Delta", max(lowcut, 1.0), min(highcut, 4.0)))

    if highcut > 4.0:
        bands.append(("Theta", max(lowcut, 4.0), min(highcut, 8.0)))

    if highcut > 8.0:
        bands.append(("Alpha", max(lowcut, 8.0), min(highcut, 13.0)))

    if highcut > 13.0:
        bands.append(("Beta", max(lowcut, 13.0), min(highcut, 30.0)))

    if highcut > 30.0:
        bands.append(("Low gamma", max(lowcut, 30.0), min(highcut, 45.0)))

    band_powers = {}

    for band_name, low, high in bands:
        if high <= low:
            continue

        mask = (freqs >= low) & (freqs < high)

        if not np.any(mask):
            power = 0.0
        else:
            power = float(np.trapezoid(mean_psd[mask], freqs[mask]))

        band_powers[band_name] = {
            "low": low,
            "high": high,
            "power": power,
        }

    total_power = sum(item["power"] for item in band_powers.values())

    if total_power <= 0:
        total_power = 1e-12

    for band_name in band_powers:
        band_powers[band_name]["relative"] = (
            band_powers[band_name]["power"] / total_power
        )

    return {
        "dominant_freq": dominant_freq,
        "total_power": total_power,
        "bands": band_powers,
    }


def robust_ylim(channel_epochs: np.ndarray) -> tuple[float, float]:
    low = np.percentile(channel_epochs, 1)
    high = np.percentile(channel_epochs, 99)

    if low == high:
        margin = max(abs(low) * 0.1, 1.0)
        return low - margin, high + margin

    margin = 0.15 * (high - low)

    return low - margin, high + margin


def build_channel_text(
    channel_idx: int,
    n_epochs: int,
    n_samples: int,
    sfreq: float,
    units: str,
    stats: dict,
    psd: dict,
) -> str:
    epoch_duration = n_samples / sfreq

    psd_lines = [
        "PSD summary",
        f"Dominant freq: {psd['dominant_freq']:.2f} Hz",
    ]

    for band_name, item in psd["bands"].items():
        psd_lines.append(
            f"{band_name} {item['low']:.1f}-{item['high']:.1f} Hz: "
            f"{item['relative'] * 100:.1f}%"
        )

    psd_text = "\n".join(psd_lines)

    return (
        f"Channel {channel_idx}\n"
        f"Epochs: {n_epochs}\n"
        f"Samples/epoch: {n_samples}\n"
        f"Epoch length: {epoch_duration:.2f} s\n\n"
        f"Time-domain ({units})\n"
        f"Mean: {stats['mean']:.3f}\n"
        f"Std: {stats['std']:.3f}\n"
        f"Min / Max: {stats['min']:.1f} / {stats['max']:.1f}\n"
        f"Abs max: {stats['abs_max']:.1f}\n"
        f"Median P2P: {stats['p2p_median']:.1f}\n"
        f"Max P2P: {stats['p2p_max']:.1f}\n"
        f"Mean RMS: {stats['rms_mean']:.1f}\n"
        f"Flat epochs: {stats['flat_epochs']}\n"
        f"High-amp epochs: {stats['high_amp_epochs']}\n\n"
        f"{psd_text}"
    )


def inspect_random_subject(condition: str):
    dataset = EEGDataset(condition=condition)

    subject_idx = random.randrange(len(dataset))
    subject_id, eeg_tensor, label_tensor = dataset[subject_idx]

    eeg = eeg_tensor.numpy()
    n_epochs, n_channels, n_samples = eeg.shape

    selected_channels = random.sample(range(n_channels), k=3)

    sample = dataset.samples[subject_idx]
    label_name = sample["label_name"]
    label_code = int(label_tensor.item())
    sfreq = float(sample["metadata"]["sfreq"])
    units = "μV" if dataset.scale_to_uv else "V"

    time_sec = np.arange(n_samples) / sfreq

    notch_text = "None" if dataset.notch is None else f"{dataset.notch} Hz"

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(18, 13),
        gridspec_kw={"width_ratios": [2.6, 1.0]},
    )

    fig.suptitle(
        (
            f"EEG subject inspection | ID{subject_id} | "
            f"{dataset.condition} | {label_name} ({label_code})\n"
            f"Config: average reference | bandpass {dataset.lowcut}-{dataset.highcut} Hz | "
            f"notch {notch_text} | fs {dataset.target_fs} Hz | "
            f"window {dataset.window}s | overlap {dataset.overlap} | scale {units}"
        ),
        fontsize=13,
    )

    for row, channel_idx in enumerate(selected_channels):
        channel_epochs = eeg[:, channel_idx, :]

        ax_signal = axes[row, 0]
        ax_text = axes[row, 1]

        ax_signal.plot(
            time_sec,
            channel_epochs.T,
            linewidth=0.45,
            alpha=0.22,
        )

        y_min, y_max = robust_ylim(channel_epochs)
        ax_signal.set_ylim(y_min, y_max)

        ax_signal.set_title(f"Channel {channel_idx} | all epochs")
        ax_signal.set_xlabel("Time (s)")
        ax_signal.set_ylabel(f"Amplitude ({units})")
        ax_signal.grid(True, alpha=0.25)

        stats = compute_channel_stats(channel_epochs)

        psd = compute_psd_features(
            channel_epochs=channel_epochs,
            sfreq=sfreq,
            lowcut=float(dataset.lowcut),
            highcut=float(dataset.highcut),
        )

        text = build_channel_text(
            channel_idx=channel_idx,
            n_epochs=n_epochs,
            n_samples=n_samples,
            sfreq=sfreq,
            units=units,
            stats=stats,
            psd=psd,
        )

        ax_text.axis("off")
        ax_text.text(
            0.0,
            1.0,
            text,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            linespacing=1.15,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"subject_ID{subject_id}_{dataset.condition}_inspection.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    print(f"Subject ID: {subject_id}")
    print(f"Label: {label_name} ({label_code})")
    print(f"Condition: {dataset.condition}")
    print(f"EEG shape: {eeg.shape}")
    print(f"Selected channels: {selected_channels}")
    print(f"Config path: {dataset.config_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="closed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_random_subject(condition=args.condition)