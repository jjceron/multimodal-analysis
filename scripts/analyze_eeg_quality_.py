from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.build_eeg import EEGDataset


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug_eeg"


def to_uv(eeg: np.ndarray, scale_to_uv: bool) -> np.ndarray:
    if scale_to_uv:
        return eeg

    return eeg * 1e6


def analyze_subject(
    subject_id: int,
    label_name: str,
    label_code: int,
    eeg_uv: np.ndarray,
) -> tuple[dict, list[dict]]:
    n_epochs, n_channels, n_samples = eeg_uv.shape

    p2p = np.ptp(eeg_uv, axis=2)
    abs_max = np.max(np.abs(eeg_uv), axis=2)
    rms = np.sqrt(np.mean(eeg_uv**2, axis=2))

    bad_epoch_200 = np.any(p2p > 200.0, axis=1)
    bad_epoch_300 = np.any(p2p > 300.0, axis=1)
    bad_epoch_500 = np.any(p2p > 500.0, axis=1)
    bad_epoch_1000 = np.any(p2p > 1000.0, axis=1)

    worst_channel = int(np.argmax(np.max(p2p, axis=0)))

    subject_row = {
        "subject_id": subject_id,
        "label_name": label_name,
        "label_code": label_code,
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "bad_epochs_p2p_gt_200": int(np.sum(bad_epoch_200)),
        "bad_epochs_p2p_gt_300": int(np.sum(bad_epoch_300)),
        "bad_epochs_p2p_gt_500": int(np.sum(bad_epoch_500)),
        "bad_epochs_p2p_gt_1000": int(np.sum(bad_epoch_1000)),
        "pct_bad_epochs_p2p_gt_200": float(np.mean(bad_epoch_200) * 100),
        "pct_bad_epochs_p2p_gt_300": float(np.mean(bad_epoch_300) * 100),
        "pct_bad_epochs_p2p_gt_500": float(np.mean(bad_epoch_500) * 100),
        "pct_bad_epochs_p2p_gt_1000": float(np.mean(bad_epoch_1000) * 100),
        "global_max_abs_uv": float(np.max(abs_max)),
        "global_max_p2p_uv": float(np.max(p2p)),
        "median_p2p_uv": float(np.median(p2p)),
        "mean_rms_uv": float(np.mean(rms)),
        "worst_channel": worst_channel,
        "worst_channel_max_p2p_uv": float(np.max(p2p[:, worst_channel])),
    }

    channel_rows = []

    for channel_idx in range(n_channels):
        channel_p2p = p2p[:, channel_idx]
        channel_abs = abs_max[:, channel_idx]
        channel_rms = rms[:, channel_idx]

        row = {
            "subject_id": subject_id,
            "label_name": label_name,
            "label_code": label_code,
            "channel": channel_idx,
            "n_epochs": n_epochs,
            "mean_uv": float(np.mean(eeg_uv[:, channel_idx, :])),
            "std_uv": float(np.std(eeg_uv[:, channel_idx, :])),
            "median_p2p_uv": float(np.median(channel_p2p)),
            "mean_p2p_uv": float(np.mean(channel_p2p)),
            "max_p2p_uv": float(np.max(channel_p2p)),
            "mean_rms_uv": float(np.mean(channel_rms)),
            "max_abs_uv": float(np.max(channel_abs)),
            "flat_epochs_p2p_lt_1": int(np.sum(channel_p2p < 1.0)),
            "high_epochs_p2p_gt_200": int(np.sum(channel_p2p > 200.0)),
            "high_epochs_p2p_gt_300": int(np.sum(channel_p2p > 300.0)),
            "high_epochs_p2p_gt_500": int(np.sum(channel_p2p > 500.0)),
            "high_epochs_p2p_gt_1000": int(np.sum(channel_p2p > 1000.0)),
            "pct_high_epochs_p2p_gt_300": float(np.mean(channel_p2p > 300.0) * 100),
        }

        channel_rows.append(row)

    return subject_row, channel_rows


def main(condition: str):
    dataset = EEGDataset(condition=condition)

    subject_rows = []
    channel_rows = []

    for idx in range(len(dataset)):
        subject_id, eeg_tensor, label_tensor = dataset[idx]

        sample = dataset.samples[idx]
        label_name = sample["label_name"]
        label_code = int(label_tensor.item())

        eeg = eeg_tensor.numpy()
        eeg_uv = to_uv(eeg, scale_to_uv=dataset.scale_to_uv)

        subject_row, subject_channel_rows = analyze_subject(
            subject_id=int(subject_id),
            label_name=label_name,
            label_code=label_code,
            eeg_uv=eeg_uv,
        )

        subject_rows.append(subject_row)
        channel_rows.extend(subject_channel_rows)

    subjects_df = pd.DataFrame(subject_rows)
    channels_df = pd.DataFrame(channel_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_csv = OUTPUT_DIR / f"eeg_quality_subjects_{condition}.csv"
    channel_csv = OUTPUT_DIR / f"eeg_quality_channels_{condition}.csv"

    subjects_df.to_csv(subject_csv, index=False)
    channels_df.to_csv(channel_csv, index=False)

    print("\nEEG quality analysis")
    print(f"Condition: {condition}")
    print(f"Config path: {dataset.config_path}")
    print(f"Scale to uV in dataset: {dataset.scale_to_uv}")
    print(f"Reported units: uV")
    print(f"Subjects: {len(subjects_df)}")
    print(f"Subject CSV: {subject_csv}")
    print(f"Channel CSV: {channel_csv}")

    print("\nDataset preprocessing")
    print(f"Bandpass: {dataset.lowcut}-{dataset.highcut} Hz")
    print(f"Notch: {dataset.notch}")
    print(f"Target fs: {dataset.target_fs} Hz")
    print(f"Window: {dataset.window} s")
    print(f"Overlap: {dataset.overlap}")

    print("\nTop subjects by percentage of bad epochs, P2P > 300 uV")
    cols = [
        "subject_id",
        "label_name",
        "n_epochs",
        "pct_bad_epochs_p2p_gt_300",
        "global_max_p2p_uv",
        "global_max_abs_uv",
        "worst_channel",
        "worst_channel_max_p2p_uv",
    ]

    print(
        subjects_df.sort_values(
            "pct_bad_epochs_p2p_gt_300",
            ascending=False,
        )[cols]
        .head(15)
        .to_string(index=False)
    )

    print("\nTop subject-channel pairs by high-amplitude epochs, P2P > 300 uV")
    cols = [
        "subject_id",
        "label_name",
        "channel",
        "n_epochs",
        "high_epochs_p2p_gt_300",
        "pct_high_epochs_p2p_gt_300",
        "max_p2p_uv",
        "max_abs_uv",
        "median_p2p_uv",
    ]

    print(
        channels_df.sort_values(
            ["high_epochs_p2p_gt_300", "max_p2p_uv"],
            ascending=False,
        )[cols]
        .head(30)
        .to_string(index=False)
    )

    print("\nTop subject-channel pairs by maximum P2P")
    print(
        channels_df.sort_values(
            "max_p2p_uv",
            ascending=False,
        )[cols]
        .head(30)
        .to_string(index=False)
    )

    print("\nClass-level summary")
    print(
        subjects_df.groupby("label_name")[
            [
                "pct_bad_epochs_p2p_gt_300",
                "global_max_p2p_uv",
                "median_p2p_uv",
                "mean_rms_uv",
            ]
        ]
        .agg(["mean", "median", "max"])
        .round(3)
        .to_string()
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="closed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(condition=args.condition)