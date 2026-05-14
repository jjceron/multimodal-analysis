from __future__ import annotations

from pathlib import Path
import re

import yaml
import numpy as np
import pandas as pd


def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_subject_id(file_path: str | Path) -> int:
    file_path = Path(file_path)

    match = re.search(r"ID(\d+)", file_path.stem, flags=re.IGNORECASE)

    if match is None:
        raise ValueError(f"Could not extract subject ID from {file_path.name}")

    return int(match.group(1))


def normalize_condition(condition: str) -> str:
    condition = condition.lower().strip()

    aliases = {
        "complete": "complete",
        "all": "complete",
        "full": "complete",
        "open": "open",
        "closed": "closed",
    }

    if condition not in aliases:
        raise ValueError("condition must be one of: complete, open, closed")

    return aliases[condition]


def assign_epoch_conditions(
    epoch_start_sec: np.ndarray,
    epoch_end_sec: np.ndarray,
    eyes_split_sec: float = 300.0,
) -> np.ndarray:
    labels = np.full(len(epoch_start_sec), "transition", dtype=object)

    labels[epoch_end_sec <= eyes_split_sec] = "open"
    labels[epoch_start_sec >= eyes_split_sec] = "closed"

    return labels


def build_condition_masks(condition_labels: np.ndarray) -> dict[str, np.ndarray]:
    complete_mask = np.ones(len(condition_labels), dtype=bool)
    open_mask = condition_labels == "open"
    closed_mask = condition_labels == "closed"
    transition_mask = condition_labels == "transition"

    return {
        "complete": complete_mask,
        "open": open_mask,
        "closed": closed_mask,
        "transition": transition_mask,
    }


def build_label_table(questionnaire_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(questionnaire_path)

    id_col = "ID"
    pain_col = "Pain Score (Actual Pain of Brief Pain Inventory)"

    subject_ids = pd.to_numeric(df[id_col], errors="coerce")
    pain_scores = pd.to_numeric(df[pain_col], errors="coerce")

    pain_labels = pd.cut(
        pain_scores,
        bins=[0, 3, 6, 10],
        labels=["low", "moderate", "severe"],
        include_lowest=True,
    )

    pain_codes = pd.Categorical(
        pain_labels,
        categories=["low", "moderate", "severe"],
        ordered=True,
    ).codes

    label_table = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "pain_scale": pain_scores,
            "pain_scale_label": pain_labels.astype("string"),
            "pain_scale_code": pain_codes,
        }
    )

    label_table = label_table.dropna(
        subset=["subject_id", "pain_scale", "pain_scale_label"]
    ).copy()

    label_table["subject_id"] = label_table["subject_id"].astype(int)
    label_table["pain_scale_code"] = label_table["pain_scale_code"].astype(int)

    duplicated = label_table["subject_id"].duplicated(keep=False)

    if duplicated.any():
        duplicated_ids = sorted(
            label_table.loc[duplicated, "subject_id"].unique().tolist()
        )
        raise ValueError(f"Duplicate subject IDs in questionnaire: {duplicated_ids}")

    return label_table


def pick_eeg_channels(raw):
    eeg_channels = [
        channel
        for channel, channel_type in zip(raw.ch_names, raw.get_channel_types())
        if channel_type == "eeg"
    ]

    if len(eeg_channels) == 0:
        raw.set_channel_types(
            {
                channel: "eeg"
                for channel in raw.ch_names
            }
        )
        eeg_channels = raw.ch_names.copy()

    raw.pick(eeg_channels)

    return raw


def replace_non_finite_raw_values(raw):
    data = raw.get_data()

    if not np.isfinite(data).all():
        raw._data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return raw


def build_epoch_metadata(
    eeg_data: np.ndarray,
    epochs,
    raw,
    original_sfreq: float,
    original_duration_sec: float,
    window: float,
    overlap: float,
    eyes_split_sec: float,
) -> dict:
    sfreq = float(raw.info["sfreq"])
    duration_sec = raw.n_times / sfreq

    epoch_start_sec = (
        epochs.events[:, 0].astype(float) - float(raw.first_samp)
    ) / sfreq

    epoch_start_sec = np.maximum(epoch_start_sec, 0.0)
    epoch_end_sec = epoch_start_sec + window

    condition_labels = assign_epoch_conditions(
        epoch_start_sec=epoch_start_sec,
        epoch_end_sec=epoch_end_sec,
        eyes_split_sec=eyes_split_sec,
    )

    masks = build_condition_masks(condition_labels)

    return {
        "original_sfreq": original_sfreq,
        "sfreq": sfreq,
        "original_duration_sec": original_duration_sec,
        "duration_sec": duration_sec,
        "n_channels": int(eeg_data.shape[1]),
        "n_samples": int(eeg_data.shape[2]),
        "complete_n_epochs": int(np.sum(masks["complete"])),
        "open_n_epochs": int(np.sum(masks["open"])),
        "closed_n_epochs": int(np.sum(masks["closed"])),
        "transition_n_epochs": int(np.sum(masks["transition"])),
        "complete_shape": tuple(eeg_data[masks["complete"]].shape),
        "open_shape": tuple(eeg_data[masks["open"]].shape),
        "closed_shape": tuple(eeg_data[masks["closed"]].shape),
        "condition_masks": {
            "complete": masks["complete"],
            "open": masks["open"],
            "closed": masks["closed"],
        },
    }