from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter


def extract_subject_id(file_path: str | Path) -> int:
    file_path = Path(file_path)

    match = re.search(r"ID(\d+)", file_path.stem, flags=re.IGNORECASE)

    if match is None:
        raise ValueError(f"Could not extract subject ID from {file_path.name}.")

    return int(match.group(1))


def normalize_condition(condition: str) -> str:
    condition = condition.lower().strip()

    aliases = {
        "all": "complete",
        "full": "complete",
        "complete": "complete",
        "open": "open",
        "closed": "closed",
    }

    if condition not in aliases:
        raise ValueError("condition must be one of: complete, all, full, open, closed.")

    return aliases[condition]


def assign_epoch_conditions(
    epoch_start_sec: np.ndarray,
    epoch_end_sec: np.ndarray,
    eyes_split_sec: float,
) -> np.ndarray:
    labels = np.full(len(epoch_start_sec), "transition", dtype=object)

    labels[epoch_end_sec <= eyes_split_sec] = "open"
    labels[epoch_start_sec >= eyes_split_sec] = "closed"

    return labels


def build_label_table(questionnaire_path, questionnaire_cfg) -> pd.DataFrame:
    id_col = questionnaire_cfg.get("id_column", "ID")
    pain_col = questionnaire_cfg["pain_scale"]["source_column"]
    bins = questionnaire_cfg["pain_scale"]["bins"]
    labels = questionnaire_cfg["pain_scale"]["labels"]

    df = pd.read_excel(questionnaire_path)

    rows_to_drop = questionnaire_cfg.get("rows_to_drop", [])
    if rows_to_drop:
        df = df.drop(index=rows_to_drop, errors="ignore")

    if id_col not in df.columns:
        raise KeyError(f"Column not found in questionnaire: {id_col}")

    if pain_col not in df.columns:
        raise KeyError(f"Column not found in questionnaire: {pain_col}")

    subject_ids = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    pain_scores = pd.to_numeric(df[pain_col], errors="coerce")

    pain_labels = pd.cut(
        pain_scores,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    pain_codes = pd.Categorical(
        pain_labels,
        categories=labels,
        ordered=True,
    ).codes

    pain_codes = pd.Series(pain_codes, index=df.index).replace({-1: pd.NA}).astype("Int64")

    label_table = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "pain_scale_label": pain_labels.astype("string"),
            "pain_scale_code": pain_codes,
        }
    )

    label_table = label_table.dropna(
        subset=["subject_id", "pain_scale_label", "pain_scale_code"]
    ).copy()

    label_table["subject_id"] = label_table["subject_id"].astype(int)
    label_table["pain_scale_code"] = label_table["pain_scale_code"].astype(int)

    duplicated = label_table["subject_id"].duplicated(keep=False)

    if duplicated.any():
        duplicated_ids = sorted(label_table.loc[duplicated, "subject_id"].unique().tolist())
        raise ValueError(f"Duplicate subject IDs in questionnaire: {duplicated_ids}")

    return label_table


def summarize_dataset_sizes(dataset) -> pd.DataFrame:
    summary = dataset.get_summary_dataframe()

    rows = []

    for condition, epoch_col in [
        ("complete", "complete_n_epochs"),
        ("open", "open_n_epochs"),
        ("closed", "closed_n_epochs"),
    ]:
        total_epochs = int(summary[epoch_col].sum())
        n_channels = int(summary["n_channels"].iloc[0])
        n_samples = int(summary["n_samples"].iloc[0])

        rows.append(
            {
                "condition": condition,
                "subjects": int(len(summary)),
                "total_epochs": total_epochs,
                "shape": (total_epochs, n_channels, n_samples),
            }
        )

    return pd.DataFrame(rows)


def summarize_kfold_partitions(
    dataset,
    k: int = 10,
    inner_splits: int = 5,
    random_state: int = 3407,
) -> pd.DataFrame:
    subject_ids = dataset.get_subject_ids()
    labels = dataset.get_labels()
    summary = dataset.get_summary_dataframe()

    label_by_subject = {
        row["subject_id"]: row["label_code"]
        for _, row in summary.iterrows()
    }

    epochs_by_subject = {
        row["subject_id"]: {
            "complete": int(row["complete_n_epochs"]),
            "open": int(row["open_n_epochs"]),
            "closed": int(row["closed_n_epochs"]),
        }
        for _, row in summary.iterrows()
    }

    n_channels = int(summary["n_channels"].iloc[0])
    n_samples = int(summary["n_samples"].iloc[0])

    outer_gkf = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        outer_gkf.split(np.zeros(len(labels)), labels, groups=subject_ids),
        start=1,
    ):
        trainval_idx = np.asarray(trainval_idx)
        test_idx = np.asarray(test_idx)

        inner_gkf = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state,
        )

        train_idx, val_idx = next(
            inner_gkf.split(
                np.zeros(len(trainval_idx)),
                [labels[i] for i in trainval_idx],
                groups=[subject_ids[i] for i in trainval_idx],
            )
        )

        split_indices = {
            "train": trainval_idx[train_idx],
            "val": trainval_idx[val_idx],
            "test": test_idx,
        }

        for split_name, indices in split_indices.items():
            split_subjects = [subject_ids[i] for i in indices]
            split_labels = [label_by_subject[subject] for subject in split_subjects]

            complete_epochs = sum(
                epochs_by_subject[subject]["complete"]
                for subject in split_subjects
            )

            open_epochs = sum(
                epochs_by_subject[subject]["open"]
                for subject in split_subjects
            )

            closed_epochs = sum(
                epochs_by_subject[subject]["closed"]
                for subject in split_subjects
            )

            rows.append(
                {
                    "fold": fold_idx,
                    "split": split_name,
                    "n_subjects": len(split_subjects),
                    "subjects": ", ".join(f"ID{subject}" for subject in split_subjects),
                    "label_counts": dict(Counter(split_labels)),
                    "complete_shape": (complete_epochs, n_channels, n_samples),
                    "open_shape": (open_epochs, n_channels, n_samples),
                    "closed_shape": (closed_epochs, n_channels, n_samples),
                }
            )

    return pd.DataFrame(rows)