from __future__ import annotations

from pathlib import Path
from collections import Counter
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

from src.utils import (
    find_project_root,
    load_yaml,
    extract_subject_id,
    normalize_condition,
    assign_epoch_conditions,
    build_label_table,
)


class EEGDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        raw_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        questionnaire_path: str | Path | None = None,
        condition: str = "complete",
        lowcut: float | None = None,
        highcut: float | None = None,
        notch: float | None = None,
        window_duration: float | None = None,
        overlap: float | None = None,
        target_fs: float | None = None,
        eyes_split_sec: float = 300.0,
        scale_to_uv: bool = False,
        reject_by_annotation: bool = False,
        strict_labels: bool = True,
        pick_channels: list[str] | None = None,
        exclude_channels: list[str] | None = None,
    ):
        self.root = Path(root).resolve() if root is not None else find_project_root()

        self.config_path = (
            Path(config_path).resolve()
            if config_path is not None
            else self.root / "configs" / "preprocessing.yaml"
        )

        self.cfg = load_yaml(self.config_path)

        eeg_cfg = self.cfg.get("eeg", {})
        questionnaire_cfg = self.cfg.get("questionnaire", {})

        self.raw_dir = (
            Path(raw_dir).resolve()
            if raw_dir is not None
            else self.root / eeg_cfg.get("raw_data_dir", "data/raw")
        )

        self.questionnaire_path = (
            Path(questionnaire_path).resolve()
            if questionnaire_path is not None
            else self.root
            / questionnaire_cfg.get("raw_data_path", "data/raw/Demo_Questionnaires.xlsx")
        )

        self.condition = normalize_condition(condition)

        self.lowcut = lowcut if lowcut is not None else eeg_cfg.get("lowcut", 0.5)
        self.highcut = highcut if highcut is not None else eeg_cfg.get("highcut", 60.0)
        self.notch = notch if notch is not None else eeg_cfg.get("notch", 50.0)

        self.window_duration = (
            window_duration
            if window_duration is not None
            else eeg_cfg.get("window_duration", eeg_cfg.get("window", 2.0))
        )

        self.overlap = overlap if overlap is not None else eeg_cfg.get("overlap", 0.5)
        self.target_fs = target_fs if target_fs is not None else eeg_cfg.get("target_fs", None)

        self.eyes_split_sec = eyes_split_sec
        self.scale_to_uv = scale_to_uv
        self.reject_by_annotation = reject_by_annotation
        self.strict_labels = strict_labels
        self.pick_channels = pick_channels
        self.exclude_channels = exclude_channels or []

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Missing raw EEG directory: {self.raw_dir}")

        if not self.questionnaire_path.exists():
            raise FileNotFoundError(f"Missing questionnaire file: {self.questionnaire_path}")

        if not (0 <= self.overlap < 1):
            raise ValueError("overlap must be in [0, 1).")

        self.label_table = build_label_table(
            questionnaire_path=self.questionnaire_path,
            questionnaire_cfg=questionnaire_cfg,
        )

        self.id_to_label_code = dict(
            zip(
                self.label_table["subject_id"].astype(int),
                self.label_table["pain_scale_code"].astype(int),
            )
        )

        self.id_to_label_name = dict(
            zip(
                self.label_table["subject_id"].astype(int),
                self.label_table["pain_scale_label"].astype(str),
            )
        )

        self.samples = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError("No EEG subjects were loaded.")

    def _load_samples(self) -> None:
        gdf_files = sorted(self.raw_dir.glob("*.gdf"))

        if not gdf_files:
            raise FileNotFoundError(f"No .gdf files were found in {self.raw_dir}")

        for file_path in gdf_files:
            subject_id = extract_subject_id(file_path)

            if subject_id not in self.id_to_label_code:
                message = f"Subject ID{subject_id} has no pain-scale label."

                if self.strict_labels:
                    raise ValueError(message)

                warnings.warn(message)
                continue

            eeg_data, metadata = self._process_gdf(file_path)

            self.samples.append(
                {
                    "subject_id": subject_id,
                    "file": file_path.name,
                    "path": str(file_path),
                    "eeg_complete": torch.tensor(eeg_data, dtype=torch.float32),
                    "label": torch.tensor(
                        self.id_to_label_code[subject_id],
                        dtype=torch.long,
                    ),
                    "label_name": self.id_to_label_name[subject_id],
                    "metadata": metadata,
                }
            )

    def _process_gdf(self, file_path: Path) -> tuple[np.ndarray, dict]:
        raw = mne.io.read_raw_gdf(
            str(file_path),
            preload=True,
            verbose=False,
        )

        raw = self._select_channels(raw)

        original_sfreq = float(raw.info["sfreq"])
        original_duration_sec = raw.n_times / original_sfreq

        data = raw.get_data()

        if not np.isfinite(data).all():
            warnings.warn(f"NaN or Inf detected in {file_path.name}. Replacing values with zero.")
            raw._data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        raw.set_eeg_reference("average", verbose=False)

        nyquist = float(raw.info["sfreq"]) / 2.0

        if self.notch is not None and self.notch < nyquist:
            raw.notch_filter(freqs=[self.notch], verbose=False)

        highcut = self.highcut

        if highcut is not None and highcut >= nyquist:
            highcut = nyquist - 1.0
            warnings.warn(f"Adjusted highcut for {file_path.name} to {highcut:.2f} Hz.")

        raw.filter(
            l_freq=self.lowcut,
            h_freq=highcut,
            fir_design="firwin",
            verbose=False,
        )

        if self.target_fs is not None and float(self.target_fs) != float(raw.info["sfreq"]):
            raw.resample(float(self.target_fs), npad="auto", verbose=False)

        sfreq = float(raw.info["sfreq"])
        duration_sec = raw.n_times / sfreq
        overlap_sec = self.window_duration * self.overlap

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=self.window_duration,
            overlap=overlap_sec,
            preload=True,
            reject_by_annotation=self.reject_by_annotation,
            verbose=False,
        )

        eeg_data = epochs.get_data()

        if eeg_data.size == 0:
            raise ValueError(f"No epochs were generated for {file_path.name}.")

        if self.scale_to_uv:
            eeg_data = eeg_data * 1e6

        epoch_start_sec = (epochs.events[:, 0].astype(float) - float(raw.first_samp)) / sfreq
        epoch_start_sec = np.maximum(epoch_start_sec, 0.0)
        epoch_end_sec = epoch_start_sec + self.window_duration

        condition_labels = assign_epoch_conditions(
            epoch_start_sec=epoch_start_sec,
            epoch_end_sec=epoch_end_sec,
            eyes_split_sec=self.eyes_split_sec,
        )

        complete_mask = np.ones(len(condition_labels), dtype=bool)
        open_mask = condition_labels == "open"
        closed_mask = condition_labels == "closed"
        transition_mask = condition_labels == "transition"

        metadata = {
            "file": file_path.name,
            "original_sfreq": original_sfreq,
            "sfreq": sfreq,
            "original_duration_sec": original_duration_sec,
            "duration_sec": duration_sec,
            "n_channels": int(eeg_data.shape[1]),
            "n_samples": int(eeg_data.shape[2]),
            "complete_n_epochs": int(np.sum(complete_mask)),
            "open_n_epochs": int(np.sum(open_mask)),
            "closed_n_epochs": int(np.sum(closed_mask)),
            "transition_n_epochs": int(np.sum(transition_mask)),
            "complete_shape": tuple(eeg_data[complete_mask].shape),
            "open_shape": tuple(eeg_data[open_mask].shape),
            "closed_shape": tuple(eeg_data[closed_mask].shape),
            "window_duration": float(self.window_duration),
            "overlap": float(self.overlap),
            "overlap_sec": float(overlap_sec),
            "eyes_split_sec": float(self.eyes_split_sec),
            "channel_names": raw.ch_names,
            "epoch_start_sec": epoch_start_sec,
            "epoch_end_sec": epoch_end_sec,
            "condition_labels": condition_labels,
            "condition_masks": {
                "complete": complete_mask,
                "open": open_mask,
                "closed": closed_mask,
            },
        }

        return eeg_data, metadata

    def _select_channels(self, raw):
        if self.pick_channels is not None:
            missing = [
                channel
                for channel in self.pick_channels
                if channel not in raw.ch_names
            ]

            if missing:
                raise ValueError(f"Missing requested channels: {missing}")

            raw.pick(self.pick_channels)
            return raw

        existing_exclusions = [
            channel
            for channel in self.exclude_channels
            if channel in raw.ch_names
        ]

        if existing_exclusions:
            raw.drop_channels(existing_exclusions)

        channel_types = raw.get_channel_types()

        eeg_channels = [
            channel
            for channel, channel_type in zip(raw.ch_names, channel_types)
            if channel_type == "eeg"
        ]

        if not eeg_channels:
            raw.set_channel_types({channel: "eeg" for channel in raw.ch_names})
            eeg_channels = raw.ch_names

        raw.pick(eeg_channels)

        return raw

    def set_condition(self, condition: str) -> None:
        self.condition = normalize_condition(condition)

    def get_eeg(self, idx: int, condition: str | None = None) -> torch.Tensor:
        condition = self.condition if condition is None else normalize_condition(condition)

        sample = self.samples[idx]
        mask = sample["metadata"]["condition_masks"][condition]
        indices = np.where(mask)[0]

        return sample["eeg_complete"][indices]

    def get_subject_ids(self) -> list[int]:
        return [sample["subject_id"] for sample in self.samples]

    def get_labels(self) -> list[int]:
        return [int(sample["label"].item()) for sample in self.samples]

    def get_summary_dataframe(self) -> pd.DataFrame:
        rows = []

        for idx, sample in enumerate(self.samples):
            metadata = sample["metadata"]
            selected_eeg = self.get_eeg(idx)

            rows.append(
                {
                    "subject_id": sample["subject_id"],
                    "file": sample["file"],
                    "label_name": sample["label_name"],
                    "label_code": int(sample["label"].item()),
                    "selected_condition": self.condition,
                    "original_sfreq": metadata["original_sfreq"],
                    "sfreq": metadata["sfreq"],
                    "duration_sec": metadata["duration_sec"],
                    "n_channels": metadata["n_channels"],
                    "n_samples": metadata["n_samples"],
                    "complete_n_epochs": metadata["complete_n_epochs"],
                    "open_n_epochs": metadata["open_n_epochs"],
                    "closed_n_epochs": metadata["closed_n_epochs"],
                    "transition_n_epochs": metadata["transition_n_epochs"],
                    "complete_shape": metadata["complete_shape"],
                    "open_shape": metadata["open_shape"],
                    "closed_shape": metadata["closed_shape"],
                    "selected_shape": tuple(selected_eeg.shape),
                }
            )

        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return sample["subject_id"], self.get_eeg(idx), sample["label"]


def create_kfold_dataloaders(dataset, k=10, batch_size=32, shuffle=True):

    subjects, labels, eeg_data = [], [], []

    for i in range(len(dataset)):
        name, eeg_tensor, label = dataset[i]
        subjects.append(name)
        labels.append(label.item())
        eeg_data.append(eeg_tensor)

    outer_gkf = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=3407
    )
    folds = []

    class FoldDataset(Dataset):
        def __init__(self, eeg_list, label_list, subject_indices):
            self.X, self.y, self.names = [], [], []

            for idx in subject_indices:
                eeg = eeg_list[idx]
                label = label_list[idx]
                name = subjects[idx]

                for epoch in eeg:
                    self.X.append(epoch)
                    self.y.append(label)
                    self.names.append(name)

            self.X = torch.stack(self.X)
            self.y = torch.tensor(self.y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.names[idx]

    # ================= OUTER LOOP =================
    for trainval_idx, test_idx in outer_gkf.split(
        eeg_data, labels, groups=subjects
    ):

        # -------- Split TRAIN / VAL --------
        inner_gkf = StratifiedGroupKFold(
            n_splits=5,   
            shuffle=True,
            random_state=3407
        )

        train_idx, val_idx = next(
            inner_gkf.split(
                [eeg_data[i] for i in trainval_idx],
                [labels[i] for i in trainval_idx],
                groups=[subjects[i] for i in trainval_idx]
            )
        )


        train_subjects = [trainval_idx[i] for i in train_idx]
        val_subjects   = [trainval_idx[i] for i in val_idx]

        # -------- Datasets --------
        train_dataset = FoldDataset(eeg_data, labels, train_subjects)
        val_dataset   = FoldDataset(eeg_data, labels, val_subjects)
        test_dataset  = FoldDataset(eeg_data, labels, test_idx)

        # -------- Dataloaders --------
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


if __name__ == "__main__":
    dataset = EEGDataset(condition="closed")
    summary = dataset.get_summary_dataframe()

    print("\nEEG dataset")
    print(f"Subjects: {len(dataset)}")
    print(f"Raw EEG directory: {dataset.raw_dir}")
    print(f"Questionnaire: {dataset.questionnaire_path}")
    print(f"Selected condition: {dataset.condition}")

    print("\nSignal configuration")
    print(f"Original sampling frequency: {summary['original_sfreq'].unique().tolist()}")
    print(f"Final sampling frequency: {summary['sfreq'].unique().tolist()}")
    print(f"Channels: {summary['n_channels'].unique().tolist()}")
    print(f"Samples per epoch: {summary['n_samples'].unique().tolist()}")
    print(f"Window duration: {dataset.window_duration} s")
    print(f"Overlap: {dataset.overlap}")
    print(f"Eyes split: {dataset.eyes_split_sec} s")

    print("\nClass distribution")
    class_counts = summary["label_name"].value_counts()
    for label_name, count in class_counts.items():
        print(f"{label_name}: {count}")

    print("\nDuration")
    print(
        "seconds min/median/max:",
        round(summary["duration_sec"].min(), 3),
        round(summary["duration_sec"].median(), 3),
        round(summary["duration_sec"].max(), 3),
    )

    print("\nCondition sizes")
    for condition, column in [
        ("complete", "complete_n_epochs"),
        ("open", "open_n_epochs"),
        ("closed", "closed_n_epochs"),
    ]:
        total_epochs = int(summary[column].sum())
        n_channels = int(summary["n_channels"].iloc[0])
        n_samples = int(summary["n_samples"].iloc[0])

        print(f"{condition}")
        print(f"  total shape: ({total_epochs}, {n_channels}, {n_samples})")
        print(
            "  epochs per subject min/median/max:",
            int(summary[column].min()),
            int(summary[column].median()),
            int(summary[column].max()),
        )

    print("\nTransition epochs")
    print(f"Total: {int(summary['transition_n_epochs'].sum())}")
    print(
        "Per subject min/median/max:",
        int(summary["transition_n_epochs"].min()),
        int(summary["transition_n_epochs"].median()),
        int(summary["transition_n_epochs"].max()),
    )

    folds = create_kfold_dataloaders(
        dataset,
        k=10,
        batch_size=32,
        shuffle=True,
    )

    print("\nK-fold check")
    print(f"Number of folds: {len(folds)}")

    train_loader, val_loader, test_loader = folds[0]

    split_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    print("\nFold 1")
    for split_name, loader in split_loaders.items():
        X = loader.dataset.X
        y = loader.dataset.y
        names = loader.dataset.names

        subject_ids = sorted(set(int(subject) for subject in names))
        label_counts = Counter(y.tolist())

        print(f"{split_name}")
        print(f"  subjects ({len(subject_ids)}): {', '.join(f'ID{s}' for s in subject_ids)}")
        print(f"  X shape: {tuple(X.shape)}")
        print(f"  y shape: {tuple(y.shape)}")
        print(f"  label counts: {dict(label_counts)}")

    X_batch, y_batch, subject_batch = next(iter(train_loader))

    print("\nFirst train batch")
    print(f"X shape: {tuple(X_batch.shape)}")
    print(f"y shape: {tuple(y_batch.shape)}")
    print(f"subjects in batch: {sorted(set(int(s) for s in subject_batch.tolist()))}")
