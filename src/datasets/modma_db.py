from __future__ import annotations

import argparse
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset


mne.set_log_level("WARNING")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESTING_DIR_NAME = "EEG_LZU_2015_2_resting state"
SCALES_FILE_NAME = "Lanzhou University Second Hospital MODMA participants scales.xlsx"
DEFAULT_SCALE_SHEET = "128-electrodes EEG scale"


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def normalize_subject_id(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    match = re.search(r"sub[-_]?(\d+)", text)

    if match is None:
        return None

    return f"sub-{int(match.group(1)):03d}"


def normalize_raw_subject_number(value: Any) -> str | None:
    if pd.isna(value):
        return None

    if isinstance(value, float):
        value = int(value)

    digits = re.sub(r"\D+", "", str(value))

    if not digits:
        return None

    if len(digits) == 7 and digits.startswith("2"):
        digits = "0" + digits

    return digits


def infer_label_from_raw_subject(raw_subject: str | None) -> tuple[int | None, str | None]:
    if raw_subject is None:
        return None, None

    if raw_subject.startswith("0201") or raw_subject.startswith("201"):
        return 1, "MDD"

    if (
        raw_subject.startswith("0202")
        or raw_subject.startswith("0203")
        or raw_subject.startswith("202")
        or raw_subject.startswith("203")
    ):
        return 0, "Control"

    return None, None


def infer_label_from_subject_index(subject: str | None) -> tuple[int | None, str | None]:
    if subject is None:
        return None, None

    match = re.search(r"sub-(\d+)", subject)

    if match is None:
        return None, None

    idx = int(match.group(1))

    if 1 <= idx <= 24:
        return 1, "MDD"

    if 25 <= idx <= 53:
        return 0, "Control"

    return None, None


def safe_read_tsv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        lines = [line.rstrip("\n\r") for line in f]

    if not lines:
        return pd.DataFrame()

    columns = lines[0].split("\t")
    rows = []

    for line in lines[1:]:
        if not line.strip():
            continue

        values = line.split("\t")

        if len(values) < len(columns):
            values = values + [None] * (len(columns) - len(values))

        if len(values) > len(columns):
            values = values[: len(columns)]

        rows.append(values)

    return pd.DataFrame(rows, columns=columns)


def read_edf_header(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.io.read_raw_edf(path.as_posix(), preload=False, verbose="ERROR")


def get_eeg_channel_names(raw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    if len(picks) == 0:
        return list(raw.ch_names)

    return [raw.ch_names[i] for i in picks]


def get_bad_channels_from_tsv(path: Path | None) -> list[str]:
    channels_df = safe_read_tsv(path)

    if channels_df is None:
        return []

    if "name" not in channels_df.columns or "status" not in channels_df.columns:
        return []

    bad_channels = channels_df.loc[
        channels_df["status"].fillna("").astype(str).str.lower().eq("bad"),
        "name",
    ]

    return bad_channels.astype(str).tolist()


def load_modma_scale_metadata(
    bids_root: Path,
    sheet_name: str = DEFAULT_SCALE_SHEET,
) -> pd.DataFrame:
    scales_file = bids_root / SCALES_FILE_NAME

    if not scales_file.exists():
        raise FileNotFoundError(f"MODMA scales file does not exist: {scales_file}")

    df = pd.read_excel(scales_file, sheet_name=sheet_name)
    df = df.copy()

    subject_col = "BIDS subjects number"
    raw_col = "Raw subjects number"

    if subject_col not in df.columns:
        raise ValueError(f"Column not found in MODMA scales file: {subject_col}")

    df["subject"] = df[subject_col].apply(normalize_subject_id)

    if raw_col in df.columns:
        df["raw_subject"] = df[raw_col].apply(normalize_raw_subject_number)
    else:
        df["raw_subject"] = None

    df = df[df["subject"].notna()].copy()

    labels = []
    label_names = []

    for _, row in df.iterrows():
        label, label_name = infer_label_from_raw_subject(row.get("raw_subject"))

        if label is None:
            label, label_name = infer_label_from_subject_index(row.get("subject"))

        labels.append(label)
        label_names.append(label_name)

    df["label"] = labels
    df["label_name"] = label_names

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    return df


class MODMADataset(Dataset):
    """
    MODMA 128-channel resting-state EEG.

    Labels:
        0 -> Control
        1 -> MDD

    Returns:
        name: str
        eeg_tensor: Tensor[C, T]
        label: LongTensor scalar
    """

    def __init__(
        self,
        root: str | Path = PROJECT_ROOT / "data/raw/modma/MODMA_EEG_BIDS_format",
        lowcut: float | None = 0.5,
        highcut: float | None = 60.0,
        notch: float | None = 50.0,
        target_fs: float | None = None,
        duration_sec: float | None = None,
        pp_as: str = "tensor",
        channel_strategy: str = "all",
        bad_channel_policy: str = "none",
        scale_sheet: str = DEFAULT_SCALE_SHEET,
    ):
        self.root = Path(root)
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch = notch
        self.target_fs = None if target_fs is None else float(target_fs)
        self.duration_sec = duration_sec
        self.pp_as = pp_as
        self.channel_strategy = channel_strategy
        self.bad_channel_policy = bad_channel_policy
        self.scale_sheet = scale_sheet

        if self.pp_as not in {"tensor", "list"}:
            raise ValueError("pp_as must be 'tensor' or 'list'.")

        if self.channel_strategy not in {"common", "all"}:
            raise ValueError("channel_strategy must be 'common' or 'all'.")

        if self.bad_channel_policy not in {"none", "mark", "drop-subject"}:
            raise ValueError("bad_channel_policy must be 'none', 'mark', or 'drop-subject'.")

        if not self.root.exists():
            raise FileNotFoundError(f"MODMA BIDS root does not exist: {self.root}")

        self.resting_dir = self.root / RESTING_DIR_NAME

        if not self.resting_dir.exists():
            raise FileNotFoundError(f"MODMA resting-state directory does not exist: {self.resting_dir}")

        self.metadata = load_modma_scale_metadata(
            bids_root=self.root,
            sheet_name=self.scale_sheet,
        )

        self.records = self._discover_records()

        if len(self.records) == 0:
            raise ValueError(f"No MODMA resting-state EDF records found in {self.resting_dir}")

        self.channel_names = self._resolve_channel_names()
        self.samples = []

        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError("No MODMA samples were loaded.")

        if self.pp_as == "tensor":
            self._crop_to_min_time()

    def _discover_records(self) -> list[dict]:
        records = []

        metadata_by_subject = {
            row["subject"]: row.to_dict()
            for _, row in self.metadata.iterrows()
        }

        for subject_dir in sorted(self.resting_dir.glob("sub-*")):
            if not subject_dir.is_dir():
                continue

            subject = normalize_subject_id(subject_dir.name)

            if subject is None or subject not in metadata_by_subject:
                continue

            eeg_dir = subject_dir / "eeg"
            edf_path = eeg_dir / f"{subject}_task-Resting-state_eeg.EDF"

            if not edf_path.exists():
                matches = sorted(eeg_dir.glob("*_task-Resting-state_eeg.EDF"))
                if not matches:
                    matches = sorted(eeg_dir.glob("*_eeg.EDF"))
                if not matches:
                    matches = sorted(eeg_dir.glob("*_eeg.edf"))
                if not matches:
                    continue
                edf_path = matches[0]

            channels_path = eeg_dir / f"{subject}_task-Resting-state_channels.tsv"

            if not channels_path.exists():
                matches = sorted(eeg_dir.glob("*_channels.tsv"))
                channels_path = matches[0] if matches else None

            metadata = metadata_by_subject[subject]

            records.append(
                {
                    "path": edf_path,
                    "channels_path": channels_path,
                    "subject": subject,
                    "name": subject,
                    "label": int(metadata["label"]),
                    "label_name": metadata["label_name"],
                    "raw_subject": metadata.get("raw_subject"),
                    "age": metadata.get("Age"),
                    "gender": metadata.get("Gender"),
                    "education": metadata.get("Education"),
                    "phq9": metadata.get("Patient Health Questionnaire-9 (PHQ-9)"),
                    "gad7": metadata.get("Generalized Anxiety Disorder, GAD-7"),
                    "psqi": metadata.get("Pittsburgh Sleep Quality Index,PSQI"),
                }
            )

        return sorted(records, key=lambda x: (x["label"], x["subject"]))

    def _resolve_channel_names(self) -> list[str]:
        channel_lists = []

        for record in self.records:
            raw = read_edf_header(record["path"])
            channel_lists.append(get_eeg_channel_names(raw))

        if self.channel_strategy == "all":
            first = channel_lists[0]

            for idx, chs in enumerate(channel_lists[1:], start=1):
                if chs != first:
                    raise ValueError(
                        "Not all MODMA EDF files have identical EEG channel names. "
                        "Use channel_strategy='common'. "
                        f"First mismatch: {self.records[idx]['path']}"
                    )

            return first

        common = set(channel_lists[0])

        for chs in channel_lists[1:]:
            common &= set(chs)

        channel_names = [ch for ch in channel_lists[0] if ch in common]

        if len(channel_names) == 0:
            raise ValueError("No common channels found across MODMA EDF files.")

        return channel_names

    def _load_samples(self) -> None:
        for record in self.records:
            eeg = self._process_edf(record)

            if eeg is None:
                continue

            self.samples.append(
                {
                    "name": record["name"],
                    "subject": record["subject"],
                    "label_name": record["label_name"],
                    "raw_subject": record["raw_subject"],
                    "eeg": torch.tensor(eeg, dtype=torch.float32),
                    "label": torch.tensor(record["label"], dtype=torch.long),
                    "metadata": record,
                }
            )

    def _process_edf(self, record: dict):
        path = record["path"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")

        missing = [ch for ch in self.channel_names if ch not in raw.ch_names]

        if missing:
            print(f"Skipping {record['subject']}: missing channels {missing}")
            return None

        bad_channels = get_bad_channels_from_tsv(record.get("channels_path"))

        if self.bad_channel_policy == "drop-subject" and len(bad_channels) > 0:
            print(f"Skipping {record['subject']}: bad channels found {bad_channels}")
            return None

        raw.pick(self.channel_names)
        raw.reorder_channels(self.channel_names)

        if self.bad_channel_policy == "mark":
            raw.info["bads"] = [ch for ch in bad_channels if ch in raw.ch_names]

        raw.set_eeg_reference("average", verbose=False)

        fs = float(raw.info["sfreq"])

        if self.notch is not None and self.notch < fs / 2:
            raw.notch_filter([self.notch], verbose=False)

        highcut = self.highcut

        if highcut is not None:
            highcut = min(highcut, fs / 2 - 1e-3)

            if self.lowcut is not None and highcut <= self.lowcut:
                highcut = None

        if self.lowcut is not None or highcut is not None:
            raw.filter(
                l_freq=self.lowcut,
                h_freq=highcut,
                fir_design="firwin",
                verbose=False,
            )

        if self.target_fs is not None and not np.isclose(fs, self.target_fs):
            raw.resample(self.target_fs, npad="auto", verbose=False)

        eeg = raw.get_data().astype(np.float32)
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

        if self.duration_sec is not None:
            final_fs = float(raw.info["sfreq"])
            n_samples = int(float(self.duration_sec) * final_fs)
            eeg = eeg[:, :n_samples]

        return eeg

    def _crop_to_min_time(self):
        min_t = min(sample["eeg"].shape[1] for sample in self.samples)

        for sample in self.samples:
            sample["eeg"] = sample["eeg"][:, :min_t].contiguous()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return sample["name"], sample["eeg"], sample["label"]


def create_dataloaders(
    dataset: MODMADataset,
    k_folder: int = 5,
    batch_size: int = 2,
    shuffle: bool = True,
    split_seed: int = 42,
    inner_split: int = 5,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    pp_as = dataset.pp_as

    subjects, labels, eeg_data = [], [], []

    for sample in dataset.samples:
        subjects.append(sample["subject"])
        labels.append(int(sample["label"].item()))
        eeg_data.append(sample["eeg"])

    outer_gkf = StratifiedGroupKFold(
        n_splits=k_folder,
        shuffle=shuffle,
        random_state=split_seed,
    )

    folds = []

    class FoldDatasets(Dataset):
        def __init__(self, eeg_list, label_list, subject_indices):
            self.X, self.y, self.names, self.subjects = [], [], [], []

            for idx in subject_indices:
                self.X.append(eeg_list[idx])
                self.y.append(label_list[idx])
                self.names.append(dataset.samples[idx]["name"])
                self.subjects.append(subjects[idx])

            self.y = torch.tensor(self.y, dtype=torch.long)

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            return self.names[idx], self.X[idx], self.y[idx]

    def collate_list(batch):
        names, X, y = zip(*batch)

        return (
            list(names),
            list(X),
            torch.stack(y),
        )

    collate_fn = collate_list if pp_as == "list" else None

    for train_val_idx, test_idx in outer_gkf.split(
        eeg_data,
        labels,
        groups=subjects,
    ):
        inner_gkf = StratifiedGroupKFold(
            n_splits=inner_split,
            shuffle=shuffle,
            random_state=split_seed,
        )

        train_idx, val_idx = next(
            inner_gkf.split(
                [eeg_data[i] for i in train_val_idx],
                [labels[i] for i in train_val_idx],
                groups=[subjects[i] for i in train_val_idx],
            )
        )

        train_subjects = [train_val_idx[i] for i in train_idx]
        val_subjects = [train_val_idx[i] for i in val_idx]

        train_dataset = FoldDatasets(eeg_data, labels, train_subjects)
        val_dataset = FoldDatasets(eeg_data, labels, val_subjects)
        test_dataset = FoldDatasets(eeg_data, labels, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


def format_loader_shape(loader, pp_as):
    shapes = [tuple(eeg.shape) for eeg in loader.dataset.X]
    channels = sorted(set(shape[0] for shape in shapes))
    lengths = [shape[1] for shape in shapes]

    C = channels[0] if len(channels) == 1 else channels
    T = min(lengths) if min(lengths) == max(lengths) else f"{min(lengths)}-{max(lengths)}"

    return f"C={C} | T={T}"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "data/raw/modma/MODMA_EEG_BIDS_format"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--target-fs", type=parse_optional_float, default=None)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)

    parser.add_argument("--pp-as", type=str, default="tensor", choices=["tensor", "list"])
    parser.add_argument("--channel-strategy", type=str, default="all", choices=["common", "all"])
    parser.add_argument("--bad-channel-policy", type=str, default="none", choices=["none", "mark", "drop-subject"])
    parser.add_argument("--scale-sheet", type=str, default=DEFAULT_SCALE_SHEET)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = MODMADataset(
        root=args.root,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as=args.pp_as,
        channel_strategy=args.channel_strategy,
        bad_channel_policy=args.bad_channel_policy,
        scale_sheet=args.scale_sheet,
    )

    print(f"\nLoaded {len(dataset)} samples.")

    labels = []
    shapes = []
    subjects = []

    for sample in dataset.samples:
        labels.append(sample["label"].item())
        shapes.append(tuple(sample["eeg"].shape))
        subjects.append(sample["subject"])

    label_count = Counter(labels)
    shape_count = Counter(shapes)

    print("\nDataset")
    print("Task: Resting-state")
    print(f"Subjects/files: {len(dataset)}")
    print(f"Unique subjects: {len(set(subjects))}")
    print(f"Control 0: {label_count.get(0, 0)}")
    print(f"MDD     1: {label_count.get(1, 0)}")

    print("\nPreprocessing:")
    print(f"lowcut:             {dataset.lowcut}")
    print(f"highcut:            {dataset.highcut}")
    print(f"notch:              {dataset.notch}")
    print(f"target_fs:          {dataset.target_fs}")
    print(f"duration_sec:       {dataset.duration_sec}")
    print(f"pp_as:              {dataset.pp_as}")
    print(f"channel_strategy:   {dataset.channel_strategy}")
    print(f"bad_channel_policy: {dataset.bad_channel_policy}")
    print(f"n_channels:         {len(dataset.channel_names)}")
    print(f"channels:           {dataset.channel_names}")

    if args.pp_as == "tensor":
        if len(shape_count) == 1:
            C, T = next(iter(shape_count))

            print("\nFull tensor:")
            print(f"N x C x T = ({len(dataset)}, {C}, {T})")
        else:
            print("\nFolds were not created because tensor mode requires same C x T.")
            print(f"Shapes: {shape_count}")
            return

    else:
        channels = sorted(set(shape[0] for shape in shapes))
        lengths = [shape[1] for shape in shapes]

        C = channels[0] if len(channels) == 1 else channels

        print("\nFull list:")
        print(f"N = {len(dataset)}")
        print(f"C = {C}")
        print(f"T min = {min(lengths)}")
        print(f"T max = {max(lengths)}")

    folds = create_dataloaders(
        dataset=dataset,
        k_folder=args.k,
        batch_size=args.batch_size,
        shuffle=True,
        split_seed=42,
        inner_split=args.inner_splits,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nFolds:")
    for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
        for split_name, loader in [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]:
            y = loader.dataset.y.tolist()
            count = Counter(y)
            shape_text = format_loader_shape(loader, args.pp_as)

            print(
                f"Fold {fold_id:02d} | "
                f"{split_name:5s} | "
                f"N={len(loader.dataset):3d} | "
                f"{shape_text} | "
                f"Control={count.get(0, 0):3d} | "
                f"MDD={count.get(1, 0):3d} | "
                f"Batches={len(loader)}"
            )


if __name__ == "__main__":
    main()
