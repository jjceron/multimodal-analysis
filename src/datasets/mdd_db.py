from __future__ import annotations

import argparse
import re
import warnings
from collections import Counter
from pathlib import Path

import mne
mne.set_log_level("WARNING")
import numpy as np
import torch

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def tokenize_path(path: Path) -> list[str]:
    text = path.as_posix().upper()
    return [tok for tok in re.split(r"[/\\_.\-\s()]+", text) if tok]


def infer_label(tokens: list[str]) -> tuple[int | None, str | None]:
    if "MDD" in tokens:
        return 1, "MDD"

    if any(tok in tokens for tok in ["H", "HC", "HEALTHY", "CONTROL", "CONTROLS"]):
        return 0, "H"

    return None, None


def infer_condition(tokens: list[str]) -> str | None:
    joined = "".join(tokens)

    if "EC" in tokens or "EYESCLOSED" in joined or "CLOSED" in tokens:
        return "EC"

    if "EO" in tokens or "EYESOPEN" in joined or "OPEN" in tokens:
        return "EO"

    if "TASK" in tokens or "P300" in tokens:
        return "TASK"

    return None


def infer_subject(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix().upper()
    stem = path.stem.upper()

    patterns = [
        r"\b(MDD|HC|H)\s*S?(\d{1,4})\b",
        r"\b(MDD|HC|H)[-_ ]?(\d{1,4})\b",
        r"\bSUB[-_ ]?([A-Z0-9]+)\b",
        r"\bS(?:UBJECT)?[-_ ]?(\d{1,4})\b",
    ]

    for source in [stem, rel]:
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return f"{groups[0]}{int(groups[1]):02d}"
                return groups[0]

    return path.stem


def infer_metadata(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    tokens = tokenize_path(Path(rel))

    label, label_name = infer_label(tokens)
    condition = infer_condition(tokens)
    subject = infer_subject(path, root)

    return {
        "path": path,
        "rel_path": rel,
        "name": path.name,
        "subject": subject,
        "condition": condition,
        "label": label,
        "label_name": label_name,
    }


def read_edf_header(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.io.read_raw_edf(path.as_posix(), preload=False, verbose="ERROR")


def get_eeg_channel_names(raw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    if len(picks) == 0:
        return list(raw.ch_names)

    return [raw.ch_names[i] for i in picks]


class MDDDataset(Dataset):
    """
    EEG Data New / MDD dataset.

    Labels:
        0 -> H / healthy control
        1 -> MDD

    Conditions:
        EC -> eyes closed
        EO -> eyes open

    Returns:
        name: str
        eeg_tensor: Tensor[C, T]
        label: LongTensor scalar
    """

    def __init__(
        self,
        root: str | Path = PROJECT_ROOT / "data/raw/mdd_db",
        condition: str = "EC",
        lowcut: float | None = 0.5,
        highcut: float | None = 60.0,
        notch: float | None = 50.0,
        target_fs: float | None = None,
        duration_sec: float | None = None,
        pp_as: str = "tensor",
        channel_strategy: str = "common",
    ):
        self.root = Path(root)
        self.condition = condition.upper()
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch = notch
        self.target_fs = None if target_fs is None else float(target_fs)
        self.duration_sec = duration_sec
        self.pp_as = pp_as
        self.channel_strategy = channel_strategy

        if self.condition not in {"EC", "EO"}:
            raise ValueError("condition must be EC or EO.")

        if self.pp_as not in {"tensor", "list"}:
            raise ValueError("pp_as must be 'tensor' or 'list'.")

        if self.channel_strategy not in {"common", "all"}:
            raise ValueError("channel_strategy must be 'common' or 'all'.")

        if not self.root.exists():
            raise FileNotFoundError(f"MDD root does not exist: {self.root}")

        self.records = self._discover_records()

        if len(self.records) == 0:
            raise ValueError(
                f"No EDF records found for condition={self.condition} in {self.root}"
            )

        self.channel_names = self._resolve_channel_names()
        self.samples = []

        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError("No MDD samples were loaded.")

        if self.pp_as == "tensor":
            self._crop_to_min_time()

    def _discover_records(self) -> list[dict]:
        records = []

        for path in sorted(self.root.rglob("*.edf")):
            meta = infer_metadata(path, self.root)

            if meta["label"] is None:
                continue

            if meta["condition"] != self.condition:
                continue

            records.append(meta)

        return sorted(records, key=lambda x: (x["label"], x["subject"], x["name"]))

    def _resolve_channel_names(self) -> list[str]:
        channel_lists = []

        for record in self.records:
            raw = read_edf_header(record["path"])
            channel_lists.append(get_eeg_channel_names(raw))

        if self.channel_strategy == "all":
            first = channel_lists[0]
            bad = [
                self.records[i]["rel_path"]
                for i, chs in enumerate(channel_lists)
                if chs != first
            ]

            if bad:
                raise ValueError(
                    "Not all EDF files have identical channel names. "
                    "Use channel_strategy='common'. "
                    f"First mismatch: {bad[0]}"
                )

            return first

        common = set(channel_lists[0])

        for chs in channel_lists[1:]:
            common &= set(chs)

        channel_names = [ch for ch in channel_lists[0] if ch in common]

        if len(channel_names) == 0:
            raise ValueError("No common channels found across MDD EDF files.")

        return channel_names

    def _load_samples(self) -> None:
        for record in self.records:
            eeg = self._process_edf(record["path"])

            if eeg is None:
                continue

            self.samples.append(
                {
                    "name": record["name"],
                    "subject": record["subject"],
                    "condition": record["condition"],
                    "label_name": record["label_name"],
                    "eeg": torch.tensor(eeg, dtype=torch.float32),
                    "label": torch.tensor(record["label"], dtype=torch.long),
                }
            )

    def _process_edf(self, path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")

        missing = [ch for ch in self.channel_names if ch not in raw.ch_names]

        if missing:
            print(f"Skipping {path.name}: missing channels {missing}")
            return None

        raw.pick(self.channel_names)
        raw.reorder_channels(self.channel_names)

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
    dataset: MDDDataset,
    k_folder: int = 5,
    batch_size: int = 16,
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

            if pp_as == "tensor":
                self.X = torch.stack(self.X)

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
    if pp_as == "tensor":
        C = loader.dataset.X.shape[1]
        T = loader.dataset.X.shape[2]
        return f"C={C} | T={T}"

    shapes = [tuple(eeg.shape) for eeg in loader.dataset.X]
    channels = sorted(set(shape[0] for shape in shapes))
    lengths = [shape[1] for shape in shapes]

    C = channels[0] if len(channels) == 1 else channels
    T = min(lengths) if min(lengths) == max(lengths) else f"{min(lengths)}-{max(lengths)}"

    return f"C={C} | T={T}"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "data/raw/mdd_db"))
    parser.add_argument("--condition", type=str, default="EC", choices=["EC", "EO"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--target-fs", type=parse_optional_float, default=None)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)

    parser.add_argument("--pp-as", type=str, default="tensor", choices=["tensor", "list"])
    parser.add_argument("--channel-strategy", type=str, default="common", choices=["common", "all"])
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = MDDDataset(
        root=args.root,
        condition=args.condition,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as=args.pp_as,
        channel_strategy=args.channel_strategy,
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
    print(f"Condition: {dataset.condition}")
    print(f"Subjects/files: {len(dataset)}")
    print(f"Unique subjects: {len(set(subjects))}")
    print(f"H    0: {label_count.get(0, 0)}")
    print(f"MDD  1: {label_count.get(1, 0)}")

    print("\nPreprocessing:")
    print(f"lowcut:           {dataset.lowcut}")
    print(f"highcut:          {dataset.highcut}")
    print(f"notch:            {dataset.notch}")
    print(f"target_fs:        {dataset.target_fs}")
    print(f"duration_sec:     {dataset.duration_sec}")
    print(f"pp_as:            {dataset.pp_as}")
    print(f"channel_strategy: {dataset.channel_strategy}")
    print(f"n_channels:       {len(dataset.channel_names)}")
    print(f"channels:         {dataset.channel_names}")

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
    for fold_id, (train_loader, val_loader, test_loader) in enumerate(
        folds,
        start=1,
    ):
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
                f"H={count.get(0, 0):3d} | "
                f"MDD={count.get(1, 0):3d} | "
                f"Batches={len(loader)}"
            )


if __name__ == "__main__":
    main()