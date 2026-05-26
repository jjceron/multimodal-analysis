from __future__ import annotations

from pathlib import Path
from collections import Counter

import mne
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader


OPEN = "instructed_toOpenEyes"
CLOSE = "instructed_toCloseEyes"
REST_START = "resting_start"
BREAK = "break cnt"


class HBNRestingStateDataset(Dataset):
    """
    HBN / EEG2025 RestingState dataset for subject-level regression.

    Each item is:
        subject_id: str
        X: Tensor[C, T_s]
        y: Tensor scalar float32

    condition:
        "EO": eyes open only
        "EC": eyes closed only

    This dataset uses list-style variable-length EEG records.
    It does not crop, window, pad, or stack subjects.
    """

    def __init__(
        self,
        root: str | Path,
        condition: str,
        target: str = "externalizing",
        participants_file: str = "participants.tsv",
        task: str = "RestingState",
        scale_uv: bool = True,
        preload: bool = False,
        cache: bool = False,
        cache_dir: str | Path = "data/processed/hbn_db",
        refresh_cache: bool = False,
    ):
        self.root = Path(root)
        self.condition = condition.upper()
        self.target = target
        self.participants_file = participants_file
        self.task = task
        self.scale_uv = scale_uv
        self.preload = preload
        self.pp_as = "list"
        self.cache = cache
        self.cache_dir = Path(cache_dir)
        self.refresh_cache = refresh_cache
        self.cache_path = self._resolve_cache_path()

        if self.condition not in {"EO", "EC"}:
            raise ValueError("condition must be 'EO' or 'EC'.")

        participants_path = self.root / participants_file

        if not participants_path.exists():
            raise FileNotFoundError(participants_path)

        participants = pd.read_csv(participants_path, sep="\t")

        required_columns = {"participant_id", target}
        missing = required_columns - set(participants.columns)

        if missing:
            raise ValueError(f"Missing columns in participants.tsv: {missing}")

        participants = participants[
            ["participant_id", target, "age", "sex", "ehq_total"]
        ].copy()

        participants = participants.dropna(subset=[target])
        participants["participant_id"] = participants["participant_id"].astype(str)

        self.targets = {
            row["participant_id"]: float(row[target])
            for _, row in participants.iterrows()
        }

        self.metadata = {
            row["participant_id"]: {
                "age": row.get("age", np.nan),
                "sex": row.get("sex", None),
                "ehq_total": row.get("ehq_total", np.nan),
            }
            for _, row in participants.iterrows()
        }

        if self.cache and self.cache_path.exists() and not self.refresh_cache:
            self._load_cache()
            return

        self.samples = self._build_index()

        if len(self.samples) == 0:
            raise ValueError("No usable HBN RestingState samples were found.")

        if len(self.samples) == 0:
            raise ValueError("No usable HBN RestingState samples were found.")

        if self.cache:
            self._build_and_save_cache()
        elif self.preload:
            self.samples = [
                {
                    **sample,
                    "X": self._load_eeg_condition(sample),
                }
                for sample in self.samples
            ]

    def _build_index(self):
        bdfs = sorted(
            self.root.rglob(f"*task-{self.task}_eeg.bdf")
        )

        samples = []

        for bdf_path in bdfs:
            subject_id = self._subject_from_path(bdf_path)

            if subject_id not in self.targets:
                continue

            events_path = bdf_path.with_name(
                bdf_path.name.replace("_eeg.bdf", "_events.tsv")
            )

            channels_path = bdf_path.with_name(
                bdf_path.name.replace("_eeg.bdf", "_channels.tsv")
            )

            if not events_path.exists():
                continue

            events = pd.read_csv(events_path, sep="\t")
            eo_intervals, ec_intervals, status = build_resting_intervals(events)

            if status != "ok":
                continue

            intervals = eo_intervals if self.condition == "EO" else ec_intervals

            if len(intervals) == 0:
                continue

            samples.append(
                {
                    "subject_id": subject_id,
                    "bdf_path": bdf_path,
                    "events_path": events_path,
                    "channels_path": channels_path,
                    "intervals": intervals,
                    "y": self.targets[subject_id],
                    "metadata": self.metadata.get(subject_id, {}),
                }
            )

        return samples

    @staticmethod
    def _subject_from_path(path: Path) -> str:
        for part in path.parts:
            if part.startswith("sub-"):
                return part

        return path.name.split("_")[0]

    def _load_eeg_condition(self, sample: dict) -> torch.Tensor:
        raw = mne.io.read_raw_bdf(
            sample["bdf_path"],
            preload=False,
            verbose=False,
        )

        picks = self._get_eeg_picks(raw, sample["channels_path"])

        sfreq = float(raw.info["sfreq"])

        blocks = []

        for onset, offset in sample["intervals"]:
            start = int(round(onset * sfreq))
            stop = int(round(offset * sfreq))

            if stop <= start:
                continue

            block = raw.get_data(
                picks=picks,
                start=start,
                stop=stop,
            ).astype(np.float32)

            if self.scale_uv:
                # MNE returns EEG in volts. Convert to microvolts.
                block = block * 1e6

            blocks.append(block)

        if len(blocks) == 0:
            raise RuntimeError(
                f"No valid {self.condition} blocks for {sample['subject_id']}"
            )

        eeg = np.concatenate(blocks, axis=1)
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(eeg, dtype=torch.float32)

    @staticmethod
    def _get_eeg_picks(raw, channels_path: Path | None):
        """
        Use channels.tsv when available.

        In R1, all 129 channels are EEG/good, including Cz.
        This keeps C=129.
        """
        if channels_path is not None and channels_path.exists():
            channels = pd.read_csv(channels_path, sep="\t")

            if {"name", "type", "status"}.issubset(channels.columns):
                eeg_names = channels.loc[
                    (channels["type"].astype(str).str.upper() == "EEG")
                    & (channels["status"].astype(str).str.lower() == "good"),
                    "name",
                ].tolist()

                eeg_names = [ch for ch in eeg_names if ch in raw.ch_names]

                if len(eeg_names) > 0:
                    return eeg_names

        return mne.pick_types(raw.info, eeg=True, exclude=[])

    def _resolve_cache_path(self) -> Path:
        root_name = self.root.name

        filename = (
            f"{root_name}_"
            f"task-{self.task}_"
            f"condition-{self.condition}_"
            f"target-{self.target}_"
            f"list.pt"
        )

        return self.cache_dir / filename


    def _load_cache(self) -> None:
        payload = torch.load(
            self.cache_path,
            map_location="cpu",
            weights_only=False,
        )

        meta = payload.get("meta", {})

        if meta.get("condition") != self.condition:
            raise ValueError(
                f"Cache condition mismatch: "
                f"{meta.get('condition')} != {self.condition}"
            )

        if meta.get("target") != self.target:
            raise ValueError(
                f"Cache target mismatch: "
                f"{meta.get('target')} != {self.target}"
            )

        self.samples = payload["samples"]
        self.pp_as = "list"
        self.preload = True

        if len(self.samples) == 0:
            raise ValueError(f"Cache is empty: {self.cache_path}")

        print(f"\nLoaded cache: {self.cache_path}")
        print(f"Cached samples: {len(self.samples)}")


    def _build_and_save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached_samples = []

        print(f"\nBuilding cache: {self.cache_path}")
        print(f"Samples to cache: {len(self.samples)}")

        for i, sample in enumerate(self.samples, start=1):
            subject_id = sample["subject_id"]

            print(
                f"  [{i:03d}/{len(self.samples):03d}] "
                f"{subject_id} | condition={self.condition}"
            )

            X = self._load_eeg_condition(sample)

            cached_samples.append(
                {
                    "subject_id": subject_id,
                    "X": X.cpu(),
                    "y": float(sample["y"]),
                    "condition": self.condition,
                    "target": self.target,
                    "metadata": sample.get("metadata", {}),
                    "intervals": sample.get("intervals", []),
                    "source_bdf": str(sample.get("bdf_path", "")),
                    "source_events": str(sample.get("events_path", "")),
                    "source_channels": str(sample.get("channels_path", "")),
                }
            )

        payload = {
            "meta": {
                "root": str(self.root),
                "root_name": self.root.name,
                "task": self.task,
                "condition": self.condition,
                "target": self.target,
                "pp_as": "list",
                "scale_uv": self.scale_uv,
                "n_samples": len(cached_samples),
            },
            "samples": cached_samples,
        }

        torch.save(payload, self.cache_path)

        self.samples = cached_samples
        self.preload = True

        print(f"\nSaved cache: {self.cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if "X" in sample:
            X = sample["X"]
        else:
            X = self._load_eeg_condition(sample)

        y = torch.tensor(sample["y"], dtype=torch.float32)

        return sample["subject_id"], X, y


def build_resting_intervals(events: pd.DataFrame):
    """
    Build EO and EC intervals from HBN RestingState events.tsv.

    Logic:
        resting_start -> first instructed_toOpenEyes is treated as EC.
        instructed_toOpenEyes -> next instruction is EO.
        instructed_toCloseEyes -> next instruction is EC.
        first break cnt after resting_start marks the end of RestingState.
    """
    events = events.copy()
    events = events.sort_values("onset").reset_index(drop=True)

    if "value" not in events.columns or "onset" not in events.columns:
        return [], [], "missing_columns"

    rest_rows = events[events["value"] == REST_START]

    if len(rest_rows) == 0:
        return [], [], "missing_resting_start"

    rest_start = float(rest_rows.iloc[0]["onset"])

    after_rest = events[events["onset"] >= rest_start].copy()

    break_rows = after_rest[after_rest["value"] == BREAK]

    if len(break_rows) > 0:
        rest_end = float(break_rows.iloc[0]["onset"])
    else:
        rest_end = float(after_rest["onset"].max())

    condition_events = after_rest[
        after_rest["value"].isin([OPEN, CLOSE])
    ].copy()

    condition_events = condition_events[
        condition_events["onset"] < rest_end
    ].reset_index(drop=True)

    if len(condition_events) == 0:
        return [], [], "missing_open_close"

    eo_intervals = []
    ec_intervals = []

    first_event = condition_events.iloc[0]
    first_onset = float(first_event["onset"])

    if first_event["value"] == OPEN and first_onset > rest_start:
        ec_intervals.append((rest_start, first_onset))

    for i in range(len(condition_events)):
        onset = float(condition_events.iloc[i]["onset"])
        value = condition_events.iloc[i]["value"]

        if i + 1 < len(condition_events):
            next_onset = float(condition_events.iloc[i + 1]["onset"])
        else:
            next_onset = rest_end

        if next_onset <= onset:
            continue

        if value == OPEN:
            eo_intervals.append((onset, next_onset))
        elif value == CLOSE:
            ec_intervals.append((onset, next_onset))

    return eo_intervals, ec_intervals, "ok"


def create_k_folders(
    dataset: HBNRestingStateDataset,
    k_folder: int = 5,
    batch_size: int = 4,
    shuffle: bool = True,
    split_seed: int = 3407,
    inner_split: int = 5,
    n_bins: int = 5,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Create train/val/test folds for continuous externalizing regression.

    Since y is continuous, we create quantile bins of y only for stratification.
    The model still sees the original continuous target.
    """
    subjects = []
    labels = []
    y_values = []

    for i in range(len(dataset)):
        sample = dataset.samples[i]
        subjects.append(sample["subject_id"])
        y = float(sample["y"])
        y_values.append(y)

    y_series = pd.Series(y_values)

    try:
        labels = pd.qcut(
            y_series,
            q=min(n_bins, y_series.nunique()),
            labels=False,
            duplicates="drop",
        ).astype(int).tolist()
    except ValueError:
        labels = pd.cut(
            y_series,
            bins=min(n_bins, y_series.nunique()),
            labels=False,
            duplicates="drop",
        ).astype(int).tolist()

    outer_gkf = StratifiedGroupKFold(
        n_splits=k_folder,
        shuffle=shuffle,
        random_state=split_seed,
    )

    folds = []

    class FoldDataset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = list(indices)

            self.names = [
                base_dataset.samples[i]["subject_id"]
                for i in self.indices
            ]

            self.y = torch.tensor(
                [
                    float(base_dataset.samples[i]["y"])
                    for i in self.indices
                ],
                dtype=torch.float32,
            )

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.base_dataset[self.indices[idx]]

    def collate_list(batch):
        names, X, y = zip(*batch)

        return (
            list(names),
            list(X),
            torch.stack(y),
        )

    dummy_X = np.zeros(len(dataset))

    for train_val_idx, test_idx in outer_gkf.split(
        dummy_X,
        labels,
        groups=subjects,
    ):
        train_val_idx = np.asarray(train_val_idx)
        test_idx = np.asarray(test_idx)

        inner_labels = [labels[i] for i in train_val_idx]
        inner_subjects = [subjects[i] for i in train_val_idx]

        inner_gkf = StratifiedGroupKFold(
            n_splits=inner_split,
            shuffle=shuffle,
            random_state=split_seed,
        )

        inner_train_idx, inner_val_idx = next(
            inner_gkf.split(
                np.zeros(len(train_val_idx)),
                inner_labels,
                groups=inner_subjects,
            )
        )

        train_idx = train_val_idx[inner_train_idx]
        val_idx = train_val_idx[inner_val_idx]

        train_dataset = FoldDataset(dataset, train_idx)
        val_dataset = FoldDataset(dataset, val_idx)
        test_dataset = FoldDataset(dataset, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_list,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_list,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_list,
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


def inspect_dataset(dataset: HBNRestingStateDataset):
    print("\nDataset")
    print(f"Condition: {dataset.condition}")
    print(f"Target:    {dataset.target}")
    print(f"Subjects:  {len(dataset)}")
    print(f"pp_as:     {dataset.pp_as}")

    y = [float(sample["y"]) for sample in dataset.samples]
    print("\nTarget summary:")
    print(pd.Series(y).describe())

    print("\nLoading shapes. This may take a bit...")

    shapes = []

    for i in range(len(dataset)):
        _, X, _ = dataset[i]
        shapes.append(tuple(X.shape))

    channels = sorted(set(s[0] for s in shapes))
    lengths = [s[1] for s in shapes]

    print("\nSignal shapes:")
    print(f"C: {channels}")
    print(f"T min:  {min(lengths)}")
    print(f"T mean: {np.mean(lengths):.1f}")
    print(f"T max:  {max(lengths)}")

    seconds = np.asarray(lengths) / 100.0

    print("\nDuration seconds:")
    print(pd.Series(seconds).describe())


def inspect_folds(folds):
    print("\nFolds")

    for fold_id, (train_loader, val_loader, test_loader) in enumerate(
        folds,
        start=1,
    ):
        for split_name, loader in [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]:
            y = loader.dataset.y.numpy()

            print(
                f"Fold {fold_id:02d} | "
                f"{split_name:5s} | "
                f"N={len(loader.dataset):3d} | "
                f"y_mean={y.mean(): .3f} | "
                f"y_std={y.std(): .3f} | "
                f"y_min={y.min(): .3f} | "
                f"y_max={y.max(): .3f} | "
                f"Batches={len(loader)}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/hbn_db/R1_L100_bdf",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["EO", "EC"],
        required=True,
    )
    parser.add_argument("--target", type=str, default="externalizing")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-split", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/processed/hbn_db",
    )
    parser.add_argument("--refresh-cache", action="store_true")

    args = parser.parse_args()

    dataset = HBNRestingStateDataset(
        root=args.root,
        condition=args.condition,
        target=args.target,
        preload=args.preload,
        cache=args.cache,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )

    inspect_dataset(dataset)

    folds = create_k_folders(
        dataset=dataset,
        k_folder=args.k,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        inner_split=args.inner_split,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    inspect_folds(folds)