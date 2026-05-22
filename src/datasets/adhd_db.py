from __future__ import annotations

import os
import argparse
from pathlib import Path
from collections import Counter

import mne
import numpy as np
import torch

from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EEGDataset(Dataset):
    def __init__(
        self,
        adhd_dir: str | Path,
        control_dir: str | Path,
        lowcut: float | None = 0.5,
        highcut: float | None = 60.0,
        notch: float | None = 50,
        default_fs: float = 128.0,
        target_fs: float | None = None,
        duration_sec: int | None = None,
        pp_as: str = "tensor",
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch = notch

        self.default_fs = float(default_fs)
        self.target_fs = None if target_fs is None else float(target_fs)
        self.output_fs = self.target_fs if self.target_fs is not None else self.default_fs

        self.duration_sec = duration_sec
        self.pp_as = pp_as

        if self.pp_as not in {"tensor", "list"}:
            raise ValueError("pp_as must be 'tensor' or 'list'.")

        self.samples = []
        self._process_folder(control_dir, label=0)
        self._process_folder(adhd_dir, label=1)

        if len(self.samples) == 0:
            raise ValueError("Subjects were not loaded.")

        if self.pp_as == "tensor":
            self._crop_to_min_time()

    def _process_folder(self, folder, label):
        """Process all .mat files in the given folder and add them to the dataset."""
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".mat"):
                mat_path = os.path.join(folder, fname)
                eeg = self._process_mat(mat_path)

                if eeg is not None:
                    eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
                    label_tensor = torch.tensor(label, dtype=torch.long)
                    self.samples.append((fname, eeg_tensor, label_tensor))

    def _extract_signal_matrix(self, mat: dict):
        """Search for the main EEG signal matrix in the .mat file."""
        candidate_keys = []

        for key, value in mat.items():
            if key.startswith("__"):
                continue

            if not isinstance(value, np.ndarray):
                continue

            if value.ndim != 2:
                continue

            if not np.issubdtype(value.dtype, np.number):
                continue

            if value.shape[0] <= 1 or value.shape[1] <= 1:
                continue

            candidate_keys.append((key, value, value.size))

        if len(candidate_keys) == 0:
            return None, None

        candidate_keys = sorted(
            candidate_keys,
            key=lambda item: item[2],
            reverse=True,
        )

        key_signal, signal_matrix, _ = candidate_keys[0]

        return key_signal, signal_matrix

    def _get_fs(self, mat):
        """Search for sampling frequency in the .mat file."""
        fs_keywords = [
            "fs",
            "sfreq",
            "freq",
            "srate",
            "sampling",
            "sample_rate",
            "sampling_rate",
        ]

        for key, value in mat.items():
            if key.startswith("__"):
                continue

            key_lower = key.lower()

            if not any(token in key_lower for token in fs_keywords):
                continue

            arr = np.asarray(value).squeeze()

            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                fs = float(arr.item())

                if fs > 0:
                    return fs, False

        return self.default_fs, True

    def _orient_channels(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is in C x T format."""
        data = np.asarray(data)

        if np.iscomplexobj(data):
            data = np.real(data)

        if data.shape[0] > data.shape[1]:
            data = data.T

        return data

    def _crop_to_min_time(self):
        """Crop all EEG samples to the minimum time length across the dataset."""
        min_t = min(eeg.shape[1] for _, eeg, _ in self.samples)

        self.samples = [
            (name, eeg[:, :min_t].contiguous(), label)
            for name, eeg, label in self.samples
        ]

    def _process_mat(self, file_path):
        """Load and preprocess EEG data from a .mat file."""
        mat = loadmat(file_path)

        key_signal, data = self._extract_signal_matrix(mat)

        if key_signal is None:
            print(f"There's not EEG matrix in {file_path}")
            return None

        data = self._orient_channels(data)
        data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        fs, fs_was_assumed = self._get_fs(mat)

        ch_names = [f"ch{i}" for i in range(data.shape[0])]

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=["eeg"] * len(ch_names),
        )

        raw = mne.io.RawArray(data, info, verbose=False)

        raw.set_eeg_reference("average", verbose=False)

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

        if not np.isclose(fs, self.output_fs):
            raw.resample(
                self.output_fs,
                npad="auto",
                verbose=False,
            )

        final_fs = float(raw.info["sfreq"])

        eeg = raw.get_data().astype(np.float32)

        if self.duration_sec is not None:
            n_samples = int(float(self.duration_sec) * final_fs)
            eeg = eeg[:, :n_samples]

        return eeg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloaders(
    dataset: EEGDataset,
    k_folder: int = 10,
    batch_size: int = 16,
    shuffle: bool = True,
    split_seed: int = 42,
    inner_split: int = 5,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    pp_as = dataset.pp_as

    subjects, labels, eeg_data = [], [], []

    for i in range(len(dataset)):
        name, eeg_tensor, label = dataset[i]
        subjects.append(name)
        labels.append(label.item())
        eeg_data.append(eeg_tensor)

    outer_gkf = StratifiedGroupKFold(
        n_splits=k_folder,
        shuffle=shuffle,
        random_state=split_seed,
    )

    folds = []

    class FoldDatasets(Dataset):
        def __init__(self, eeg_list, label_list, subject_ind):
            self.X, self.y, self.names = [], [], []

            for idx in subject_ind:
                self.X.append(eeg_list[idx])
                self.y.append(label_list[idx])
                self.names.append(subjects[idx])

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
        eeg_data, labels, groups=subjects
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--default-fs", type=float, default=128.0)
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--duration-sec", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pp-as", type=str, default="tensor", choices=["tensor", "list"])

    return parser.parse_args()


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


def main():
    args = parse_args()

    adhd_dir = PROJECT_ROOT / "data/raw/adhd_control/ADHD"
    control_dir = PROJECT_ROOT / "data/raw/adhd_control/Control"

    dataset = EEGDataset(
        adhd_dir=adhd_dir,
        control_dir=control_dir,
        default_fs=args.default_fs,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as=args.pp_as,
    )

    print(f"\nLoaded {len(dataset)} samples.")

    labels = []
    shapes = []

    for _, eeg, label in dataset:
        labels.append(label.item())
        shapes.append(tuple(eeg.shape))

    label_count = Counter(labels)
    shape_count = Counter(shapes)

    print("\nDataset")
    print(f"Subjects: {len(dataset)}")
    print(f"Control 0: {label_count.get(0, 0)}")
    print(f"ADHD    1: {label_count.get(1, 0)}")

    print("\nPreprocessing:")
    print(f"lowcut:     {dataset.lowcut}")
    print(f"highcut:    {dataset.highcut}")
    print(f"notch:      {dataset.notch}")
    print(f"default_fs: {dataset.default_fs}")
    print(f"output_fs:  {dataset.output_fs}")
    print(f"pp_as:      {dataset.pp_as}")


    if args.pp_as == "tensor":
        if len(shape_count) == 1:
            C, T = next(iter(shape_count))

            print(f"\nFull tensor:")
            print(f"N x C x T = ({len(dataset)}, {C}, {T})")
        else:
            print("\nFolds were not created because tensor mode requires same C x T.")
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
        k_folder=10,
        batch_size=args.batch_size,
        shuffle=True,
        split_seed=42,
        inner_split=5,
        num_workers=0,
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
                f"Control={count.get(0, 0):3d} | "
                f"ADHD={count.get(1, 0):3d} | "
                f"Batches={len(loader)}"
            )


if __name__ == "__main__":
    main()