from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import mne
import numpy as np
import pandas as pd
import torch

from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EEGDataset_ADHD(Dataset):
    """
    Dataset iraní ADHD/control a nivel de sujeto.

    Cada sujeto queda como:

        X_s in R^{C x T}
        y_s in {0, 1}

    El dataset NO crea ventanas. Primero carga cada .mat, detecta la matriz EEG,
    orienta a (C, T_s), aplica el preprocesamiento opcional y luego recorta todos
    los sujetos a una longitud temporal común.

    Regla por defecto:

        duration_sec=None  -> usar el sujeto más corto del propio dataset.

    Es decir, si el sujeto más corto después del preprocesamiento tiene T_min
    muestras, todos los sujetos se recortan a T_min y:

        selected_duration_sec = T_min / target_fs

    Si se pasa duration_sec, se pide esa duración, pero nunca se excede al sujeto
    más corto:

        crop_n_samples = min(round(duration_sec * target_fs), T_min)
    """

    def __init__(
        self,
        adhd_dir: str | Path,
        control_dir: str | Path,
        lowcut: float | None = 0.5,
        highcut: float | None = 60.0,
        notch: float | None = 50.0,
        target_fs: float = 128.0,
        default_fs: float = 128.0,
        duration_sec: float | None = None,
        crop_from: str = "start",
        max_channels: int = 64,
        scale: float = 1.0,
        apply_reference: bool = True,
        apply_notch: bool = True,
        apply_filter: bool = True,
        apply_resample: bool = True,
        verbose: bool = True,
    ) -> None:
        self.adhd_dir = self._resolve_existing_dir(Path(adhd_dir))
        self.control_dir = self._resolve_existing_dir(Path(control_dir))

        self.lowcut = lowcut
        self.highcut = highcut
        self.notch = notch
        self.target_fs = float(target_fs)
        self.default_fs = float(default_fs)
        self.duration_sec = duration_sec
        self.crop_from = crop_from
        self.max_channels = int(max_channels)
        self.scale = float(scale)
        self.apply_reference = bool(apply_reference)
        self.apply_notch = bool(apply_notch)
        self.apply_filter = bool(apply_filter)
        self.apply_resample = bool(apply_resample)
        self.verbose = bool(verbose)

        if self.target_fs <= 0:
            raise ValueError("target_fs debe ser > 0.")

        if self.default_fs <= 0:
            raise ValueError("default_fs debe ser > 0.")

        if self.crop_from not in {"start", "center"}:
            raise ValueError("crop_from must be 'start' or 'center'.")

        loaded_samples: list[dict] = []
        loaded_samples.extend(
            self._load_folder(
                folder=self.adhd_dir,
                label=1,
                label_name="ADHD",
            )
        )
        loaded_samples.extend(
            self._load_folder(
                folder=self.control_dir,
                label=0,
                label_name="control",
            )
        )

        if len(loaded_samples) == 0:
            raise ValueError("No se cargaron sujetos. Revisa las rutas y los .mat.")

        self._validate_channel_count(loaded_samples)

        sample_lengths = np.array(
            [int(sample["eeg"].shape[1]) for sample in loaded_samples],
            dtype=int,
        )
        min_n_samples = int(sample_lengths.min())
        max_n_samples = int(sample_lengths.max())
        min_subject_ids = [
            str(sample["subject_id"])
            for sample in loaded_samples
            if int(sample["eeg"].shape[1]) == min_n_samples
        ]

        self.min_n_samples_after_preprocess = min_n_samples
        self.max_n_samples_after_preprocess = max_n_samples
        self.min_duration_sec_after_preprocess = min_n_samples / self.target_fs
        self.min_duration_subject_ids = min_subject_ids

        if duration_sec is None:
            requested_n_samples = None
            crop_n_samples = min_n_samples
            crop_policy = "dataset_min"
        else:
            requested_n_samples = int(round(float(duration_sec) * self.target_fs))
            crop_policy = "requested_duration"

            if requested_n_samples <= 0:
                raise ValueError("--duration-sec debe producir al menos 1 muestra.")

            crop_n_samples = min(requested_n_samples, min_n_samples)

            if crop_n_samples < requested_n_samples and self.verbose:
                print(
                    "WARNING: --duration-sec solicita "
                    f"{requested_n_samples} muestras, pero el sujeto más corto tiene "
                    f"{min_n_samples}. Se usará {crop_n_samples}."
                )

        self.crop_n_samples = int(crop_n_samples)
        self.selected_duration_sec = self.crop_n_samples / self.target_fs
        self.crop_policy = crop_policy
        self.requested_n_samples = requested_n_samples

        self.samples: list[dict] = []

        for sample in loaded_samples:
            eeg = self._crop_eeg(sample["eeg"], self.crop_n_samples)
            eeg = eeg * self.scale

            metadata = dict(sample["metadata"])
            metadata.update(
                {
                    "crop_policy": self.crop_policy,
                    "crop_n_samples": self.crop_n_samples,
                    "selected_duration_sec": self.selected_duration_sec,
                    "requested_n_samples": requested_n_samples,
                    "duration_sec_requested": duration_sec,
                    "selected_shape": tuple(eeg.shape),
                    "crop_from": self.crop_from,
                    "scale": self.scale,
                    "is_min_duration_subject": bool(
                        int(sample["eeg"].shape[1]) == min_n_samples
                    ),
                    "dataset_min_n_samples_after_preprocess": min_n_samples,
                    "dataset_max_n_samples_after_preprocess": max_n_samples,
                    "dataset_min_duration_sec_after_preprocess": (
                        self.min_duration_sec_after_preprocess
                    ),
                }
            )

            self.samples.append(
                {
                    "subject_id": sample["subject_id"],
                    "file": sample["file"],
                    "eeg": torch.tensor(eeg, dtype=torch.float32),
                    "label": torch.tensor(sample["label"], dtype=torch.long),
                    "label_name": sample["label_name"],
                    "metadata": metadata,
                }
            )

    @staticmethod
    def _resolve_existing_dir(folder: Path) -> Path:
        """Resuelve diferencias comunes de mayúsculas/minúsculas: control/Control."""
        if folder.exists():
            return folder

        parent = folder.parent
        if parent.exists():
            for child in parent.iterdir():
                if child.is_dir() and child.name.lower() == folder.name.lower():
                    return child

        return folder

    def _load_folder(
        self,
        folder: Path,
        label: int,
        label_name: str,
    ) -> list[dict]:
        if not folder.exists():
            raise FileNotFoundError(f"No existe la carpeta: {folder}")

        mat_files = sorted(folder.glob("*.mat"))

        if len(mat_files) == 0 and self.verbose:
            print(f"WARNING: no encontré archivos .mat en {folder}")

        samples: list[dict] = []

        for file_path in mat_files:
            try:
                eeg, metadata = self._process_mat(file_path)

                samples.append(
                    {
                        "subject_id": file_path.stem,
                        "file": file_path.name,
                        "eeg": eeg,
                        "label": int(label),
                        "label_name": label_name,
                        "metadata": metadata,
                    }
                )

            except Exception as exc:
                if self.verbose:
                    print(f"WARNING: no pude procesar {file_path.name}: {exc}")

        return samples

    def _process_mat(self, file_path: Path) -> tuple[np.ndarray, dict]:
        mat = loadmat(file_path)

        data, signal_key = self._extract_signal_matrix(mat)
        fs, fs_key = self._extract_sampling_frequency(mat)

        data, was_transposed = self._orient_to_channels_time(data)
        data = np.asarray(data, dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if data.ndim != 2:
            raise ValueError(f"EEG debe ser 2D, llegó shape={data.shape}")

        n_channels, n_samples_before = data.shape

        if n_channels > self.max_channels:
            raise ValueError(
                f"Demasiados canales detectados: {n_channels}. "
                f"Shape={data.shape}. Probablemente la matriz no es EEG."
            )

        original_fs = float(fs)
        original_duration_sec = n_samples_before / original_fs

        raw = self._build_raw(data, fs=original_fs)

        did_reference = False
        if self.apply_reference:
            raw.set_eeg_reference("average", verbose=False)
            did_reference = True

        did_notch = False
        did_filter = False

        current_fs = float(raw.info["sfreq"])
        nyquist = current_fs / 2.0

        if (
            self.apply_notch
            and self.notch is not None
            and 0.0 < float(self.notch) < nyquist
        ):
            raw.notch_filter([float(self.notch)], verbose=False)
            did_notch = True

        if self.apply_filter and (self.lowcut is not None or self.highcut is not None):
            current_fs = float(raw.info["sfreq"])
            nyquist = current_fs / 2.0
            highcut = self.highcut

            if highcut is not None:
                highcut = min(float(highcut), nyquist - 1e-3)

                if self.lowcut is not None and highcut <= float(self.lowcut):
                    highcut = None

            if self.lowcut is not None or highcut is not None:
                raw.filter(
                    l_freq=self.lowcut,
                    h_freq=highcut,
                    fir_design="firwin",
                    verbose=False,
                )
                did_filter = True

        pre_resample_fs = float(raw.info["sfreq"])
        did_resample = False

        if self.apply_resample and not np.isclose(pre_resample_fs, self.target_fs):
            raw.resample(
                self.target_fs,
                npad="auto",
                verbose=False,
            )
            did_resample = True

        eeg = raw.get_data()
        final_fs = float(raw.info["sfreq"])

        metadata = {
            "signal_key": signal_key,
            "fs_key": fs_key,
            "fs_was_assumed": fs_key is None,
            "original_fs": original_fs,
            "target_fs": self.target_fs,
            "pre_resample_fs": pre_resample_fs,
            "final_fs": final_fs,
            "original_duration_sec": original_duration_sec,
            "duration_sec_after_resample": eeg.shape[1] / final_fs,
            "n_channels": int(eeg.shape[0]),
            "n_samples_before": int(n_samples_before),
            "n_samples_after_resample": int(eeg.shape[1]),
            "was_transposed": bool(was_transposed),
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "notch": self.notch,
            "apply_reference": self.apply_reference,
            "apply_notch": self.apply_notch,
            "apply_filter": self.apply_filter,
            "apply_resample": self.apply_resample,
            "did_reference": did_reference,
            "did_notch": did_notch,
            "did_filter": did_filter,
            "did_resample": did_resample,
        }

        return eeg, metadata

    def _extract_signal_matrix(self, mat: dict) -> tuple[np.ndarray, str]:
        candidates = []

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

            candidates.append((key, value, value.size))

        if len(candidates) == 0:
            raise ValueError("No encontré una matriz EEG 2D numérica en el .mat.")

        candidates = sorted(
            candidates,
            key=lambda item: item[2],
            reverse=True,
        )

        key, data, _ = candidates[0]
        return data, key

    def _extract_sampling_frequency(self, mat: dict) -> tuple[float, str | None]:
        fs_keywords = [
            "fs",
            "sfreq",
            "srate",
            "freq",
            "sampling",
            "sampling_rate",
            "sample_rate",
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
                    return fs, key

        return self.default_fs, None

    def _orient_to_channels_time(self, data: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Devuelve EEG como (C, T).

        Caso común:
            (T, 19) -> transpose -> (19, T)
        """
        data = np.asarray(data)

        if np.iscomplexobj(data):
            data = np.real(data)

        rows, cols = data.shape

        if rows <= self.max_channels and cols > self.max_channels:
            return data, False

        if cols <= self.max_channels and rows > self.max_channels:
            return data.T, True

        if rows <= cols:
            return data, False

        return data.T, True

    def _build_raw(self, data: np.ndarray, fs: float) -> mne.io.RawArray:
        n_channels = int(data.shape[0])
        ch_names = [f"Ch{i + 1}" for i in range(n_channels)]

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=float(fs),
            ch_types=["eeg"] * n_channels,
        )

        return mne.io.RawArray(
            data,
            info,
            verbose=False,
        )

    def _crop_eeg(
        self,
        eeg: np.ndarray,
        crop_n_samples: int,
    ) -> np.ndarray:
        n_samples = int(eeg.shape[1])

        if n_samples < crop_n_samples:
            raise ValueError(
                f"No se puede recortar a {crop_n_samples}; "
                f"la señal tiene {n_samples}."
            )

        if self.crop_from == "start":
            start = 0
        else:
            start = (n_samples - crop_n_samples) // 2

        end = start + crop_n_samples
        return eeg[:, start:end]

    def _validate_channel_count(self, samples: list[dict]) -> None:
        counts = Counter(int(sample["eeg"].shape[0]) for sample in samples)

        if len(counts) != 1:
            raise ValueError(
                "No todos los sujetos tienen el mismo número de canales: "
                f"{dict(counts)}"
            )

    def get_eeg(self, idx: int):
        return self.samples[idx]["eeg"]

    def get_subject_ids(self):
        return [sample["subject_id"] for sample in self.samples]

    def get_labels(self):
        return [int(sample["label"].item()) for sample in self.samples]

    def get_summary_dataframe(self):
        rows = []

        for sample in self.samples:
            metadata = sample["metadata"]

            rows.append(
                {
                    "subject_id": sample["subject_id"],
                    "file": sample["file"],
                    "label_name": sample["label_name"],
                    "label_code": int(sample["label"].item()),
                    "signal_key": metadata["signal_key"],
                    "fs_key": metadata["fs_key"],
                    "fs_was_assumed": metadata["fs_was_assumed"],
                    "original_fs": metadata["original_fs"],
                    "target_fs": metadata["target_fs"],
                    "pre_resample_fs": metadata["pre_resample_fs"],
                    "final_fs": metadata["final_fs"],
                    "original_duration_sec": metadata["original_duration_sec"],
                    "duration_sec_after_resample": metadata[
                        "duration_sec_after_resample"
                    ],
                    "selected_duration_sec": metadata["selected_duration_sec"],
                    "n_channels": metadata["n_channels"],
                    "n_samples_before": metadata["n_samples_before"],
                    "n_samples_after_resample": metadata[
                        "n_samples_after_resample"
                    ],
                    "crop_policy": metadata["crop_policy"],
                    "crop_n_samples": metadata["crop_n_samples"],
                    "selected_shape": metadata["selected_shape"],
                    "is_min_duration_subject": metadata["is_min_duration_subject"],
                    "dataset_min_n_samples_after_preprocess": metadata[
                        "dataset_min_n_samples_after_preprocess"
                    ],
                    "dataset_max_n_samples_after_preprocess": metadata[
                        "dataset_max_n_samples_after_preprocess"
                    ],
                    "dataset_min_duration_sec_after_preprocess": metadata[
                        "dataset_min_duration_sec_after_preprocess"
                    ],
                    "was_transposed": metadata["was_transposed"],
                    "crop_from": metadata["crop_from"],
                    "lowcut": metadata["lowcut"],
                    "highcut": metadata["highcut"],
                    "notch": metadata["notch"],
                    "apply_reference": metadata["apply_reference"],
                    "apply_notch": metadata["apply_notch"],
                    "apply_filter": metadata["apply_filter"],
                    "apply_resample": metadata["apply_resample"],
                    "did_reference": metadata["did_reference"],
                    "did_notch": metadata["did_notch"],
                    "did_filter": metadata["did_filter"],
                    "did_resample": metadata["did_resample"],
                    "scale": metadata["scale"],
                }
            )

        return pd.DataFrame(rows)

    def print_audit(self, max_rows: int = 20) -> None:
        summary = self.get_summary_dataframe()

        print("\nIranies EEG dataset audit")
        print(f"  subjects:           {len(self)}")
        print(f"  labels:             {dict(Counter(self.get_labels()))}")
        print(f"  channels:           {summary['n_channels'].unique().tolist()}")
        print(f"  original_fs:        {sorted(summary['original_fs'].unique().tolist())}")
        print(f"  final_fs:           {sorted(summary['final_fs'].unique().tolist())}")
        print(f"  fs_was_assumed:     {summary['fs_was_assumed'].value_counts().to_dict()}")
        print(f"  did_resample:       {summary['did_resample'].value_counts().to_dict()}")
        print(f"  crop_policy:        {self.crop_policy}")
        print(f"  crop_n_samples:     {self.crop_n_samples}")
        print(f"  selected_duration:  {self.selected_duration_sec:.6f} s")
        print(f"  shortest_subjects:  {self.min_duration_subject_ids}")
        print(
            "  min/max after preprocess: "
            f"{self.min_n_samples_after_preprocess}/"
            f"{self.max_n_samples_after_preprocess} samples"
        )

        cols = [
            "subject_id",
            "label_name",
            "signal_key",
            "fs_key",
            "fs_was_assumed",
            "original_fs",
            "final_fs",
            "n_samples_before",
            "n_samples_after_resample",
            "duration_sec_after_resample",
            "is_min_duration_subject",
            "crop_n_samples",
            "selected_duration_sec",
            "did_resample",
            "was_transposed",
        ]

        print("\nShortest subjects after preprocessing")
        print(
            summary[cols]
            .sort_values("n_samples_after_resample")
            .head(max_rows)
            .to_string(index=False)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return (
            sample["subject_id"],
            sample["eeg"],
            sample["label"],
        )


def create_kfold_dataloaders_(
    dataset: EEGDataset_ADHD,
    k: int = 10,
    batch_size: int = 4,
    shuffle: bool = True,
    split_seed: int = 3407,
    inner_splits: int = 5,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Crea particiones train/val/test a nivel sujeto.

    Cada batch contiene:

        batch["X"]:          (B, C, T)
        batch["y"]:          (B,)
        batch["subject_id"]: list[str]
        batch["lengths"]:    (B,), todos T
        batch["mask"]:       (B, T), todo True

    La etiqueta temporal y_time in {0,1}^{B x T'} se construye en train_iranies.py,
    después de conocer T' a la salida del encoder EEGNet.
    """

    subjects = np.array(dataset.get_subject_ids())
    labels = np.array(dataset.get_labels())

    label_counts = Counter(labels.tolist())

    if min(label_counts.values()) < k:
        raise ValueError(
            f"k={k} es muy grande para la distribución {dict(label_counts)}. "
            f"Usa k <= {min(label_counts.values())}."
        )

    def collate_subjects(batch):
        subject_ids, eeg_list, label_list = zip(*batch)

        n_channels_set = {int(eeg.shape[0]) for eeg in eeg_list}
        n_samples_set = {int(eeg.shape[1]) for eeg in eeg_list}

        if len(n_channels_set) != 1:
            raise ValueError(
                f"Todos los sujetos deben tener los mismos canales. "
                f"Encontrado: {n_channels_set}"
            )

        if len(n_samples_set) != 1:
            raise ValueError(
                f"Todos los sujetos deben tener el mismo T. "
                f"Encontrado: {n_samples_set}"
            )

        X = torch.stack(eeg_list, dim=0)
        y = torch.stack(
            [
                torch.as_tensor(label, dtype=torch.long).reshape(())
                for label in label_list
            ],
            dim=0,
        )

        batch_size_actual = X.shape[0]
        T = X.shape[-1]

        lengths = torch.full(
            (batch_size_actual,),
            fill_value=T,
            dtype=torch.long,
        )

        mask = torch.ones(
            batch_size_actual,
            T,
            dtype=torch.bool,
        )

        return {
            "X": X,
            "y": y,
            "subject_id": list(subject_ids),
            "lengths": lengths,
            "mask": mask,
        }

    outer_gkf = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=split_seed,
    )

    folds = []

    for trainval_idx, test_idx in outer_gkf.split(
        X=np.zeros(len(labels)),
        y=labels,
        groups=subjects,
    ):
        trainval_idx = np.array(trainval_idx)
        test_idx = np.array(test_idx)

        trainval_labels = labels[trainval_idx]
        trainval_subjects = subjects[trainval_idx]
        trainval_label_counts = Counter(trainval_labels.tolist())

        effective_inner_splits = min(
            inner_splits,
            min(trainval_label_counts.values()),
        )

        if effective_inner_splits < 2:
            raise ValueError(
                "No hay suficientes sujetos por clase para crear validación."
            )

        inner_gkf = StratifiedGroupKFold(
            n_splits=effective_inner_splits,
            shuffle=True,
            random_state=split_seed,
        )

        train_rel_idx, val_rel_idx = next(
            inner_gkf.split(
                X=np.zeros(len(trainval_idx)),
                y=trainval_labels,
                groups=trainval_subjects,
            )
        )

        train_idx = trainval_idx[train_rel_idx]
        val_idx = trainval_idx[val_rel_idx]

        train_dataset = Subset(dataset, train_idx.tolist())
        val_dataset = Subset(dataset, val_idx.tolist())
        test_dataset = Subset(dataset, test_idx.tolist())

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adhd-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "ADHD"),
    )
    parser.add_argument(
        "--control-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "control"),
    )

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--target-fs", type=float, default=128.0)
    parser.add_argument("--default-fs", type=float, default=128.0)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)
    parser.add_argument("--crop-from", type=str, default="start", choices=["start", "center"])
    parser.add_argument("--max-channels", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)

    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-notch", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-resample", action="store_true")

    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=20)

    args = parser.parse_args()

    dataset = EEGDataset_ADHD(
        adhd_dir=args.adhd_dir,
        control_dir=args.control_dir,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        default_fs=args.default_fs,
        duration_sec=args.duration_sec,
        crop_from=args.crop_from,
        max_channels=args.max_channels,
        scale=args.scale,
        apply_reference=not args.no_reference,
        apply_notch=not args.no_notch,
        apply_filter=not args.no_filter,
        apply_resample=not args.no_resample,
        verbose=True,
    )

    dataset.print_audit(max_rows=args.max_rows)

    if args.summary_csv is not None:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.get_summary_dataframe().to_csv(summary_path, index=False)
        print(f"\nSaved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
