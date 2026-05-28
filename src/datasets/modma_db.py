from __future__ import annotations

import argparse
import csv
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import mne
mne.set_log_level("WARNING")
import numpy as np
import torch

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


LABEL_NAMES = {
    0: "HC",
    1: "MDD",
}


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def tokenize_path(path: Path | str) -> list[str]:
    text = str(path).upper()
    return [tok for tok in re.split(r"[/\\_.\-\s()]+", text) if tok]


def normalize_subject_id(value: Any) -> str | None:
    """Normalize subject identifiers so file names and CSV rows can be matched.

    The MODMA archives are not always distributed with identical column names,
    so this function keeps matching conservative but flexible.
    """
    if value is None:
        return None

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None

    text = Path(text).stem
    text = text.upper()
    text = re.sub(r"\.(MAT|EDF|SET|FIF|CSV|TXT|NPY|NPZ)$", "", text)
    text = re.sub(r"[^A-Z0-9]+", "", text)

    if text == "":
        return None

    # If the identifier is purely numeric, remove leading zeros for matching.
    if text.isdigit():
        return str(int(text))

    # Preserve clinically meaningful prefixes, but normalize the numeric suffix.
    m = re.match(r"^(MDD|HC|H|CONTROL|NC|NORMAL|PATIENT|DEPRESSED|DEPRESSION|SUB|SUBJECT|S)(\d+)$", text)
    if m:
        prefix, num = m.groups()
        if prefix in {"CONTROL", "NC", "NORMAL"}:
            prefix = "HC"
        if prefix in {"PATIENT", "DEPRESSED", "DEPRESSION"}:
            prefix = "MDD"
        if prefix in {"SUBJECT", "SUB"}:
            prefix = "S"
        return f"{prefix}{int(num)}"

    return text


def subject_aliases(value: Any) -> set[str]:
    """Generate possible aliases for a subject id or path stem."""
    aliases: set[str] = set()
    norm = normalize_subject_id(value)

    if norm is None:
        return aliases

    aliases.add(norm)

    m = re.search(r"(\d+)$", norm)
    if m:
        num = str(int(m.group(1)))
        aliases.add(num)
        aliases.add(num.zfill(2))
        aliases.add(num.zfill(3))
        aliases.add(num.zfill(4))

        prefix = norm[: -len(m.group(1))]
        if prefix:
            aliases.add(f"{prefix}{num}")
            aliases.add(f"{prefix}{num.zfill(2)}")
            aliases.add(f"{prefix}{num.zfill(3)}")
            aliases.add(f"{prefix}{num.zfill(4)}")

    return {a for a in aliases if a}


def infer_label_from_value(value: Any) -> int | None:
    if value is None:
        return None

    text = str(value).strip().upper()

    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None

    # Numeric labels are common in prepared CSV files.
    try:
        number = float(text)
        if np.isfinite(number):
            if int(number) == 0:
                return 0
            if int(number) == 1:
                return 1
    except Exception:
        pass

    tokens = tokenize_path(text)

    mdd_tokens = {
        "MDD",
        "DEPRESSED",
        "DEPRESSION",
        "DEPRESSIVE",
        "PATIENT",
        "PATIENTS",
        "CASE",
        "CASES",
    }
    hc_tokens = {
        "HC",
        "H",
        "HEALTHY",
        "CONTROL",
        "CONTROLS",
        "NORMAL",
        "NC",
    }

    if any(tok in mdd_tokens for tok in tokens):
        return 1

    if any(tok in hc_tokens for tok in tokens):
        return 0

    # Some CSVs use Chinese/long strings. Keep basic English fragments too.
    compact = "".join(tokens)
    if "MDD" in compact or "DEPRESS" in compact or "PATIENT" in compact:
        return 1
    if "CONTROL" in compact or "HEALTH" in compact or "NORMAL" in compact:
        return 0

    return None


def infer_label_from_path(path: Path) -> int | None:
    # Check parent folders first, then file name.
    for part in reversed(path.parts):
        label = infer_label_from_value(part)
        if label is not None:
            return label
    return None


def infer_subject_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    stem = path.stem

    patterns = [
        r"\b(MDD|HC|H|NC)\s*[-_ ]?S?0*(\d{1,5})\b",
        r"\b(PATIENT|CONTROL|NORMAL)\s*[-_ ]?0*(\d{1,5})\b",
        r"\bSUB(?:JECT)?\s*[-_ ]?0*([A-Z0-9]+)\b",
        r"\bS\s*[-_ ]?0*(\d{1,5})\b",
    ]

    for source in [stem.upper(), rel.upper()]:
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    prefix, number = groups
                    prefix = prefix.upper()
                    if prefix in {"CONTROL", "NORMAL", "NC"}:
                        prefix = "HC"
                    if prefix == "PATIENT":
                        prefix = "MDD"
                    return f"{prefix}{int(number)}"
                return normalize_subject_id(groups[0]) or path.stem

    return normalize_subject_id(stem) or path.stem


def discover_metadata_csvs(root: Path, explicit_csv: str | Path | None = None) -> list[Path]:
    if explicit_csv is not None:
        path = Path(explicit_csv)
        if not path.is_absolute():
            path = root / path
        return [path] if path.exists() else []

    candidates = []
    for path in sorted(root.rglob("*.csv")):
        # Skip extremely large feature CSVs if any are later generated in-place.
        if path.stat().st_size > 50_000_000:
            continue
        candidates.append(path)

    return candidates


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    encodings = ["utf-8-sig", "utf-8", "latin1"]

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                sample = f.read(4096)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except Exception:
                    dialect = csv.excel

                reader = csv.DictReader(f, dialect=dialect)
                return [dict(row) for row in reader]
        except UnicodeDecodeError:
            continue

    return []


def build_metadata_index(
    root: Path,
    metadata_csv: str | Path | None = None,
    subject_col: str | None = None,
    label_col: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a subject -> metadata dictionary from all small CSV files.

    If a row can be matched to a subject and a label can be inferred from one of
    its columns, the row is added to every known alias of that subject.
    """
    index: dict[str, dict[str, Any]] = {}
    csv_paths = discover_metadata_csvs(root, explicit_csv=metadata_csv)

    subject_keywords = [
        "subject",
        "sub",
        "participant",
        "id",
        "name",
        "file",
        "filename",
        "eeg",
    ]
    label_keywords = [
        "type",
        "group",
        "label",
        "class",
        "diagnosis",
        "diagnose",
        "category",
        "target",
    ]

    for csv_path in csv_paths:
        rows = read_csv_rows(csv_path)

        if not rows:
            continue

        columns = sorted(
            {
                str(c).strip()
                for row in rows
                for c in row.keys()
                if c is not None and str(c).strip() != ""
            }
        )

        if not columns:
            continue

        lower_map = {str(c).lower().strip(): c for c in columns if c is not None}

        if subject_col is not None:
            subject_candidates = [subject_col] if subject_col in columns else []
        else:
            subject_candidates = [
                c for c in columns
                if any(k in c.lower().strip() for k in subject_keywords)
            ]

        if label_col is not None:
            label_candidates = [label_col] if label_col in columns else []
        else:
            label_candidates = [
                c for c in columns
                if any(k in c.lower().strip() for k in label_keywords)
            ]

        # Use exact common variants when available.
        for key in ["subject id", "subject_id", "participant_id", "id"]:
            if key in lower_map and lower_map[key] not in subject_candidates:
                subject_candidates.insert(0, lower_map[key])

        for key in ["type", "group", "label", "diagnosis", "class"]:
            if key in lower_map and lower_map[key] not in label_candidates:
                label_candidates.insert(0, lower_map[key])

        for row in rows:
            subject = None

            for col in subject_candidates:
                subject = normalize_subject_id(row.get(col))
                if subject is not None:
                    break

            if subject is None:
                continue

            label = None

            for col in label_candidates:
                label = infer_label_from_value(row.get(col))
                if label is not None:
                    break

            # Some files encode the label in a subject id like MDD12/HC01.
            if label is None:
                label = infer_label_from_value(subject)

            metadata = {
                "subject": subject,
                "label": label,
                "label_name": LABEL_NAMES.get(label) if label is not None else None,
                "csv_path": str(csv_path),
                "csv_row": row,
            }

            for alias in subject_aliases(subject):
                index[alias] = metadata

            # Also index original subject string variants from candidate columns.
            for col in subject_candidates:
                for alias in subject_aliases(row.get(col)):
                    index[alias] = metadata

    return index


def _iter_scipy_arrays(obj: Any, prefix: str = ""):
    """Yield numeric arrays from nested scipy.io.loadmat objects."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("__"):
                continue
            yield from _iter_scipy_arrays(value, f"{prefix}/{key}")
        return

    # scipy.io.loadmat(..., struct_as_record=False) returns mat_struct
    # objects for MATLAB structs. Net Station exports can store signal
    # matrices inside nested structs, so walk these fields too.
    if hasattr(obj, "_fieldnames"):
        for name in getattr(obj, "_fieldnames", []) or []:
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            yield from _iter_scipy_arrays(value, f"{prefix}/{name}")
        return

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for name in obj.dtype.names:
                try:
                    yield from _iter_scipy_arrays(obj[name], f"{prefix}/{name}")
                except Exception:
                    pass
            return

        if obj.dtype == object:
            # Avoid walking huge object arrays deeply unless they are tiny.
            if obj.size <= 1000:
                for i, item in enumerate(obj.flat):
                    yield from _iter_scipy_arrays(item, f"{prefix}/{i}")
            return

        if np.issubdtype(obj.dtype, np.number) and obj.ndim >= 2:
            yield prefix.strip("/"), obj


def _iter_h5_arrays(h5obj: Any, prefix: str = ""):
    if h5py is None:
        return

    if isinstance(h5obj, h5py.Dataset):
        if h5obj.ndim >= 2 and np.issubdtype(h5obj.dtype, np.number):
            try:
                yield prefix.strip("/"), np.asarray(h5obj)
            except Exception:
                return
        return

    if isinstance(h5obj, h5py.Group):
        for key in h5obj.keys():
            yield from _iter_h5_arrays(h5obj[key], f"{prefix}/{key}")


def orient_signal(data: np.ndarray, expected_channels: int | None = 128) -> np.ndarray:
    """Return EEG-like data in C x T format."""
    data = np.asarray(data)

    if np.iscomplexobj(data):
        data = np.real(data)

    data = np.squeeze(data)

    if data.ndim < 2:
        raise ValueError("Signal candidate must have at least 2 dimensions.")

    if data.ndim > 2:
        # Prefer an axis with the expected channel count. Keep that axis first
        # and concatenate all remaining axes as time/trials.
        if expected_channels is not None and expected_channels in data.shape:
            ch_axis = list(data.shape).index(expected_channels)
        else:
            # EEG channel axes are usually the smallest non-singleton dimension.
            dims = list(data.shape)
            ch_axis = int(np.argmin(dims))

        data = np.moveaxis(data, ch_axis, 0)
        data = data.reshape(data.shape[0], -1)

    if data.ndim != 2:
        raise ValueError("Signal candidate could not be reshaped to 2D.")

    rows, cols = data.shape

    if expected_channels is not None:
        if rows == expected_channels:
            pass
        elif cols == expected_channels:
            data = data.T
        elif rows > cols:
            data = data.T
    elif rows > cols:
        data = data.T

    return data.astype(np.float32, copy=False)


def candidate_score(name: str, arr: np.ndarray, expected_channels: int | None = 128) -> tuple[int, int]:
    name_lower = name.lower()
    bad_tokens = [
        "event",
        "events",
        "latency",
        "time",
        "times",
        "chanloc",
        "channel",
        "label",
        "labels",
        "epoch",
        "trigger",
        "stim",
        "class",
        "target",
    ]
    good_tokens = ["data", "eeg", "signal", "signals", "wave", "waveform", "record"]

    score = 0

    if any(tok in name_lower for tok in good_tokens):
        score += 20

    if any(tok in name_lower for tok in bad_tokens):
        score -= 20

    shape = tuple(int(x) for x in np.squeeze(arr).shape)

    if expected_channels is not None and expected_channels in shape:
        score += 100

    # Resting EEG should contain many more time points than channels.
    if len(shape) >= 2:
        dims = sorted(shape)
        if dims[-1] >= 1000:
            score += 20
        if dims[0] <= 1:
            score -= 10

    score += int(np.prod(shape) // 1000)

    return score, int(np.prod(shape))


def read_mat_signal(path: Path, expected_channels: int | None = 128) -> np.ndarray:
    candidates: list[tuple[int, int, str, np.ndarray]] = []

    if loadmat is not None:
        try:
            mat = loadmat(path, squeeze_me=False, struct_as_record=False)
            for name, arr in _iter_scipy_arrays(mat):
                try:
                    oriented = orient_signal(arr, expected_channels=expected_channels)
                except Exception:
                    continue

                if oriented.shape[0] <= 1 or oriented.shape[1] <= 10:
                    continue

                score, size = candidate_score(name, arr, expected_channels=expected_channels)
                candidates.append((score, size, name, oriented))
        except NotImplementedError:
            pass
        except ValueError:
            pass
        except Exception:
            pass

    if not candidates and h5py is not None:
        try:
            with h5py.File(path, "r") as f:
                for name, arr in _iter_h5_arrays(f):
                    try:
                        oriented = orient_signal(arr, expected_channels=expected_channels)
                    except Exception:
                        continue

                    if oriented.shape[0] <= 1 or oriented.shape[1] <= 10:
                        continue

                    score, size = candidate_score(name, arr, expected_channels=expected_channels)
                    candidates.append((score, size, name, oriented))
        except Exception:
            pass

    if not candidates:
        raise ValueError(f"No EEG-like matrix found in {path}")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][3]


def read_text_matrix(path: Path, delimiter: str | None = None) -> np.ndarray:
    if delimiter is None:
        # Let numpy infer whitespace for txt/asc/dat; csv is handled explicitly.
        delimiter = "," if path.suffix.lower() == ".csv" else None
    data = np.loadtxt(path.as_posix(), delimiter=delimiter)
    return orient_signal(data, expected_channels=None)


def read_numpy_matrix(path: Path) -> np.ndarray:
    obj = np.load(path.as_posix(), allow_pickle=True)

    if isinstance(obj, np.lib.npyio.NpzFile):
        arrays = []
        for key in obj.files:
            arr = obj[key]
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2:
                arrays.append((arr.size, key, arr))
        if not arrays:
            raise ValueError(f"No numeric matrix found in {path}")
        arrays.sort(reverse=True)
        data = arrays[0][2]
    else:
        data = obj

    return orient_signal(data, expected_channels=None)


def read_edf_header(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.io.read_raw_edf(path.as_posix(), preload=False, verbose="ERROR")


def get_eeg_channel_names(raw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    if len(picks) == 0:
        return list(raw.ch_names)

    return [raw.ch_names[i] for i in picks]


class MODMADataset(Dataset):
    """MODMA 128-channel resting-state EEG dataset for MDD vs HC classification.

    Labels:
        0 -> HC / healthy control
        1 -> MDD

    Returns:
        name: str
        eeg_tensor: Tensor[C, T]
        label: LongTensor scalar
    """

    SUPPORTED_SUFFIXES = {".mat", ".edf", ".set", ".fif", ".fif.gz", ".csv", ".txt", ".dat", ".asc", ".npy", ".npz"}

    def __init__(
        self,
        root: str | Path = PROJECT_ROOT / "data/raw/modma_db",
        eeg_dir: str | Path | None = "EEG_128channels_resting_lanzhou_2015",
        metadata_csv: str | Path | None = None,
        subject_col: str | None = None,
        label_col: str | None = None,
        lowcut: float | None = 0.5,
        highcut: float | None = 60.0,
        notch: float | None = 50.0,
        default_fs: float = 250.0,
        target_fs: float | None = None,
        duration_sec: float | None = None,
        pp_as: str = "tensor",
        channel_strategy: str = "common",
        expected_channels: int | None = 128,
    ):
        self.root = Path(root)
        self.eeg_dir = None if eeg_dir is None else Path(eeg_dir)
        self.metadata_csv = metadata_csv
        self.subject_col = subject_col
        self.label_col = label_col
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch = notch
        self.default_fs = float(default_fs)
        self.target_fs = None if target_fs is None else float(target_fs)
        self.duration_sec = duration_sec
        self.pp_as = pp_as
        self.channel_strategy = channel_strategy
        self.expected_channels = expected_channels

        if self.pp_as not in {"tensor", "list"}:
            raise ValueError("pp_as must be 'tensor' or 'list'.")

        if self.channel_strategy not in {"common", "all"}:
            raise ValueError("channel_strategy must be 'common' or 'all'.")

        if not self.root.exists():
            raise FileNotFoundError(f"MODMA root does not exist: {self.root}")

        self.data_root = self._resolve_data_root()
        self.metadata_index = build_metadata_index(
            root=self.root,
            metadata_csv=self.metadata_csv,
            subject_col=self.subject_col,
            label_col=self.label_col,
        )
        self.records = self._discover_records()

        if len(self.records) == 0:
            raise ValueError(
                f"No labeled MODMA resting EEG files found in {self.data_root}. "
                "Run the inspect script to check file names, CSV labels, and extensions."
            )

        self.n_channels = self._resolve_n_channels()
        self.channel_names = [f"ch{i:03d}" for i in range(1, self.n_channels + 1)]
        self.samples: list[dict[str, Any]] = []

        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError("No MODMA samples were loaded after preprocessing.")

        if self.pp_as == "tensor":
            self._crop_to_min_time()

    def _resolve_data_root(self) -> Path:
        if self.eeg_dir is None or str(self.eeg_dir).lower() in {".", "none", "null"}:
            return self.root

        if self.eeg_dir.is_absolute():
            data_root = self.eeg_dir
        else:
            data_root = self.root / self.eeg_dir

        if not data_root.exists():
            raise FileNotFoundError(f"MODMA EEG directory does not exist: {data_root}")

        return data_root

    def _is_supported_file(self, path: Path) -> bool:
        name_lower = path.name.lower()

        if name_lower.endswith(".fif.gz"):
            return True

        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def _lookup_metadata(self, path: Path, subject: str) -> dict[str, Any] | None:
        keys = set()
        keys.update(subject_aliases(subject))
        keys.update(subject_aliases(path.stem))
        keys.update(subject_aliases(path.name))

        for key in keys:
            if key in self.metadata_index:
                return self.metadata_index[key]

        # Last resort: if a CSV subject id is a substring of the stem, match it.
        stem_norm = normalize_subject_id(path.stem) or path.stem.upper()
        for key, meta in self.metadata_index.items():
            if key and (key in stem_norm or stem_norm in key):
                return meta

        return None

    def _discover_records(self) -> list[dict[str, Any]]:
        records = []

        for path in sorted(self.data_root.rglob("*")):
            if not path.is_file() or not self._is_supported_file(path):
                continue

            # Avoid accidentally using root-level metadata CSVs as EEG.
            if path.suffix.lower() == ".csv" and path.parent == self.root:
                continue

            subject = infer_subject_from_path(path, self.data_root)
            metadata = self._lookup_metadata(path, subject)

            label = None
            label_name = None
            csv_path = None

            if metadata is not None:
                label = metadata.get("label")
                label_name = metadata.get("label_name")
                csv_path = metadata.get("csv_path")

            if label is None:
                label = infer_label_from_path(path)
                label_name = LABEL_NAMES.get(label) if label is not None else None

            if label is None:
                continue

            if label_name is None:
                label_name = LABEL_NAMES.get(label)

            records.append(
                {
                    "path": path,
                    "rel_path": path.relative_to(self.root).as_posix(),
                    "name": path.name,
                    "subject": subject,
                    "label": int(label),
                    "label_name": label_name,
                    "csv_path": csv_path,
                }
            )

        return sorted(records, key=lambda x: (x["label"], x["subject"], x["name"]))

    def _read_raw_signal_for_shape(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        name_lower = path.name.lower()

        if suffix == ".mat":
            return read_mat_signal(path, expected_channels=self.expected_channels)

        if suffix == ".edf":
            raw = read_edf_header(path)
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(picks) == 0:
                n_ch = len(raw.ch_names)
            else:
                n_ch = len(picks)
            # Only shape is needed here, so make a tiny placeholder.
            return np.empty((n_ch, max(1, int(raw.n_times))), dtype=np.float32)

        if suffix == ".set":
            raw = mne.io.read_raw_eeglab(path.as_posix(), preload=False, verbose="ERROR")
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            n_ch = len(picks) if len(picks) else len(raw.ch_names)
            return np.empty((n_ch, max(1, int(raw.n_times))), dtype=np.float32)

        if suffix == ".fif" or name_lower.endswith(".fif.gz"):
            raw = mne.io.read_raw_fif(path.as_posix(), preload=False, verbose="ERROR")
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            n_ch = len(picks) if len(picks) else len(raw.ch_names)
            return np.empty((n_ch, max(1, int(raw.n_times))), dtype=np.float32)

        if suffix in {".npy", ".npz"}:
            return read_numpy_matrix(path)

        if suffix in {".csv", ".txt", ".dat", ".asc"}:
            return read_text_matrix(path)

        raise ValueError(f"Unsupported file extension: {path}")

    def _resolve_n_channels(self) -> int:
        channel_counts = []

        for record in self.records:
            try:
                data = self._read_raw_signal_for_shape(record["path"])
                channel_counts.append(int(data.shape[0]))
            except Exception as exc:
                print(f"Could not inspect {record['rel_path']}: {exc}")

        if not channel_counts:
            raise ValueError("Could not inspect channel counts from MODMA files.")

        unique_counts = sorted(set(channel_counts))

        if self.channel_strategy == "all":
            if len(unique_counts) != 1:
                raise ValueError(
                    "Not all MODMA files have the same channel count. "
                    f"Found: {unique_counts}. Use channel_strategy='common'."
                )
            return unique_counts[0]

        # For common strategy, use expected_channels when present in all files;
        # otherwise trim to the minimum channel count.
        if self.expected_channels is not None and all(c >= self.expected_channels for c in channel_counts):
            return int(self.expected_channels)

        return int(min(channel_counts))

    def _read_preprocessed_signal(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        name_lower = path.name.lower()

        if suffix == ".edf":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(picks):
                raw.pick([raw.ch_names[i] for i in picks])
            data = raw.get_data().astype(np.float32)
            fs = float(raw.info["sfreq"])
            return self._preprocess_array(data, fs)

        if suffix == ".set":
            raw = mne.io.read_raw_eeglab(path.as_posix(), preload=True, verbose="ERROR")
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(picks):
                raw.pick([raw.ch_names[i] for i in picks])
            data = raw.get_data().astype(np.float32)
            fs = float(raw.info["sfreq"])
            return self._preprocess_array(data, fs)

        if suffix == ".fif" or name_lower.endswith(".fif.gz"):
            raw = mne.io.read_raw_fif(path.as_posix(), preload=True, verbose="ERROR")
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(picks):
                raw.pick([raw.ch_names[i] for i in picks])
            data = raw.get_data().astype(np.float32)
            fs = float(raw.info["sfreq"])
            return self._preprocess_array(data, fs)

        if suffix == ".mat":
            data = read_mat_signal(path, expected_channels=self.expected_channels)
            return self._preprocess_array(data, self.default_fs)

        if suffix in {".npy", ".npz"}:
            data = read_numpy_matrix(path)
            return self._preprocess_array(data, self.default_fs)

        if suffix in {".csv", ".txt", ".dat", ".asc"}:
            data = read_text_matrix(path)
            return self._preprocess_array(data, self.default_fs)

        raise ValueError(f"Unsupported file extension: {path}")

    def _preprocess_array(self, data: np.ndarray, fs: float) -> np.ndarray:
        data = orient_signal(data, expected_channels=self.expected_channels)
        data = data.astype(np.float32, copy=False)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if data.shape[0] < self.n_channels:
            raise ValueError(
                f"Signal has {data.shape[0]} channels, but {self.n_channels} are required."
            )

        # Trim channels in common strategy. For MODMA 128ch this normally keeps 128.
        data = data[: self.n_channels, :]

        ch_names = [f"ch{i:03d}" for i in range(1, data.shape[0] + 1)]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=float(fs),
            ch_types=["eeg"] * len(ch_names),
        )

        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_eeg_reference("average", verbose=False)

        current_fs = float(raw.info["sfreq"])

        if self.notch is not None and self.notch < current_fs / 2:
            raw.notch_filter([self.notch], verbose=False)

        highcut = self.highcut

        if highcut is not None:
            highcut = min(highcut, current_fs / 2 - 1e-3)

            if self.lowcut is not None and highcut <= self.lowcut:
                highcut = None

        if self.lowcut is not None or highcut is not None:
            raw.filter(
                l_freq=self.lowcut,
                h_freq=highcut,
                fir_design="firwin",
                verbose=False,
            )

        if self.target_fs is not None and not np.isclose(current_fs, self.target_fs):
            raw.resample(self.target_fs, npad="auto", verbose=False)

        eeg = raw.get_data().astype(np.float32)
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

        if self.duration_sec is not None:
            final_fs = float(raw.info["sfreq"])
            n_samples = int(float(self.duration_sec) * final_fs)
            eeg = eeg[:, :n_samples]

        return eeg

    def _load_samples(self) -> None:
        for record in self.records:
            try:
                eeg = self._read_preprocessed_signal(record["path"])
            except Exception as exc:
                print(f"Skipping {record['rel_path']}: {exc}")
                continue

            self.samples.append(
                {
                    "name": record["name"],
                    "subject": record["subject"],
                    "rel_path": record["rel_path"],
                    "label_name": record["label_name"],
                    "eeg": torch.tensor(eeg, dtype=torch.float32),
                    "label": torch.tensor(record["label"], dtype=torch.long),
                }
            )

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
        return list(names), list(X), torch.stack(y)

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

    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "data/raw/modma_db"))
    parser.add_argument("--eeg-dir", type=str, default="EEG_128channels_resting_lanzhou_2015")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--subject-col", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--default-fs", type=float, default=250.0)
    parser.add_argument("--target-fs", type=parse_optional_float, default=None)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)
    parser.add_argument("--expected-channels", type=int, default=128)

    parser.add_argument("--pp-as", type=str, default="tensor", choices=["tensor", "list"])
    parser.add_argument("--channel-strategy", type=str, default="common", choices=["common", "all"])
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = MODMADataset(
        root=args.root,
        eeg_dir=args.eeg_dir,
        metadata_csv=args.metadata_csv,
        subject_col=args.subject_col,
        label_col=args.label_col,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        default_fs=args.default_fs,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as=args.pp_as,
        channel_strategy=args.channel_strategy,
        expected_channels=args.expected_channels,
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
    print("Dataset: MODMA 128-channel resting-state EEG")
    print(f"Root: {dataset.root}")
    print(f"EEG dir: {dataset.data_root}")
    print(f"Subjects/files: {len(dataset)}")
    print(f"Unique subjects: {len(set(subjects))}")
    print(f"HC   0: {label_count.get(0, 0)}")
    print(f"MDD  1: {label_count.get(1, 0)}")

    print("\nPreprocessing:")
    print(f"lowcut:           {dataset.lowcut}")
    print(f"highcut:          {dataset.highcut}")
    print(f"notch:            {dataset.notch}")
    print(f"default_fs:       {dataset.default_fs}")
    print(f"target_fs:        {dataset.target_fs}")
    print(f"duration_sec:     {dataset.duration_sec}")
    print(f"pp_as:            {dataset.pp_as}")
    print(f"channel_strategy: {dataset.channel_strategy}")
    print(f"n_channels:       {dataset.n_channels}")

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
                f"HC={count.get(0, 0):3d} | "
                f"MDD={count.get(1, 0):3d} | "
                f"Batches={len(loader)}"
            )


if __name__ == "__main__":
    main()
