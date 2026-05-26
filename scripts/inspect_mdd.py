from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mne
except Exception as exc:
    mne = None
    MNE_IMPORT_ERROR = repr(exc)
else:
    MNE_IMPORT_ERROR = None

try:
    from scipy.io import loadmat
except Exception as exc:
    loadmat = None
    SCIPY_IMPORT_ERROR = repr(exc)
else:
    SCIPY_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data/raw/mdd_db"
DEFAULT_OUT = PROJECT_ROOT / "outputs/mdd_db/inspection"

RAW_EXTENSIONS = {
    ".set",
    ".vhdr",
    ".edf",
    ".bdf",
    ".gdf",
    ".cnt",
    ".fif",
}

MAT_EXTENSIONS = {".mat"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}
CANDIDATE_EXTENSIONS = RAW_EXTENSIONS | MAT_EXTENSIONS | TABLE_EXTENSIONS
EEG_EXTENSIONS = RAW_EXTENSIONS | MAT_EXTENSIONS

LABEL_ALIASES = {
    "MDD": 1,
    "DEPRESSED": 1,
    "DEPRESSION": 1,
    "PATIENT": 1,
    "PATIENTS": 1,
    "H": 0,
    "HC": 0,
    "CONTROL": 0,
    "CONTROLS": 0,
    "HEALTHY": 0,
    "NORMAL": 0,
}

LABEL_NAMES = {
    0: "H",
    1: "MDD",
}

CONDITION_ALIASES = {
    "EC": "EC",
    "CLOSED": "EC",
    "CLOSE": "EC",
    "EYESCLOSED": "EC",
    "EYESCLOSE": "EC",
    "EO": "EO",
    "OPEN": "EO",
    "EYESOPEN": "EO",
    "TASK": "TASK",
    "P300": "TASK",
}


def human_bytes(n: int) -> str:
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def tokenize_path(path: Path) -> list[str]:
    text = path.as_posix().upper()
    return [tok for tok in re.split(r"[/\\_.\-\s()]+", text) if tok]


def infer_label(tokens: list[str]) -> tuple[int | None, str | None]:
    for token in tokens:
        if token in LABEL_ALIASES:
            label = LABEL_ALIASES[token]
            return label, LABEL_NAMES[label]
    return None, None


def infer_condition(tokens: list[str]) -> str | None:
    joined = "".join(tokens)

    for token in tokens:
        if token in CONDITION_ALIASES:
            return CONDITION_ALIASES[token]

    for key, value in CONDITION_ALIASES.items():
        if key in joined:
            return value

    return None


def infer_subject(path: Path, root: Path) -> str:
    rel = safe_rel(path, root)
    text = rel.upper()
    stem = path.stem.upper()

    patterns = [
        r"(MDD|HC|H)[-_ ]?(\d{1,4})",
        r"SUB[-_ ]?([A-Z0-9]+)",
        r"S(?:UBJECT)?[-_ ]?(\d{1,4})",
        r"(\d{1,4})",
    ]

    for source in [stem, text]:
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return "".join(match.groups())

    return path.stem


def infer_metadata(path: Path, root: Path) -> dict:
    rel = safe_rel(path, root)
    tokens = tokenize_path(Path(rel))
    label, label_name = infer_label(tokens)
    condition = infer_condition(tokens)
    subject = infer_subject(path, root)

    return {
        "path": rel,
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "size": human_bytes(path.stat().st_size),
        "label": label,
        "label_name": label_name,
        "condition": condition,
        "subject": subject,
        "is_training_candidate": (
            label is not None
            and condition in {"EC", "EO"}
            and path.suffix.lower() in EEG_EXTENSIONS
        ),
    }


def get_mne_reader(path: Path):
    if mne is None:
        return None

    mapping = {
        ".set": "read_raw_eeglab",
        ".vhdr": "read_raw_brainvision",
        ".edf": "read_raw_edf",
        ".bdf": "read_raw_bdf",
        ".gdf": "read_raw_gdf",
        ".cnt": "read_raw_cnt",
        ".fif": "read_raw_fif",
    }

    reader_name = mapping.get(path.suffix.lower())
    if reader_name is None:
        return None

    return getattr(mne.io, reader_name, None)


def inspect_raw_file(path: Path) -> dict:
    reader = get_mne_reader(path)

    if reader is None:
        return {
            "read_ok": False,
            "kind": "raw",
            "error": MNE_IMPORT_ERROR or f"No MNE reader for {path.suffix}",
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = reader(path.as_posix(), preload=False, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    n_channels = int(len(raw.ch_names))

    return {
        "read_ok": True,
        "kind": "raw",
        "reader": reader.__name__,
        "n_channels": n_channels,
        "n_times": n_times,
        "sfreq": sfreq,
        "duration_sec": n_times / sfreq if sfreq else None,
        "ch_names_first_12": raw.ch_names[:12],
        "channel_types": dict(Counter(raw.get_channel_types())),
    }


def find_fs_in_mat(mat: dict) -> float | None:
    fs_tokens = [
        "FS",
        "SFREQ",
        "FREQ",
        "SRATE",
        "SAMPLING",
        "SAMPLE_RATE",
        "SAMPLING_RATE",
    ]

    for key, value in mat.items():
        if str(key).startswith("__"):
            continue

        key_upper = str(key).upper()
        if not any(token in key_upper for token in fs_tokens):
            continue

        arr = np.asarray(value).squeeze()

        if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
            fs = float(arr.item())
            if fs > 0:
                return fs

    return None


def inspect_mat_file(path: Path) -> dict:
    if loadmat is None:
        return {
            "read_ok": False,
            "kind": "mat",
            "error": SCIPY_IMPORT_ERROR or "scipy.io.loadmat unavailable",
        }

    try:
        mat = loadmat(path.as_posix(), simplify_cells=True)
    except TypeError:
        mat = loadmat(path.as_posix())

    candidates = []

    for key, value in mat.items():
        if str(key).startswith("__"):
            continue

        arr = np.asarray(value)

        if not np.issubdtype(arr.dtype, np.number):
            continue

        if arr.ndim < 2 or arr.size <= 1:
            continue

        candidates.append(
            {
                "key": str(key),
                "shape": list(arr.shape),
                "ndim": int(arr.ndim),
                "size": int(arr.size),
                "dtype": str(arr.dtype),
            }
        )

    candidates = sorted(candidates, key=lambda x: x["size"], reverse=True)
    best = candidates[0] if candidates else None
    fs = find_fs_in_mat(mat)

    if best is None:
        return {
            "read_ok": False,
            "kind": "mat",
            "fs_candidate": fs,
            "error": "No numeric matrix candidate found.",
        }

    shape = best["shape"]

    if len(shape) == 2:
        n_channels_guess = min(shape)
        n_times_guess = max(shape)
    else:
        n_channels_guess = None
        n_times_guess = None

    return {
        "read_ok": True,
        "kind": "mat",
        "signal_key_guess": best["key"],
        "shape": shape,
        "n_channels": n_channels_guess,
        "n_times": n_times_guess,
        "sfreq": fs,
        "duration_sec": (n_times_guess / fs) if fs and n_times_guess else None,
        "numeric_arrays_top_5": candidates[:5],
    }


def inspect_table_file(path: Path) -> dict:
    try:
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path, nrows=8)
        else:
            sep = "\t" if path.suffix.lower() == ".tsv" else None
            df = pd.read_csv(path, sep=sep, engine="python", nrows=8, on_bad_lines="skip")

        return {
            "read_ok": True,
            "kind": "table",
            "columns": list(map(str, df.columns[:40])),
            "preview_shape": list(df.shape),
        }
    except Exception as exc:
        return {
            "read_ok": False,
            "kind": "table",
            "error": repr(exc),
        }


def inspect_file(path: Path) -> dict:
    try:
        if path.suffix.lower() in RAW_EXTENSIONS:
            return inspect_raw_file(path)

        if path.suffix.lower() in MAT_EXTENSIONS:
            return inspect_mat_file(path)

        if path.suffix.lower() in TABLE_EXTENSIONS:
            return inspect_table_file(path)

        return {
            "read_ok": False,
            "error": f"Unsupported extension: {path.suffix}",
        }

    except Exception as exc:
        return {
            "read_ok": False,
            "error": repr(exc),
        }


def choose_deep_read_files(paths: list[Path], root: Path, per_group: int) -> list[Path]:
    grouped = defaultdict(list)

    for path in paths:
        meta = infer_metadata(path, root)
        key = (
            meta["condition"] or "UNKNOWN_CONDITION",
            meta["label_name"] or "UNKNOWN_LABEL",
            meta["suffix"],
        )
        grouped[key].append(path)

    selected = []

    for key in sorted(grouped):
        selected.extend(sorted(grouped[key])[:per_group])

    return selected


def summarize_counts(metas: list[dict]) -> pd.DataFrame:
    rows = []

    for meta in metas:
        if not meta["is_training_candidate"]:
            continue

        rows.append(
            {
                "condition": meta["condition"],
                "label_name": meta["label_name"],
                "subject": meta["subject"],
                "suffix": meta["suffix"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["condition", "label_name"])
        .agg(
            files=("subject", "size"),
            subjects=("subject", "nunique"),
        )
        .reset_index()
        .sort_values(["condition", "label_name"])
    )

    return summary


def summarize_deep_read(deep_rows: list[dict]) -> pd.DataFrame:
    rows = []

    for item in deep_rows:
        meta = item["meta"]
        read = item["read"]

        rows.append(
            {
                "condition": meta["condition"],
                "label_name": meta["label_name"],
                "suffix": meta["suffix"],
                "read_ok": bool(read.get("read_ok")),
                "n_channels": read.get("n_channels"),
                "sfreq": read.get("sfreq"),
                "n_times": read.get("n_times"),
                "duration_sec": read.get("duration_sec"),
                "error": read.get("error"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    ok = df[df["read_ok"]].copy()

    if ok.empty:
        return df

    summary = (
        ok.groupby(["condition", "label_name", "suffix"])
        .agg(
            inspected=("read_ok", "size"),
            channels=("n_channels", lambda x: sorted(set(int(v) for v in x.dropna()))),
            fs=("sfreq", lambda x: sorted(set(float(v) for v in x.dropna()))),
            t_min=("n_times", "min"),
            t_max=("n_times", "max"),
            dur_min=("duration_sec", "min"),
            dur_max=("duration_sec", "max"),
        )
        .reset_index()
        .sort_values(["condition", "label_name", "suffix"])
    )

    return summary


def compact_list(values, max_items: int = 8) -> str:
    values = list(values)

    if len(values) <= max_items:
        return ", ".join(map(str, values))

    return ", ".join(map(str, values[:max_items])) + f", ... +{len(values) - max_items}"


def print_relevant_log(
    root: Path,
    files: list[Path],
    candidates: list[Path],
    metas: list[dict],
    deep_rows: list[dict],
    out_dir: Path,
):
    total_size = sum(p.stat().st_size for p in files)
    training_metas = [m for m in metas if m["is_training_candidate"]]

    ext_counter = Counter(p.suffix.lower() or "<no_ext>" for p in files)
    training_ext_counter = Counter(m["suffix"] for m in training_metas)

    subjects_by_condition_label = defaultdict(set)

    for meta in training_metas:
        subjects_by_condition_label[(meta["condition"], meta["label_name"])].add(meta["subject"])

    print("\nMDD inspection for training")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Total files: {len(files)} | size: {human_bytes(total_size)}")
    print(f"Candidate EEG/table files: {len(candidates)}")
    print(f"Training candidates EC/EO with known label: {len(training_metas)}")
    print("Target classification: H/control = 0 vs MDD = 1")

    print("\nTraining file extensions:")
    if training_ext_counter:
        for ext, count in sorted(training_ext_counter.items()):
            print(f"  {ext:8s} {count}")
    else:
        print("  NONE")

    print("\nEC/EO label balance inferred from paths:")
    count_summary = summarize_counts(metas)

    if count_summary.empty:
        print("  No EC/EO training candidates found.")
    else:
        for _, row in count_summary.iterrows():
            print(
                f"  condition={row['condition']:2s} | "
                f"label={row['label_name']:3s} | "
                f"files={int(row['files']):4d} | "
                f"subjects={int(row['subjects']):4d}"
            )

    print("\nHeader/sample-read summary needed for mdd_db.py:")
    deep_summary = summarize_deep_read(deep_rows)

    if deep_summary.empty:
        print("  No files were read.")
    else:
        if "inspected" in deep_summary.columns:
            for _, row in deep_summary.iterrows():
                print(
                    f"  condition={row['condition']:2s} | "
                    f"label={row['label_name']:3s} | "
                    f"ext={row['suffix']:5s} | "
                    f"n={int(row['inspected']):3d} | "
                    f"C={row['channels']} | "
                    f"fs={row['fs']} | "
                    f"T={int(row['t_min'])}-{int(row['t_max'])} | "
                    f"dur={float(row['dur_min']):.2f}-{float(row['dur_max']):.2f}s"
                )
        else:
            errors = deep_summary[~deep_summary["read_ok"]]
            for _, row in errors.iterrows():
                print(
                    f"  ERR | condition={row['condition']} | "
                    f"label={row['label_name']} | ext={row['suffix']} | "
                    f"{row['error']}"
                )

    print("\nExamples actually read:")
    for item in deep_rows[:20]:
        meta = item["meta"]
        read = item["read"]

        if read.get("read_ok"):
            print(
                f"  OK  | {meta['path']} | "
                f"label={meta['label_name']} condition={meta['condition']} "
                f"C={read.get('n_channels')} fs={read.get('sfreq')} "
                f"T={read.get('n_times')}"
            )
        else:
            print(
                f"  ERR | {meta['path']} | "
                f"label={meta['label_name']} condition={meta['condition']} | "
                f"{read.get('error')}"
            )

    unknown_label = sum(1 for m in metas if m["label"] is None and m["suffix"] in EEG_EXTENSIONS)
    unknown_condition = sum(1 for m in metas if m["condition"] is None and m["suffix"] in EEG_EXTENSIONS)

    print("\nChecks:")
    print(f"  EEG files with unknown label: {unknown_label}")
    print(f"  EEG files with unknown condition: {unknown_condition}")

    if MNE_IMPORT_ERROR:
        print(f"  MNE import error: {MNE_IMPORT_ERROR}")

    if SCIPY_IMPORT_ERROR:
        print(f"  SciPy import error: {SCIPY_IMPORT_ERROR}")

    print("\nSaved for implementation:")
    print(f"  {out_dir / 'mdd_db_inventory.json'}")
    print(f"  {out_dir / 'mdd_db_candidates.csv'}")
    print(f"  {out_dir / 'mdd_db_deep_read.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--read-all", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    files = sorted([p for p in root.rglob("*") if p.is_file()])
    candidates = [
        p for p in files
        if p.suffix.lower() in CANDIDATE_EXTENSIONS
        and not p.name.startswith(".")
    ]

    metas = [infer_metadata(p, root) for p in candidates]
    training_paths = [
        root / meta["path"]
        for meta in metas
        if meta["is_training_candidate"]
    ]

    if args.read_all:
        deep_paths = training_paths
    else:
        deep_paths = choose_deep_read_files(training_paths, root, per_group=args.per_group)

    deep_rows = []

    for path in deep_paths:
        meta = infer_metadata(path, root)
        read = inspect_file(path)
        deep_rows.append(
            {
                "meta": meta,
                "read": read,
            }
        )

    inventory = {
        "root": str(root),
        "total_files": len(files),
        "total_size_bytes": sum(p.stat().st_size for p in files),
        "candidate_files": len(candidates),
        "candidate_meta": metas,
        "deep_read": deep_rows,
    }

    json_path = out_dir / "mdd_db_inventory.json"
    candidates_path = out_dir / "mdd_db_candidates.csv"
    deep_path = out_dir / "mdd_db_deep_read.csv"

    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame(metas).to_csv(candidates_path, index=False)

    flat_deep = []

    for item in deep_rows:
        meta = item["meta"]
        read = item["read"]
        flat_deep.append(
            {
                **meta,
                "read_ok": read.get("read_ok"),
                "kind": read.get("kind"),
                "reader": read.get("reader"),
                "n_channels": read.get("n_channels"),
                "sfreq": read.get("sfreq"),
                "n_times": read.get("n_times"),
                "duration_sec": read.get("duration_sec"),
                "shape": read.get("shape"),
                "signal_key_guess": read.get("signal_key_guess"),
                "ch_names_first_12": json.dumps(read.get("ch_names_first_12")),
                "error": read.get("error"),
            }
        )

    pd.DataFrame(flat_deep).to_csv(deep_path, index=False)

    print_relevant_log(
        root=root,
        files=files,
        candidates=candidates,
        metas=metas,
        deep_rows=deep_rows,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()