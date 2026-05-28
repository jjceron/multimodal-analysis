# py -m scripts.inspect_modma_db

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import pandas as pd


_INVALID_JSON_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')


@dataclass
class SubjectEEGReport:
    subject: str
    subject_dir: str
    eeg_dir_exists: bool
    edf_file: str | None
    channels_file: str | None
    electrodes_file: str | None
    json_file: str | None
    edf_size_mb: float | None
    edf_read_status: str
    n_channels_edf: int | None
    sfreq: float | None
    n_times: int | None
    duration_sec: float | None
    duration_min: float | None
    channels_tsv_rows: int | None
    electrodes_tsv_rows: int | None
    channels_tsv_malformed_rows: int | None
    electrodes_tsv_malformed_rows: int | None
    channel_types: dict[str, int] | None
    channel_units: dict[str, int] | None
    channel_status: dict[str, int] | None
    json_keys: list[str] | None
    json_parse_status: str | None
    manufacturer: str | None
    power_line_frequency: Any | None
    recording_type: str | None
    status: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_bids_root() -> Path:
    return project_root() / "data" / "raw" / "modma" / "MODMA_EEG_BIDS_format"


def default_output_dir() -> Path:
    return project_root() / "outputs" / "modma_inspection"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)

    return value


def read_json_lenient(path: Path) -> tuple[dict[str, Any], str | None]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")

    try:
        return json.loads(text), "ok"
    except json.JSONDecodeError as original_error:
        repaired_text = _INVALID_JSON_BACKSLASH.sub(r"\\\\", text)

        try:
            return json.loads(repaired_text), f"repaired_invalid_escape: {original_error}"
        except json.JSONDecodeError as repaired_error:
            return {}, f"json_decode_error: {repaired_error}"


def file_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return round(path.stat().st_size / (1024**2), 3)


def find_first(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def safe_read_tsv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        lines = [line.rstrip("\n\r") for line in f]

    if not lines:
        df = pd.DataFrame()
        df.attrs["malformed_rows"] = 0
        df.attrs["source_file"] = str(path)
        return df

    columns = lines[0].split("\t")
    rows = []
    malformed_rows = 0

    for line in lines[1:]:
        if not line.strip():
            continue

        values = line.split("\t")

        if len(values) != len(columns):
            malformed_rows += 1

        if len(values) < len(columns):
            values = values + [None] * (len(columns) - len(values))

        if len(values) > len(columns):
            values = values[: len(columns)]

        rows.append(values)

    df = pd.DataFrame(rows, columns=columns)
    df.attrs["malformed_rows"] = malformed_rows
    df.attrs["source_file"] = str(path)

    return df


def summarize_value_counts(df: pd.DataFrame | None, column: str) -> dict[str, int] | None:
    if df is None or column not in df.columns:
        return None

    counts = df[column].fillna("NA").astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def inspect_edf(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "edf_read_status": "missing",
            "n_channels_edf": None,
            "sfreq": None,
            "n_times": None,
            "duration_sec": None,
            "duration_min": None,
        }

    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        duration_sec = n_times / sfreq

        return {
            "edf_read_status": "ok",
            "n_channels_edf": int(len(raw.ch_names)),
            "sfreq": sfreq,
            "n_times": n_times,
            "duration_sec": round(duration_sec, 3),
            "duration_min": round(duration_sec / 60, 3),
        }
    except Exception as error:
        return {
            "edf_read_status": f"error: {error}",
            "n_channels_edf": None,
            "sfreq": None,
            "n_times": None,
            "duration_sec": None,
            "duration_min": None,
        }


def inspect_subject(subject_dir: Path) -> SubjectEEGReport:
    subject = subject_dir.name
    eeg_dir = subject_dir / "eeg"

    edf_file = find_first(eeg_dir, f"{subject}_task-Resting-state_eeg.EDF")
    if edf_file is None:
        edf_file = find_first(eeg_dir, "*_eeg.EDF")
    if edf_file is None:
        edf_file = find_first(eeg_dir, "*_eeg.edf")

    channels_file = find_first(eeg_dir, f"{subject}_task-Resting-state_channels.tsv")
    if channels_file is None:
        channels_file = find_first(eeg_dir, "*_channels.tsv")

    electrodes_file = find_first(eeg_dir, f"{subject}_task-Resting-state_electrodes.tsv")
    if electrodes_file is None:
        electrodes_file = find_first(eeg_dir, "*_electrodes.tsv")

    json_file = find_first(eeg_dir, f"{subject}_task-Resting-state_eeg.json")
    if json_file is None:
        json_file = find_first(eeg_dir, "*_eeg.json")

    channels_df = safe_read_tsv(channels_file)
    electrodes_df = safe_read_tsv(electrodes_file)
    edf_info = inspect_edf(edf_file)

    json_data, json_parse_status = (
        read_json_lenient(json_file)
        if json_file is not None and json_file.exists()
        else ({}, None)
    )

    required_files = [edf_file, channels_file, electrodes_file, json_file]
    status = (
        "ok"
        if eeg_dir.exists()
        and all(p is not None and p.exists() for p in required_files)
        and edf_info["edf_read_status"] == "ok"
        else "incomplete"
    )

    return SubjectEEGReport(
        subject=subject,
        subject_dir=str(subject_dir),
        eeg_dir_exists=eeg_dir.exists(),
        edf_file=str(edf_file) if edf_file else None,
        channels_file=str(channels_file) if channels_file else None,
        electrodes_file=str(electrodes_file) if electrodes_file else None,
        json_file=str(json_file) if json_file else None,
        edf_size_mb=file_size_mb(edf_file),
        edf_read_status=edf_info["edf_read_status"],
        n_channels_edf=edf_info["n_channels_edf"],
        sfreq=edf_info["sfreq"],
        n_times=edf_info["n_times"],
        duration_sec=edf_info["duration_sec"],
        duration_min=edf_info["duration_min"],
        channels_tsv_rows=len(channels_df) if channels_df is not None else None,
        electrodes_tsv_rows=len(electrodes_df) if electrodes_df is not None else None,
        channels_tsv_malformed_rows=channels_df.attrs.get("malformed_rows") if channels_df is not None else None,
        electrodes_tsv_malformed_rows=electrodes_df.attrs.get("malformed_rows") if electrodes_df is not None else None,
        channel_types=summarize_value_counts(channels_df, "type"),
        channel_units=summarize_value_counts(channels_df, "units"),
        channel_status=summarize_value_counts(channels_df, "status"),
        json_keys=sorted(json_data.keys()) if json_data else None,
        json_parse_status=json_parse_status,
        manufacturer=json_data.get("Manufacturer"),
        power_line_frequency=json_data.get("PowerLineFrequency"),
        recording_type=json_data.get("RecordingType"),
        status=status,
    )


def inspect_sourcedata(resting_dir: Path) -> dict[str, Any]:
    sourcedata = resting_dir / "sourcedata"

    mat_files = sorted(sourcedata.glob("*.mat")) if sourcedata.exists() else []
    raw_files = sorted(sourcedata.glob("*.raw")) if sourcedata.exists() else []
    all_files = mat_files + raw_files

    return {
        "exists": sourcedata.exists(),
        "path": str(sourcedata),
        "mat_files": len(mat_files),
        "raw_files": len(raw_files),
        "total_size_mb": round(sum(p.stat().st_size for p in all_files) / (1024**2), 3),
        "first_files": [str(p) for p in all_files[:10]],
    }


def inspect_scales_file(path: Path, output_dir: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    xls = pd.ExcelFile(path)
    sheets = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        safe_sheet_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sheet))
        csv_path = output_dir / f"participants_scales__{safe_sheet_name}.csv"

        df.to_csv(csv_path, index=False)

        missing_values = {str(k): int(v) for k, v in df.isna().sum().items()}

        sheets[str(sheet)] = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": [str(col) for col in df.columns],
            "missing_values": missing_values,
            "exported_csv": str(csv_path),
        }

    return {
        "exists": True,
        "path": str(path),
        "sheet_names": [str(sheet) for sheet in xls.sheet_names],
        "sheets": sheets,
    }


def compact_subject_table(reports: list[SubjectEEGReport]) -> pd.DataFrame:
    rows = []

    for report in reports:
        row = asdict(report)
        row["channel_types"] = json.dumps(row["channel_types"], ensure_ascii=False)
        row["channel_units"] = json.dumps(row["channel_units"], ensure_ascii=False)
        row["channel_status"] = json.dumps(row["channel_status"], ensure_ascii=False)
        row["json_keys"] = json.dumps(row["json_keys"], ensure_ascii=False)
        rows.append(row)

    return pd.DataFrame(rows)


def describe_numeric(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    available = [col for col in columns if col in df.columns]

    if not available or df.empty:
        return {}

    summary = df[available].describe().to_dict()
    return to_jsonable(summary)


def collect_task_files(resting_dir: Path) -> dict[str, Any]:
    edf_files = sorted(resting_dir.rglob("*_eeg.EDF")) + sorted(resting_dir.rglob("*_eeg.edf"))
    json_files = sorted(resting_dir.rglob("*_eeg.json"))
    channels_files = sorted(resting_dir.rglob("*_channels.tsv"))
    electrodes_files = sorted(resting_dir.rglob("*_electrodes.tsv"))

    return {
        "edf_files": len(edf_files),
        "json_files": len(json_files),
        "channels_files": len(channels_files),
        "electrodes_files": len(electrodes_files),
        "first_edf_files": [str(p) for p in edf_files[:10]],
    }


def build_report(bids_root: Path, output_dir: Path) -> dict[str, Any]:
    resting_dir = bids_root / "EEG_LZU_2015_2_resting state"
    dot_probe_dir = bids_root / "EEG_LZU_2015_2_dot probe"
    scales_file = bids_root / "Lanzhou University Second Hospital MODMA participants scales.xlsx"

    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(p for p in resting_dir.glob("sub-*") if p.is_dir()) if resting_dir.exists() else []
    subject_reports = [inspect_subject(subject_dir) for subject_dir in subject_dirs]

    subjects_df = compact_subject_table(subject_reports)
    subjects_csv = output_dir / "modma_resting_subjects_report.csv"
    subjects_df.to_csv(subjects_csv, index=False)

    edf_summary = describe_numeric(
        subjects_df,
        [
            "edf_size_mb",
            "n_channels_edf",
            "sfreq",
            "n_times",
            "duration_sec",
            "duration_min",
            "channels_tsv_rows",
            "electrodes_tsv_rows",
            "channels_tsv_malformed_rows",
            "electrodes_tsv_malformed_rows",
        ],
    )

    report = {
        "paths": {
            "project_root": str(project_root()),
            "bids_root": str(bids_root),
            "resting_dir": str(resting_dir),
            "dot_probe_dir": str(dot_probe_dir),
            "scales_file": str(scales_file),
            "output_dir": str(output_dir),
        },
        "composition": {
            "bids_root_exists": bids_root.exists(),
            "resting_dir_exists": resting_dir.exists(),
            "dot_probe_dir_exists": dot_probe_dir.exists(),
            "scales_file_exists": scales_file.exists(),
            "n_subjects": len(subject_reports),
            "n_complete_subjects": sum(r.status == "ok" for r in subject_reports),
            "n_incomplete_subjects": sum(r.status != "ok" for r in subject_reports),
            "subjects": [r.subject for r in subject_reports],
        },
        "task_files": collect_task_files(resting_dir),
        "sourcedata": inspect_sourcedata(resting_dir),
        "edf_summary": edf_summary,
        "scales": inspect_scales_file(scales_file, output_dir),
        "outputs": {
            "subjects_csv": str(subjects_csv),
            "inspection_json": str(output_dir / "modma_bids_inspection_report.json"),
        },
    }

    report_json = output_dir / "modma_bids_inspection_report.json"

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)

    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\nMODMA BIDS inspection")
    print(f"BIDS root: {report['paths']['bids_root']}")
    print(f"Resting-state directory: {report['paths']['resting_dir']}")
    print(f"Dot-probe directory: {report['paths']['dot_probe_dir']}")
    print(f"Scales file: {report['paths']['scales_file']}")

    print("\nDataset composition")
    print(f"Resting-state directory exists: {report['composition']['resting_dir_exists']}")
    print(f"Dot-probe directory exists: {report['composition']['dot_probe_dir_exists']}")
    print(f"Scales file exists: {report['composition']['scales_file_exists']}")
    print(f"Subject folders: {report['composition']['n_subjects']}")
    print(f"Complete subjects: {report['composition']['n_complete_subjects']}")
    print(f"Incomplete subjects: {report['composition']['n_incomplete_subjects']}")

    print("\nResting-state task files")
    print(f"EDF files: {report['task_files']['edf_files']}")
    print(f"JSON sidecars: {report['task_files']['json_files']}")
    print(f"Channels TSV files: {report['task_files']['channels_files']}")
    print(f"Electrodes TSV files: {report['task_files']['electrodes_files']}")

    print("\nSourcedata")
    print(f"Exists: {report['sourcedata']['exists']}")
    print(f".mat files: {report['sourcedata']['mat_files']}")
    print(f".raw files: {report['sourcedata']['raw_files']}")
    print(f"Total size MB: {report['sourcedata']['total_size_mb']}")

    print("\nEDF summary")
    for key, value in report["edf_summary"].items():
        print(f"{key}: {value}")

    print("\nScales")
    print(f"Exists: {report['scales']['exists']}")
    if report["scales"]["exists"]:
        print(f"Sheets: {report['scales']['sheet_names']}")

    print("\nOutputs")
    for key, value in report["outputs"].items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, default=default_bids_root())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.bids_root.exists():
        print(f"BIDS root does not exist: {args.bids_root}", file=sys.stderr)
        return 1

    report = build_report(
        bids_root=args.bids_root,
        output_dir=args.output_dir,
    )

    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())