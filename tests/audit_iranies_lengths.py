from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

from src.datasets.adhd_dataset import EEGDataset_ADHD


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_control_dir(project_root: Path, control_dir_arg: str | None) -> Path:
    if control_dir_arg is not None:
        return Path(control_dir_arg)

    candidates = [
        project_root / "data" / "iraniesdataset" / "control",
        project_root / "data" / "iraniesdataset" / "Control",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audita la longitud temporal T_s de cada sujeto del dataset iraní "
            "antes del recorte común."
        )
    )

    parser.add_argument(
        "--adhd-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "iraniesdataset" / "ADHD"),
    )
    parser.add_argument(
        "--control-dir",
        type=str,
        default=None,
    )

    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=60.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--target-fs", type=float, default=128.0)
    parser.add_argument("--default-fs", type=float, default=128.0)
    parser.add_argument("--crop-from", type=str, default="start", choices=["start", "center"])
    parser.add_argument("--max-channels", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)

    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-notch", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-resample", action="store_true")

    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "iraniesdataset" / "lengths_by_subject.csv"),
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Imprime los 121 sujetos ordenados por T_s. Por defecto imprime los 20 más cortos.",
    )

    args = parser.parse_args()

    project_root = PROJECT_ROOT
    control_dir = resolve_control_dir(project_root, args.control_dir)

    # duration_sec=None es intencional: queremos que el dataset calcule T_min
    # a partir de los T_s reales después del preprocesamiento y antes del recorte.
    dataset = EEGDataset_ADHD(
        adhd_dir=args.adhd_dir,
        control_dir=control_dir,
        lowcut=None if args.no_filter else args.lowcut,
        highcut=None if args.no_filter else args.highcut,
        notch=None if args.no_notch else args.notch,
        target_fs=args.target_fs,
        default_fs=args.default_fs,
        duration_sec=None,
        crop_from=args.crop_from,
        max_channels=args.max_channels,
        scale=args.scale,
        apply_reference=not args.no_reference,
        apply_resample=not args.no_resample,
    )

    summary = dataset.get_summary_dataframe().copy()

    required_cols = [
        "subject_id",
        "label_name",
        "signal_key",
        "fs_key",
        "fs_was_assumed",
        "original_fs",
        "final_fs",
        "did_resample",
        "n_channels",
        "n_samples_before",
        "n_samples_after_resample",
        "duration_sec_after_resample",
        "is_min_duration_subject",
        "crop_n_samples",
        "selected_duration_sec",
        "selected_shape",
        "was_transposed",
    ]

    cols = [c for c in required_cols if c in summary.columns]
    summary_sorted = summary.sort_values(
        ["n_samples_after_resample", "subject_id"],
        ascending=[True, True],
    )[cols]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_sorted.to_csv(out_csv, index=False)

    lengths = summary["n_samples_after_resample"].astype(int)
    durations = summary["duration_sec_after_resample"].astype(float)

    t_min = int(lengths.min())
    t_max = int(lengths.max())
    t_median = float(lengths.median())
    t_mean = float(lengths.mean())
    selected_t = int(dataset.crop_n_samples)
    selected_duration_sec = float(dataset.selected_duration_sec)

    shortest = summary.loc[
        summary["n_samples_after_resample"].astype(int) == t_min,
        "subject_id",
    ].astype(str).tolist()

    print("\nIranies dataset length audit")
    print(f"  subjects:                  {len(dataset)}")
    print(f"  labels:                    {dict(Counter(dataset.get_labels()))}")
    print(f"  channels:                  {sorted(summary['n_channels'].unique().tolist())}")
    print(f"  original_fs values:         {sorted(summary['original_fs'].unique().tolist())}")
    print(f"  final_fs values:            {sorted(summary['final_fs'].unique().tolist())}")
    print(f"  fs_was_assumed counts:      {summary['fs_was_assumed'].value_counts(dropna=False).to_dict()}")
    print(f"  did_resample counts:        {summary['did_resample'].value_counts(dropna=False).to_dict()}")
    print("\nT_s before common crop")
    print(f"  min T_s:                   {t_min} samples")
    print(f"  min duration:              {t_min / args.target_fs:.6f} s at {args.target_fs:g} Hz")
    print(f"  shortest subject(s):        {shortest}")
    print(f"  median T_s:                {t_median:.1f} samples")
    print(f"  mean T_s:                  {t_mean:.1f} samples")
    print(f"  max T_s:                   {t_max} samples")
    print(f"  max duration:              {t_max / args.target_fs:.6f} s at {args.target_fs:g} Hz")
    print("\nCommon crop selected by EEGDataset_ADHD")
    print(f"  crop_n_samples:            {selected_t} samples")
    print(f"  selected_duration_sec:     {selected_duration_sec:.6f} s")
    print(f"  expected selected T == min: {selected_t == t_min}")
    print(f"\nSaved full sorted table to: {out_csv}")

    n_rows = len(summary_sorted) if args.print_all else min(20, len(summary_sorted))
    print(f"\nShortest {n_rows} subjects by T_s")
    print(summary_sorted.head(n_rows).to_string(index=False))

    if args.print_all:
        print("\nAll subjects by T_s")
        print(summary_sorted.to_string(index=False))

    if selected_t != t_min:
        raise RuntimeError(
            f"Inconsistencia: crop_n_samples={selected_t}, pero min(T_s)={t_min}."
        )


if __name__ == "__main__":
    main()
