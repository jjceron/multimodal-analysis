# scripts/inspect_hbn_resting_global.py

from pathlib import Path
import pandas as pd
import mne

ROOT = Path("data/raw/hbn_db/R1_L100_bdf")

OPEN = "instructed_toOpenEyes"
CLOSE = "instructed_toCloseEyes"
REST_START = "resting_start"
BREAK = "break cnt"

def subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return path.stem.split("_")[0]

def build_intervals(events: pd.DataFrame):
    events = events.copy()
    events = events.sort_values("onset").reset_index(drop=True)

    rest_rows = events[events["value"] == REST_START]
    if len(rest_rows) == 0:
        return [], [], None, None, "missing_resting_start"

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

    if len(condition_events) == 0:
        return [], [], rest_start, rest_end, "missing_open_close"

    condition_events = condition_events[
        condition_events["onset"] < rest_end
    ].reset_index(drop=True)

    eo_intervals = []
    ec_intervals = []

    # Initial segment: resting_start -> first explicit instruction.
    first_event = condition_events.iloc[0]
    first_onset = float(first_event["onset"])

    # If first instruction is "open eyes", then previous state was likely EC.
    if first_event["value"] == OPEN and first_onset > rest_start:
        ec_intervals.append((rest_start, first_onset))

    # Alternating explicit intervals.
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

    return eo_intervals, ec_intervals, rest_start, rest_end, "ok"

def duration(intervals):
    return sum(max(0.0, b - a) for a, b in intervals)

rows = []

bdfs = sorted(ROOT.rglob("*task-RestingState*_eeg.bdf"))
print(f"RestingState BDF files found: {len(bdfs)}")

for bdf_path in bdfs:
    subject_id = subject_from_path(bdf_path)

    events_path = bdf_path.with_name(
        bdf_path.name.replace("_eeg.bdf", "_events.tsv")
    )

    row = {
        "subject_id": subject_id,
        "bdf_path": str(bdf_path),
        "events_path": str(events_path),
        "has_events": events_path.exists(),
        "sfreq": None,
        "n_channels": None,
        "last_channel": None,
        "rest_start": None,
        "rest_end": None,
        "n_eo_blocks": 0,
        "n_ec_blocks": 0,
        "eo_sec": 0.0,
        "ec_sec": 0.0,
        "status": None,
    }

    try:
        raw = mne.io.read_raw_bdf(
            bdf_path,
            preload=False,
            verbose=False,
        )

        row["sfreq"] = float(raw.info["sfreq"])
        row["n_channels"] = len(raw.ch_names)
        row["last_channel"] = raw.ch_names[-1]

    except Exception as e:
        row["status"] = f"raw_error: {e}"
        rows.append(row)
        continue

    if not events_path.exists():
        row["status"] = "missing_events"
        rows.append(row)
        continue

    try:
        events = pd.read_csv(events_path, sep="\t")
        eo, ec, rest_start, rest_end, status = build_intervals(events)

        row["rest_start"] = rest_start
        row["rest_end"] = rest_end
        row["n_eo_blocks"] = len(eo)
        row["n_ec_blocks"] = len(ec)
        row["eo_sec"] = duration(eo)
        row["ec_sec"] = duration(ec)
        row["status"] = status

    except Exception as e:
        row["status"] = f"events_error: {e}"

    rows.append(row)

summary = pd.DataFrame(rows)

out_path = ROOT / "restingstate_inspection_summary.csv"
summary.to_csv(out_path, index=False)

print("\nSaved:")
print(out_path)

print("\nStatus counts:")
print(summary["status"].value_counts(dropna=False))

print("\nChannels:")
print(summary["n_channels"].value_counts(dropna=False).sort_index())

print("\nSampling frequency:")
print(summary["sfreq"].value_counts(dropna=False).sort_index())

print("\nEO duration seconds:")
print(summary["eo_sec"].describe())

print("\nEC duration seconds:")
print(summary["ec_sec"].describe())

print("\nSubjects with suspicious EO/EC duration:")
bad = summary[
    (summary["status"] != "ok")
    | (summary["eo_sec"] < 30)
    | (summary["ec_sec"] < 30)
]
print(bad[[
    "subject_id",
    "status",
    "n_channels",
    "sfreq",
    "n_eo_blocks",
    "n_ec_blocks",
    "eo_sec",
    "ec_sec",
    "last_channel",
]].to_string(index=False))