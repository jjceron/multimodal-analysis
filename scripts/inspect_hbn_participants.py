# scripts/inspect_hbn_participants.py

from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path("data/raw/hbn_db/R1_L100_bdf")

participants_path = ROOT / "participants.tsv"
participants_json_path = ROOT / "participants.json"
resting_summary_path = ROOT / "restingstate_inspection_summary.csv"

if not participants_path.exists():
    raise FileNotFoundError(participants_path)

df = pd.read_csv(participants_path, sep="\t")

print("\nparticipants.tsv")
print(f"shape: {df.shape}")

print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"{i:03d}: {col}")

patterns = [
    "participant",
    "subject",
    "external",
    "internal",
    "attention",
    "factor",
    "p_factor",
    "age",
    "sex",
    "gender",
    "hand",
]

regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
candidate_cols = [c for c in df.columns if regex.search(c)]

print("\nCandidate columns:")
for col in candidate_cols:
    nonnull = df[col].notna().sum()
    print(f"{col:40s} non-null={nonnull:4d} dtype={df[col].dtype}")

print("\nHead of candidate columns:")
print(df[candidate_cols].head(20).to_string(index=False))

print("\nNumeric summaries for candidate columns:")
for col in candidate_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"\n{col}")
        print(df[col].describe())

if participants_json_path.exists():
    print("\nparticipants.json matching candidate keys:")
    with open(participants_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for key, value in meta.items():
        if regex.search(key):
            print("\n" + "-" * 80)
            print(key)
            print(value)

if resting_summary_path.exists():
    resting = pd.read_csv(resting_summary_path)

    print("\nRestingState overlap")
    print(f"Resting subjects: {len(resting)}")

    possible_id_cols = [
        c for c in df.columns
        if c.lower() in {"participant_id", "subject_id", "participant", "subject"}
        or "participant" in c.lower()
        or "subject" in c.lower()
    ]

    print(f"Possible ID columns: {possible_id_cols}")

    for id_col in possible_id_cols:
        left = set(resting["subject_id"].astype(str))
        right = set(df[id_col].astype(str))

        overlap = left & right

        # Also try without "sub-" prefix.
        right_sub = {
            x if x.startswith("sub-") else f"sub-{x}"
            for x in df[id_col].astype(str)
        }

        overlap_sub = left & right_sub

        print(
            f"{id_col:30s} | "
            f"direct overlap={len(overlap):3d} | "
            f"with sub- overlap={len(overlap_sub):3d}"
        )