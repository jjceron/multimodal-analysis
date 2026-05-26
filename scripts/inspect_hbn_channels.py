# scripts/inspect_hbn_channels.py

from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/hbn_db/R1_L100_bdf")

channels_files = sorted(ROOT.rglob("*task-RestingState_channels.tsv"))

print(f"RestingState channels.tsv files found: {len(channels_files)}")

if len(channels_files) == 0:
    print("No RestingState channels.tsv files found.")
    raise SystemExit

path = channels_files[0]
print(f"\nExample:\n{path}")

channels = pd.read_csv(path, sep="\t")

print("\nColumns:")
print(channels.columns.tolist())

print("\nHead:")
print(channels.head(20).to_string(index=False))

print("\nTail:")
print(channels.tail(20).to_string(index=False))

for col in channels.columns:
    print(f"\nValue counts: {col}")
    print(channels[col].value_counts(dropna=False).head(30))

print("\nAcross all subjects:")

rows = []

for path in channels_files:
    subject_id = next(part for part in path.parts if part.startswith("sub-"))
    ch = pd.read_csv(path, sep="\t")

    row = {
        "subject_id": subject_id,
        "n_channels": len(ch),
        "last_name": ch["name"].iloc[-1] if "name" in ch.columns else None,
        "last_type": ch["type"].iloc[-1] if "type" in ch.columns else None,
        "last_status": ch["status"].iloc[-1] if "status" in ch.columns else None,
    }

    rows.append(row)

summary = pd.DataFrame(rows)

print("\nNumber of channels:")
print(summary["n_channels"].value_counts(dropna=False))

print("\nLast channel name:")
print(summary["last_name"].value_counts(dropna=False).head(20))

print("\nLast channel type:")
print(summary["last_type"].value_counts(dropna=False).head(20))

print("\nLast channel status:")
print(summary["last_status"].value_counts(dropna=False).head(20))