"""
MODMA metadata extraction: Build label vector (MDD/HC) and psychometric features.
Reads from participants scales xlsx and merges with EEG subjects.
"""
import os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

META_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/Lanzhou University Second Hospital MODMA participants scales.xlsx'
OUT_PATH = 'data/processed/modma_meta_features.npz'

def main():
    df = pd.read_excel(META_PATH)
    print("Raw metadata shape:", df.shape)
    print("Raw columns:", [str(c) for c in df.columns[:5]])

    # The metadata has 2 columns describing group compositions
    # We need to extract subject-level information
    # Try reading with header=None to get all rows
    df_raw = pd.read_excel(META_PATH, header=None)
    print("\nRaw values (first 10 rows):")
    for i in range(min(10, len(df_raw))):
        print(f"  Row {i}: {[str(v) for v in df_raw.iloc[i].values[:5]]}")

    # Look for actual subject data - it might be in a separate file
    # Check if there's a participants.tsv in BIDS
    bids_dir = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
    for sub in sorted(os.listdir(bids_dir))[:3]:
        sub_path = os.path.join(bids_dir, sub)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path + '/eeg'):
                if f.endswith('.tsv') or f.endswith('.json'):
                    print(f"  {sub}: {f}")

    # The actual subject labels are in the sourcedata
    sourcedata_dir = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/sub-001/sourcedata'
    if os.path.exists(sourcedata_dir):
        for f in os.listdir(sourcedata_dir):
            print(f"  sourcedata: {f}")

    # Build labels: from the xlsx we know 24 Depressive, 29 Normal
    # But we need the actual subject-to-group mapping
    # Try to find a participants.tsv in BIDS root
    root_tsv = None
    for root, dirs, files in os.walk('data/raw/modma/MODMA_EEG_BIDS_format'):
        for f in files:
            if f == 'participants.tsv':
                root_tsv = os.path.join(root, f)
                break
        if root_tsv: break

    if root_tsv:
        print(f"\nFound participants.tsv: {root_tsv}")
        participants = pd.read_csv(root_tsv, sep='\t')
        print(participants.head())
        subjects = participants['participant_id'].tolist()
        # Check if there's a group column
        for col in participants.columns:
            if 'group' in col.lower() or 'diagnosis' in col.lower() or 'patient' in col.lower():
                groups = participants[col].tolist()
                print(f"Groups from {col}: {set(groups)}")
                break
    else:
        # Fallback: hardcode labels based on subject order
        # MODMA has 53 subjects: 24 Depressive (first 24), 29 Normal (last 29)
        # But we don't know the exact order
        # Let's try to find the order from sourcedata or xlsx details
        subjects = [f"sub-{i:03d}" for i in range(1, 54)]
        # We need to find which subjects are Depressive
        # The xlsx says: 24 Depressive, 29 Normal for 53 subjects
        # Without per-subject info, we can't know
        print("\nNo participants.tsv found. Cannot map subjects to groups.")
        print("Using placeholder: all subjects as HC")
        groups = [0] * 53

    # Save what we have
    if 'groups' in dir():
        np.savez(OUT_PATH,
                 subjects=np.array(subjects),
                 groups=np.array(groups))
        print(f"\nSaved {len(subjects)} subjects with labels to: {OUT_PATH}")

if __name__ == '__main__':
    main()
