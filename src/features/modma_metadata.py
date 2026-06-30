"""MODMA metadata module.
Reads BIDS participants.tsv, maps subject IDs to groups (MDD/HC),
extracts psychometric variables (PHQ-9, GAD-7, PSQI, age, gender, education).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def load_participants(participants_path):
    """Load participants.tsv, handling the file with 9-10 columns (extra empty field).
    Returns DataFrame with columns: participant_id, gender, age, education, PHQ-9, group, GAD-7, PSQI
    """
    p = pd.read_csv(participants_path, sep='\t', header=None, skiprows=1,
                     on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 4, 5, 6, 7, 8]]
    p.columns = ['participant_id', 'gender', 'age', 'education', 'PHQ-9', 'group', 'GAD-7', 'PSQI']
    return p


def get_subject_groups(participants):
    """Returns dict: subject_id -> 'MDD' or 'HC'."""
    return dict(zip(
        [str(s) for s in participants['participant_id'].tolist()],
        participants['group']
    ))


def get_psychometric_features(participants, subject_ids):
    """Extract psychometric features for a list of subject IDs.
    Returns numpy array [n_subjects, 6] with columns: gender, age, education, PHQ-9, GAD-7, PSQI."""
    rows = []
    for sid in subject_ids:
        row = participants[participants['participant_id'] == sid]
        if len(row) == 0:
            rows.append([0.0] * 6)
            continue
        r = row.iloc[0]
        rows.append([
            0.0 if pd.isna(r['gender']) else (1.0 if str(r['gender']).upper().startswith('M') else 0.0),
            0.0 if pd.isna(r['age']) else float(r['age']),
            0.0 if pd.isna(r['education']) else float(r['education']),
            0.0 if pd.isna(r['PHQ-9']) else float(r['PHQ-9']),
            0.0 if pd.isna(r['GAD-7']) else float(r['GAD-7']),
            0.0 if pd.isna(r['PSQI']) else float(r['PSQI']),
        ])
    return np.array(rows, dtype=np.float32)
