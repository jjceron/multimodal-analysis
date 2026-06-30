"""MODMA multimodal feature matrix builder v2.
Combine EEG, audio, and psychometric features. Correct audio mapping via Excel.
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

EEG_V3_PATH = 'data/processed/modma_eeg_features_v3.npz'
AUDIO_PATH = 'data/processed/modma_audio_features.npz'
AUDIO_XLSX = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
OUT_PATH = 'data/processed/modma_multimodal_features.npz'

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def load_participants():
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                      on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 5, 6, 7, 8, 9]]
    p.columns = ['participant_id', 'gender', 'age', 'education', 'group', 'PHQ9', 'GAD7', 'PSQI']
    return p


def main():
    print("="*70)
    print("  MODMA MULTIMODAL FEATURE MATRIX BUILDER v2 (correct mapping)")
    print("="*70)

    participants = load_participants()
    sub_to_group = dict(zip(
        [str(s) for s in participants['participant_id'].tolist()],
        participants['group']
    ))

    eeg = np.load(EEG_V3_PATH, allow_pickle=True)
    eeg_subs = list(eeg['subjects'])
    eeg_X = eeg['X'].astype(np.float32)
    eeg_feats = list(eeg['feature_names'])

    audio = np.load(AUDIO_PATH, allow_pickle=True)
    audio_dirs = list(audio['subjects'])
    audio_X = audio['X'].astype(np.float32)
    audio_feats = [f'audio_{f}' for f in list(audio['feature_names'])]

    # Load Excel for audio-to-participant mapping via PHQ9 matching
    df_xlsx = pd.read_excel(AUDIO_XLSX)

    # Map audio directory to participant by matching PHQ-9, GAD-7, PSQI
    audio_to_part = {}
    for i, row in df_xlsx.iterrows():
        matches = participants[
            (participants['PHQ9'] == row['PHQ-9']) &
            (participants['GAD7'] == row['GAD-7']) &
            (participants['PSQI'] == row['PSQI'])
        ]
        if len(matches) == 1:
            audio_to_part[audio_dirs[i]] = str(matches['participant_id'].values[0])

    print(f"Audio-to-EEG mapping: {len(audio_to_part)}/{len(audio_dirs)} subjects")

    common = []
    for e_sub in eeg_subs:
        a_sub = None
        for adir, part_id in audio_to_part.items():
            if part_id == e_sub:
                a_sub = adir
                break
        if a_sub is None:
            continue
        g = sub_to_group.get(e_sub)
        if g in ('MDD', 'HC'):
            common.append((e_sub, a_sub, 1 if g == 'MDD' else 0))

    print(f"Subjects with EEG + audio + labels: {len(common)}")
    if not common:
        return

    common = sorted(common, key=lambda x: x[0])
    eeg_ids = [c[0] for c in common]
    audio_ids = [c[1] for c in common]
    y = np.array([c[2] for c in common])
    print(f"  MDD: {np.sum(y==1)}, HC: {np.sum(y==0)}")

    eeg_dict = dict(zip(eeg_subs, eeg_X))
    audio_dict = dict(zip(audio_dirs, audio_X))

    eeg_rows = np.array([eeg_dict[s] for s in eeg_ids])
    audio_rows = np.array([audio_dict[s] for s in audio_ids])

    eeg_imp = SimpleImputer(strategy='constant', fill_value=0.0).fit_transform(eeg_rows)
    audio_imp = SimpleImputer(strategy='constant', fill_value=0.0).fit_transform(audio_rows)

    eeg_sc = StandardScaler().fit_transform(eeg_imp)
    audio_sc = StandardScaler().fit_transform(audio_imp)

    X = np.hstack([eeg_sc, audio_sc])
    feat_names = [f'eeg_{f}' for f in eeg_feats] + audio_feats

    psych_features = ['gender', 'age', 'education', 'PHQ9', 'GAD7', 'PSQI']
    psych_rows = []
    for e_sub, a_sub, _ in common:
        row = participants[participants['participant_id'] == e_sub]
        if len(row) == 0:
            psych_rows.append([0.0] * len(psych_features))
            continue
        r = row.iloc[0]
        psych_rows.append([
            0.0 if pd.isna(r['gender']) else (1.0 if str(r['gender']).upper().startswith('M') else 0.0),
            0.0 if pd.isna(r['age']) else float(r['age']),
            0.0 if pd.isna(r['education']) else float(r['education']),
            0.0 if pd.isna(r['PHQ9']) else float(r['PHQ9']),
            0.0 if pd.isna(r['GAD7']) else float(r['GAD7']),
            0.0 if pd.isna(r['PSQI']) else float(r['PSQI']),
        ])
    psych_arr = np.array(psych_rows, dtype=np.float32)
    psych_imp = SimpleImputer(strategy='constant', fill_value=0.0).fit_transform(psych_arr)
    psych_sc = StandardScaler().fit_transform(psych_imp)

    X = np.hstack([X, psych_sc])
    feat_names += [f'psych_{f}' for f in psych_features]

    print(f"Final matrix: {X.shape} ({X.shape[1]} features)")
    np.savez(OUT_PATH, subjects=np.array(eeg_ids), X=X, y=y, feature_names=np.array(feat_names))
    print(f"Saved to: {OUT_PATH}")


if __name__ == '__main__':
    main()
