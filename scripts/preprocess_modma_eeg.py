"""
MODMA EEG preprocessing: Load EDF files from BIDS structure, extract band power.
128-channel resting state EEG, 250Hz, ~5min per subject.
"""
import os, glob, warnings
import numpy as np, pandas as pd
from scipy.signal import welch
import mne

warnings.filterwarnings('ignore')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
META_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/Lanzhou University Second Hospital MODMA participants scales.xlsx'
OUT_PATH = 'data/processed/modma_eeg_features.npz'

BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
WINDOW_SEC = 2.0; OVERLAP = 0.5

def load_subjects():
    """Load all subjects with their EDF files and labels from BIDS structure."""
    meta = pd.read_excel(META_PATH)
    cols = list(meta.columns)
    print("Metadata columns:", [str(c) for c in cols])
    sub_dir = EEG_DIR
    subjects_data = {}
    sub_dirs = sorted(glob.glob(os.path.join(sub_dir, 'sub-*')))
    print(f"Found {len(sub_dirs)} subject directories")
    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        eeg_files = glob.glob(os.path.join(sd, 'eeg', '*Resting-state*eeg.EDF'))
        if not eeg_files:
            eeg_files = glob.glob(os.path.join(sd, 'eeg', '*.EDF'))
        if not eeg_files:
            print(f"  {sub_id}: no EDF file found")
            continue
        edf_path = eeg_files[0]
        channels_tsv = os.path.join(sd, 'eeg', '*Resting-state*channels.tsv')
        ch_files = glob.glob(channels_tsv)
        if not ch_files:
            ch_files = glob.glob(os.path.join(sd, 'eeg', '*channels.tsv'))
        ch_names = None
        if ch_files:
            try:
                ch_df = pd.read_csv(ch_files[0], sep='\t')
                if 'name' in ch_df.columns:
                    ch_names = ch_df['name'].tolist()
            except:
                pass
        subjects_data[sub_id] = {'edf_path': edf_path, 'ch_names': ch_names}
    return subjects_data

def extract_bandpower(edf_path, ch_names, n_target_channels=64, window_sec=2.0, overlap=0.5):
    """Extract band power from EDF file, averaged across windows."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  Error reading {edf_path}: {e}")
        return None, None
    sfreq = int(raw.info['sfreq'])
    data = raw.get_data()
    actual_ch = raw.ch_names
    n_ch_data, n_samples = data.shape
    if n_ch_data < n_target_channels:
        return None, None
    if n_samples < int(window_sec * sfreq) * 2:
        return None, None
    ws = int(window_sec * sfreq)
    stride = int(ws * (1 - overlap))
    n_w = (n_samples - ws) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(data, ws, axis=1)[:, ::stride].transpose(1,0,2)
    windows = windows[:n_w].astype(np.float32)
    bp = np.zeros((windows.shape[0], n_target_channels, len(BANDS)), dtype=np.float32)
    for ci in range(n_target_channels):
        if ci >= n_ch_data: break
        for bi, (lo, hi) in enumerate(BANDS.values()):
            f, psd = welch(windows[:, ci, :], fs=sfreq, nperseg=ws, noverlap=ws//2, axis=1)
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0:
                bp[:, ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)
    return bp, actual_ch

def main():
    print("="*70)
    print("  MODMA EEG PREPROCESSING — band power features")
    print("="*70)
    subjects_data = load_subjects()
    print(f"\nSubjects with EDF: {len(subjects_data)}")
    if len(subjects_data) == 0:
        print("No EDF files found. Checking alternative locations...")
        alt_dirs = [
            'data/raw/modma/854301_EEG_128Channels_Resting_Lanzhou_2015',
            'data/raw/modma/MODMA_EEG_BIDS_format',
        ]
        for d in alt_dirs:
            if os.path.exists(d):
                sub_dirs = sorted(glob.glob(os.path.join(d, 'sub-*')))
                print(f"  {d}: {len(sub_dirs)} sub dirs")
        return

    results = {}
    errors = []
    for sub_id, info in subjects_data.items():
        edf_path = info['edf_path']
        try:
            bp, ch_names = extract_bandpower(edf_path, info['ch_names'], n_target_channels=64)
            if bp is None:
                errors.append(sub_id)
                continue
            avg_bp = bp.mean(axis=0)  # [64, 5]
            results[sub_id] = {
                'bp_per_window': bp,
                'bp_avg': avg_bp.flatten(),
                'n_windows': bp.shape[0],
                'n_channels': bp.shape[1],
                'actual_ch_names': ch_names,
            }
            print(f"  {sub_id}: {bp.shape[0]} windows x {bp.shape[1]} ch x {bp.shape[2]} bands")
        except Exception as e:
            print(f"  {sub_id}: error {e}")
            errors.append(sub_id)

    print(f"\nProcessed: {len(results)}/{len(subjects_data)} subjects")
    print(f"Errors: {len(errors)}")

    # Save
    if results:
        sub_ids = sorted(results.keys())
        X = np.array([results[s]['bp_avg'] for s in sub_ids])
        X_win = np.array([results[s]['bp_per_window'] for s in sub_ids], dtype=object)
        np.savez(OUT_PATH,
                 subjects=np.array(sub_ids),
                 X=X, X_windows=X_win,
                 n_channels=np.array([results[s]['n_channels'] for s in sub_ids]))
        print(f"\nSaved to: {OUT_PATH}")
        print(f"  Feature matrix: {X.shape} (subjects x 64ch*5bands)")
    else:
        print("No subjects processed successfully.")

if __name__ == '__main__':
    main()
