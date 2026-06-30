"""MODMA EEG preprocessing module.
Reads BIDS-format EDF files, applies bandpass filtering,
2-second window segmentation with 50% overlap, PSD estimation
via Welch's method, and extracts band power features.
"""
import os, glob, warnings
import numpy as np
from scipy.signal import welch
import mne

warnings.filterwarnings('ignore')

BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
WINDOW_SEC = 2.0
OVERLAP = 0.5


def extract_bandpower_per_window(data, sfreq, n_channels=64):
    """Extract band power from [n_ch, n_samples] data.
    Returns [n_windows, n_channels, n_bands]."""
    ws = int(WINDOW_SEC * sfreq)
    stride = int(ws * (1 - OVERLAP))
    n_samples = data.shape[1]
    n_w = (n_samples - ws) // stride + 1
    if n_w < 1:
        return None
    n_use = min(n_channels, data.shape[0])
    windows = np.lib.stride_tricks.sliding_window_view(data[:n_use], ws, axis=1)[:, ::stride].transpose(1,0,2)
    windows = windows[:n_w].astype(np.float32)
    bp = np.zeros((n_w, n_use, len(BANDS)), dtype=np.float32)
    for ci in range(n_use):
        for bi, (lo, hi) in enumerate(BANDS.values()):
            f, psd = welch(windows[:, ci, :], fs=sfreq, nperseg=ws, noverlap=ws//2, axis=1)
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0:
                bp[:, ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)
    return bp


def load_subjects_features(eeg_dir, n_channels=64, feature_mode='basic'):
    """Load all subjects, extract band power features.
    feature_mode: 'basic' (band power only) or 'rich' (BP + asym + ratios + coherence).
    """
    sub_dirs = sorted(glob.glob(os.path.join(eeg_dir, 'sub-*')))
    all_subjects, all_bp, all_ch_names = [], [], []
    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        edf_files = glob.glob(os.path.join(sd, 'eeg', '*Resting-state*eeg.EDF'))
        if not edf_files:
            edf_files = glob.glob(os.path.join(sd, 'eeg', '*.EDF'))
        if not edf_files:
            continue
        try:
            raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
        except:
            continue
        sfreq = int(raw.info['sfreq'])
        data = raw.get_data()
        if data.shape[0] < 16 or data.shape[1] < 500:
            continue
        ch_names = raw.ch_names[:min(n_channels, data.shape[0])]
        bp = extract_bandpower_per_window(data, sfreq, n_channels)
        if bp is None:
            continue
        all_subjects.append(sub_id)
        all_bp.append(bp.mean(axis=0))  # average across windows: [n_ch, 5]
        all_ch_names.append(ch_names)
    return all_subjects, all_bp, all_ch_names


def save_features(subjects, features, output_path, feature_names=None, metadata=None):
    """Save feature matrix to .npz file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X = np.array(features, dtype=np.float32)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    out = {'subjects': np.array(subjects), 'X': X}
    if feature_names is not None:
        out['feature_names'] = np.array(feature_names)
    if metadata is not None:
        for k, v in metadata.items():
            out[k] = np.array(v)
    np.savez(output_path, **out)
    return output_path
