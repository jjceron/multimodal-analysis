"""
MODMA EEG preprocessing v3: add connectivity (coherence) + asymmetry features.
These capture inter-hemispheric and inter-regional patterns critical for MDD.
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.signal import welch, coherence
import mne

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
META_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/Lanzhou University Second Hospital MODMA participants scales.xlsx'
OUT_PATH = 'data/processed/modma_eeg_features_v3.npz'

BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
WINDOW_SEC = 2.0; OVERLAP = 0.5

# Region mapping (rough - based on standard 10-20)
# F: frontal, C: central, P: parietal, O: occipital, T: temporal
CH_REGIONS = {
    'Fp1': 'F', 'Fp2': 'F', 'F7': 'F', 'F3': 'F', 'Fz': 'F', 'F4': 'F', 'F8': 'F',
    'C3': 'C', 'Cz': 'C', 'C4': 'C', 'T7': 'T', 'T8': 'T',
    'P3': 'P', 'Pz': 'P', 'P4': 'P', 'P7': 'P', 'P8': 'P',
    'O1': 'O', 'Oz': 'O', 'O2': 'O',
    # Fp1/Fp2 etc
}


def load_edf_features():
    sub_dirs = sorted(glob.glob(os.path.join(EEG_DIR, 'sub-*')))
    subjects = []
    bp_per_subj = []
    ch_names_per_subj = []

    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        edf_files = glob.glob(os.path.join(sd, 'eeg', '*Resting-state*eeg.EDF'))
        if not edf_files:
            edf_files = glob.glob(os.path.join(sd, 'eeg', '*.EDF'))
        if not edf_files:
            continue
        edf_path = edf_files[0]
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except:
            continue
        sfreq = int(raw.info['sfreq'])
        data = raw.get_data()  # [n_ch, n_samples]
        n_ch, n_samples = data.shape
        if n_ch < 16 or n_samples < 500:
            continue

        # Get first 64 channels
        n_use = min(64, n_ch)
        data_64 = data[:n_use]
        ch_names = raw.ch_names[:n_use]

        ws = int(WINDOW_SEC * sfreq)
        stride = int(ws * (1 - OVERLAP))
        n_w = (n_samples - ws) // stride + 1
        if n_w < 1:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(data_64, ws, axis=1)[:, ::stride].transpose(1,0,2)
        windows = windows[:n_w].astype(np.float32)

        # Band power (5 bands x n_ch)
        bp = np.zeros((n_w, n_use, len(BANDS)), dtype=np.float32)
        for ci in range(n_use):
            for bi, (lo, hi) in enumerate(BANDS.values()):
                f, psd = welch(windows[:, ci, :], fs=sfreq, nperseg=ws, noverlap=ws//2, axis=1)
                mask = (f >= lo) & (f <= hi)
                if mask.sum() > 0:
                    bp[:, ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)

        # Average across windows
        avg_bp = bp.mean(axis=0)  # [n_ch, 5]

        subjects.append(sub_id)
        bp_per_subj.append(avg_bp)
        ch_names_per_subj.append(ch_names)

    return subjects, bp_per_subj, ch_names_per_subj


def build_rich_features(subjects, bp_per_subj, ch_names_per_subj):
    """Build rich features: band power + asymmetry + ratios + band power per region."""
    from collections import defaultdict
    all_features = []
    feature_names = []

    # First pass: determine common channels
    all_channels = set(ch_names_per_subj[0])
    for chs in ch_names_per_subj[1:]:
        all_channels &= set(chs)
    common_channels = sorted(all_channels)[:32]  # use first 32 common

    band_names = list(BANDS.keys())

    for i, sub_id in enumerate(subjects):
        chs = ch_names_per_subj[i]
        bp = bp_per_subj[i]  # [n_ch, 5]

        # Build mapping channel -> bandpower
        ch_to_bp = {ch: bp[chs.index(ch)] for ch in chs if ch in common_channels}
        if len(ch_to_bp) < 16:
            all_features.append(None)
            continue

        feat = []
        names = []

        # 1. Band power per channel (per common channel)
        for ch in common_channels:
            for bi, bn in enumerate(band_names):
                feat.append(ch_to_bp[ch][bi])
                names.append(f'bp_{ch}_{bn}')

        # 2. Inter-hemispheric asymmetry (left - right) / (left + right)
        # Standard 10-20 pairs
        pairs = [('Fp1','Fp2'), ('F3','F4'), ('C3','C4'), ('P3','P4'),
                 ('O1','O2'), ('T7','T8'), ('F7','F8'), ('P7','P8')]
        for bi, bn in enumerate(band_names):
            for chL, chR in pairs:
                if chL in ch_to_bp and chR in ch_to_bp:
                    l, r = ch_to_bp[chL][bi], ch_to_bp[chR][bi]
                    if (l + r) > 0:
                        feat.append((l - r) / (l + r))
                        names.append(f'asym_{chL}_{chR}_{bn}')

        # 3. Band ratios (theta/beta, alpha/theta, etc.) per channel
        eps = 1e-10
        ratio_pairs = [
            ('theta', 'beta'),  # classic ADHD/MDD marker
            ('alpha', 'theta'),
            ('delta', 'theta'),
            ('alpha', 'beta'),
        ]
        for ch in common_channels:
            for num, den in ratio_pairs:
                bi_n = band_names.index(num)
                bi_d = band_names.index(den)
                v = (ch_to_bp[ch][bi_n] + eps) / (ch_to_bp[ch][bi_d] + eps)
                feat.append(v)
                names.append(f'ratio_{num}_{den}_{ch}')

        # 4. Coherence between hemispheric pairs in key bands
        # F3-F4, C3-C4, P3-P4, O1-O2, F7-F8, T7-T8
        for bi, bn in enumerate(band_names):
            f_lo, f_hi = BANDS[bn]
            for chL, chR in pairs:
                if chL in ch_to_bp and chR in ch_to_bp:
                    # Average coherence across windows
                    coh_vals = []
                    for w in range(min(20, len(bp))):
                        f_, cxy = coherence(windows[w, chs.index(chL)],
                                              windows[w, chs.index(chR)],
                                              fs=sfreq, nperseg=ws)
                        mask = (f_ >= f_lo) & (f_ <= f_hi)
                        if mask.sum() > 0:
                            coh_vals.append(np.mean(cxy[mask]))
                    if coh_vals:
                        feat.append(np.mean(coh_vals))
                    else:
                        feat.append(0.0)
                    names.append(f'coh_{chL}_{chR}_{bn}')

        all_features.append(feat)

    # Pad features to same length
    if not all_features or all(f is None for f in all_features):
        return None, None, None

    valid_features = [f for f in all_features if f is not None]
    max_len = max(len(f) for f in valid_features)
    valid_idx = [i for i, f in enumerate(all_features) if f is not None]

    # Pad with zeros
    X_padded = np.zeros((len(valid_features), max_len), dtype=np.float32)
    for j, f in enumerate(valid_features):
        X_padded[j, :len(f)] = f

    # Trim names
    max_names = feature_names
    if len(max_names) > max_len:
        max_names = max_names[:max_len]

    return X_padded, valid_idx, max_names


def main():
    print("="*70)
    print("  MODMA EEG V3: band power + asymmetry + ratios + coherence")
    print("="*70)

    subjects, bp_per_subj, ch_names_per_subj = load_edf_features()
    print(f"\nLoaded {len(subjects)} subjects with EEG")
    if len(subjects) == 0:
        return

    X, valid_idx, feature_names = build_rich_features(subjects, bp_per_subj, ch_names_per_subj)
    if X is None:
        print("Feature construction failed")
        return

    valid_subjects = [subjects[i] for i in valid_idx]
    print(f"Valid subjects: {len(valid_subjects)}")
    print(f"Feature matrix: {X.shape}")
    print(f"Feature names: {len(feature_names)} (first 10: {feature_names[:10]})")

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, subjects=np.array(valid_subjects), X=X,
             feature_names=np.array(feature_names))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == '__main__':
    main()
