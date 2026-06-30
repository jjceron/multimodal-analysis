"""MODMA audio preprocessing module.
Reads .wav files, extracts spectral features (band-wise energy,
temporal statistics, spectral statistics).
No NLP text extraction (audio is in Chinese language).
"""
import os, glob, warnings
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import spectrogram

warnings.filterwarnings('ignore')

BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}


def extract_spectral_features_from_wav(wav_path):
    """Extract spectral features from a single .wav file."""
    try:
        sr, audio = wav.read(wav_path)
    except:
        return None
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=512, noverlap=256)
    feat = {}
    feat['duration_sec'] = float(len(audio) / sr)
    feat['sample_rate'] = int(sr)
    feat['spectral_mean'] = float(np.mean(Sxx))
    feat['spectral_std'] = float(np.std(Sxx))
    feat['spectral_max'] = float(np.max(Sxx))
    feat['spectral_energy'] = float(np.sum(Sxx ** 2))
    for bname, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f <= hi)
        feat[f'band_{bname}'] = float(np.mean(Sxx[mask, :])) if mask.sum() > 0 else 0.0
    feat['rms'] = float(np.sqrt(np.mean(audio ** 2)))
    feat['zero_crossing_rate'] = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))
    feat['spectral_centroid'] = float(np.sum(f * Sxx.mean(axis=1)) / (Sxx.mean(axis=1).sum() + 1e-10))
    feat['spectral_spread'] = float(np.sqrt(np.sum((f - feat['spectral_centroid'])**2 * Sxx.mean(axis=1)) / (Sxx.mean(axis=1).sum() + 1e-10)))
    return feat


def load_audio_features(audio_dir):
    """Load all subjects, extract per-subject mean spectral features."""
    sub_dirs = sorted(glob.glob(os.path.join(audio_dir, '*')))
    subjects, all_features = [], []
    feature_names = None
    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        wav_files = sorted(glob.glob(os.path.join(sd, '*.wav')))
        if not wav_files:
            continue
        subj_feats = []
        for wf in wav_files:
            f = extract_spectral_features_from_wav(wf)
            if f is not None:
                subj_feats.append(f)
        if not subj_feats:
            continue
        if feature_names is None:
            feature_names = sorted(subj_feats[0].keys())
        avg_feat = {k: np.mean([sf[k] for sf in subj_feats if k in sf]) for k in feature_names}
        subjects.append(sub_id)
        all_features.append([avg_feat[k] for k in feature_names])
    return subjects, all_features, feature_names
