"""
MODMA audio preprocessing: Extract spectral features from .wav files.
Audio is in Chinese language - no NLP transcription, only spectral analysis.
"""
import os, sys, glob, warnings, json
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import spectrogram

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

AUDIO_DIR = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015'
OUT_PATH = 'data/processed/modma_audio_features.npz'
LOG_PATH = 'results/modma_audio_preprocess.log'

os.makedirs('results', exist_ok=True)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def extract_spectral_features(wav_path, n_mels=64, n_mfcc=13):
    """Extract mel-spectrogram and MFCC-like features from a wav file."""
    try:
        sr, audio = wav.read(wav_path)
    except Exception as e:
        return None

    # Handle stereo - convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalize
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Compute mel-spectrogram
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=512, noverlap=256)
    # Sxx shape: (n_freq, n_time)

    # Statistical features
    feat = {}
    feat['duration_sec'] = float(len(audio) / sr)
    feat['sample_rate'] = int(sr)

    # Spectral statistics
    feat['spectral_mean'] = float(np.mean(Sxx))
    feat['spectral_std'] = float(np.std(Sxx))
    feat['spectral_max'] = float(np.max(Sxx))
    feat['spectral_energy'] = float(np.sum(Sxx ** 2))

    # Band-wise energy (5 bands)
    bands = {'delta':(1,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,50)}
    for bname, (lo, hi) in bands.items():
        mask = (f >= lo) & (f <= hi)
        if mask.sum() > 0:
            feat[f'band_{bname}'] = float(np.mean(Sxx[mask, :]))
        else:
            feat[f'band_{bname}'] = 0.0

    # Temporal statistics
    feat['rms'] = float(np.sqrt(np.mean(audio ** 2)))
    feat['zero_crossing_rate'] = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))
    feat['spectral_centroid'] = float(np.sum(f * Sxx.mean(axis=1)) / (Sxx.mean(axis=1).sum() + 1e-10))
    feat['spectral_spread'] = float(np.sqrt(np.sum((f - feat['spectral_centroid'])**2 * Sxx.mean(axis=1)) / (Sxx.mean(axis=1).sum() + 1e-10)))

    return feat


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("="*70)
    log("  MODMA AUDIO PREPROCESSING - spectral features")
    log("="*70)

    if not os.path.exists(AUDIO_DIR):
        log(f"Audio directory not found: {AUDIO_DIR}")
        return

    sub_dirs = sorted(glob.glob(os.path.join(AUDIO_DIR, '*')))
    log(f"Found {len(sub_dirs)} subject directories")

    all_features = []
    all_ids = []
    errors = []

    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        wav_files = sorted(glob.glob(os.path.join(sd, '*.wav')))

        if not wav_files:
            continue

        # Aggregate across all wav files for this subject
        subj_feats = []
        for wf in wav_files:
            f = extract_spectral_features(wf)
            if f is not None:
                subj_feats.append(f)

        if not subj_feats:
            errors.append(sub_id)
            continue

        # Average across files for this subject
        keys = subj_feats[0].keys()
        avg_feat = {k: np.mean([sf[k] for sf in subj_feats if k in sf]) for k in keys}
        all_features.append(avg_feat)
        all_ids.append(sub_id)
        log(f"  {sub_id}: {len(wav_files)} wav files, dur={avg_feat['duration_sec']:.1f}s")

    if not all_features:
        log("No subjects processed.")
        return

    log(f"\nProcessed: {len(all_features)} subjects")
    log(f"Errors: {len(errors)}")

    # Build feature matrix
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features])

    np.savez(OUT_PATH,
             subjects=np.array(all_ids),
             X=X, feature_names=np.array(feature_names))
    log(f"\nSaved to: {OUT_PATH}")
    log(f"  Feature matrix: {X.shape}")
    log(f"  Features: {feature_names}")

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


if __name__ == '__main__':
    main()
