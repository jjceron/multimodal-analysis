"""Run the full MODMA preprocessing + feature matrix + baseline pipeline."""
import os, sys, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.preprocessing.modma_eeg import load_subjects_features, save_features, extract_bandpower_per_window, BANDS as EEG_BANDS
from src.preprocessing.modma_audio import load_audio_features
from src.features.modma_metadata import load_participants, get_subject_groups, get_psychometric_features
from src.features.rich_eeg import extract_rich_features
from src.features.modma_matrix import build_multimodal_matrix
from src.models.baseline_classifier import run_benchmark, get_default_models

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
AUDIO_DIR = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'

OUT_EEG = 'data/processed/modma_eeg_features.npz'
OUT_AUDIO = 'data/processed/modma_audio_features.npz'
OUT_MATRIX = 'data/processed/modma_multimodal_features.npz'
OUT_RESULTS = 'results/modma_multimodal_baseline.json'

os.makedirs('data/processed', exist_ok=True)
os.makedirs('results', exist_ok=True)


def main():
    print("="*70)
    print("  MODMA FULL PIPELINE")
    print("  EEG + audio + psychometric features for MDD/HC classification")
    print("="*70)

    # 1. EEG features (basic band power)
    print("\n[1/3] EEG preprocessing (BIDS-EDF, 64-channel, 5 bands)")
    eeg_subs, eeg_bp, eeg_ch_names = load_subjects_features(EEG_DIR, n_channels=64)
    print(f"  Loaded {len(eeg_subs)} subjects")
    eeg_feature_names = [f'{ch}_{bn}' for ch in eeg_ch_names[0] for bn in EEG_BANDS.keys()]
    eeg_feature_names = eeg_feature_names[:64*5]
    save_features(eeg_subs, eeg_bp, OUT_EEG,
                  feature_names=eeg_feature_names)
    print(f"  Saved to: {OUT_EEG} ({64*5} features per subject)")

    # 2. Audio features
    print("\n[2/3] Audio preprocessing (.wav, spectral features)")
    audio_subs, audio_X, audio_feats = load_audio_features(AUDIO_DIR)
    print(f"  Loaded {len(audio_subs)} subjects")
    save_features(audio_subs, audio_X, OUT_AUDIO, feature_names=audio_feats)
    print(f"  Saved to: {OUT_AUDIO} ({len(audio_feats)} features per subject)")

    # 3. Build multimodal matrix + benchmark
    print("\n[3/3] Multimodal feature matrix + benchmark")
    participants = load_participants(PARTICIPANTS_PATH)
    sub_to_group = get_subject_groups(participants)

    eeg_data = np.load(OUT_EEG, allow_pickle=True)
    audio_data = np.load(OUT_AUDIO, allow_pickle=True)

    psych_subs = eeg_data['subjects'].tolist()
    psych_X = get_psychometric_features(participants, psych_subs)

    X, y, common_subs, feat_names = build_multimodal_matrix(
        eeg_data['subjects'].tolist(), eeg_data['X'].astype(np.float32), eeg_feature_names,
        audio_data['subjects'].tolist(), audio_data['X'].astype(np.float32), audio_feats,
        psych_subs, psych_X,
        sub_to_group
    )

    if X is None:
        print("  No overlapping subjects found")
        return

    print(f"  Matrix: {X.shape} ({len(feat_names)} features)")
    print(f"  Subjects: MDD={int(np.sum(y==1))}, HC={int(np.sum(y==0))}")

    np.savez(OUT_MATRIX, subjects=common_subs, X=X, y=y, feature_names=feat_names)
    print(f"  Saved to: {OUT_MATRIX}")

    # 4. Baseline benchmark
    print(f"\n[4/4] Baseline classification benchmark")
    groups = np.array([str(s) for s in common_subs])
    results, best_key = run_benchmark(X, y, groups, n_folds=5,
                                       models=get_default_models(),
                                       noise_levels=[0.0])

    print(f"\nResults (5-fold SGKF):")
    print(f"{'Config':>45s} | {'bacc':>7s} {'acc':>7s} {'f1':>7s}")
    print("-"*75)
    for k, v in sorted(results.items()):
        print(f"  {k:>45s} | {v['mean_bacc']:>7.3f} {v['mean_acc']:>7.3f} {v['mean_f1']:>7.3f}")

    best = results[best_key]
    print(f"\nBEST: {best_key}")
    print(f"  Balanced Accuracy: {best['mean_bacc']:.3f} +/- {np.std(best['fold_baccs']):.3f}")

    output = {
        'n_subjects': int(len(common_subs)),
        'n_MDD': int(np.sum(y==1)),
        'n_HC': int(np.sum(y==0)),
        'n_features': int(X.shape[1]),
        'results': results,
        'best': {'config': best_key, **best},
    }
    with open(OUT_RESULTS, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {OUT_RESULTS}")


if __name__ == '__main__':
    main()
