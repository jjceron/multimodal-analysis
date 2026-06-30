# Activity 2 Deliverables – MODMA Multimodal Feature Pipeline

## 1. Overview

This document describes the implementation of the activity: **preprocessing and feature extraction by modality, including filtering, segmentation and normalization of EEG recordings, standardization of psychometric variables, and preparation of multimodal feature matrices ready for training**.

The pipeline was developed on the MODMA dataset (Multimodal Open Dataset for Mental-disorder Analysis) for binary classification of Major Depressive Disorder (MDD) vs Healthy Controls (HC).

## 2. Repository Structure

```
MultimodalAnalysis/
├── data/
│   ├── raw/                    # (gitignored) BIDS-formatted MODMA data
│   └── processed/              # Generated feature matrices
│       ├── modma_eeg_features.npz          # EEG band power (320 features)
│       ├── modma_eeg_features_v3.npz       # EEG rich features (288 features)
│       ├── modma_audio_features.npz        # Audio spectral (15 features)
│       └── modma_multimodal_features.npz  # Combined (309 features, 30 subjects)
├── src/
│   ├── preprocessing/
│   │   ├── modma_eeg.py         # BIDS-EDF reader, band power, segmentation
│   │   ├── modma_audio.py       # WAV reader, spectral features
│   │   └── run_pipeline.py      # Full preprocessing + feature pipeline
│   ├── features/
│   │   ├── modma_metadata.py    # participants.tsv, psychometric
│   │   ├── rich_eeg.py          # Asymmetry, ratios, coherence
│   │   └── modma_matrix.py      # Multimodal fusion
│   └── models/
│       └── baseline_classifier.py  # Subject-level CV with multiple models
├── scripts/                      # Standalone runnable scripts (legacy)
├── results/                      # JSON results + log files
└── docs/                         # This report and detailed results
```

## 3. Activity Requirements Coverage

| Requirement | Implementation | Status |
|---|---|---|
| EEG preprocessing (filtering) | MNE read_raw_edf, automatic bandpass | Done |
| EEG segmentation | 2-second windows, 50% overlap | Done |
| EEG normalization | Per-channel z-score (in the rich features) | Done |
| Psychometric standardization | StandardScaler after imputation | Done |
| Audio preprocessing | WAV read, spectral features (band energy, RMS, ZCR) | Done |
| Text preparation (NLP) | Not applicable (Chinese audio, no transcription) | N/A |
| Consolidated feature matrix | `modma_multimodal_features.npz` (30x309) | Done |
| Processing scripts | All in `src/` and `scripts/` | Done |

## 4. Methodology

### 4.1 Subject-level data partition

All validation uses **Stratified Group K-Fold (SGKF)** with the subject as group. This ensures that all windows/audio files from one subject are in the same fold, preventing information leakage between training and test.

### 4.2 EEG preprocessing

- Read BIDS-format EDF files (128-channel → first 64 selected for consistency)
- 2-second windows with 50% overlap
- Power Spectral Density via Welch's method
- Band power extraction in 5 canonical bands: delta, theta, alpha, beta, gamma
- Z-score normalization per channel
- Rich features: inter-hemispheric asymmetry, band ratios, inter-hemispheric coherence

### 4.3 Audio preprocessing

- Read WAV files per subject (~29 files per subject, 53 subjects)
- Mono conversion, normalize to float
- Spectral features: RMS, zero-crossing rate, spectral centroid, spectral spread
- Band-wise energy: delta, theta, alpha, beta, gamma

### 4.4 Psychometric standardization

- Read participants.tsv (BIDS metadata with PHQ-9, GAD-7, PSQI)
- Extract 6 variables: gender (binary), age, education, PHQ-9, GAD-7, PSQI
- SimpleImputer (constant=0 for missing), then StandardScaler 

### 4.5 Multimodal fusion

Concatenation of EEG + audio + psych features after per-modality standardization. Saved as `modma_multimodal_features.npz`.

## 5. Results Summary

| Modality | Subjects | Best bacc | Best acc |
|---|---|---|---|
| EEG (v3, 288 features) | 53 | 0.577 | 0.585 |
| Audio (15 features) | 52 | 0.728 | 0.865 |
| **Multimodal (309 features)** | **30** | **0.880** | **0.900** |

## 6. Files Generated

- `data/processed/modma_eeg_features.npz`: EEG band power (320 features, 53 subjects)
- `data/processed/modma_eeg_features_v3.npz`: EEG rich features (288 features, 53 subjects)
- `data/processed/modma_audio_features.npz`: Audio spectral (15 features, 53 subjects)
- `data/processed/modma_multimodal_features.npz`: Combined (309 features, 30 subjects)
- `results/modma_eeg_baseline.json/.log`: EEG-only baseline results
- `results/modma_eeg_baseline_v2.json/.log`: EEG rich features results
- `results/modma_eeg_baseline_v3.json/.log`: EEG rich features + feature selection
- `results/modma_audio_baseline.json/.log`: Audio-only baseline results
- `results/modma_multimodal_baseline.json/.log`: Multimodal results
- `results/modma_audio_preprocess.log`: Audio preprocessing log

## 7. Reproducibility

Run `python src/preprocessing/run_pipeline.py` from the repo root to reproduce the entire pipeline. The pipeline:
1. Loads BIDS EEG and extracts band power
2. Loads WAV audio and extracts spectral features
3. Reads participants.tsv and extracts psychometric features
4. Builds multimodal feature matrix
5. Runs baseline classification

## 8. Important Limitations

1. **Subject overlap**: Multimodal requires all three modalities per subject. Only 30 subjects have EEG + audio + psych + labels. Audio group (52 subjects) overlaps partially with EEG (53).
2. **Class imbalance**: 23 MDD vs 7 HC in multimodal. High accuracy may reflect majority class.
3. **No NLP text**: Audio is in Chinese, no transcription, no linguistic analysis. Only spectral features.
4. **Psychometric circularity**: PHQ-9 is used to define depression, but it's also in the features. This may inflate results.
5. **EEG v1 bacc below target**: EEG alone at 0.577 is below the 60-65% target. With more sophisticated EEG features or larger sample, this could improve.
6. **Audio alone is high but caveat**: 0.728 bacc on imbalanced group (44 MDD, 8 HC). Acc=0.865 inflates by majority class.
