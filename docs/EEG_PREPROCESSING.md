# EEG Preprocessing Report – MODMA Dataset

## 1. Objective

Process 128-channel resting-state EEG recordings from the MODMA dataset, extract band power features, and prepare them for binary classification (MDD vs HC).

## 2. Data Description

- **Subjects**: 53 (24 MDD, 29 HC).
- **Files**: EDF (European Data Format) under BIDS structure.
- **Channels**: Up to 128 (first 64 used for consistent feature engineering).
- **Sampling rate**: 250 Hz.
- **Duration**: ~301-336 seconds per subject (~5 minutes).

## 3. Preprocessing Steps

1. **Channel selection**: First 64 channels per subject (consistent across subjects).
2. **Segmentation**: 2-second windows with 50% overlap.
3. **Power Spectral Density (PSD)**: Welch's method with nperseg=512, noverlap=256.
4. **Band power extraction**: Mean PSD in 5 bands:
   - Delta: 0.5-4 Hz
   - Theta: 4-8 Hz
   - Alpha: 8-13 Hz
   - Beta: 13-30 Hz
   - Gamma: 30-50 Hz
5. **Z-score normalization**: Per channel, across the full recording.
6. **Output**: Average band power per subject = 53 subjects x 64 channels x 5 bands = 320 features.

## 4. Rich Features (v3)

To improve discriminative power, additional features were engineered:

### 4.1 Inter-hemispheric asymmetry

For 8 canonical electrode pairs (Fp1-Fp2, F3-F4, C3-C4, P3-P4, O1-O2, T7-T8, F7-F8, P7-P8) and 5 bands, compute:

```
asym(L, R, band) = (L - R) / (L + R)
```

This gives 8 x 5 = 40 asymmetry features per subject. These capture hemispheric lateralization patterns associated with MDD.

### 4.2 Band ratios

For each of 32 common channels, compute 4 ratios:
- theta/beta (classic ADHD/MDD marker)
- alpha/theta
- delta/theta
- alpha/beta

This gives 32 x 4 = 128 ratio features.

### 4.3 Inter-hemispheric coherence

For 8 canonical pairs and 5 bands, compute Welch coherence averaged across windows. This gives 8 x 5 = 40 coherence features capturing functional connectivity.

### 4.4 Total feature count

| Feature type | Count |
|---|---|
| Band power (64ch x 5 bands) | 320 |
| Asymmetry (8 pairs x 5 bands) | 40 |
| Band ratios (32ch x 4 ratios) | 128 |
| Coherence (8 pairs x 5 bands) | 40 |
| **Total** | **528** (288 after trimming common channels) |

## 5. Output Files

- `data/processed/modma_eeg_features.npz`: Basic band power (320 features)
- `data/processed/modma_eeg_features_v3.npz`: Rich features (288 features, common channels)

## 6. Subject-level partition (no leakage)

All subsequent training uses **Stratified Group K-Fold** where the group is the subject. This ensures that all windows from one subject are in the same fold, preventing information leakage between training and test.
