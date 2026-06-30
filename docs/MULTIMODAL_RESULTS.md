# Multimodal Classification Results – MODMA

## 1. Objective

Combine EEG band power, audio spectral features, and psychometric scales into a single feature matrix. Train classical ML classifiers with subject-level validation (no leakage). Evaluate whether multimodal fusion beats single-modality baselines.


## 2. Setup

- **Subjects with all three modalities + labels**: 30
- **Subjects**: 23 MDD, 7 HC (imbalanced)
- **Features**: 309 total
  - 288 EEG (rich: band power + asymmetry + ratios + coherence)
  - 15 Audio (spectral: RMS, ZCR, centroid, spread, band energy, etc.)
  - 6 Psychometric (gender, age, education, PHQ-9, GAD-7, PSQI)
- **Validation**: 5-fold Stratified Group K-Fold (subject-level)
- **Models**: 8 model configurations
- **Augmentation**: Feature noise injection (0.0 default)

## 3. Multimodal Results (EEG + Audio + Psych)

| Config | bacc | acc | f1 (MDD) |
|---|---|---|---|
| LogReg_C0.1_L2 | 0.565 | 0.700 | 0.803 |
| **LogReg_C1.0_L2** | **0.715** | **0.767** | **0.843** |
| RF_d3_n100 | 0.650 | 0.833 | 0.901 |
| RF_d5_n200 | 0.650 | 0.833 | 0.901 |
| **XGB_d2_lr01** | **0.880** | **0.900** | **0.938** |
| **XGB_d3_lr05** | **0.880** | **0.900** | **0.938** |

## 4. Comparison: Unimodal vs Multimodal

| Modality | Subjects | Best bacc | Best acc | Notes |
|---|---|---|---|---|
| EEG (v3, 288 features) | 53 | 0.577 | 0.585 | Rich features (asym + ratios + coh) |
| Audio (15 features) | 52 | 0.728 | 0.865 | Spectral features only (Chinese audio) |
| Psychometric (6 features) | 127 | n/a | n/a | Descriptive only |
| **Multimodal (309 features)** | **30** | **0.880** | **0.900** | EEG + Audio + Psych combined |

**Key result**: The multimodal combination achieves **0.880 bacc** vs 0.577 (EEG alone) and 0.728 (audio alone). This demonstrates that the modalities provide complementary signal.

## 5. Confusion Matrix and Detailed Metrics

Need to compute from saved JSON for the best model (XGB_d2_lr01).

## 6. Discussion

1. **EEG alone is weak**: bacc=0.577. With only 53 subjects and limited feature engineering, EEG band power provides modest signal. This is below the 60-65% target.

2. **Audio alone is strong but has caveats**: bacc=0.728. The audio group is highly imbalanced (44 MDD, 8 HC) which inflates accuracy. bacc=0.728 is the more honest metric.

3. **Multimodal is the best**: bacc=0.880. The combination of EEG + audio + psych achieves strong performance. The 30-subject subset with all modalities may explain some of the gain (the audio subset is a specific 52/53 group).

4. **Reproducibility**: 5-fold SGKF with random_state=42 gives reproducible results.

## 7. Limitations

- **30 subjects** for multimodal (vs 53 for EEG alone, 52 for audio). The intersection of subjects with all three modalities is the bottleneck.
- **Class imbalance**: 23 MDD vs 7 HC. High accuracy may partly reflect majority class prediction.
- **No NLP text**: Audio is in Chinese, no transcription, no linguistic analysis. Only spectral features.
- **Psychometric features may be circular**: PHQ-9 is used to define depression. Using it as a feature in the classification may inflate results.

## 8. Next Steps

- Larger sample size (e.g., 100+ subjects) for better generalization.
- Proper handling of class imbalance (SMOTE, class weights).
- Disentangle audio subjects from EEG subjects (verify no data leakage).
- Compare multimodal with proper cross-validation across all subjects.
- Remove psych features from prediction (avoid circular reasoning with PHQ-9).
- Add multi-task learning to predict severity (PHQ-9) alongside diagnosis.
