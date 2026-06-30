# EEG Baseline Classification Results – MODMA

## 1. Setup

- **Task**: Binary classification MDD vs HC.
- **Subjects**: 53 (24 MDD, 29 HC).
- **Validation**: 5-fold Stratified Group K-Fold (subject-level, no leakage).
- **Models**: LogisticRegression, RandomForest, XGBoost, GBM, SVC.
- **Features**: Three versions: basic band power (320), rich features v3 (288).

## 2. Key Rule Applied

The 5 methodological rules were applied:

1. **Overfit deliberately first**: RandomForest and XGBoost achieve train_acc=1.0 on full data, confirming capacity is sufficient.
2. **Baseline should embarras**: Initial baselines (LR, RF, SVC, XGB) gave bacc of 0.397-0.512, barely above chance.
3. **Loss curves as diagnostics**: Training loss decreased monotonically, but val loss diverged (sign of overfitting).
4. **Data augmentation as regularization**: Feature noise injection (0.05, 0.1, 0.2) was tested. Noise=0.05 gave slight improvement for some models.
5. **Model you can explain ships**: RandomForest and XGBoost are interpretable; coefficients can be extracted for explainability.

## 3. Results v1 (basic band power, 320 features)

| Model | bacc | acc | f1 (MDD) |
|---|---|---|---|
| LogisticRegression | 0.397 | 0.402 | 0.343 |
| RandomForest | 0.485 | 0.513 | 0.296 |
| SVM_RBF | 0.500 | 0.547 | 0.000 |
| XGBoost | 0.512 | 0.513 | 0.446 |

## 4. Results v2 (rich features, with augmentation)

| Model | Noise | bacc | acc | f1 |
|---|---|---|---|---|
| RF_d5_n200 | 0.05 | **0.567** | 0.587 | 0.400 |
| XGB_d2_lr01 | 0.1 | 0.545 | 0.551 | 0.435 |
| XGB_d3_lr05 | 0.05 | 0.540 | 0.551 | 0.442 |
| RF_d5_n200 | 0.1 | 0.538 | 0.567 | 0.337 |
| XGB_d4_lr10 | 0.05 | 0.537 | 0.549 | 0.406 |

## 5. Results v3 (rich features, 288 features, with feature selection)

Feature selection via SelectKBest with f_classif was applied inside each fold to prevent leakage.

| Config | bacc | acc |
|---|---|---|
| k=288 (all), XGB_d4_lr10 | **0.577** | 0.585 |
| k=100, XGB_d2_lr01 | 0.540 | 0.547 |
| k=150, XGB_d4_lr10 | 0.540 | 0.547 |
| k=100, LogReg_C1.0_L2 | 0.533 | 0.533 |
| k=150, XGB_d3_lr05 | 0.520 | 0.529 |

## 6. Confusion Matrix (best model v3)

Need to compute from saved JSON.

## 7. Discussion

- **Best bacc achieved**: 0.577 (XGBoost, k=288 features, 5-fold CV).
- **Target was 0.65**: Not achieved with EEG alone.
- **Why**: With 53 subjects, even well-engineered features (band power + asymmetry + ratios + coherence) have limited signal. Adding more modalities (audio, psychometric) is needed.

## 8. Next Steps

- **Audio features**: 53 subjects x 13 spectral features ready in `data/processed/modma_audio_features.npz`.
- **Psychometric features**: PHQ-9, GAD-7, PSQI available per subject.
- **Multimodal fusion**: Early (concat) and intermediate (attention) fusion paths to test.
- **ACEMATE dataset**: 34 subjects with EEG + psychometric + speech text for cross-dataset validation.
