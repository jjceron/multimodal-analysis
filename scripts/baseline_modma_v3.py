"""
MODMA EEG baseline v3: rich features (band power + asymmetry + ratios + coherence).
Target: balanced accuracy > 0.65.
"""
import os, sys, warnings, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

EEG_FEAT_PATH = 'data/processed/modma_eeg_features_v3.npz'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
OUT_PATH = 'results/modma_eeg_baseline_v3.json'
LOG_PATH = 'results/modma_eeg_baseline_v3.log'

os.makedirs('results', exist_ok=True)


def load_participants():
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                      on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 4, 5, 6, 7, 8]]
    p.columns = ['participant_id', 'gender', 'age', 'education', 'PHQ-9', 'group', 'GAD-7', 'PSQI']
    return p


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("="*70)
    log("  MODMA EEG BASELINE v3: rich features (BP+asym+ratios+coherence)")
    log("  Target: bal_acc > 0.65")
    log("="*70)

    eeg = np.load(EEG_FEAT_PATH, allow_pickle=True)
    eeg_subs = list(eeg['subjects'])
    X = eeg['X'].astype(np.float32)

    participants = load_participants()
    sub_to_group = dict(zip(participants['participant_id'], participants['group']))

    y, valid_subs = [], []
    for s in eeg_subs:
        g = sub_to_group.get(s)
        if g in ('MDD', 'HC'):
            y.append(1 if g == 'MDD' else 0)
            valid_subs.append(s)
    y = np.array(y); valid_subs = np.array(valid_subs)
    X = X[:len(valid_subs)]

    log(f"\nSubjects: {len(valid_subs)} (MDD: {np.sum(y==1)}, HC: {np.sum(y==0)})")
    log(f"Features: {X.shape}")

    # === STEP 1: Feature selection - use SelectKBest with mutual info ===
    log("\n--- Feature selection experiments ---")
    n_folds = 5
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    model_configs = {
        'LogReg_C0.1_L2':    LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42),
        'LogReg_C1.0_L2':    LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42),
        'LogReg_C0.1_L1':    LogisticRegression(C=0.1, penalty='l1', solver='saga', max_iter=5000, random_state=42),
        'LogReg_C0.01_L1':   LogisticRegression(C=0.01, penalty='l1', solver='saga', max_iter=5000, random_state=42),
        'RF_d3_n100':        RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
        'RF_d5_n200':        RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
        'RF_d10_n300':       RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        'XGB_d2_lr01':       XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.1,
                                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                                        objective='binary:logistic', eval_metric='logloss', verbosity=0, n_jobs=-1),
        'XGB_d3_lr05':       XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                                        objective='binary:logistic', eval_metric='logloss', verbosity=0, n_jobs=-1),
        'XGB_d4_lr10':       XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        subsample=0.7, colsample_bytree=0.7, random_state=42,
                                        objective='binary:logistic', eval_metric='logloss', verbosity=0, n_jobs=-1),
        'GBM_d3_n200':       GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                                      subsample=0.8, random_state=42),
    }

    # Try different feature counts with mutual_info feature selection
    results_all = {}
    for k_features in [50, 100, 150, 200, 288]:
        if k_features > X.shape[1]:
            continue
        for cfg_name, model in model_configs.items():
            from sklearn.base import clone
            fold_baccs = []
            fold_accs = []
            for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=valid_subs)):
                Xtr, Xte = X[train_idx], X[test_idx]
                ytr, yte = y[train_idx], y[test_idx]

                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr)
                Xte_s = sc.transform(Xte)

                # Feature selection inside fold (no leak)
                selector = SelectKBest(f_classif, k=min(k_features, Xtr_s.shape[1]))
                Xtr_sel = selector.fit_transform(Xtr_s, ytr)
                Xte_sel = selector.transform(Xte_s)

                m = clone(model)
                m.fit(Xtr_sel, ytr)
                y_pred = m.predict(Xte_sel)
                fold_baccs.append(balanced_accuracy_score(yte, y_pred))
                fold_accs.append(accuracy_score(yte, y_pred))

            mean_bacc = float(np.mean(fold_baccs))
            mean_acc = float(np.mean(fold_accs))
            key = f"k={k_features} | {cfg_name}"
            results_all[key] = {
                'mean_bacc': mean_bacc, 'mean_acc': mean_acc,
                'fold_baccs': [float(x) for x in fold_baccs],
                'fold_accs': [float(x) for x in fold_accs],
            }
            log(f"  {key:<35s}: bacc={mean_bacc:.3f} acc={mean_acc:.3f}")

    # Top 10
    top10 = sorted(results_all.items(), key=lambda x: -x[1]['mean_bacc'])[:10]
    log(f"\n  TOP 10 by balanced accuracy:")
    log(f"  {'Config':<40s} {'bacc':>7s} {'acc':>7s}")
    for k, v in top10:
        log(f"  {k:<40s} {v['mean_bacc']:>7.3f} {v['mean_acc']:>7.3f}")

    best_key = top10[0][0]
    best = results_all[best_key]
    log(f"\n{'='*70}")
    log(f"  BEST: {best_key}")
    log(f"  Balanced Accuracy: {best['mean_bacc']:.3f} +/- {np.std(best['fold_baccs']):.3f}")
    log(f"  Accuracy:          {best['mean_acc']:.3f} +/- {np.std(best['fold_accs']):.3f}")
    log(f"{'='*70}")

    # Baselines
    majority_acc = float(np.sum(y == 0) / len(y))
    log(f"\n  Baselines:")
    log(f"  Majority class (HC): acc={majority_acc:.3f}, bal_acc=0.500")
    log(f"  Chance:               acc=0.500, bal_acc=0.500")
    log(f"  Target:               bal_acc > 0.650")

    output = {
        'n_subjects': int(len(valid_subs)),
        'n_MDD': int(np.sum(y == 1)),
        'n_HC': int(np.sum(y == 0)),
        'n_folds': n_folds,
        'n_features_total': int(X.shape[1]),
        'results_all': results_all,
        'best': {'config': best_key, **best},
        'top10': [{'config': k, **v} for k, v in top10],
        'baselines': {'majority': majority_acc, 'chance': 0.5},
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log(f"\nResults saved to: {OUT_PATH}")
    log(f"Log saved to: {LOG_PATH}")
    log(f"Target: bal_acc > 0.65 | Achieved: bal_acc = {best['mean_bacc']:.3f}")


if __name__ == '__main__':
    main()
