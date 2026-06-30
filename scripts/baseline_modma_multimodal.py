"""MODMA multimodal baseline: EEG + audio + psychometric features.
30 subjects, 5-fold Stratified Group K-Fold."""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import warnings as _w
_w.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

DATA_PATH = 'data/processed/modma_multimodal_features.npz'
OUT_PATH = 'results/modma_multimodal_baseline.json'
LOG_PATH = 'results/modma_multimodal_baseline.log'

os.makedirs('results', exist_ok=True)


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    subjects = data['subjects']
    X = data['X'].astype(np.float32)
    y = data['y']
    groups = np.array([str(s) for s in subjects])

    print("="*70)
    print("  MODMA MULTIMODAL BASELINE (EEG + audio + psych)")
    print(f"  Subjects: {len(subjects)} (MDD: {np.sum(y==1)}, HC: {np.sum(y==0)})")
    print(f"  Features: {X.shape[1]}")
    print("="*70)

    models = {
        'LogReg_C0.1_L2': LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42),
        'LogReg_C1.0_L2': LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42),
        'RF_d3_n100': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
        'RF_d5_n200': RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
        'XGB_d2_lr01': XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     objective='binary:logistic', eval_metric='logloss', verbosity=0, n_jobs=-1),
        'XGB_d3_lr05': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     objective='binary:logistic', eval_metric='logloss', verbosity=0, n_jobs=-1),
    }

    n_folds = 5
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    print(f"\nResults ({n_folds}-fold SGKF):")
    print(f"{'Model':>22s} | {'bacc':>7s} {'acc':>7s} {'f1':>7s}")
    print("-"*45)

    log_lines = []
    for cfg, model in models.items():
        from sklearn.base import clone
        baccs, accs, f1s = [], [], []
        for train_idx, test_idx in skf.split(X, y, groups=groups):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            m = clone(model)
            m.fit(Xtr_s, ytr)
            yp = m.predict(Xte_s)
            baccs.append(balanced_accuracy_score(yte, yp))
            accs.append(accuracy_score(yte, yp))
            f1s.append(f1_score(yte, yp, pos_label=1, zero_division=0))
        mb = float(np.mean(baccs))
        ma = float(np.mean(accs))
        mf = float(np.mean(f1s))
        results[cfg] = {'mean_bacc': mb, 'mean_acc': ma, 'mean_f1': mf,
                        'fold_baccs': [float(x) for x in baccs],
                        'fold_accs': [float(x) for x in accs]}
        print(f"  {cfg:>22s} | {mb:>7.3f} {ma:>7.3f} {mf:>7.3f}")

    best = max(results, key=lambda k: results[k]['mean_bacc'])
    print(f"\nBEST: {best} bacc={results[best]['mean_bacc']:.3f}")
    print(f"Majority baseline: acc={float(np.sum(y==0)/len(y)):.3f}, bal_acc=0.500")

    output = {
        'n_subjects': int(len(subjects)),
        'n_MDD': int(np.sum(y==1)),
        'n_HC': int(np.sum(y==0)),
        'n_features': int(X.shape[1]),
        'results': results,
        'best': {'config': best, **results[best]},
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == '__main__':
    main()
