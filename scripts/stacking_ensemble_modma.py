"""
MODMA Stacking Ensemble: RF + XGB + LogReg -> meta-LogReg.
5-fold SGKF. Subject-level. Expected +2-5% over best unimodal.
"""
import os, sys, warnings, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

sys.path.insert(0, '.')
from src.utils.training_logger import log_header, log_epoch, log_fold_test, log_summary

warnings.filterwarnings('ignore')

EEG_PATH = 'data/processed/modma_eeg_features_v3.npz'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
RANDOM_STATE = 42; N_CH = 64; N_BANDS = 5


def load_data():
    eeg = np.load(EEG_PATH, allow_pickle=True)
    subs = eeg['subjects']
    X = eeg['X'].astype(np.float32)
    if X.ndim == 3: X = X.reshape(X.shape[0], -1)
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1, on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 5, 6, 7, 8, 9]]
    p.columns = ['participant_id','gender','age','education','group','PHQ9','GAD7','PSQI']
    sg = dict(zip(p['participant_id'], p['group']))
    y = []; vs = []
    for s in subs:
        g = sg.get(s); 
        if g in ('MDD','HC'): y.append(1 if g=='MDD' else 0); vs.append(s)
    return X[:len(vs)], np.array(y), np.array(vs)


def main():
    print("="*70)
    print("  MODMA STACKING ENSEMBLE: RF + XGB + LogReg -> meta-LogReg")
    print("  5-fold SGKF, subject-level")
    print("="*70)

    X, y, subs = load_data()
    print(f"\nSubjects: {len(subs)} (MDD: {np.sum(y==1)}, HC: {np.sum(y==0)})")
    print(f"Features: {X.shape[1]}")

    base_models = {
        'RF': RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1),
        'XGB': XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                             random_state=RANDOM_STATE, objective='binary:logistic', eval_metric='logloss',
                             verbosity=0, n_jobs=-1),
        'LogReg': LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
    }
    meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)

    n_folds = 5
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {'Stacking': [], 'RF': [], 'XGB': [], 'LogReg': []}

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(subs)), y, groups=subs)):
        Xtr, Xte = X[tvi], X[tei]; ytr, yte = y[tvi], y[tei]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

        # Train base models and get their out-of-fold predictions for stacking
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+fi)
        meta_X = np.zeros((len(tvi), len(base_models)))
        meta_Xte = np.zeros((len(tei), len(base_models)))

        for mi, (name, model) in enumerate(base_models.items()):
            # Out-of-fold predictions for stacking
            m = clone(model)
            meta_X[:, mi] = cross_val_predict(m, Xtr_s, ytr, cv=inner, groups=subs[tvi], method='predict_proba')[:, 1]
            # Train on all training and predict test
            m.fit(Xtr_s, ytr)
            meta_Xte[:, mi] = m.predict_proba(Xte_s)[:, 1]
            fold_metrics[name].append({
                'bacc': balanced_accuracy_score(yte, (meta_Xte[:, mi] >= 0.5).astype(int)),
                'acc': accuracy_score(yte, (meta_Xte[:, mi] >= 0.5).astype(int)),
                'f1': f1_score(yte, (meta_Xte[:, mi] >= 0.5).astype(int), zero_division=0),
            })

        # Train meta-model
        meta_model.fit(meta_X, ytr)
        meta_preds = meta_model.predict_proba(meta_Xte)[:, 1]
        fold_metrics['Stacking'].append({
            'bacc': balanced_accuracy_score(yte, (meta_preds >= 0.5).astype(int)),
            'acc': accuracy_score(yte, (meta_preds >= 0.5).astype(int)),
            'f1': f1_score(yte, (meta_preds >= 0.5).astype(int), zero_division=0),
        })

    # Aggregate
    print(f"\n{'Model':<12s} {'bacc':>7s} {'acc':>7s} {'f1':>7s} {'std_bacc':>7s}")
    print("-"*40)
    for name, metrics in fold_metrics.items():
        mb = float(np.mean([m['bacc'] for m in metrics]))
        ma = float(np.mean([m['acc'] for m in metrics]))
        mf = float(np.mean([m['f1'] for m in metrics]))
        sb = float(np.std([m['bacc'] for m in metrics]))
        print(f"{name:<12s} {mb:>7.3f} {ma:>7.3f} {mf:>7.3f} {sb:>7.3f}")

    # Compare
    best_stacking = float(np.mean([m['bacc'] for m in fold_metrics['Stacking']]))
    best_individual = max(float(np.mean([m['bacc'] for m in fold_metrics[n]])) for n in ['RF','XGB','LogReg'])
    improvement = best_stacking - best_individual
    print(f"\nStacking bacc: {best_stacking:.3f} | Best individual: {best_individual:.3f} | Delta: {improvement:+.3f}")
    print(f"Baseline classical EEG: XGBoost v3 rich bacc=0.577")

    # Save
    results = {'models': {n: {'mean_bacc': float(np.mean([m['bacc'] for m in fm]))} for n, fm in fold_metrics.items()},
               'stacking_improvement': float(improvement)}
    with open('results/modma_stacking_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/modma_stacking_baseline.json")


if __name__ == '__main__': main()
