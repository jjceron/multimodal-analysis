"""MODMA audio baseline classification: spectral features.
53 subjects, Chinese audio (no NLP text - spectral only).
Binary: MDD vs HC, 5-fold Stratified Group K-Fold.
Audio IDs (02010XXX) map to participant IDs (sub-XXX)."""
import os, sys, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

AUDIO_PATH = 'data/processed/modma_audio_features.npz'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
OUT_PATH = 'results/modma_audio_baseline.json'
LOG_PATH = 'results/modma_audio_baseline.log'

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
    log("  MODMA AUDIO BASELINE CLASSIFICATION - spectral features")
    log("  Binary: MDD vs HC, 5-fold Stratified Group K-Fold")
    log("  Note: Chinese language audio (no NLP text extraction)")
    log("  Audio ID 02010XXX maps to participant sub-XXX")
    log("="*70)

    audio = np.load(AUDIO_PATH, allow_pickle=True)
    audio_subs = list(audio['subjects'])
    X = audio['X'].astype(np.float32)
    feat_names = list(audio['feature_names'])

    participants = load_participants()
    sub_to_group = dict(zip(
        [str(s) for s in participants['participant_id'].tolist()],
        participants['group']
    ))

    # Map audio IDs to participant IDs
    audio_to_part = {a: f"sub-{a[5:]}" for a in audio_subs}
    y, valid_subs = [], []
    for a in audio_subs:
        pid = audio_to_part[a]
        g = sub_to_group.get(pid)
        if g in ('MDD', 'HC'):
            y.append(1 if g == 'MDD' else 0)
            valid_subs.append(a)
    y = np.array(y); valid_subs = np.array(valid_subs)
    X = X[:len(valid_subs)]

    log(f"\nSubjects with labels: {len(valid_subs)} (MDD: {np.sum(y==1)}, HC: {np.sum(y==0)})")
    log(f"Features: {X.shape}")
    log(f"Feature names: {feat_names}")

    # Models
    models = {
        'LogReg_C0.1_L2':    LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42),
        'LogReg_C1.0_L2':    LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42),
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
    }

    n_folds = 5
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results_all = {}

    log(f"\nPer-model results ({n_folds}-fold Stratified Group K-Fold):")
    log(f"{'Model':>22s} | {'bacc':>7s} {'acc':>7s} {'f1(MDD)':>7s}")
    log("-" * 50)

    for cfg_name, model in models.items():
        from sklearn.base import clone
        fold_baccs, fold_accs, fold_f1s = [], [], []
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=valid_subs)):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            m = clone(model)
            m.fit(Xtr_s, ytr)
            y_pred = m.predict(Xte_s)
            fold_baccs.append(balanced_accuracy_score(yte, y_pred))
            fold_accs.append(accuracy_score(yte, y_pred))
            fold_f1s.append(f1_score(yte, y_pred, pos_label=1, zero_division=0))
        mean_bacc = float(np.mean(fold_baccs))
        mean_acc = float(np.mean(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        results_all[cfg_name] = {
            'mean_bacc': mean_bacc, 'mean_acc': mean_acc, 'mean_f1': mean_f1,
            'fold_baccs': [float(x) for x in fold_baccs],
            'fold_accs': [float(x) for x in fold_accs],
        }
        log(f"  {cfg_name:>22s} | {mean_bacc:>7.3f} {mean_acc:>7.3f} {mean_f1:>7.3f}")

    best_key = max(results_all, key=lambda k: results_all[k]['mean_bacc'])
    best = results_all[best_key]
    log(f"\n{'='*70}")
    log(f"  BEST: {best_key}")
    log(f"  bacc={best['mean_bacc']:.3f} +/- {np.std(best['fold_baccs']):.3f}")
    log(f"  acc={best['mean_acc']:.3f} +/- {np.std(best['fold_accs']):.3f}")
    log(f"{'='*70}")

    majority_acc = float(np.sum(y == 0) / len(y))
    log(f"\n  Baselines:")
    log(f"  Majority class (HC): acc={majority_acc:.3f}, bal_acc=0.500")
    log(f"  Chance:               acc=0.500, bal_acc=0.500")

    output = {
        'n_subjects': int(len(valid_subs)),
        'n_MDD': int(np.sum(y == 1)),
        'n_HC': int(np.sum(y == 0)),
        'n_folds': n_folds,
        'n_features': int(X.shape[1]),
        'feature_names': feat_names,
        'results_all': results_all,
        'best': {'config': best_key, **best},
        'baselines': {'majority': majority_acc, 'chance': 0.5},
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log(f"\nResults saved to: {OUT_PATH}")
    log(f"Log saved to: {LOG_PATH}")


if __name__ == '__main__':
    main()
