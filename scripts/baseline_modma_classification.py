"""
MODMA baseline classification: classical ML on EEG band power.
Binary classification MDD vs HC, Stratified Group K-Fold (subject-level).
"""
import os, sys, warnings, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

EEG_FEAT_PATH = 'data/processed/modma_eeg_features.npz'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
OUT_PATH = 'results/modma_eeg_baseline.json'
LOG_PATH = 'results/modma_eeg_baseline.log'


def load_participants():
    """Load participants.tsv with correct column handling.
    The file has 8 header columns but 10 data columns (extra empty field after sub-XXX).
    Data layout: sub-XXX, '', gender, age, education, PHQ-9, group, GAD-7, PSQI, extra
    """
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                      on_bad_lines='skip', engine='python')
    # p has 10 columns: 0=sub, 1=empty, 2=gender, 3=age, 4=education,
    # 5=PHQ-9, 6=group, 7=GAD-7, 8=PSQI, 9=extra
    p = p[[0, 2, 3, 4, 5, 6, 7, 8]]
    p.columns = ['participant_id', 'gender', 'age', 'education', 'PHQ-9', 'group', 'GAD-7', 'PSQI']
    return p


def main():
    os.makedirs('results', exist_ok=True)
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("="*70)
    log("  MODMA BASELINE CLASSIFICATION - EEG band power")
    log("  Binary: MDD vs HC, Stratified Group K-Fold (subject-level)")
    log("="*70)

    # Load features
    eeg = np.load(EEG_FEAT_PATH, allow_pickle=True)
    eeg_subs = list(eeg['subjects'])
    X = eeg['X']  # [53, 320]
    log(f"\nEEG features: {X.shape} (subjects x 64ch*5bands)")

    # Load participants
    participants = load_participants()
    log(f"Participants: {participants.shape}")
    log(f"Groups: {participants['group'].value_counts().to_dict()}")

    # Map sub IDs to groups
    sub_to_group = dict(zip(participants['participant_id'], participants['group']))
    y = []
    valid_subs = []
    for s in eeg_subs:
        g = sub_to_group.get(s)
        if g in ('MDD', 'HC'):
            y.append(1 if g == 'MDD' else 0)
            valid_subs.append(s)
    y = np.array(y)
    valid_subs = np.array(valid_subs)
    X = X[:len(valid_subs)]
    log(f"\nValid subjects with labels: {len(valid_subs)}")
    log(f"  MDD: {np.sum(y==1)}, HC: {np.sum(y==0)}")

    # Models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest':       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'SVM_RBF':            SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'XGBoost':            XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                         subsample=0.8, random_state=42, objective='binary:logistic',
                                         eval_metric='logloss', verbosity=0, n_jobs=-1),
    }

    def compute_metrics(y_true, y_pred):
        return {
            'accuracy':          float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'f1_MDD':            float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            'sensitivity':       float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            'specificity':       float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        }

    # Stratified Group K-Fold
    n_folds = 5
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = {name: [] for name in models}

    log(f"\nPer-fold metrics ({n_folds}-fold Stratified Group K-Fold):")
    log(f"{'Model':>22s} | {'Acc':>7s} {'BalAcc':>7s} {'F1':>7s} {'Sens':>7s} {'Spec':>7s}")
    log("-" * 70)

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=valid_subs)):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        for name, model in models.items():
            from sklearn.base import clone
            m = clone(model)
            m.fit(Xtr_s, ytr)
            y_pred = m.predict(Xte_s)
            m_metrics = compute_metrics(yte, y_pred)
            fold_metrics[name].append(m_metrics)

    # Aggregate
    summary = {}
    log("")
    for name, mlist in fold_metrics.items():
        avg = {k: float(np.mean([m[k] for m in mlist])) for k in mlist[0]}
        std = {k: float(np.std([m[k] for m in mlist])) for k in mlist[0]}
        summary[name] = {'mean': avg, 'std': std, 'per_fold': mlist}
        log(f"{name:>22s} | {avg['accuracy']:>7.3f} {avg['balanced_accuracy']:>7.3f} "
            f"{avg['f1_MDD']:>7.3f} {avg['sensitivity']:>7.3f} {avg['specificity']:>7.3f}")

    results = {
        'n_subjects': int(len(valid_subs)),
        'n_MDD': int(np.sum(y == 1)),
        'n_HC': int(np.sum(y == 0)),
        'n_folds': n_folds,
        'feature_dim': int(X.shape[1]),
        'models': summary,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log(f"\nResults saved to: {OUT_PATH}")
    log(f"Log saved to: {LOG_PATH}")

    majority_acc = float(np.sum(y == 0) / len(y))
    log(f"\nBaselines:")
    log(f"  Majority class (HC): acc={majority_acc:.3f}, bal_acc=0.500")
    log(f"  Chance:               acc=0.500, bal_acc=0.500")


if __name__ == '__main__':
    main()
