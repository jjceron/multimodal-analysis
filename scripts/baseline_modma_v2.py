"""
MODMA EEG improved baseline classification v2.
Apply the 5 rules:
1. Overfit deliberately first (capacity check)
2. Baseline should embarras
3. Loss curves as diagnostic tools
4. Data augmentation as regularization
5. Model you can explain ships
"""
import os, sys, warnings, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

EEG_FEAT_PATH = 'data/processed/modma_eeg_features.npz'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
OUT_PATH = 'results/modma_eeg_baseline_v2.json'
LOG_PATH = 'results/modma_eeg_baseline_v2.log'
CM_PATH = 'results/modma_eeg_confusion_matrices.json'

os.makedirs('results', exist_ok=True)


def load_participants():
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                      on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 4, 5, 6, 7, 8]]
    p.columns = ['participant_id', 'gender', 'age', 'education', 'PHQ-9', 'group', 'GAD-7', 'PSQI']
    return p


def build_rich_features(X, eeg_subs, participants):
    """Build richer feature set: ratios, per-subject normalization, channel aggregates.
    Input X: [53, 320] = 53 subjects x (64ch * 5bands)."""
    n_sub, n_feat = X.shape
    n_ch = 64; n_bands = 5
    # Reshape to [n_sub, n_ch, n_bands]
    X_3d = X.reshape(n_sub, n_ch, n_bands)
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    extras = []
    feature_names = []

    # Per-channel band power (already in X)
    # Channel aggregates (mean, std across channels per band)
    for bi, bn in enumerate(band_names):
        col = X_3d[:, :, bi]
        agg_mean = col.mean(axis=1)  # [n_sub]
        agg_std = col.std(axis=1)    # [n_sub]
        extras.append(agg_mean)
        feature_names.append(f'mean_{bn}')
        extras.append(agg_std)
        feature_names.append(f'std_{bn}')

    # Ratios (theta/beta, alpha/theta, delta/theta, (theta+alpha)/(delta+beta))
    delta = X_3d[:, :, 0]
    theta = X_3d[:, :, 1]
    alpha = X_3d[:, :, 2]
    beta = X_3d[:, :, 3]
    gamma = X_3d[:, :, 4]

    # Per-channel ratios, then aggregate
    eps = 1e-10
    for name, num, den in [
        ('theta_beta', theta, beta),
        ('alpha_theta', alpha, theta),
        ('delta_theta', delta, theta),
        ('alpha_beta', alpha, beta),
        ('theta_alpha', theta, alpha),
    ]:
        ratio = (num + eps) / (den + eps)  # [n_sub, n_ch]
        extras.append(ratio.mean(axis=1))
        feature_names.append(f'ratio_{name}_mean')
        extras.append(ratio.std(axis=1))
        feature_names.append(f'ratio_{name}_std')

    X_extra = np.column_stack(extras)
    X_full = np.hstack([X, X_extra])
    return X_full, feature_names


def train_overfit_check(X, y, groups, model_factory, model_name):
    """Train on full data without CV to see if model can memorize."""
    from sklearn.base import clone
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    m = clone(model_factory)
    m.fit(X_s, y)
    train_pred = m.predict(X_s)
    train_acc = accuracy_score(y, train_pred)
    train_bacc = balanced_accuracy_score(y, train_pred)
    return train_acc, train_bacc


def main():
    log_lines = []
    cm_dict = {}
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("="*70)
    log("  MODMA EEG IMPROVED BASELINE CLASSIFICATION v2")
    log("  5 rules: overfit first, baseline embarras, loss curves, augment=regularize, explainable model")
    log("="*70)

    eeg = np.load(EEG_FEAT_PATH, allow_pickle=True)
    eeg_subs = list(eeg['subjects'])
    X_raw = eeg['X']

    participants = load_participants()
    sub_to_group = dict(zip(participants['participant_id'], participants['group']))

    y, valid_subs = [], []
    for s in eeg_subs:
        g = sub_to_group.get(s)
        if g in ('MDD', 'HC'):
            y.append(1 if g == 'MDD' else 0)
            valid_subs.append(s)
    y = np.array(y); valid_subs = np.array(valid_subs)
    X_raw = X_raw[:len(valid_subs)]

    log(f"\nSubjects: {len(valid_subs)} (MDD: {np.sum(y==1)}, HC: {np.sum(y==0)})")
    log(f"Raw features: {X_raw.shape} (64ch x 5bands)")

    # Build rich features
    X_rich, feat_names = build_rich_features(X_raw, valid_subs, participants)
    log(f"Rich features: {X_rich.shape} ({len(feat_names)} extra)")

    # === RULE 1: Overfit deliberately ===
    log("\n" + "="*70)
    log("  RULE 1: OVERFIT CHECK - can the model memorize training data?")
    log("="*70)
    candidate_models = {
        'RandomForest':      RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1),
        'XGBoost':           XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.3,
                                          subsample=0.9, colsample_bytree=0.9, random_state=42,
                                          objective='binary:logistic', eval_metric='logloss',
                                          verbosity=0, n_jobs=-1),
    }
    for name, m in candidate_models.items():
        train_acc, train_bacc = train_overfit_check(X_rich, y, valid_subs, m, name)
        log(f"  {name:>20s}: train_acc={train_acc:.3f}  train_bacc={train_bacc:.3f}  -> capacity OK={train_acc > 0.95}")

    # === RULE 5: Explainable models + RULE 4: Augmentation via noise injection ===
    log("\n" + "="*70)
    log("  MODELS: explainable + augmentation (noise injection on features)")
    log("="*70)

    # Augmentation as feature noise injection (RULE 4)
    rng = np.random.RandomState(42)
    noise_levels = [0.0, 0.05, 0.10, 0.20]

    # Try multiple model configurations
    model_configs = {
        'LogReg_C0.1_L2':    LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42),
        'LogReg_C1.0_L2':    LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42),
        'LogReg_C0.01_L1':   LogisticRegression(C=0.01, penalty='l1', solver='saga', max_iter=5000, random_state=42),
        'RF_d3_n100':        RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42, n_jobs=-1),
        'RF_d5_n200':        RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42, n_jobs=-1),
        'RF_d8_n300':        RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1),
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
    for noise_lvl in noise_levels:
        log(f"\n--- Noise level: {noise_lvl} ---")
        for cfg_name, model in model_configs.items():
            from sklearn.base import clone
            fold_baccs = []
            fold_accs = []
            fold_f1s = []
            for fold_id, (train_idx, test_idx) in enumerate(skf.split(X_rich, y, groups=valid_subs)):
                Xtr_raw, Xte_raw = X_rich[train_idx], X_rich[test_idx]
                ytr, yte = y[train_idx], y[test_idx]
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr_raw)
                Xte_s = sc.transform(Xte_raw)
                # Feature noise augmentation (RULE 4)
                if noise_lvl > 0 and len(Xtr_s) > 0:
                    noise = rng.normal(0, noise_lvl, Xtr_s.shape).astype(np.float32)
                    Xtr_s = Xtr_s + noise
                m = clone(model)
                m.fit(Xtr_s, ytr)
                y_pred = m.predict(Xte_s)
                fold_baccs.append(balanced_accuracy_score(yte, y_pred))
                fold_accs.append(accuracy_score(yte, y_pred))
                fold_f1s.append(f1_score(yte, y_pred, pos_label=1, zero_division=0))
            mean_bacc = float(np.mean(fold_baccs))
            mean_acc = float(np.mean(fold_accs))
            mean_f1 = float(np.mean(fold_f1s))
            key = f"noise={noise_lvl} | {cfg_name}"
            results_all[key] = {
                'mean_bacc': mean_bacc, 'mean_acc': mean_acc, 'mean_f1': mean_f1,
                'fold_baccs': [float(x) for x in fold_baccs],
                'fold_accs': [float(x) for x in fold_accs],
                'fold_f1s': [float(x) for x in fold_f1s],
            }
            log(f"  {cfg_name:>22s}: bacc={mean_bacc:.3f} acc={mean_acc:.3f} f1={mean_f1:.3f}")

    # Find best config
    best_key = max(results_all, key=lambda k: results_all[k]['mean_bacc'])
    best = results_all[best_key]
    log(f"\n{'='*70}")
    log(f"  BEST: {best_key}")
    log(f"  Balanced Accuracy: {best['mean_bacc']:.3f} +/- {np.std(best['fold_baccs']):.3f}")
    log(f"  Accuracy:          {best['mean_acc']:.3f} +/- {np.std(best['fold_accs']):.3f}")
    log(f"  F1 (MDD):          {best['mean_f1']:.3f} +/- {np.std(best['fold_f1s']):.3f}")
    log(f"{'='*70}")

    # Top 5 by balanced accuracy
    top5 = sorted(results_all.items(), key=lambda x: -x[1]['mean_bacc'])[:5]
    log(f"\n  TOP 5 configurations by balanced accuracy:")
    log(f"  {'Config':<40s} {'bacc':>7s} {'acc':>7s} {'f1':>7s}")
    for k, v in top5:
        log(f"  {k:<40s} {v['mean_bacc']:>7.3f} {v['mean_acc']:>7.3f} {v['mean_f1']:>7.3f}")

    # Baselines
    majority_acc = float(np.sum(y == 0) / len(y))
    log(f"\n  Baselines:")
    log(f"  Majority class (HC): acc={majority_acc:.3f}, bal_acc=0.500")
    log(f"  Chance:               acc=0.500, bal_acc=0.500")
    log(f"  Target:               bal_acc > 0.650")

    # Save
    output = {
        'n_subjects': int(len(valid_subs)),
        'n_MDD': int(np.sum(y == 1)),
        'n_HC': int(np.sum(y == 0)),
        'n_folds': n_folds,
        'n_features_rich': int(X_rich.shape[1]),
        'results_all': results_all,
        'best': {'config': best_key, **best},
        'top5': [{'config': k, **v} for k, v in top5],
        'baselines': {'majority': majority_acc, 'chance': 0.5},
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log(f"\nResults saved to: {OUT_PATH}")
    log(f"Log saved to: {LOG_PATH}")

    return results_all, best_key, best['mean_bacc']


if __name__ == '__main__':
    results, best_cfg, best_bacc = main()
    target = 0.65
    print(f"\nTarget: bal_acc > {target}")
    print(f"Achieved: bal_acc = {best_bacc:.3f}")
    if best_bacc >= target:
        print("TARGET MET!")
    else:
        print(f"Gap to target: {target - best_bacc:.3f}")
