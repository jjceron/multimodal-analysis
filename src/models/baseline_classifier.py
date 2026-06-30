"""Baseline binary classifier with subject-level CV.
Supports multiple models and noise-based augmentation.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def get_default_models():
    """Return dict of model name -> instantiated classifier."""
    return {
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
                                        objective='binary:logilogistic', eval_metric='logloss', verbosity=0, n_jobs=-1),
    }


def evaluate_model(X, y, groups, model, n_folds=5, noise_level=0.0, seed=42, random_state=42):
    """Run StratifiedGroupKFold evaluation. Returns dict of metrics per fold."""
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    rng = np.random.RandomState(seed)
    fold_baccs, fold_accs, fold_f1s = [], [], []
    for train_idx, test_idx in skf.split(X, y, groups=groups):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        # Feature noise augmentation
        if noise_level > 0:
            noise = rng.normal(0, noise_level, Xtr_s.shape).astype(np.float32)
            Xtr_s = Xtr_s + noise
        from sklearn.base import clone
        m = clone(model)
        m.fit(Xtr_s, ytr)
        yp = m.predict(Xte_s)
        fold_baccs.append(balanced_accuracy_score(yte, yp))
        fold_accs.append(accuracy_score(yte, yp))
        fold_f1s.append(f1_score(yte, yp, pos_label=1, zero_division=0))
    return {
        'fold_baccs': [float(x) for x in fold_baccs],
        'fold_accs': [float(x) for x in fold_accs],
        'fold_f1s': [float(x) for x in fold_f1s],
        'mean_bacc': float(np.mean(fold_baccs)),
        'mean_acc': float(np.mean(fold_accs)),
        'mean_f1': float(np.mean(fold_f1s)),
    }


def run_benchmark(X, y, groups, n_folds=5, models=None, noise_levels=None, seed=42):
    """Run full benchmark across models and noise levels."""
    if models is None:
        models = get_default_models()
    if noise_levels is None:
        noise_levels = [0.0]

    results = {}
    for noise in noise_levels:
        for name, model in models.items():
            res = evaluate_model(X, y, groups, model, n_folds=n_folds,
                                 noise_level=noise, seed=seed)
            key = f"noise={noise} | {name}"
            results[key] = res

    best_key = max(results, key=lambda k: results[k]['mean_bacc'])
    return results, best_key
