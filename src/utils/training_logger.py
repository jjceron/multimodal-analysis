"""
Standardized training logger for classification and regression pipelines.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, mean_absolute_error


# ── Metrics computation ────────────────────────────────────────────────

def regression_metrics(true, pred):
    """Compute regression metrics at subject level."""
    t = np.array(true, dtype=float)
    p = np.array(pred, dtype=float)
    mae = mean_absolute_error(t, p) if len(t) > 1 else float('nan')
    r2 = r2_score(t, p) if len(t) > 1 else float('nan')
    nrmse = np.sqrt(np.mean((t - p) ** 2)) / (t.std() + 1e-10) if len(t) > 1 else float('nan')
    sr, sp = stats.spearmanr(t, p) if len(t) > 2 else (float('nan'), 1.0)
    pr, pp = stats.pearsonr(t, p) if len(t) > 2 else (float('nan'), 1.0)
    return {'mae': mae, 'r2': r2, 'nrmse': nrmse,
            'spear': sr, 'spear_p': sp, 'pear': pr, 'pear_p': pp}


def classification_metrics(true, pred):
    """Compute classification metrics."""
    t = np.array(true, dtype=int)
    p = np.array(pred, dtype=int)
    if len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
        return {'acc': 0.0, 'bacc': 0.0, 'f1': 0.0, 'sens': 0.0, 'spec': 0.0}
    return {
        'acc':  float(accuracy_score(t, p)),
        'bacc': float(balanced_accuracy_score(t, p)),
        'f1':   float(f1_score(t, p, zero_division=0)),
        'sens': float(f1_score(t, p, pos_label=1, zero_division=0)),
        'spec': float(f1_score(t, p, pos_label=0, zero_division=0)),
    }


def subject_aggregate(preds_window, trues_window, cods, subjects):
    """Aggregate per-window predictions into per-subject predictions."""
    true_s, pred_s = [], []
    offset = 0
    for cod in cods:
        nw = len(subjects[cod]['windows'])
        pred_s.append(np.mean(preds_window[offset:offset + nw]))
        sub_trues = np.array(trues_window[offset:offset + nw])
        true_s.append(sub_trues[0] if len(sub_trues) > 0 else subjects[cod].get('cog', 0))
        offset += nw
    return np.array(true_s), np.array(pred_s)


# ── Epoch-level logging ────────────────────────────────────────────────

def log_header(mode='regr'):
    """Print header for per-epoch training log."""
    if mode == 'regr':
        print(f"  {'Epoch':>5s} | {'T_loss':>8s} {'V_loss':>8s} "
              f"{'T_mae':>7s} {'V_mae':>7s} | "
              f"{'V_r2':>7s} {'V_spear':>7s} {'V_pear':>7s} {'V_nrmse':>8s} | pat")
    else:
        print(f"  {'Epoch':>5s} | {'T_loss':>8s} {'V_loss':>8s} "
              f"{'T_acc':>6s} {'V_acc':>6s} | "
              f"{'V_bacc':>6s} {'V_f1':>6s} {'V_sens':>6s} {'V_spec':>6s} | pat")


def log_epoch(epoch, tr_loss, vl_loss, tr_m, vl_m, patience, mode='regr'):
    """Print a single epoch line matching the header format."""
    if mode == 'regr':
        print(f"  {epoch:5d} | {tr_loss:8.4f} {vl_loss:8.4f} "
              f"{tr_m['mae']:7.3f} {vl_m['mae']:7.3f} | "
              f"{vl_m['r2']:7.3f} {vl_m['spear']:7.3f} "
              f"{vl_m['pear']:7.3f} {vl_m['nrmse']:8.3f} | {patience:2d}")
    else:
        print(f"  {epoch:5d} | {tr_loss:8.4f} {vl_loss:8.4f} "
              f"{tr_m['acc']:6.3f} {vl_m['acc']:6.3f} | "
              f"{vl_m['bacc']:6.3f} {vl_m['f1']:6.3f} "
              f"{vl_m['sens']:6.3f} {vl_m['spec']:6.3f} | {patience:2d}")


# ── Fold test log ──────────────────────────────────────────────────────

def log_fold_test(test_true, test_pred, mode='regr'):
    """Print test results for a single fold."""
    if mode == 'regr':
        m = regression_metrics(test_true, test_pred)
        print(f"  >>> test: mae={m['mae']:.3f} r2={m['r2']:+.3f} "
              f"spear={m['spear']:+.3f} pear={m['pear']:+.3f} nrmse={m['nrmse']:.3f}")
        return m
    else:
        m = classification_metrics(test_true, test_pred)
        print(f"  >>> test: acc={m['acc']:.3f} bacc={m['bacc']:.3f} "
              f"f1={m['f1']:.3f}")
        return m


# ── Final summary tables ───────────────────────────────────────────────

_REGR_KEYS = ['r2', 'mae', 'spear', 'pear', 'nrmse']
_CLAS_KEYS = ['acc', 'bacc', 'f1', 'sens', 'spec']


def log_summary(fold_metrics, n_folds=None, mode='regr', split_type='gkf'):
    """Print final aggregated results table.
    fold_metrics: list of dicts (one per fold), each from regression_metrics or classification_metrics
    """
    if n_folds is None:
        n_folds = len(fold_metrics)
    keys = _REGR_KEYS if mode == 'regr' else _CLAS_KEYS
    arrays = {k: np.array([m[k] for m in fold_metrics]) for k in keys}

    print(f"\n{'=' * 60}")
    title = 'GKF' if split_type == 'gkf' else 'LOSO'
    print(f"  {title} RESULT ({n_folds} folds)")
    print(f"  {'':>7s} | {'mean':>8s} {'+-':>2s} {'std':>8s}")
    print(f"  {'':->7s}-+-{'-' * 20}")
    for k in keys:
        mn, sd = np.mean(arrays[k]), np.std(arrays[k])
        print(f"  {k:>7s} | {mn:>8.3f} {'+-':>2s} {sd:>8.3f}")

    if split_type == 'gkf' and n_folds > 1:
        print()
        fold_hdr = ' | '.join([f"{k:>8s}" for k in keys])
        print(f"  {'Fold':>7s} | {fold_hdr}")
        print(f"  {'':->7s}-+-{'-' * (9 * len(keys) - 1)}")
        for fi in range(n_folds):
            vals = ' | '.join([f"{arrays[k][fi]:>8.3f}" for k in keys])
            print(f"  {fi + 1:>7d} | {vals}")

    print(f"{'=' * 60}")
