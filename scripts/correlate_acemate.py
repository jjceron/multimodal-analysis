"""
Spearman correlation + Ridge regression: EEG band power vs Barratt impulsivity
Reuses the same 90 features (18ch x 5bands, OA+OC avg) from baseline_acemate.py
"""
import glob, os, re, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from statsmodels.stats.multitest import multipletests

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
SFREQ = 250

BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}

CHANNEL_18 = ['FP1','F7','T7','P7','F3','C3','P3','O1','FZ','PZ',
              'FP2','F8','T8','P8','F4','C4','P4','O2']

CDMS_MAP = {n:n for n in CHANNEL_18}
CSO_MAP = {'FP1-Cz':'FP1','FP2-Cz':'FP2','FZ-Cz':'FZ','PZ-Cz':'PZ',
    'F3-Cz':'F3','F4-Cz':'F4','F7-Cz':'F7','F8-Cz':'F8',
    'C3-Cz':'C3','C4-Cz':'C4','T3-Cz':'T7','T4-Cz':'T8','T5-Cz':'P7','T6-Cz':'P8',
    'P3-Cz':'P3','P4-Cz':'P4','O1-Cz':'O1','O2-Cz':'O2'}
BIOSEMI_MAP = {'A1':'FP1','A5':'F3','A7':'F7','A12':'C3','A15':'T7','A18':'P3',
    'A21':'P7','A23':'O1','B1':'FP2','B3':'FZ','B5':'F4','B7':'F8',
    'B12':'C4','B15':'T8','B18':'P4','B21':'P8','B24':'O2','B27':'PZ'}

FEATURE_NAMES = [f"{ch}_{bn}" for ch in CHANNEL_18 for bn in ['delta','theta','alpha','beta','gamma']]


def read_eeg(filepath):
    try:
        import mne
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        return raw.get_data(), list(raw.ch_names), int(raw.info['sfreq'])
    except Exception:
        import h5py
        fdt = filepath.replace('.set','.fdt')
        with h5py.File(filepath,'r') as h5:
            nch=int(h5['nbchan'][0][0]); pnts=int(h5['pnts'][0][0]); sr=int(h5['srate'][0][0])
            labs=h5['chanlocs']['labels'][:]
            ch_names=[]
            for l in labs:
                if isinstance(l,bytes): ch_names.append(l.decode('utf-8',errors='replace').strip('\x00'))
                elif hasattr(l,'tobytes'): ch_names.append(l.tobytes().decode('utf-8',errors='replace').strip('\x00'))
                else: ch_names.append(str(l))
        data=np.fromfile(fdt,dtype=np.float32).reshape(nch,pnts,order='F')
        return data,ch_names,sr


def load_data():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1'] = meta[['3.','6.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    raw = {}
    for fpath in all_files:
        bn = os.path.basename(fpath)
        cod = re.sub(r'_(OA|OC)\.set$','',bn)
        cond = 'OA' if '_OA.set' in fpath else 'OC'
        try:
            data, ch_names, sfreq = read_eeg(fpath)
        except:
            continue
        nch = data.shape[0]
        if nch == 32: cm = CDMS_MAP
        elif nch == 19: cm = CSO_MAP
        elif nch >= 137: cm = BIOSEMI_MAP; keep=[i for i,n in enumerate(ch_names) if n in BIOSEMI_MAP]
        else: continue

        # Build ch_data dict
        ch_data = {}
        for on, tn in cm.items():
            if on in ch_names: ch_data[tn] = data[ch_names.index(on)]

        windows = []
        window_s = 2 * sfreq
        n_windows = data.shape[1] // window_s
        if n_windows < 1: continue
        for target_ch in CHANNEL_18:
            if target_ch not in ch_data: continue
        trimmed = {ch: d[:n_windows*window_s] for ch, d in ch_data.items()}

        bp_window = np.zeros((len(CHANNEL_18), len(BANDS), n_windows))
        for ci, target_ch in enumerate(CHANNEL_18):
            if target_ch not in trimmed: continue
            sig = trimmed[target_ch].reshape(n_windows, window_s)
            for bi, (lo, hi) in enumerate(BANDS.values()):
                f, psd = welch(sig, fs=sfreq, nperseg=window_s, noverlap=window_s//2, axis=1)
                mask = (f >= lo) & (f <= hi)
                if mask.sum() > 0:
                    bp_window[ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)

        bp = bp_window.mean(axis=2)
        raw.setdefault(cod,{})[cond] = bp

    subjects = {}
    for cod, conds in raw.items():
        bps = [v for v in conds.values() if v is not None]
        if not bps: continue
        avg = np.mean(bps, axis=0)
        if cod not in meta.index: continue
        subjects[cod] = {'bp': avg.flatten()}
        for t in ['MOT','TOTAL','COG','NPLAN','MOT_V4','COG_V1']:
            subjects[cod][t] = meta.loc[cod, t]

    cods = sorted(subjects.keys())
    X = np.array([subjects[c]['bp'] for c in cods])
    y = {t: np.array([subjects[c][t] for c in cods]) for t in ['MOT','TOTAL','COG','NPLAN','MOT_V4','COG_V1']}

    return cods, X, y


def main():
    print("="*70)
    print("  CORRELACION EEG BAND POWER <-> BARRATT IMPULSIVIDAD")
    print("="*70+"\n")

    cods, X, y = load_data()
    n = len(cods)
    print(f"Sujetos: {n}")
    print(f"Features: {X.shape[1]} (18 canales x 5 bandas)\n")

    TARGETS = ['MOT','MOT_V4','TOTAL','NPLAN','COG']
    TARGET_LABEL = {
        'MOT': 'Impulsividad Motora',
        'MOT_V4': 'MOT_V4 (items 8+13+16+21+23)',
        'TOTAL': 'TOTAL impulsividad',
        'NPLAN': 'No Planificacion',
        'COG': 'Impulsividad Cognitiva',
    }

    # ── 1. Spearman correlation per feature per target ────────────────
    print("-"*60)
    print("  1. SPEARMAN CORRELATIONS + FDR (Benjamini-Hochberg)")
    print("-"*60)

    for target in TARGETS:
        yt = y[target]
        rho = np.zeros(X.shape[1])
        pval = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            rho[j], pval[j] = stats.spearmanr(X[:, j], yt)

        # FDR correction across all 90 features
        _, pval_corrected, _, _ = multipletests(pval, method='fdr_bh')
        sig_mask = pval_corrected < 0.05
        n_sig = sig_mask.sum()

        print(f"\n  [{target}] {TARGET_LABEL[target]}")
        print(f"  Rango: [{yt.min():.0f}, {yt.max():.0f}],  Media: {yt.mean():.1f},  Std: {yt.std():.1f}")
        print(f"  Significativas (FDR<0.05): {n_sig}/{X.shape[1]}")

        if n_sig > 0:
            print(f"  Top features (+ = mas impulsividad con mas potencia):")
            top_idx = np.argsort(pval_corrected)[:min(n_sig, 15)]
            max_abs = max(abs(rho[i]) for i in top_idx)
            for i in top_idx:
                bar = '+' * max(1, int(abs(rho[i]) / max_abs * 30)) if rho[i] > 0 else '-' * max(1, int(abs(rho[i]) / max_abs * 30))
                print(f"    {FEATURE_NAMES[i]:25s}  rho={rho[i]:+.3f}  p(unc)={pval[i]:.3f}  p(fdr)={pval_corrected[i]:.3f}  {bar}")
        else:
            # Show top 5 by raw p-value even if none significant
            print(f"  Top 5 by raw p-value (none significant after FDR):")
            top_idx = np.argsort(pval)[:5]
            for i in top_idx:
                print(f"    {FEATURE_NAMES[i]:25s}  rho={rho[i]:+.3f}  p(unc)={pval[i]:.3f}  p(fdr)={pval_corrected[i]:.3f}")

    # ── 2. Ridge regression (LOSO) ───────────────────────────────────
    print(f"\n{'-'*60}")
    print("  2. RIDGE REGRESSION (Leave-One-Subject-Out)")
    print(f"{'-'*60}")

    sc = StandardScaler()
    X_norm = sc.fit_transform(X)

    for target in TARGETS:
        yt = y[target]
        logo = LeaveOneGroupOut()
        # Use subject identity as group
        all_pred, all_true = [], []
        for train_idx, test_idx in logo.split(X_norm, yt, groups=cods):
            X_tr, X_te = X_norm[train_idx], X_norm[test_idx]
            y_tr, y_te = yt[train_idx], yt[test_idx]
            # RidgeCV finds best alpha internally
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
            ridge.fit(X_tr, y_tr)
            pred = ridge.predict(X_te)[0]
            all_pred.append(pred)
            all_true.append(y_te[0])

        r2 = r2_score(all_true, all_pred)
        mae = mean_absolute_error(all_true, all_pred)
        r_pearson, p_pearson = stats.pearsonr(all_true, all_pred)
        r_spearman, p_spearman = stats.spearmanr(all_true, all_pred)

        print(f"\n  [{target}] {TARGET_LABEL[target]}")
        print(f"    R2      = {r2:+.3f}")
        print(f"    MAE     = {mae:.3f}  (baseline: y_mean={yt.mean():.1f})")
        print(f"    Pearson = {r_pearson:+.3f}  (p={p_pearson:.3f})")
        print(f"    Spearman= {r_spearman:+.3f}  (p={p_spearman:.3f})")
        print(f"    Relative MAE = {mae / yt.std():.2f} sigma")

    # ── 3. Summary table ─────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  3. SUMMARY")
    print(f"{'-'*60}")

    print(f"\n  {'Target':<12s} {'R2':>8s} {'MAE':>8s} {'Pearson r':>10s} {'Spearman r':>12s} {'FDR<0.05':>10s}")
    print(f"  {'-'*60}")

    for target in TARGETS:
        yt = y[target]
        pvals = np.array([stats.spearmanr(X[:, j], yt)[1] for j in range(X.shape[1])])
        _, pval_corr, _, _ = multipletests(pvals, method='fdr_bh')
        n_sig = (pval_corr < 0.05).sum()

        logo = LeaveOneGroupOut()
        all_pred, all_true = [], []
        for train_idx, test_idx in logo.split(X_norm, yt, groups=cods):
            X_tr, X_te = X_norm[train_idx], X_norm[test_idx]
            y_tr, y_te = yt[train_idx], yt[test_idx]
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
            ridge.fit(X_tr, y_tr)
            all_pred.append(ridge.predict(X_te)[0])
            all_true.append(y_te[0])

        r2 = r2_score(all_true, all_pred)
        mae = mean_absolute_error(all_true, all_pred)
        r_p, _ = stats.pearsonr(all_true, all_pred)
        r_s, _ = stats.spearmanr(all_true, all_pred)

        print(f"  {target:<12s} {r2:>+8.3f} {mae:>8.2f} {r_p:>+10.3f} {r_s:>+12.3f} {n_sig:>10d}")

    # ── 4. Top features for MOT specifically ─────────────────────────
    print(f"\n{'-'*60}")
    print("  4. TOP FEATURES FOR MOT (best candidate)")
    print(f"{'-'*60}")

    yt = y['MOT']
    top_rho = []
    for j in range(X.shape[1]):
        rho, p = stats.spearmanr(X[:, j], yt)
        top_rho.append((rho, p, FEATURE_NAMES[j]))
    top_rho.sort(key=lambda x: abs(x[0]), reverse=True)

    # Group by band
    band_rho = {b: [] for b in BANDS}
    for rho, p, name in top_rho:
        band = name.split('_')[-1]
        band_rho[band].append(abs(rho))
    print(f"\n  Mean |rho| per band (MOT):")
    for b in ['delta','theta','alpha','beta','gamma']:
        vals = band_rho[b]
        print(f"    {b:8s}: mean|rho|={np.mean(vals):.3f}  max|rho|={np.max(vals):.3f}")

    # Group by region
    regions = {
        'Frontal': ['FP1','FP2','F3','F4','F7','F8','FZ'],
        'Central': ['C3','C4','T7','T8'],
        'Parietal': ['P3','P4','P7','P8','PZ'],
        'Occipital': ['O1','O2'],
    }
    region_rho = {}
    for reg, chs in regions.items():
        vals = []
        for rho, p, name in top_rho:
            ch = name.split('_')[0]
            if ch in chs: vals.append(abs(rho))
        region_rho[reg] = vals
    print(f"\n  Mean |rho| per region (MOT):")
    for reg, vals in region_rho.items():
        print(f"    {reg:12s}: mean|rho|={np.mean(vals):.3f}  max|rho|={np.max(vals):.3f}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
