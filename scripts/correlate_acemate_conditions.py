"""
Spearman correlations EEG band power vs Barratt targets – OA / OC / OA+OC / reactivity
Simple + partial (controlled by School_year) + FDR
"""
import glob, os, re, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from statsmodels.stats.multitest import multipletests

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

FEATURE_NAMES = [f"{ch}_{bn}" for ch in CHANNEL_18 for bn in BANDS]

REGIONS = {
    'Frontal':  ['FP1','FP2','F3','F4','F7','F8','FZ'],
    'Central':  ['C3','C4','T7','T8'],
    'Parietal': ['P3','P4','P7','P8','PZ'],
    'Occipital':['O1','O2'],
}

TARGETS = ['MOT','MOT_V4','COG','NPLAN']
TARGET_LABEL = {
    'MOT':'Impulsividad Motora',
    'MOT_V4':'MOT_V4 (items 8+13+16+21+23)',
    'COG':'Impulsividad Cognitiva',
    'NPLAN':'No Planificacion',
}


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


def partial_spearman(x, y, z):
    """Spearman partial correlation between x and y, controlling for z."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 5:
        return np.nan, 1.0
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    rx_res = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry_res = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(rx_res, ry_res)


def load_data():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))

    bp_data = {}
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
        elif nch >= 137:
            cm = BIOSEMI_MAP
            keep = [i for i,n in enumerate(ch_names) if n in BIOSEMI_MAP]
            data = data[keep]; ch_names = [ch_names[i] for i in keep]
        else: continue

        ch_data = {}
        for on, tn in cm.items():
            if on in ch_names: ch_data[tn] = data[ch_names.index(on)]

        window_s = 2 * sfreq
        n_windows = data.shape[1] // window_s
        if n_windows < 1: continue

        bp = np.zeros((len(CHANNEL_18), len(BANDS), n_windows))
        for ci, target_ch in enumerate(CHANNEL_18):
            if target_ch not in ch_data: continue
            sig = ch_data[target_ch][:n_windows * window_s].reshape(n_windows, window_s)
            for bi, (lo, hi) in enumerate(BANDS.values()):
                f, psd = welch(sig, fs=sfreq, nperseg=window_s, noverlap=window_s//2, axis=1)
                mask = (f >= lo) & (f <= hi)
                if mask.sum() > 0:
                    bp[ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)

        bp_avg = bp.mean(axis=2)
        bp_data.setdefault(cod,{})[cond] = bp_avg

    cods = sorted(bp_data.keys())
    X_oa, X_oc, X_both = [], [], []
    y_vals = {t: [] for t in TARGETS}
    school_years = []
    valid_cods = []

    for cod in cods:
        if cod not in meta.index: continue
        oa = bp_data[cod].get('OA', None)
        oc = bp_data[cod].get('OC', None)
        if oa is None or oc is None: continue  # need both conditions

        X_oa.append(oa.flatten())
        X_oc.append(oc.flatten())
        X_both.append((oa + oc).flatten() / 2)

        sy = meta.loc[cod, 'School year']
        school_years.append(float(sy) if not pd.isna(sy) else np.nan)

        for t in TARGETS:
            v = meta.loc[cod, t]
            y_vals[t].append(float(v) if not pd.isna(v) else np.nan)
        valid_cods.append(cod)

    school_years = np.array(school_years)
    return (np.array(X_oa), np.array(X_oc), np.array(X_both),
            {t: np.array(y_vals[t]) for t in TARGETS},
            valid_cods, school_years)


def summarise_correlations(rho, pval, pval_fdr, target_name):
    n_sig = (pval_fdr < 0.05).sum()
    n_unc = (pval < 0.05).sum()
    mean_abs = np.mean(np.abs(rho))
    max_abs = np.max(np.abs(rho))
    best_i = np.argmax(np.abs(rho))
    best_feat = FEATURE_NAMES[best_i]
    best_r = rho[best_i]
    best_p = pval[best_i]
    return {
        'n_fdr': n_sig, 'n_unc': n_unc, 'mean|rho|': mean_abs,
        'max|rho|': max_abs, 'best_feat': best_feat,
        'best_r': best_r, 'best_p': best_p,
    }


def print_band_region_analysis(rho_all):
    print(f"  Mean |rho| per band:")
    for bn in BANDS:
        idxs = [i for i, fn in enumerate(FEATURE_NAMES) if fn.endswith(f"_{bn}")]
        vals = [abs(rho_all[j]) for j in idxs]
        print(f"    {bn:8s}: mean={np.mean(vals):.3f}  max={np.max(vals):.3f}")

    print(f"  Mean |rho| per region:")
    for reg, chs in REGIONS.items():
        vals = [abs(rho_all[i]) for i, fn in enumerate(FEATURE_NAMES)
                if any(fn.startswith(ch + '_') for ch in chs)]
        print(f"    {reg:12s}: mean={np.mean(vals):.3f}  max={np.max(vals):.3f}")


def main():
    print("=" * 70)
    print("  CORRELACIONES EEG BAND POWER — OA / OC / OA+OC / REACTIVIDAD")
    print("  Simples + parciales (control: School year) + FDR")
    print("=" * 70 + "\n")

    X_oa, X_oc, X_both, y_dict, cods, school = load_data()
    n = X_oa.shape[0]
    print(f"Sujetos: {n}")
    print(f"Features: {X_oa.shape[1]} (18ch x 5 bands)\n")

    conditions = {
        'OA':    X_oa,
        'OC':    X_oc,
        'OA+OC': X_both,
        'OC-OA': X_oc - X_oa,
    }

    for cond_name, X in conditions.items():
        print("=" * 70)
        print(f"  CONDICION: {cond_name} ({'reactividad' if cond_name == 'OC-OA' else 'band power'})")
        print("=" * 70)

        for target in TARGETS:
            yt = y_dict[target]
            valid = ~np.isnan(yt)
            Xv, yv = X[valid], yt[valid]

            print(f"\n  --- [{target}] {TARGET_LABEL[target]} ---")
            print(f"  Rango: [{yv.min():.0f}, {yv.max():.0f}], mean={yv.mean():.1f}, std={yv.std():.1f}")

            # Simple Spearman
            rho_simple = np.zeros(X.shape[1])
            pval_simple = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                rho_simple[j], pval_simple[j] = stats.spearmanr(Xv[:, j], yv)

            _, p_fdr_simple, _, _ = multipletests(pval_simple, method='fdr_bh')
            s = summarise_correlations(rho_simple, pval_simple, p_fdr_simple, target)

            print(f"  > SIMPLE: FDR<0.05: {s['n_fdr']}/90 | p<0.05: {s['n_unc']}/90")
            print(f"    mean|rho|={s['mean|rho|']:.3f}  max|rho|={s['max|rho|']:.3f}  "
                  f"best: {s['best_feat']} (rho={s['best_r']:+.3f}, p={s['best_p']:.3f})")
            if s['n_unc'] > 0:
                top5 = np.argsort(pval_simple)[:min(5, s['n_unc'])]
                for i in top5:
                    print(f"      {FEATURE_NAMES[i]:25s}  rho={rho_simple[i]:+.3f}  "
                          f"p(unc)={pval_simple[i]:.3f}  p(fdr)={p_fdr_simple[i]:.3f}")

            # Partial Spearman (controlling for School year)
            school_valid = school[valid]
            rho_partial = np.zeros(X.shape[1])
            pval_partial = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                rho_partial[j], pval_partial[j] = partial_spearman(Xv[:, j], yv, school_valid)

            _, p_fdr_partial, _, _ = multipletests(pval_partial, method='fdr_bh')
            p = summarise_correlations(rho_partial, pval_partial, p_fdr_partial, target)

            print(f"  > PARCIAL (ctrl=School year): FDR<0.05: {p['n_fdr']}/90 | p<0.05: {p['n_unc']}/90")
            print(f"    mean|rho|={p['mean|rho|']:.3f}  max|rho|={p['max|rho|']:.3f}  "
                  f"best: {p['best_feat']} (rho={p['best_r']:+.3f}, p={p['best_p']:.3f})")
            if p['n_unc'] > 0:
                top5 = np.argsort(pval_partial)[:min(5, p['n_unc'])]
                for i in top5:
                    print(f"      {FEATURE_NAMES[i]:25s}  rho={rho_partial[i]:+.3f}  "
                          f"p(unc)={pval_partial[i]:.3f}  p(fdr)={p_fdr_partial[i]:.3f}")

            # Band/region analysis for the best-performing of simple vs partial
            best_rhos = rho_simple if s['mean|rho|'] >= p['mean|rho|'] else rho_partial
            print_band_region_analysis(best_rhos)

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY — mean|rho| across all 90 features")
    print(f"{'='*70}")
    print(f"\n  {'Target':<10s} {'OA':>8s} {'OC':>8s} {'OA+OC':>8s} {'OC-OA':>8s} "
          f"{'OA_p':>8s} {'OC_p':>8s} {'OA+OC_p':>8s}")
    print(f"  {'-'*78}")

    for target in TARGETS:
        yt = y_dict[target]
        valid = ~np.isnan(yt)
        xv = {}
        for cn in ['OA','OC','OA+OC','OC-OA']:
            xv[cn] = conditions[cn][valid]

        row = [target]
        for cn in ['OA','OC','OA+OC','OC-OA']:
            rhos = [stats.spearmanr(xv[cn][:, j], yt[valid])[0] for j in range(X.shape[1])]
            row.append(f"{np.mean(np.abs(rhos)):.4f}")
        # Partial versions
        for cn in ['OA','OC','OA+OC']:
            rhos = [partial_spearman(xv[cn][:, j], yt[valid], school[valid])[0]
                    for j in range(X.shape[1])]
            row.append(f"{np.mean(np.abs(rhos)):.4f}")

        print(f"  {row[0]:<10s} {row[1]:>8s} {row[2]:>8s} {row[3]:>8s} {row[4]:>8s} "
              f"{row[5]:>8s} {row[6]:>8s} {row[7]:>8s}")

    print(f"\n  OA_p / OC_p / OA+OC_p = mean|rho| de correlaciones parciales (ctrl=School year)")

    print(f"\n{'='*70}")
    print("  DONE")
    print("="*70)


if __name__ == '__main__':
    main()
