"""Quick XGBoost test on ACEMATE band power — COG regression, LOSO"""
import glob, os, re, warnings, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
SFREQ = 250
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
CH_18 = ['FP1','F7','T7','P7','F3','C3','P3','O1','FZ','PZ',
         'FP2','F8','T8','P8','F4','C4','P4','O2']
CDMS_MAP = {n:n for n in CH_18}
CSO_MAP = {'FP1-Cz':'FP1','FP2-Cz':'FP2','FZ-Cz':'FZ','PZ-Cz':'PZ',
    'F3-Cz':'F3','F4-Cz':'F4','F7-Cz':'F7','F8-Cz':'F8',
    'C3-Cz':'C3','C4-Cz':'C4','T3-Cz':'T7','T4-Cz':'T8','T5-Cz':'P7','T6-Cz':'P8',
    'P3-Cz':'P3','P4-Cz':'P4','O1-Cz':'O1','O2-Cz':'O2'}
BIOSEMI_MAP = {'A1':'FP1','A5':'F3','A7':'F7','A12':'C3','A15':'T7','A18':'P3',
    'A21':'P7','A23':'O1','B1':'FP2','B3':'FZ','B5':'F4','B7':'F8',
    'B12':'C4','B15':'T8','B18':'P4','B21':'P8','B24':'O2','B27':'PZ'}

TARGETS = ['COG','MOT','MOT_V4','NPLAN']
TARGET_LABEL = {'COG':'Imp Cognitiva','MOT':'Imp Motora','MOT_V4':'MOT_V4 (8+13+16+21+23)','NPLAN':'No Planificacion'}

def load_and_extract():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta = meta.set_index('Cod')
    all_set = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    
    bp_raw = {}
    for fpath in all_set:
        bn = os.path.basename(fpath)
        cod = re.sub(r'_(OA|OC)\.set$','',bn)
        cond = 'OA' if '_OA.set' in fpath else 'OC'
        try:
            import mne
            raw = mne.io.read_raw_eeglab(fpath, preload=True, verbose=False)
            data, ch_names, sfreq = raw.get_data(), list(raw.ch_names), int(raw.info['sfreq'])
        except:
            try:
                import h5py
                fdt = fpath.replace('.set','.fdt')
                with h5py.File(fpath,'r') as h5:
                    nch=int(h5['nbchan'][0][0]); pnts=int(h5['pnts'][0][0]); sr=int(h5['srate'][0][0])
                    labs=h5['chanlocs']['labels'][:]
                    ch_names=[]
                    for l in labs:
                        if isinstance(l,bytes): ch_names.append(l.decode('utf-8',errors='replace').strip('\x00'))
                        elif hasattr(l,'tobytes'): ch_names.append(l.tobytes().decode('utf-8',errors='replace').strip('\x00'))
                        else: ch_names.append(str(l))
                data=np.fromfile(fdt,dtype=np.float32).reshape(nch,pnts,order='F')
                sfreq=sr
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
        
        ws = 2 * sfreq
        nw = data.shape[1] // ws
        if nw < 1: continue
        bp = np.zeros((len(CH_18), len(BANDS), nw))
        for ci, target_ch in enumerate(CH_18):
            if target_ch not in ch_data: continue
            sig = ch_data[target_ch][:nw*ws].reshape(nw, ws)
            for bi, (lo, hi) in enumerate(BANDS.values()):
                f, psd = welch(sig, fs=sfreq, nperseg=ws, noverlap=ws//2, axis=1)
                mask = (f >= lo) & (f <= hi)
                if mask.sum() > 0: bp[ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)
        bp_raw.setdefault(cod,{})[cond] = bp.mean(axis=2)  # [18, 5]
    
    cods = sorted(bp_raw.keys())
    X_oa, X_oc, X_both, X_react = [], [], [], []
    y_dict = {t: [] for t in TARGETS}
    valid_cods = []
    
    for cod in cods:
        if cod not in meta.index: continue
        oa = bp_raw[cod].get('OA', None)
        oc = bp_raw[cod].get('OC', None)
        if oa is None or oc is None: continue
        X_oa.append(oa.flatten())
        X_oc.append(oc.flatten())
        X_both.append((oa+oc).flatten()/2)
        X_react.append((oc-oa).flatten())
        for t in TARGETS:
            v = meta.loc[cod, t]
            y_dict[t].append(float(v) if not pd.isna(v) else np.nan)
        valid_cods.append(cod)
    
    return (np.array(X_oa), np.array(X_oc), np.array(X_both), np.array(X_react),
            {t: np.array(y_dict[t]) for t in TARGETS}, valid_cods)


def loso_run(X, y_true, cods, model, sc):
    all_t, all_p = [], []
    for ti in range(len(cods)):
        train_mask = np.ones(len(cods), bool); train_mask[ti] = False
        X_tr, X_te = X[train_mask], X[[ti]]
        y_tr, y_te = y_true[train_mask], y_true[[ti]]
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)[0]
        all_t.append(y_te[0]); all_p.append(pred)
    t, p = np.array(all_t), np.array(all_p)
    mae = mean_absolute_error(t, p)
    r2 = r2_score(t, p)
    nrmse = np.sqrt(np.mean((t-p)**2)) / (t.std() + 1e-10)
    sr, sp = stats.spearmanr(t, p)
    pr, pp = stats.pearsonr(t, p)
    return {'mae': mae, 'r2': r2, 'nrmse': nrmse, 'spear': sr, 'spear_p': sp, 'pear': pr, 'pear_p': pp}


def main():
    print("=" * 70)
    print("  XGBoost vs Ridge - Band Power EEG -> COG / MOT / NPLAN (LOSO)")
    print("=" * 70 + "\n")
    
    X_oa, X_oc, X_both, X_react, y_dict, cods = load_and_extract()
    n = len(cods)
    print(f"Sujetos: {n}, Features: {X_oa.shape[1]} (18ch x 5bands)\n")
    
    conds = {'OC': X_oc, 'OA+OC': X_both, 'OC-OA': X_react, 'OA': X_oa}
    
    for target in TARGETS:
        yt = y_dict[target]
        valid = ~np.isnan(yt)
        if valid.sum() < 5: continue
        yv = yt[valid]
        cods_v = [c for c, v in zip(cods, valid) if v]
        
        print("-" * 60)
        print(f"  [{target}] {TARGET_LABEL[target]}: n={valid.sum()}, "
              f"range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print("-" * 60)
        
        # Best condition from correlation analysis: OC for most targets
        for cond_name, X_cond in conds.items():
            Xv = X_cond[valid]
            if Xv.shape[0] < 5: continue
            
            # Ridge baseline
            sc_r = StandardScaler()
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
            m_ridge = loso_run(Xv, yv, cods_v, ridge, sc_r)
            
            # XGBoost
            sc_x = StandardScaler()
            xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                               objective='reg:squarederror')
            m_xgb = loso_run(Xv, yv, cods_v, xgb, sc_x)
            
            print(f"  {cond_name:6s} | Ridge: r2={m_ridge['r2']:+.3f}  mae={m_ridge['mae']:.2f}  "
                  f"spear={m_ridge['spear']:+.3f}  nrmse={m_ridge['nrmse']:.2f}")
            print(f"  {'':6s} | XGBoost: r2={m_xgb['r2']:+.3f}  mae={m_xgb['mae']:.2f}  "
                  f"spear={m_xgb['spear']:+.3f}  nrmse={m_xgb['nrmse']:.2f}")
            
            # Highlight improvement
            if m_xgb['spear'] > m_ridge['spear']:
                delta = m_xgb['spear'] - m_ridge['spear']
                print(f"  {'':6s} | XGB > Ridge: spear {delta:+.3f}")
            else:
                delta = m_ridge['spear'] - m_xgb['spear']
                if delta > 0.05:
                    print(f"  {'':6s} | Ridge > XGB: spear {delta:+.3f}")
        print()
    
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
