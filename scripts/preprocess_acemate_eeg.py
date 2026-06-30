"""
ACEMATE EEG preprocessing: Load .set files, extract band power features.
Uses OC-only EEG, 2s windows, band power across 18 channels.
"""
import glob, os, re, warnings
import numpy as np, pandas as pd
from scipy.signal import welch
import mne

warnings.filterwarnings('ignore')

EEG_DIR = 'data/raw/acemate/eeg_speech/eeg_not_locch'
META_PATH = 'data/raw/acemate/eeg_speech/metadata.xlsx'
OUT_PATH = 'data/processed/acemate_eeg_features.npz'
SFREQ = 250; WINDOW_SEC = 2.0; OVERLAP = 0.5
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

FEATURE_NAMES = [f"{ch}_{bn}" for ch in CH_18 for bn in BANDS]

def read_eeg(filepath):
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        return raw.get_data(), list(raw.ch_names), int(raw.info['sfreq'])
    except Exception:
        import h5py
        fdt = filepath.replace('.set','.fdt')
        with h5py.File(filepath,'r') as h5:
            nch=int(h5['nbchan'][0][0]); pnts=int(h5['pnts'][0][0]); sr=int(h5['srate'][0][0])
            labs=h5['chanlocs']['labels'][:]; ch_names=[]
            for l in labs:
                if isinstance(l,bytes): ch_names.append(l.decode('utf-8','replace').strip('\x00'))
                elif hasattr(l,'tobytes'): ch_names.append(l.tobytes().decode('utf-8','replace').strip('\x00'))
                else: ch_names.append(str(l))
        data=np.fromfile(fdt,dtype=np.float32).reshape(nch,pnts,order='F')
        return data,ch_names,sr

def extract_bandpower_windows(data_18, sfreq):
    """Returns [n_windows, 18, 5] band power matrix."""
    ws = int(WINDOW_SEC * sfreq); stride = int(ws * (1 - OVERLAP))
    n_w = (data_18.shape[1] - ws) // stride + 1
    if n_w < 1: return None
    windows = np.lib.stride_tricks.sliding_window_view(data_18, ws, axis=1)[:, ::stride].transpose(1,0,2)
    windows = windows[:n_w].astype(np.float32)
    bp = np.zeros((n_w, len(CH_18), len(BANDS)), dtype=np.float32)
    for ci, _ in enumerate(CH_18):
        sig = windows[:, ci, :]
        for bi, (lo, hi) in enumerate(BANDS.values()):
            f, psd = welch(sig, fs=sfreq, nperseg=ws, noverlap=ws//2, axis=1)
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0:
                bp[:, ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)
    return bp

def main():
    print("="*70)
    print("  ACEMATE EEG PREPROCESSING — band power features")
    print("="*70)

    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1'] = meta[['3.','6.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    subjects_data = {}
    skipped = []

    for fpath in all_files:
        bn = os.path.basename(fpath)
        cod = re.sub(r'_(OA|OC)\.set$','',bn)
        cond = 'OA' if '_OA.set' in fpath else 'OC'
        try:
            data, ch_names, sfreq = read_eeg(fpath)
        except:
            skipped.append(bn)
            continue
        nch = data.shape[0]
        if nch == 32: cm = CDMS_MAP
        elif nch == 19: cm = CSO_MAP
        elif nch >= 137:
            cm = BIOSEMI_MAP
            keep = [i for i,n in enumerate(ch_names) if n in BIOSEMI_MAP]
            data = data[keep]; ch_names = [ch_names[i] for i in keep]
        else:
            skipped.append(bn)
            continue
        sel = []
        for ch in CH_18:
            found = False
            for on, tn in cm.items():
                if tn == ch and on in ch_names: sel.append(ch_names.index(on)); found = True; break
            if not found and ch in ch_names: sel.append(ch_names.index(ch))
        if len(sel) < 18:
            skipped.append(bn)
            continue
        data_18 = data[sel]
        data_18 = (data_18 - data_18.mean(axis=1, keepdims=True)) / (data_18.std(axis=1, keepdims=True) + 1e-10)
        bp = extract_bandpower_windows(data_18, sfreq)
        if bp is None: continue
        subjects_data.setdefault(cod, {})[cond] = bp

    print(f"\nFiles processed: {len(all_files) - len(skipped)}/{len(all_files)}")
    print(f"Skipped: {len(skipped)} files")

    # Build output arrays
    all_cods = sorted(subjects_data.keys())
    print(f"Subjects with EEG: {len(all_cods)}")

    X_oa, X_oc, X_both, X_react = [], [], [], []
    valid_cods = []
    targets = {t: [] for t in ['Tipo','MOT','COG','MOT_V4','NPLAN','COG_V1']}

    for cod in all_cods:
        if cod not in meta.index: continue
        oa = subjects_data[cod].get('OA', None)
        oc = subjects_data[cod].get('OC', None)
        if oa is None or oc is None: continue
        # Mean across windows -> [18, 5]
        oa_bp = oa.mean(axis=0)
        oc_bp = oc.mean(axis=0)
        X_oa.append(oa_bp.flatten())
        X_oc.append(oc_bp.flatten())
        X_both.append(((oa_bp + oc_bp) / 2).flatten())
        X_react.append((oc_bp - oa_bp).flatten())
        valid_cods.append(cod)
        for t in targets:
            v = meta.loc[cod, t]
            targets[t].append(float(v) if not pd.isna(v) else np.nan)

    X_oa = np.array(X_oa); X_oc = np.array(X_oc)
    X_both = np.array(X_both); X_react = np.array(X_react)
    y = {t: np.array(targets[t]) for t in targets}

    print(f"Valid subjects (with OA+OC + metadata): {len(valid_cods)}")
    print(f"Feature shape: {X_oa.shape[1]} (18ch x 5bands)")

    # Save
    np.savez(OUT_PATH,
             subjects=np.array(valid_cods),
             X_oa=X_oa, X_oc=X_oc, X_both=X_both, X_react=X_react,
             y_NPLAN=y['NPLAN'], y_COG=y['COG'], y_MOT=y['MOT'],
             y_MOT_V4=y['MOT_V4'], y_COG_V1=y['COG_V1'], y_Tipo=y['Tipo'],
             feature_names=np.array(FEATURE_NAMES))
    print(f"\nSaved to: {OUT_PATH}")
    print(f"  X_oa: {X_oa.shape}")
    print(f"  X_oc: {X_oc.shape}")
    print(f"  X_both: {X_both.shape}")
    print(f"  X_react: {X_react.shape}")
    print(f"  targets: {[f'{t}={np.sum(~np.isnan(y[t]))} valid' for t in targets]}")


if __name__ == '__main__':
    main()
