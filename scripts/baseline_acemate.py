import glob, os, re, warnings
import numpy as np
import pandas as pd
from scipy.signal import welch

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
RANDOM_STATE = 42
SFREQ = 250

BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}

CHANNEL_18 = ['FP1','F7','T7','P7','F3','C3','P3','O1','FZ','PZ',
              'FP2','F8','T8','P8','F4','C4','P4','O2']

PO_IDX = [7,8,9,15,16]  # P3,P4,PZ,O1,O2 indices
F3_IDX, F4_IDX = 4, 14
NCH, NB = len(CHANNEL_18), len(BANDS)

CDMS_MAP = {n:n for n in CHANNEL_18}
CSO_MAP = {'FP1-Cz':'FP1','FP2-Cz':'FP2','FZ-Cz':'FZ','PZ-Cz':'PZ',
    'F3-Cz':'F3','F4-Cz':'F4','F7-Cz':'F7','F8-Cz':'F8',
    'C3-Cz':'C3','C4-Cz':'C4','T3-Cz':'T7','T4-Cz':'T8','T5-Cz':'P7','T6-Cz':'P8',
    'P3-Cz':'P3','P4-Cz':'P4','O1-Cz':'O1','O2-Cz':'O2'}
BIOSEMI_MAP = {'A1':'FP1','A5':'F3','A7':'F7','A12':'C3','A15':'T7','A18':'P3',
    'A21':'P7','A23':'O1','B1':'FP2','B3':'FZ','B5':'F4','B7':'F8',
    'B12':'C4','B15':'T8','B18':'P4','B21':'P8','B24':'O2','B27':'PZ'}


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


def extract_features(data,ch_names,sfreq,ch_map):
    w=2*sfreq; nw=data.shape[1]//w
    if nw<1: return None
    d=data[:,:nw*w]; bp=np.zeros((NCH,NB,nw))
    cd={}
    for on,tn in ch_map.items():
        if on in ch_names: cd[tn]=d[ch_names.index(on)]
    for ci,target in enumerate(CHANNEL_18):
        if target not in cd: continue
        X=cd[target].reshape(nw,w)
        for bi,(lo,hi) in enumerate(BANDS.values()):
            f,P=welch(X,fs=sfreq,nperseg=w,noverlap=w//2,axis=1)
            m=(f>=lo)&(f<=hi)
            if m.sum()>0: bp[ci,bi]=np.trapezoid(P[:,m],f[m],axis=1)
    return bp.mean(axis=2)


def build_feature_vectors(raw_features):
    cods, X_clf, X_reg = [], [], []
    oa_bps, oc_bps = {}, {}
    for cod, conds in raw_features.items():
        oa = conds.get('OA', None)
        oc = conds.get('OC', None)
        if oa is None and oc is None: continue
        cods.append(cod)

        oa_bp = oa if oa is not None else np.zeros((NCH,NB))
        oc_bp = oc if oc is not None else np.zeros((NCH,NB))
        avg = (oa_bp + oc_bp) / 2

        # Classification: averaged OA+OC, 90 features
        X_clf.append(avg.flatten())

        # Regression: OA+OC separated + derived
        oa_f = oa_bp.flatten()
        oc_f = oc_bp.flatten()
        pd_=avg[:,0].mean(); pt_=avg[:,1].mean(); pa_=avg[:,2].mean()
        pb_=avg[:,3].mean(); pg_=avg[:,4].mean()
        eps=1e-10
        ratios = np.array([pa_/(pd_+eps), pt_/(pb_+eps), (pt_+pa_)/(pd_+pb_+eps), pt_/(pa_+eps)])
        peak_alpha = avg[PO_IDX,2].mean()
        faa = np.array([oa_bp[F4_IDX,2]-oa_bp[F3_IDX,2], oc_bp[F4_IDX,2]-oc_bp[F3_IDX,2]])
        X_reg.append(np.concatenate([oa_f, oc_f, ratios, [peak_alpha], faa]))
    return cods, np.array(X_clf), np.array(X_reg)


def main():
    print("="*70)
    print("  ACEMATE LOSO - EEG band power + RandomForest")
    print("="*70+'\n')

    meta=pd.read_excel(META_PATH)
    meta['MOT_V4']=meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1']=meta[['3.','6.']].sum(axis=1)
    meta=meta.set_index('Cod')

    all_set=sorted(glob.glob(os.path.join(EEG_DIR,'*.set')))
    print(f"Archivos: {len(all_set)}")

    raw_features={}; nok=0
    for fpath in all_set:
        bn=os.path.basename(fpath)
        cod=re.sub(r'_(OA|OC)\.set$','',bn); cond='OA' if '_OA.set' in fpath else 'OC'
        try: data,ch_names,sfreq=read_eeg(fpath)
        except Exception as e: print(f"  FAIL {bn}: {e}"); continue
        nch=data.shape[0]
        if nch==32: cm=CDMS_MAP
        elif nch==19: cm=CSO_MAP
        elif nch>=137:
            cm=BIOSEMI_MAP; keep=[i for i,n in enumerate(ch_names) if n in BIOSEMI_MAP]
            data=data[keep]; ch_names=[ch_names[i] for i in keep]
        else: continue
        bp=extract_features(data,ch_names,sfreq,cm)
        if bp is None: continue
        raw_features.setdefault(cod,{})[cond]=bp; nok+=1

    print(f"Procesados: {nok}/{len(all_set)}  Sujetos: {len(raw_features)}\n")

    cods,X_clf,X_reg=build_feature_vectors(raw_features)
    n=len(cods); print(f"X_clf: {X_clf.shape}, X_reg: {X_reg.shape}")

    sc_clf=StandardScaler(); X_clf=sc_clf.fit_transform(X_clf)
    sc_reg=StandardScaler(); X_reg=sc_reg.fit_transform(X_reg)

    def is_valid(v): return v is not None and not (isinstance(v,float) and np.isnan(v))

    # ── LOSO Clasificacion ──────────────────────────────────────────
    print("\n"+"-"*60)
    print("  LOSO CLASIFICACION: Tipo (high_imp / low_imp)")
    print("-"*60)
    valid_idx=[i for i,c in enumerate(cods) if c in meta.index and is_valid(meta.loc[c,'Tipo'])]
    nc=len(valid_idx)
    print(f"  Sujetos: {nc}")
    all_true, all_pred = [], []
    for ti in valid_idx:
        cod=cods[ti]
        train_val=[i for i in valid_idx if i!=ti]
        train_val_cods=[cods[i] for i in train_val]
        tr_y=np.array([meta.loc[cods[i],'Tipo'] for i in train_val])
        gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=RANDOM_STATE+ti)
        inner=list(gss.split(X_clf[train_val],tr_y,groups=train_val_cods))
        tr_i,vl_i=inner[0]; tr=[train_val[i] for i in tr_i]; vl=[train_val[i] for i in vl_i]
        clf=RandomForestClassifier(n_estimators=500,random_state=RANDOM_STATE)
        clf.fit(X_clf[tr],[meta.loc[cods[i],'Tipo'] for i in tr])
        pred=clf.predict(X_clf[[ti]])
        all_true.append(meta.loc[cod,'Tipo']); all_pred.append(pred[0])
    acc=accuracy_score(all_true,all_pred)
    bal=balanced_accuracy_score(all_true,all_pred)
    print(f"  Correctos: {sum(np.array(all_true)==np.array(all_pred))}/{nc}")
    print(f"  >>> Tipo: acc = {acc:.3f}, bal_acc = {bal:.3f}")

    # ── LOSO Regresion ──────────────────────────────────────────────
    for tgt_name in ['MOT','COG','MOT_V4','COG_V1']:
        valid_idx=[i for i,c in enumerate(cods) if c in meta.index and is_valid(meta.loc[c,tgt_name])]
        nr=len(valid_idx)
        if nr<5: print(f"\n  SKIP {tgt_name}: {nr} subjects"); continue
        print(f"\n{'-'*60}")
        print(f"  LOSO REGRESION: {tgt_name}")
        print("-"*60)
        print(f"  Sujetos: {nr}")
        all_true, all_pred = [], []
        for ti in valid_idx:
            cod=cods[ti]
            train_val=[i for i in valid_idx if i!=ti]
            train_val_cods=[cods[i] for i in train_val]
            tr_y=np.array([meta.loc[cods[i],tgt_name] for i in train_val],dtype=float)
            gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=RANDOM_STATE+ti)
            inner=list(gss.split(X_reg[train_val],tr_y,groups=train_val_cods))
            tr_i,vl_i=inner[0]; tr=[train_val[i] for i in tr_i]; vl=[train_val[i] for i in vl_i]
            reg=RandomForestRegressor(n_estimators=500,random_state=RANDOM_STATE)
            reg.fit(X_reg[tr],[meta.loc[cods[i],tgt_name] for i in tr])
            pred=reg.predict(X_reg[[ti]])[0]
            true_val=meta.loc[cod,tgt_name]
            all_true.append(true_val); all_pred.append(pred)
            print(f"  {cod}: true={true_val:.1f} pred={pred:.1f}  (|err|={abs(pred-true_val):.1f})")
        r2=r2_score(all_true,all_pred)
        mae=mean_absolute_error(all_true,all_pred)
        print(f"  >>> {tgt_name}: R2 = {r2:.3f}, MAE = {mae:.3f}")

    print(f"\n{'='*70}")
    print("  DONE")
    print("="*70)

if __name__=='__main__':
    main()
