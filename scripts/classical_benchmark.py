"""Classical ML benchmark: 3 linear + 3 non-linear on band power (LOSO)"""
import glob, os, re, warnings, sys
import numpy as np; import pandas as pd
from scipy import stats; from scipy.signal import welch
from sklearn.linear_model import RidgeCV, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
warnings.filterwarnings('ignore')
RSEED=42

EEG_DIR="data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH="data/raw/acemate/eeg_speech/metadata.xlsx"
SFREQ=250;BANDS={'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
CH_18=['FP1','F7','T7','P7','F3','C3','P3','O1','FZ','PZ','FP2','F8','T8','P8','F4','C4','P4','O2']
CDMS_MAP={n:n for n in CH_18}
CSO_MAP={'FP1-Cz':'FP1','FP2-Cz':'FP2','FZ-Cz':'FZ','PZ-Cz':'PZ','F3-Cz':'F3','F4-Cz':'F4','F7-Cz':'F7','F8-Cz':'F8','C3-Cz':'C3','C4-Cz':'C4','T3-Cz':'T7','T4-Cz':'T8','T5-Cz':'P7','T6-Cz':'P8','P3-Cz':'P3','P4-Cz':'P4','O1-Cz':'O1','O2-Cz':'O2'}
BIOSEMI_MAP={'A1':'FP1','A5':'F3','A7':'F7','A12':'C3','A15':'T7','A18':'P3','A21':'P7','A23':'O1','B1':'FP2','B3':'FZ','B5':'F4','B7':'F8','B12':'C4','B15':'T8','B18':'P4','B21':'P8','B24':'O2','B27':'PZ'}
TARGETS=['COG','MOT','MOT_V4','NPLAN']

MODELS={
    'Ridge': (RidgeCV(alphas=np.logspace(-3,3,50)), 'linear'),
    'Lasso': (Lasso(alpha=1.0, max_iter=10000, random_state=RSEED), 'linear'),
    'ElasticNet': (ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000, random_state=RSEED), 'linear'),
    'RF': (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RSEED, n_jobs=-1), 'nonlin'),
    'XGB': (XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=RSEED, objective='reg:squarederror', n_jobs=-1, verbosity=0), 'nonlin'),
    'SVRlin': (SVR(kernel='linear', C=1.0, epsilon=0.1), 'nonlin'),
}

def load_bp():
    meta=pd.read_excel(META_PATH);meta['MOT_V4']=meta[['8.','13.','16.','21.','23.']].sum(axis=1);meta=meta.set_index('Cod')
    all_set=sorted(glob.glob(os.path.join(EEG_DIR,'*.set')))
    bp_raw={}
    for fpath in all_set:
        bn=os.path.basename(fpath);cod=re.sub(r'_(OA|OC)\.set$','',bn);cond='OA' if '_OA.set' in fpath else 'OC'
        try:
            import mne;raw=mne.io.read_raw_eeglab(fpath,preload=True,verbose=False)
            data,ch_names,sfreq=raw.get_data(),list(raw.ch_names),int(raw.info['sfreq'])
        except:
            try:
                import h5py;fdt=fpath.replace('.set','.fdt')
                with h5py.File(fpath,'r') as h5:
                    nch=int(h5['nbchan'][0][0]);pnts=int(h5['pnts'][0][0]);sr=int(h5['srate'][0][0])
                    labs=h5['chanlocs']['labels'][:];ch_names=[]
                    for l in labs:
                        if isinstance(l,bytes):ch_names.append(l.decode('utf-8','replace').strip('\x00'))
                        elif hasattr(l,'tobytes'):ch_names.append(l.tobytes().decode('utf-8','replace').strip('\x00'))
                        else:ch_names.append(str(l))
                data=np.fromfile(fdt,dtype=np.float32).reshape(nch,pnts,order='F');sfreq=sr
            except:continue
        nch=data.shape[0]
        if nch==32:cm=CDMS_MAP
        elif nch==19:cm=CSO_MAP
        elif nch>=137:cm=BIOSEMI_MAP;keep=[i for i,n in enumerate(ch_names) if n in BIOSEMI_MAP];data=data[keep];ch_names=[ch_names[i] for i in keep]
        else:continue
        ch_data={}
        for on,tn in cm.items():
            if on in ch_names:ch_data[tn]=data[ch_names.index(on)]
        ws=2*sfreq;nw=data.shape[1]//ws
        if nw<1:continue
        bp=np.zeros((len(CH_18),len(BANDS),nw))
        for ci,target_ch in enumerate(CH_18):
            if target_ch not in ch_data:continue
            sig=ch_data[target_ch][:nw*ws].reshape(nw,ws)
            for bi,(lo,hi) in enumerate(BANDS.values()):
                f,psd=welch(sig,fs=sfreq,nperseg=ws,noverlap=ws//2,axis=1)
                mask=(f>=lo)&(f<=hi)
                if mask.sum()>0:bp[ci,bi]=np.trapezoid(psd[:,mask],f[mask],axis=1)
        bp_raw.setdefault(cod,{})[cond]=bp.mean(axis=2)
    cods=sorted(bp_raw.keys())
    Xo,Xc,Xb,Xr=[],[],[],[]
    yd={t:[] for t in TARGETS}
    for cod in cods:
        if cod not in meta.index:continue
        oa=bp_raw[cod].get('OA',None);oc=bp_raw[cod].get('OC',None)
        if oa is None or oc is None:continue
        Xo.append(oa.flatten());Xc.append(oc.flatten())
        Xb.append((oa+oc).flatten()/2);Xr.append((oc-oa).flatten())
        for t in TARGETS:
            v=meta.loc[cod,t];yd[t].append(float(v) if not pd.isna(v) else np.nan)
    return {k:np.array(v) for k,v in {'OA':Xo,'OC':Xc,'OA+OC':Xb,'OC-OA':Xr}.items()},\
           {t:np.array(yd[t]) for t in TARGETS}

def loso(X,y_true,model):
    all_t,all_p=[],[]
    sc=StandardScaler()
    for ti in range(len(y_true)):
        m=np.ones(len(y_true),bool);m[ti]=False
        Xtr,Xte=X[m],X[[ti]];ytr,yte=y_true[m],y_true[[ti]]
        Xtr_s=sc.fit_transform(Xtr);Xte_s=sc.transform(Xte)
        model.fit(Xtr_s,ytr)
        all_p.append(model.predict(Xte_s)[0]);all_t.append(yte[0])
    t,p=np.array(all_t),np.array(all_p)
    mae=mean_absolute_error(t,p);r2=r2_score(t,p)
    nrmse=np.sqrt(np.mean((t-p)**2))/(t.std()+1e-10)
    sr,sp=stats.spearmanr(t,p)
    return{'spear':sr,'spear_p':sp,'r2':r2,'mae':mae,'nrmse':nrmse}

def main():
    print("="*70)
    print("  CLASSICAL ML BENCHMARK: 3 linear + 3 non-linear")
    print("  Band Power EEG -> Barratt (LOSO, 34 subjects)")
    print("="*70)
    X_dict,yd=load_bp();n=len(next(iter(yd.values())))
    print(f"\nSubjects: {np.sum(~np.isnan(yd['COG']))}, Features: {X_dict['OC'].shape[1]} (18ch x 5bands)\n")
    conds=['OC','OA+OC','OC-OA','OA']

    for target in TARGETS:
        yt=yd[target];valid=~np.isnan(yt);yv=yt[valid]
        nv=valid.sum()
        print("="*70)
        print(f"  TARGET: {target} | n={nv} | range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print("="*70)
        hdr=f"  {'Model':>12s} {'Type':>6s}"
        for cn in conds:hdr+=f" {'Spear':>7s} {'p':>6s} {'R2':>7s} {'MAE':>6s} {'NRMSE':>6s} |"
        print(hdr)
        sep=f"  {'':>12s} {'':>6s}"
        for cn in conds:sep+=f" {'---':>7s} {'--':>6s} {'---':>7s} {'---':>6s} {'---':>6s} |"
        print(sep)
        for mn,(mf,mt) in MODELS.items():
            row=f"  {mn:>12s} {mt:>6s}"
            for cn in conds:
                Xv=X_dict[cn][valid];model=clone(mf)
                m=loso(Xv,yv,model)
                row+=f" {m['spear']:>7.3f} {m['spear_p']:>6.3f} {m['r2']:>7.3f} {m['mae']:>6.2f} {m['nrmse']:>6.2f} |"
            print(row)
        print()
    print("="*70+"\n  DONE\n"+"="*70)

if __name__=='__main__':main()
