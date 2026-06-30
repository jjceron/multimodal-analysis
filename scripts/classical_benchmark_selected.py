"""
Classical ML benchmark — HYPOTHESIS-DRIVEN features only
Fronto-centro-temporal channels, delta+theta bands, theta/beta ratio (frontal+central)
No data leakage: features selected by neurophysiology, not p-values
"""
import glob, os, re, warnings
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
SFREQ=250;BANDS={'delta':(0.5,4),'theta':(4,8),'beta':(13,30)}
CH_18=['FP1','F7','T7','P7','F3','C3','P3','O1','FZ','PZ','FP2','F8','T8','P8','F4','C4','P4','O2']
CDMS_MAP={n:n for n in CH_18}
CSO_MAP={'FP1-Cz':'FP1','FP2-Cz':'FP2','FZ-Cz':'FZ','PZ-Cz':'PZ','F3-Cz':'F3','F4-Cz':'F4','F7-Cz':'F7','F8-Cz':'F8','C3-Cz':'C3','C4-Cz':'C4','T3-Cz':'T7','T4-Cz':'T8','T5-Cz':'P7','T6-Cz':'P8','P3-Cz':'P3','P4-Cz':'P4','O1-Cz':'O1','O2-Cz':'O2'}
BIOSEMI_MAP={'A1':'FP1','A5':'F3','A7':'F7','A12':'C3','A15':'T7','A18':'P3','A21':'P7','A23':'O1','B1':'FP2','B3':'FZ','B5':'F4','B7':'F8','B12':'C4','B15':'T8','B18':'P4','B21':'P8','B24':'O2','B27':'PZ'}
TARGETS=['COG','MOT','MOT_V4','NPLAN']
TARGET_LABEL={'COG':'Impulsividad Cognitiva','MOT':'Impulsividad Motora','MOT_V4':'MOT_V4 (8+13+16+21+23)','NPLAN':'No Planificacion'}

# ── Hypothesis-driven channel/band selection ───────────────────────────
# Frontal (prefrontal, executive control) + Central (motor cortex) + T7/T8 (temporal-motor)
# delta: strongest band in all correlations; theta: second strongest
# theta/beta ratio: classical ADHD / inhibitory control biomarker  (frontal+central)
# No data leakage: features selected by neurophysiology, not p-values

SEL_CHANNELS = {
    'FP1': 'Frontal left',
    'F3':  'Frontal mid-left',
    'F4':  'Frontal mid-right',
    'FZ':  'Frontal midline',
    'F8':  'Frontal right-lateral',
    'C3':  'Central left (motor)',
    'C4':  'Central right (motor)',
    'T7':  'Temporal left (motor)',
    'T8':  'Temporal right (motor)',
}
SEL_BANDS = ['delta', 'theta']
RATIO_CHANNELS = ['FP1','F3','F4','FZ','F8','C3','C4']  # frontal + central for theta/beta ratio

FEATURE_NAMES = []
for ch in SEL_CHANNELS:
    for bn in SEL_BANDS:
        FEATURE_NAMES.append(f"{ch}_{bn}")
FEATURE_NAMES.append('theta_beta_ratio')  # averaged across frontal-central channels
print(f"Selected features: {len(FEATURE_NAMES)} ({len(SEL_CHANNELS)}ch x {len(SEL_BANDS)}bands + theta/beta ratio)")
for fn in FEATURE_NAMES:
    print(f"  {fn}")

MODELS={
    'Ridge': (RidgeCV(alphas=np.logspace(-3,3,50)), 'linear'),
    'Lasso': (Lasso(alpha=1.0, max_iter=10000, random_state=RSEED), 'linear'),
    'ElasticNet': (ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000, random_state=RSEED), 'linear'),
    'RF': (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RSEED, n_jobs=-1), 'nonlin'),
    'XGB': (XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=RSEED, objective='reg:squarederror', n_jobs=-1, verbosity=0), 'nonlin'),
    'SVR': (SVR(kernel='rbf', C=1.0, epsilon=0.1), 'nonlin'),
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
        # Extract band power for all 18 channels (need beta for ratio)
        full_bp=np.zeros((len(CH_18), len(BANDS.keys()), nw))
        for ci,ch in enumerate(CH_18):
            if ch not in ch_data:continue
            sig=ch_data[ch][:nw*ws].reshape(nw,ws)
            for bi,(lo,hi) in enumerate(BANDS.values()):
                f,psd=welch(sig,fs=sfreq,nperseg=ws,noverlap=ws//2,axis=1)
                mask=(f>=lo)&(f<=hi)
                if mask.sum()>0:full_bp[ci,bi]=np.trapezoid(psd[:,mask],f[mask],axis=1)
        bp_avg=full_bp.mean(axis=2)  # [18, 3]  (delta, theta, beta)
        bp_raw.setdefault(cod,{})[cond]=bp_avg

    cods=sorted(bp_raw.keys())
    X_oa,X_oc,X_both,X_react=[],[],[],[]
    y_dict={t:[] for t in TARGETS}
    for cod in cods:
        if cod not in meta.index:continue
        oa=bp_raw[cod].get('OA',None);oc=bp_raw[cod].get('OC',None)
        if oa is None or oc is None:continue

        # Build selected feature vector per condition
        def build_features(bp):
            feat=[]
            # Delta and theta for selected channels
            for ch_name in SEL_CHANNELS:
                ch_idx = CH_18.index(ch_name) if ch_name in CH_18 else -1
                if ch_idx < 0:continue
                feat.append(bp[ch_idx,0])  # delta
                feat.append(bp[ch_idx,1])  # theta
            # Theta/beta ratio averaged across frontal+central channels
            tb_ratios=[]
            for ch_name in RATIO_CHANNELS:
                ch_idx = CH_18.index(ch_name) if ch_name in CH_18 else -1
                if ch_idx < 0:continue
                theta_p = bp[ch_idx,1]
                beta_p = bp[ch_idx,2]
                tb_ratios.append(theta_p/(beta_p+1e-10))
            feat.append(np.mean(tb_ratios) if tb_ratios else 0.0)
            return np.array(feat, dtype=float)

        X_oa.append(build_features(oa))
        X_oc.append(build_features(oc))
        X_both.append((build_features(oa)+build_features(oc))/2)
        X_react.append(build_features(oc)-build_features(oa))

        for t in TARGETS:
            v=meta.loc[cod,t];y_dict[t].append(float(v) if not pd.isna(v) else np.nan)

    return {k:np.array(v) for k,v in {'OA':X_oa,'OC':X_oc,'OA+OC':X_both,'OC-OA':X_react}.items()},\
           {t:np.array(y_dict[t]) for t in TARGETS}


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
    print("  CLASSICAL ML BENCHMARK — HYPOTHESIS-DRIVEN FEATURES")
    print("  Fronto-centro-temporal ch, delta+theta bands + theta/beta ratio")
    print("="*70)
    X_dict,yd=load_bp()
    nv=np.sum(~np.isnan(yd['COG']))
    nfeat=X_dict['OC'].shape[1]
    print(f"\nSubjects: {nv}, Features: {nfeat} ({len(FEATURE_NAMES)} selected)\n")
    conds=['OC','OA+OC','OC-OA','OA']

    best_overall={}
    for target in TARGETS:
        yt=yd[target];valid=~np.isnan(yt);yv=yt[valid]
        print("="*70)
        print(f"  TARGET: {target} ({TARGET_LABEL[target]})")
        print(f"  n={valid.sum()} range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
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
                # Track best
                key=(target,cn)
                if key not in best_overall or m['spear']>best_overall[key][0]:
                    best_overall[key]=(m['spear'],mn)
            print(row)
        print()

    # Summary
    print("="*70)
    print("  BEST SPEAR PER TARGET+CONDITION")
    print("="*70)
    print(f"  {'Target':<12s} {'Cond':<8s} {'Model':<12s} {'Spear':>7s}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*7}")
    for (target,cn),(spear,mn) in best_overall.items():
        print(f"  {target:<12s} {cn:<8s} {mn:<12s} {spear:>7.3f}")
    # Transparency: number of comparisons
    n_configs = len(MODELS) * len(conds) * len(TARGETS)
    bonf = 0.05 / (len(MODELS) * len(conds))
    print(f"\n  Configurations evaluated: {n_configs} ({len(MODELS)} models x {len(conds)} conds x {len(TARGETS)} targets)")
    print(f"  Bonferroni threshold (per target): 0.05 / {len(MODELS)*len(conds)} = {bonf:.4f}")
    print(f"  All results are EXPLORATORY — no single configuration survives Bonferroni")
    print(f"\n{'='*70}\n  DONE\n{'='*70}")


if __name__=='__main__':main()
