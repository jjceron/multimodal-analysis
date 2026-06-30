"""
VAE over 90 band power features → latent [16] → interpretability + Linear eval
Pre-train on all windows (unsupervised), then LOSO regression on NPLAN, COG, etc.
"""
import glob, os, re, sys, argparse, warnings
import numpy as np; import pandas as pd
import torch; import torch.nn as nn; import torch.nn.functional as F
from scipy import stats
from scipy.signal import welch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from src.utils.training_logger import regression_metrics, log_summary

torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
RANDOM_STATE = 42; SFREQ = 250; WINDOW_SEC = 2.0; OVERLAP = 0.5
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
TARGETS = ['NPLAN','COG','MOT','MOT_V4','COG_V1']
LATENT_DIM = 16
HIDDEN = [64, 32]
FEATURE_NAMES = [f"{ch}_{bn}" for ch in CH_18 for bn in BANDS]


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
            labs=h5['chanlocs']['labels'][:]; ch_names=[]
            for l in labs:
                if isinstance(l,bytes): ch_names.append(l.decode('utf-8','replace').strip('\x00'))
                elif hasattr(l,'tobytes'): ch_names.append(l.tobytes().decode('utf-8','replace').strip('\x00'))
                else: ch_names.append(str(l))
        data=np.fromfile(fdt,dtype=np.float32).reshape(nch,pnts,order='F')
        return data,ch_names,sr


def extract_bp_windows(data_18, sfreq):
    """Extract band power per window → [n_w, 90]"""
    ws = int(WINDOW_SEC * sfreq)
    stride = int(ws * (1 - OVERLAP))
    n_w = (data_18.shape[1] - ws) // stride + 1
    if n_w < 1: return None
    windows = np.lib.stride_tricks.sliding_window_view(data_18, ws, axis=1)[:, ::stride].transpose(1,0,2)
    windows = windows[:n_w].astype(np.float32)

    bp_all = np.zeros((n_w, len(CH_18), len(BANDS)), dtype=np.float32)
    for wi in range(n_w):
        for ci, _ in enumerate(CH_18):
            sig = windows[wi, ci].reshape(1, -1)
            for bi, (lo, hi) in enumerate(BANDS.values()):
                f, psd = welch(sig, fs=sfreq, nperseg=ws, axis=1)
                mask = (f >= lo) & (f <= hi)
                if mask.sum() > 0:
                    bp_all[wi, ci, bi] = float(np.trapezoid(psd[0, mask], f[mask]))
    return bp_all.reshape(n_w, -1)  # [n_w, 90]


def load_all_bp_windows():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1'] = meta[['3.','6.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    all_X, all_cods = [], []
    subject_windows = {}

    for fpath in all_files:
        bn = os.path.basename(fpath)
        cod = re.sub(r'_(OA|OC)\.set$','',bn)
        cond = 'OA' if '_OA.set' in fpath else 'OC'
        try: data, ch_names, sfreq = read_eeg(fpath)
        except: continue
        nch = data.shape[0]
        if nch == 32: cm = CDMS_MAP
        elif nch == 19: cm = CSO_MAP
        elif nch >= 137:
            cm = BIOSEMI_MAP
            keep = [i for i,n in enumerate(ch_names) if n in BIOSEMI_MAP]
            data = data[keep]; ch_names = [ch_names[i] for i in keep]
        else: continue
        sel = []
        for ch in CH_18:
            found = False
            for on, tn in cm.items():
                if tn == ch and on in ch_names: sel.append(ch_names.index(on)); found = True; break
            if not found and ch in ch_names: sel.append(ch_names.index(ch))
        if len(sel) < 18: continue
        data_18 = data[sel]
        data_18 = (data_18 - data_18.mean(axis=1, keepdims=True)) / (data_18.std(axis=1, keepdims=True) + 1e-10)
        bp = extract_bp_windows(data_18, sfreq)
        if bp is None: continue
        all_X.append(bp)
        all_cods.extend([cod] * bp.shape[0])
        subject_windows.setdefault(cod, {})[cond] = bp

    # Build per-subject OA+OC averaged features for linear eval
    subjects = {}
    for cod, conds in subject_windows.items():
        if cod not in meta.index: continue
        oa = conds.get('OA', None); oc = conds.get('OC', None)
        if oa is None or oc is None: continue
        avg = np.concatenate([oa, oc], axis=0) if False else (oa + oc) / 2  # avg per condition
        subjects[cod] = {'bp_windows': np.concatenate([oa, oc], axis=0)}
        for t in TARGETS:
            v = meta.loc[cod, t]
            subjects[cod][t] = float(v) if not pd.isna(v) else None

    all_X = np.concatenate(all_X, axis=0).astype(np.float32)  # [n_total, 90]
    return all_X, subjects


class VAE(nn.Module):
    def __init__(self, input_dim=90, latent_dim=LATENT_DIM, hidden=HIDDEN):
        super().__init__()
        # Encoder
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        # Decoder
        layers = []
        prev = latent_dim
        for h in reversed(hidden):
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    mse = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return mse + beta * kl, mse, kl


def train_vae(X_train, args):
    """Train VAE on all windows. Returns model, latent representations."""
    X_tensor = torch.from_numpy(X_train).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    model = VAE(input_dim=90, latent_dim=LATENT_DIM, hidden=HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])

    print(f"  VAE: 90→{HIDDEN}→latent[{LATENT_DIM}]→{list(reversed(HIDDEN))}→90")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} | epochs: {args['epochs']} | batch: {args['batch_size']} | beta={args['beta']}")
    print(f"  {'Epoch':>6s} | {'Loss':>10s} {'MSE':>10s} {'KL':>10s} | {'lr':>8s}")

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, mse, kl = vae_loss(recon, x, mu, logvar, args['beta'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_mse += mse.item() * x.size(0)
            total_kl += kl.item() * x.size(0)
        scheduler.step()
        n_total = len(loader.dataset)
        show = args.get('show_epoch', 10)
        if epoch == 1 or epoch % show == 0 or epoch == args['epochs']:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  {epoch:6d} | {total_loss/n_total:10.4f} {total_mse/n_total:10.4f} "
                  f"{total_kl/n_total:10.4f} | {lr_now:8.6f}")

    # Extract latent representations
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in loader:
            mu, _ = model.encode(x.to(device))
            latents.append(mu.cpu().numpy())
    return model, np.concatenate(latents, axis=0)


def linear_eval(latents_subject, subjects, targets):
    """LOSO regression with Linear/Ridge on latent [34, 16] → target."""
    cods = sorted(subjects.keys())
    X = np.array([latents_subject[c] for c in cods])
    sc = StandardScaler()
    X_norm = sc.fit_transform(X)

    results = {}
    for tgt in targets:
        y = np.array([subjects[c].get(tgt) for c in cods], dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < 5: continue
        Xv, yv, cv = X_norm[valid], y[valid], [cods[i] for i in np.where(valid)[0]]

        all_t, all_p = [], []
        for ti in range(len(cv)):
            train_mask = np.ones(len(cv), bool); train_mask[ti] = False
            Xtr, Xte = Xv[train_mask], Xv[[ti]]
            ytr, yte = yv[train_mask], yv[[ti]]
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
            ridge.fit(Xtr, ytr)
            all_p.append(ridge.predict(Xte)[0]); all_t.append(yte[0])

        m = regression_metrics(np.array(all_t), np.array(all_p))
        results[tgt] = m

        # Latent dimension correlations with target
        latent_corrs = []
        for d in range(Xv.shape[1]):
            rho, p = stats.spearmanr(Xv[:, d], yv)
            latent_corrs.append((rho, p, d))
        latent_corrs.sort(key=lambda x: abs(x[0]), reverse=True)

    return results, latent_corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--show-epoch', type=int, default=10)
    parser.add_argument('--quick', action='store_true')
    args_ns = parser.parse_args()

    if args_ns.quick:
        args_ns.epochs = 20
        args_ns.show_epoch = 5

    print(f"Device: {device}")
    print(f"VAE on 90 band power features (18ch x 5bands), latent={LATENT_DIM}")
    print(f"Beta={args_ns.beta} (KL weight: {'low interpretability' if args_ns.beta < 0.5 else 'balanced' if args_ns.beta < 2 else 'high structure'})")
    print()

    print("Loading EEG + extracting band power per window...")
    X_all, subjects = load_all_bp_windows()
    total_wins = X_all.shape[0]
    cods_subj = sorted(subjects.keys())
    print(f"  {total_wins:,} total windows, {len(cods_subj)} subjects, feature dim={X_all.shape[1]}")
    print()

    # Phase 1: Train VAE
    print("=" * 60)
    print("  PHASE 1: Pre-train VAE (unsupervised)")
    print("=" * 60)
    train_args = {
        'batch_size': args_ns.batch_size,
        'lr': args_ns.lr,
        'wd': args_ns.wd,
        'epochs': args_ns.epochs,
        'beta': args_ns.beta,
        'show_epoch': args_ns.show_epoch,
    }
    vae_model, latents_all = train_vae(X_all, train_args)
    print()

    # Phase 2: Build per-subject latent vectors
    print("=" * 60)
    print("  PHASE 2: Extract latent [16] per subject")
    print("=" * 60)
    # Map all windows back to subjects
    all_cods = []
    for cod in cods_subj:
        nw = subjects[cod]['bp_windows'].shape[0]
        all_cods.extend([cod] * nw)

    latents_subject = {}
    for cod in cods_subj:
        mask = np.array([c == cod for c in all_cods])
        latents_subject[cod] = latents_all[mask].mean(axis=0)  # [16]

    print(f"  Latent shape per subject: [{LATENT_DIM}]")
    print()

    # Phase 3: Linear eval
    print("=" * 60)
    print("  PHASE 3: LOSO Ridge on latent [16] → targets")
    print("=" * 60)
    results, latent_corrs = linear_eval(latents_subject, subjects, TARGETS)

    print(f"  {'Target':<8s} {'Spear':>7s} {'R2':>7s} {'MAE':>6s} {'NRMSE':>6s}")
    print(f"  {'-'*35}")
    for tgt in TARGETS:
        if tgt in results:
            m = results[tgt]
            print(f"  {tgt:<8s} {m['spear']:>+7.3f} {m['r2']:>+7.3f} {m['mae']:>6.2f} {m['nrmse']:>6.2f}")
    print()

    # Phase 4: Interpretability
    print("=" * 60)
    print("  PHASE 4: Latent dimension ↔ NPLAN correlations")
    print("=" * 60)
    # Recompute for NPLAN specifically (most signal)
    cods_subj = sorted(subjects.keys())
    Xv = np.array([latents_subject[c] for c in cods_subj])
    yv = np.array([subjects[c]['NPLAN'] for c in cods_subj], dtype=float)
    valid = ~np.isnan(yv)
    Xv, yv = Xv[valid], yv[valid]

    print(f"  {'Latent Dim':>12s} {'Spear':>7s} {'p':>7s} {'Interpretation'}")
    print(f"  {'-'*40}")
    all_corrs = []
    for d in range(LATENT_DIM):
        rho, p = stats.spearmanr(Xv[:, d], yv)
        all_corrs.append((rho, p, d))
    all_corrs.sort(key=lambda x: abs(x[0]), reverse=True)
    for rho, p, d in all_corrs[:8]:
        sig = '*' if p < 0.05 else ' '
        print(f"  z_{d:<10d} {rho:>+7.3f} {p:>7.3f} {sig}")
    print()

    # Compare to XGB 19 features baseline
    print("=" * 60)
    print("  COMPARISON TO XGB 19 features (best for NPLAN)")
    print("=" * 60)
    print(f"  VAE latent → Ridge: NPLAN Spear = {results.get('NPLAN', {}).get('spear', 0):+.3f}")
    print(f"  XGB 19 feat OA+OC:     NPLAN Spear = +0.529 (p=0.001)")
    print()

    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
