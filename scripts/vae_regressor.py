"""
beta-VAE per fold: 19 hypothesis-driven features -> latent [4] -> Ridge head -> NPLAN
Within-fold training (no leakage). Supports --split {los, gkf}. Uses training_logger.
"""
import glob, os, re, sys, argparse, warnings
import numpy as np; import pandas as pd
import torch; import torch.nn as nn; import torch.nn.functional as F
from scipy import stats
from scipy.signal import welch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from src.utils.training_logger import (
    regression_metrics,
    log_header, log_epoch, log_fold_test, log_summary,
)

torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
RANDOM_STATE = 42; SFREQ = 250; WINDOW_SEC = 2.0; OVERLAP = 0.5
LATENT_DIM = 4; HIDDEN = [12, 8]

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

# 19 hypothesis-driven features
SEL_CH = ['FP1','F3','F4','FZ','F8','C3','C4','T7','T8']   # 9 ch
FEAT_BANDS = ['delta','theta','beta']                          # 3 bands
RATIO_CH = ['FP1','F3','F4','FZ','F8','C3','C4']              # 7 ch for ratio
BANDS_RANGE = {'delta':(0.5,4),'theta':(4,8),'beta':(13,30)}

ALL_TARGETS = ['NPLAN','COG','MOT','MOT_V4','COG_V1']


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


def extract_features_19(data_18, sfreq):
    """Extract 19 features per 2s window. Returns [n_w, 19]."""
    ws = int(WINDOW_SEC * sfreq)
    stride = int(ws * (1 - OVERLAP))
    n_w = (data_18.shape[1] - ws) // stride + 1
    if n_w < 1: return None
    windows = np.lib.stride_tricks.sliding_window_view(data_18, ws, axis=1)[:, ::stride].transpose(1,0,2)
    windows = windows[:n_w].astype(np.float32)

    # Band power per channel per window: [n_w, n_ch, n_bands]
    n_ch = len(SEL_CH)
    n_b = len(FEAT_BANDS)
    bp = np.zeros((n_w, n_ch, n_b), dtype=np.float32)
    for wi in range(n_w):
        for ci, ch_name in enumerate(SEL_CH):
            try: ch_idx = CH_18.index(ch_name)
            except: continue
            sig = windows[wi, ch_idx].reshape(1, -1)
            for bi, (_, (lo, hi)) in enumerate(BANDS_RANGE.items()):
                f, psd = welch(sig, fs=sfreq, nperseg=ws, axis=1)
                mask = (f >= lo) & (f <= hi)
                if mask.sum() > 0:
                    bp[wi, ci, bi] = float(np.trapezoid(psd[0, mask], f[mask]))

    # Build 19 features: delta+theta for 9ch (18) + theta/beta ratio (1)
    features = np.zeros((n_w, 19), dtype=np.float32)
    col = 0
    for ci in range(n_ch):
        for bi, bn in enumerate(FEAT_BANDS):
            if bn in ('delta','theta'):
                features[:, col] = bp[:, ci, bi]
                col += 1
    # Theta/beta ratio: mean across ratio channels
    tb = np.zeros(n_w, dtype=np.float32)
    for ch_name in RATIO_CH:
        try: ci = SEL_CH.index(ch_name)
        except: continue
        theta_p = bp[:, ci, FEAT_BANDS.index('theta')]
        beta_p = bp[:, ci, FEAT_BANDS.index('beta')]
        tb += theta_p / (beta_p + 1e-10)
    features[:, 18] = tb / max(len(RATIO_CH), 1)
    # Log-transform band power (standard in EEG)
    features[:, :18] = np.log1p(np.maximum(features[:, :18], 0))
    return features


def load_oa_oc_subjects():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1'] = meta[['3.','6.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    bp_windows = {}

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
        feat = extract_features_19(data_18, sfreq)
        if feat is None: continue
        bp_windows.setdefault(cod, {})[cond] = feat

    subjects = {}
    for cod, conds in bp_windows.items():
        if cod not in meta.index: continue
        oa = conds.get('OA', None); oc = conds.get('OC', None)
        if oa is None or oc is None: continue
        # OA+OC: concatenate windows from both conditions
        subjects[cod] = {'windows': np.concatenate([oa, oc], axis=0).astype(np.float32)}
        for t in ALL_TARGETS:
            v = meta.loc[cod, t]
            subjects[cod][t] = float(v) if not pd.isna(v) else None
    return subjects


class WindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = windows
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class BetaVAE(nn.Module):
    """beta-VAE: encoder 19->hidden->latent[4]  |  decoder 4->hidden->19"""
    def __init__(self, input_dim=19, latent_dim=LATENT_DIM, hidden=HIDDEN):
        super().__init__()
        # Encoder
        layers = []; prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()]); prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        # Decoder
        layers = []; prev = latent_dim
        for h in reversed(hidden):
            layers.extend([nn.Linear(prev, h), nn.ReLU()]); prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)
        # Ridge head (prediction from latent)
        self.head = nn.Linear(latent_dim, 1)

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
        recon = self.decoder(z)
        pred = self.head(mu)  # predict from mu (deterministic)
        return recon, mu, logvar, pred

    def predict(self, x):
        mu, _ = self.encode(x)
        return self.head(mu).squeeze(-1)


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)


def _subject_aggregate(preds_w, trues_w, cods, subjects, target_key):
    """Aggregate per-window predictions into per-subject (local helper)."""
    true_s, pred_s = [], []
    offset = 0
    for cod in cods:
        nw = len(subjects[cod]['windows'])
        pred_s.append(np.mean(preds_w[offset:offset+nw]))
        sub_trues = np.array(trues_w[offset:offset+nw])
        true_s.append(sub_trues[0] if len(sub_trues) > 0 else subjects[cod][target_key])
        offset += nw
    return np.array(true_s), np.array(pred_s)


def train_one_fold(train_cods, val_cods, test_cods, subjects, target, args):
    tr_X = np.concatenate([subjects[c]['windows'] for c in train_cods], axis=0)
    tr_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c][target]) for c in train_cods], axis=0)
    vl_X = np.concatenate([subjects[c]['windows'] for c in val_cods], axis=0) if val_cods else None
    vl_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c][target]) for c in val_cods], axis=0) if val_cods else None

    # Normalize features within fold (fit on train only, no leakage)
    sc = StandardScaler()
    tr_X = sc.fit_transform(tr_X).astype(np.float32)
    tr_y_mean, tr_y_std = float(tr_y.mean()), float(tr_y.std())
    tr_y = ((tr_y - tr_y_mean) / (tr_y_std + 1e-10)).astype(np.float32)
    if vl_X is not None:
        vl_X = sc.transform(vl_X).astype(np.float32)
        vl_y = ((vl_y - tr_y_mean) / (tr_y_std + 1e-10)).astype(np.float32)

    train_ds = WindowDataset(tr_X, tr_y)
    val_ds = WindowDataset(vl_X, vl_y) if vl_y is not None else None
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False) if val_ds else None

    model = BetaVAE(input_dim=19, latent_dim=LATENT_DIM, hidden=HIDDEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  VAE params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    best_val_spear = -float('inf')
    best_state = None
    patience = 0
    kl_epoch_avg = 0.0
    mu_var_avg = np.zeros(LATENT_DIM)

    log_header('regr')

    for epoch in range(1, args['epochs'] + 1):
        # KL annealing: beta linearly increases from 0 to target_beta over warmup epochs
        if args.get('beta_anneal'):
            beta_now = min(1.0, epoch / args['warmup']) * args['beta']
        else:
            beta_now = args['beta']

        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        tr_preds, tr_trues = [], []

        # Track posterior collapse (KL and mu variance per batch)
        batch_mu_vars, batch_kls = [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, pred = model(X)
            # VAE loss + prediction loss
            mse_vae = F.mse_loss(recon, X, reduction='sum') / X.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X.size(0)
            mse_pred = F.mse_loss(pred.squeeze(-1), y)
            loss = mse_vae + beta_now * kl + args['alpha'] * mse_pred
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * X.size(0)
            tr_count += X.size(0)
            tr_preds.extend(pred.detach().cpu().numpy().ravel())
            tr_trues.extend(y.cpu().numpy())
            # Posterior collapse monitoring
            batch_mu_vars.append(mu.detach().var(dim=0).cpu().numpy())
            batch_kls.append(kl.item())

        kl_epoch_avg = np.mean(batch_kls) if batch_kls else 0.0
        mu_var_avg = np.mean(batch_mu_vars, axis=0) if batch_mu_vars else np.zeros(LATENT_DIM)
        tr_loss = tr_loss_sum / tr_count
        # Denormalize train predictions
        tr_preds_raw = np.array(tr_preds) * tr_y_std + tr_y_mean
        tr_trues_raw = np.array(tr_trues) * tr_y_std + tr_y_mean
        tr_true_s, tr_pred_s = _subject_aggregate(tr_preds_raw, tr_trues_raw, train_cods, subjects, target)
        tr_m = regression_metrics(tr_true_s, tr_pred_s)

        if val_loader is not None:
            model.eval()
            vl_loss_sum, vl_count = 0.0, 0
            vl_preds, vl_trues = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    recon, mu, logvar, pred = model(X)
                    mse_vae = F.mse_loss(recon, X, reduction='sum') / X.size(0)
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X.size(0)
                    mse_pred = F.mse_loss(pred.squeeze(-1), y)
                    loss = mse_vae + beta_now * kl + args['alpha'] * mse_pred
                    vl_loss_sum += loss.item() * X.size(0)
                    vl_count += X.size(0)
                    vl_preds.extend(pred.cpu().numpy().ravel())
                    vl_trues.extend(y.cpu().numpy())
            vl_loss = vl_loss_sum / vl_count
            # Denormalize val predictions
            vl_preds_raw = np.array(vl_preds) * tr_y_std + tr_y_mean
            vl_trues_raw = np.array(vl_trues) * tr_y_std + tr_y_mean
            vl_true_s, vl_pred_s = _subject_aggregate(vl_preds_raw, vl_trues_raw, val_cods, subjects, target)
            vl_m = regression_metrics(vl_true_s, vl_pred_s)
        else:
            vl_loss = float('inf')
            vl_m = {k: 0.0 for k in ['mae','r2','nrmse','spear','pear']}

        val_spear_now = vl_m.get('spear', -float('inf'))
        if not np.isnan(val_spear_now) and val_spear_now > best_val_spear:
            best_val_spear = val_spear_now
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        show = args.get('show_epoch', 5)
        if epoch == 1 or epoch % show == 0 or patience == 0 or epoch == args['epochs']:
            log_epoch(epoch, tr_loss, vl_loss, tr_m, vl_m, patience, 'regr')
            # Posterior collapse: KL and per-dim mu variance
            z_str = ' '.join([f'{v:.3f}' for v in mu_var_avg])
            print(f"    KL={kl_epoch_avg:.3f}  z_var=[{z_str}]")

        if patience >= args['patience']:
            break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_true, test_pred = [], []
    with torch.no_grad():
        for cod in test_cods:
            windows = sc.transform(subjects[cod]['windows'].astype(np.float64)).astype(np.float32)
            wins_t = torch.from_numpy(windows).to(device)
            preds = []
            for chunk in wins_t.chunk(8, dim=0):
                preds.append(model.predict(chunk).cpu().numpy())
            test_pred.append(np.concatenate(preds).mean() * tr_y_std + tr_y_mean)
            test_true.append(subjects[cod][target])

    test_m = log_fold_test(np.array(test_true), np.array(test_pred), 'regr')
    return test_true, test_pred, test_m


def run_gkf(subjects, args):
    cods = sorted(subjects.keys())

    for target in args['targets']:
        valid_cods = [c for c in cods if subjects[c].get(target) is not None and not (isinstance(subjects[c][target], float) and np.isnan(subjects[c][target]))]
        nv = len(valid_cods)
        yv = np.array([subjects[c][target] for c in valid_cods])
        labels = np.array([0 if subjects[c][target] <= np.median(yv) else 1 for c in valid_cods])

        print(f"\n{'='*60}")
        print(f"  GKF k={args['k']} beta-VAE(19->{LATENT_DIM}) + Ridge — {target}")
        print(f"  n={nv} range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print(f"  beta={args['beta']} alpha={args['alpha']}")
        print(f"{'='*60}")

        outer_gkf = StratifiedGroupKFold(n_splits=args['k'], shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = []

        for fold_id, (train_val_idx, test_idx) in enumerate(outer_gkf.split(np.zeros(nv), labels, groups=valid_cods)):
            train_val_cods = [valid_cods[i] for i in train_val_idx]
            test_cods = [valid_cods[i] for i in test_idx]

            inner_labels = np.array([0 if subjects[c][target] <= np.median(yv) else 1 for c in train_val_cods])
            inner_gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+fold_id)
            tr_i, vl_i = next(inner_gkf.split(np.zeros(len(train_val_cods)), inner_labels, groups=train_val_cods))
            train_cods = [train_val_cods[i] for i in tr_i]
            val_cods = [train_val_cods[i] for i in vl_i]

            print(f"\nFold {fold_id+1}/{args['k']}: train={len(train_cods)} val={len(val_cods)} test={len(test_cods)}")
            print(f"  train windows: {sum(len(subjects[c]['windows']) for c in train_cods)}")

            set_seed(RANDOM_STATE + fold_id)
            _, _, fm = train_one_fold(train_cods, val_cods, test_cods, subjects, target, args)
            fold_metrics.append(fm)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_summary(fold_metrics, n_folds=args['k'], mode='regr', split_type='gkf')


def run_loso(subjects, args):
    for target in args['targets']:
        cods = sorted([c for c in subjects if subjects[c].get(target) is not None and not (isinstance(subjects[c][target], float) and np.isnan(subjects[c][target]))])
        nv = len(cods)
        yv = np.array([subjects[c][target] for c in cods])

        print(f"\n{'='*60}")
        print(f"  LOSO beta-VAE(19->{LATENT_DIM}) + Ridge — {target}")
        print(f"  n={nv} range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print(f"{'='*60}")

        fold_metrics = []
        for ti, test_cod in enumerate(cods):
            train_val_cods = [c for i,c in enumerate(cods) if i != ti]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+ti)
            tr_i, vl_i = next(gss.split(train_val_cods, groups=train_val_cods))
            train_cods = [train_val_cods[i] for i in tr_i]
            val_cods = [train_val_cods[i] for i in vl_i]

            set_seed(RANDOM_STATE + ti)
            print(f"\n[{ti+1:2d}/{nv}] Test={test_cod}  train={len(train_cods)} val={len(val_cods)}")
            print(f"  train windows: {sum(len(subjects[c]['windows']) for c in train_cods)}")

            _, _, fm = train_one_fold(train_cods, val_cods, [test_cod], subjects, target, args)
            fold_metrics.append(fm)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_summary(fold_metrics, n_folds=nv, mode='regr', split_type='loso')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['los','gkf'], default='los')
    p.add_argument('--k', type=int, default=5, help='k for GKF outer folds')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--beta', type=float, default=0.5, help='KL weight')
    p.add_argument('--alpha', type=float, default=0.5, help='prediction loss weight')
    p.add_argument('--beta-anneal', action='store_true', help='KL annealing (beta 0→target over warmup)')
    p.add_argument('--warmup', type=int, default=15, help='Epochs for beta annealing')
    p.add_argument('--show-epoch', type=int, default=5)
    p.add_argument('--target', type=str, default='NPLAN',
                   help='Comma-separated targets or "all" (e.g. "NPLAN", "NPLAN,COG", "all")')
    p.add_argument('--quick', action='store_true')
    return p.parse_args()


def main():
    args_ns = parse_args()

    if args_ns.quick:
        args_ns.epochs = 5; args_ns.patience = 2
        if args_ns.split == 'gkf': args_ns.k = 3

    # Parse targets
    if args_ns.target.strip().lower() == 'all':
        targets = ALL_TARGETS
    else:
        targets = [t.strip() for t in args_ns.target.split(',') if t.strip() in ALL_TARGETS]
        if not targets:
            print(f"ERROR: no valid targets. Use 'all' or comma-separated from {ALL_TARGETS}")
            return

    print(f"Device: {device}")
    print(f"beta-VAE: 19 feat -> {HIDDEN} -> latent[{LATENT_DIM}] -> Ridge -> targets: {targets}")
    print(f"beta={args_ns.beta}  alpha={args_ns.alpha}  anneal={'ON' if args_ns.beta_anneal else 'OFF'}")
    if args_ns.beta_anneal:
        print(f"KL annealing: beta 0 -> {args_ns.beta} over {args_ns.warmup} epochs")
    print(f"Within-fold training (no leakage)")
    print()

    print("Loading EEG + extracting 19 features per window...")
    subjects = load_oa_oc_subjects()
    n = len(subjects)
    total_wins = sum(len(s['windows']) for s in subjects.values())
    print(f"  {n} subjects, {total_wins} windows, 19 features")
    print(f"  Condition: OA+OC (both conditions concatenated)")
    print()

    train_args = {
        'batch_size': args_ns.batch_size,
        'lr': args_ns.lr,
        'wd': args_ns.wd,
        'epochs': args_ns.epochs,
        'patience': args_ns.patience,
        'beta': args_ns.beta,
        'alpha': args_ns.alpha,
        'beta_anneal': args_ns.beta_anneal,
        'warmup': args_ns.warmup,
        'show_epoch': args_ns.show_epoch,
        'k': args_ns.k,
        'targets': targets,
    }

    if args_ns.split == 'los':
        run_loso(subjects, train_args)
    else:
        run_gkf(subjects, train_args)

    print("\n" + "=" * 60)
    print("  Baseline: XGB 19 features OA+OC -> NPLAN Spear=+0.529 (p=0.001)")
    print("=" * 60)


if __name__ == '__main__':
    main()
