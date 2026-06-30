"""
β-VAE on raw EEG [18, 500] — end-to-end regression
Conv encoder + decoder, ~6K params. Within-fold training (no leakage).
"""
import glob, os, re, sys, argparse, warnings
import numpy as np; import pandas as pd
import torch; import torch.nn as nn; import torch.nn.functional as F
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from src.utils.training_logger import (
    regression_metrics,
    log_header, log_epoch, log_fold_test, log_summary,
)
from src.models.augmentations import GaussianNoise, ChannelDropout, TimeMasking

torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
RANDOM_STATE = 42; SFREQ = 250; WINDOW_SEC = 2.0; OVERLAP = 0.5
LATENT_DIM = 8

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


def extract_raw_windows(data_18):
    """Extract raw 2s windows: [18, 500] per window, OA+OC concatenated.
    Returns [n_w, 18, 500]."""
    ws = int(WINDOW_SEC * SFREQ)
    stride = int(ws * (1 - OVERLAP))
    n_w = (data_18.shape[1] - ws) // stride + 1
    if n_w < 1: return None
    windows = np.lib.stride_tricks.sliding_window_view(data_18, ws, axis=1)[:, ::stride].transpose(1,0,2)
    return windows[:n_w].astype(np.float32)


def load_oa_oc_subjects():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1'] = meta[['3.','6.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    raw_data = {}

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
        windows = extract_raw_windows(data_18)
        if windows is None: continue
        raw_data.setdefault(cod, {})[cond] = windows

    subjects = {}
    for cod, conds in raw_data.items():
        if cod not in meta.index: continue
        oa = conds.get('OA', None); oc = conds.get('OC', None)
        if oa is None or oc is None: continue
        subjects[cod] = {'windows': np.concatenate([oa, oc], axis=0).astype(np.float32)}
        for t in ALL_TARGETS:
            v = meta.loc[cod, t]
            subjects[cod][t] = float(v) if not pd.isna(v) else None
    return subjects


class WindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = windows; self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class ConvVAE(nn.Module):
    """Conv encoder + upsampling decoder, ~6K params."""
    def __init__(self, in_ch=18, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: 500 -> 250 -> 125 -> pool -> 31
        self.enc_conv1 = nn.Conv1d(in_ch, 8, kernel_size=15, stride=2, padding=7)
        self.enc_conv2 = nn.Conv1d(8, 4, kernel_size=9, stride=2, padding=4)
        self.pool_enc = nn.AvgPool1d(4)
        flat_dim = 4 * 31  # 124
        self.enc_fc = nn.Linear(flat_dim, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 16)
        self.dec_fc2 = nn.Linear(16, flat_dim)
        self.dec_conv1 = nn.Conv1d(4, 8, kernel_size=9, padding=4)
        self.dec_conv2 = nn.Conv1d(8, in_ch, kernel_size=13, padding=6)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.head = nn.Linear(latent_dim, 1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.pool_enc(x)
        x = x.reshape(x.size(0), -1)
        h = F.relu(self.enc_fc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        x = F.relu(self.dec_fc1(z))
        x = F.relu(self.dec_fc2(x))
        x = x.reshape(x.size(0), 4, 31)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample4(x)   # 4*7=28
        x = self.dec_conv2(x)
        x = self.upsample8(x)   # 28*8=224 → interpolate to 500
        x = F.interpolate(x, size=500, mode='linear', align_corners=False)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        pred = self.head(mu)
        return recon, mu, logvar, pred.squeeze(-1)

    def predict(self, x):
        mu, _ = self.encode(x)
        return self.head(mu).squeeze(-1)


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)


def _subject_aggregate(preds_w, trues_w, cods, subjects, target_key):
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

    # Target norm (per fold)
    tr_y_mean, tr_y_std = float(tr_y.mean()), float(tr_y.std())
    tr_y_n = ((tr_y - tr_y_mean) / (tr_y_std + 1e-10)).astype(np.float32)
    vl_y_n = ((vl_y - tr_y_mean) / (tr_y_std + 1e-10)).astype(np.float32) if vl_y is not None else None

    train_ds = WindowDataset(tr_X, tr_y_n)
    val_ds = WindowDataset(vl_X, vl_y_n) if vl_y_n is not None else None
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False) if val_ds else None

    model = ConvVAE(in_ch=18, latent_dim=LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  VAE params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    aug = nn.Sequential(GaussianNoise(snr=20.0), ChannelDropout(p=0.15),
                        TimeMasking(max_mask_ratio=0.15)) if args.get('augment') else None

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    log_header('regr')

    for epoch in range(1, args['epochs'] + 1):
        if args.get('beta_anneal'):
            beta_now = min(1.0, epoch / args['warmup']) * args['beta']
        else:
            beta_now = args['beta']

        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        tr_preds, tr_trues = [], []
        batch_mu_vars, batch_kls = [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if aug is not None:
                X = aug(X)
            optimizer.zero_grad()
            recon, mu, logvar, pred = model(X)
            mse_vae = F.mse_loss(recon, X, reduction='sum') / X.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X.size(0)
            mse_pred = F.mse_loss(pred, y)
            loss = args['recon_weight'] * mse_vae + beta_now * kl + args['alpha'] * mse_pred
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * X.size(0)
            tr_count += X.size(0)
            tr_preds.extend(pred.detach().cpu().numpy())
            tr_trues.extend(y.cpu().numpy())
            batch_mu_vars.append(mu.detach().var(dim=0).cpu().numpy())
            batch_kls.append(kl.item())

        tr_loss = tr_loss_sum / tr_count
        kl_epoch = np.mean(batch_kls) if batch_kls else 0.0
        mu_var_avg = np.mean(batch_mu_vars, axis=0) if batch_mu_vars else np.zeros(LATENT_DIM)
        tr_preds_r = np.array(tr_preds) * tr_y_std + tr_y_mean
        tr_trues_r = np.array(tr_trues) * tr_y_std + tr_y_mean
        tr_true_s, tr_pred_s = _subject_aggregate(tr_preds_r, tr_trues_r, train_cods, subjects, target)
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
                    mse_pred = F.mse_loss(pred, y)
                    loss = args['recon_weight'] * mse_vae + beta_now * kl + args['alpha'] * mse_pred
                    vl_loss_sum += loss.item() * X.size(0)
                    vl_count += X.size(0)
                    vl_preds.extend(pred.cpu().numpy())
                    vl_trues.extend(y.cpu().numpy())
            vl_loss = vl_loss_sum / vl_count
            scheduler.step(vl_loss)
            vl_preds_r = np.array(vl_preds) * tr_y_std + tr_y_mean
            vl_trues_r = np.array(vl_trues) * tr_y_std + tr_y_mean
            vl_true_s, vl_pred_s = _subject_aggregate(vl_preds_r, vl_trues_r, val_cods, subjects, target)
            vl_m = regression_metrics(vl_true_s, vl_pred_s)
        else:
            vl_loss = float('inf')
            vl_m = {k: 0.0 for k in ['mae','r2','nrmse','spear','pear']}

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        show = args.get('show_epoch', 5)
        if epoch == 1 or epoch % show == 0 or patience == 0 or epoch == args['epochs']:
            log_epoch(epoch, tr_loss, vl_loss, tr_m, vl_m, patience, 'regr')
            z_str = ' '.join([f'{v:.3f}' for v in mu_var_avg])
            print(f"    KL={kl_epoch:.3f}  z_var=[{z_str}]")

        if patience >= args['patience']:
            break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_true, test_pred = [], []
    with torch.no_grad():
        for cod in test_cods:
            windows = subjects[cod]['windows']
            wins_t = torch.from_numpy(windows).to(device)
            preds = []
            for chunk in wins_t.chunk(8, dim=0):
                preds.append(model.predict(chunk).cpu().numpy())
            test_pred.append(np.concatenate(preds).mean() * tr_y_std + tr_y_mean)
            test_true.append(subjects[cod][target])

    # Test (only log for multi-subject GKF; LOSO uses aggregated)
    if len(test_cods) > 1:
        test_m = log_fold_test(np.array(test_true), np.array(test_pred), 'regr')
    else:
        test_m = regression_metrics(np.array(test_true), np.array(test_pred))
    return test_true, test_pred, test_m


def run_gkf(subjects, args):
    cods = sorted(subjects.keys())

    for target in args['targets']:
        valid_cods = [c for c in cods if subjects[c].get(target) is not None and not (isinstance(subjects[c][target], float) and np.isnan(subjects[c][target]))]
        nv = len(valid_cods)
        yv = np.array([subjects[c][target] for c in valid_cods])
        labels = np.array([0 if subjects[c][target] <= np.median(yv) else 1 for c in valid_cods])

        print(f"\n{'='*60}")
        print(f"  GKF k={args['k']} ConvVAE raw EEG [18,500] -> {target}")
        print(f"  n={nv} range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print(f"  beta={args['beta']} alpha={args['alpha']} anneal={'ON' if args.get('beta_anneal') else 'OFF'}")
        print(f"{'='*60}")

        outer_gkf = StratifiedGroupKFold(n_splits=args['k'], shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = []

        for fold_id, (train_val_idx, test_idx) in enumerate(outer_gkf.split(np.zeros(nv), labels, groups=valid_cods)):
            train_val_cods = [valid_cods[i] for i in train_val_idx]
            test_cods = [valid_cods[i] for i in test_idx]
            inner_labels = np.array([0 if subjects[c][target] <= np.median(yv) else 1 for c in train_val_cods])
            inner_gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE+fold_id)
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
        print(f"  LOSO ConvVAE raw EEG [18,500] -> {target}")
        print(f"  n={nv} range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print(f"{'='*60}")

        all_true, all_pred = [], []
        for ti, test_cod in enumerate(cods):
            train_val_cods = [c for i,c in enumerate(cods) if i != ti]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE+ti)
            tr_i, vl_i = next(gss.split(train_val_cods, groups=train_val_cods))
            train_cods = [train_val_cods[i] for i in tr_i]
            val_cods = [train_val_cods[i] for i in vl_i]

            set_seed(RANDOM_STATE + ti)
            print(f"\n[{ti+1:2d}/{nv}] Test={test_cod}  train={len(train_cods)} val={len(val_cods)}")
            print(f"  train windows: {sum(len(subjects[c]['windows']) for c in train_cods)}")

            test_true, test_pred, _ = train_one_fold(train_cods, val_cods, [test_cod], subjects, target, args)
            all_true.extend(test_true)
            all_pred.extend(test_pred)
            print(f"  true={test_true[0]:.1f}  pred={test_pred[0]:.1f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_m = regression_metrics(np.array(all_true), np.array(all_pred))
        print(f"\n{'='*60}")
        print(f"  LOSO FINAL ({nv} folds)")
        print(f"  mae={final_m['mae']:.3f}  r2={final_m['r2']:+.3f}  spear={final_m['spear']:+.3f}  nrmse={final_m['nrmse']:.3f}")
        print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['los','gkf'], default='los')
    p.add_argument('--k', type=int, default=5, help='k for GKF outer folds')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--beta', type=float, default=0.1, help='KL weight')
    p.add_argument('--alpha', type=float, default=1.0, help='prediction loss weight')
    p.add_argument('--recon-weight', type=float, default=0.01, help='reconstruction loss weight')
    p.add_argument('--beta-anneal', action='store_true', help='KL annealing (beta 0→target over warmup)')
    p.add_argument('--warmup', type=int, default=40, help='Epochs for beta annealing')
    p.add_argument('--show-epoch', type=int, default=5)
    p.add_argument('--target', type=str, default='NPLAN',
                   help='Comma-separated targets or "all"')
    p.add_argument('--augment', action='store_true', default=False, help='Data augmentation')
    p.add_argument('--no-augment', dest='augment', action='store_false')
    p.add_argument('--quick', action='store_true')
    return p.parse_args()


def main():
    args_ns = parse_args()

    if args_ns.target.strip().lower() == 'all':
        targets = ALL_TARGETS
    else:
        targets = [t.strip() for t in args_ns.target.split(',') if t.strip() in ALL_TARGETS]
        if not targets:
            print(f"ERROR: no valid targets. Use 'all' or comma-separated from {ALL_TARGETS}")
            return

    if args_ns.quick:
        args_ns.epochs = 5; args_ns.patience = 2
        if args_ns.split == 'gkf': args_ns.k = 3

    print(f"Device: {device}")
    print(f"ConvVAE: raw EEG [18,500] -> latent[{LATENT_DIM}] -> head -> {targets}")
    print(f"beta={args_ns.beta}  alpha={args_ns.alpha}  recon_w={args_ns.recon_weight}  anneal={'ON' if args_ns.beta_anneal else 'OFF'}  augment={'ON' if args_ns.augment else 'OFF'}")
    if args_ns.beta_anneal:
        print(f"KL annealing: beta 0 -> {args_ns.beta} over {args_ns.warmup} epochs")
    print(f"Within-fold training (no leakage)")
    print()

    print("Loading raw EEG windows (OA+OC)...")
    subjects = load_oa_oc_subjects()
    n = len(subjects)
    total_wins = sum(len(s['windows']) for s in subjects.values())
    dummy = subjects[next(iter(subjects))]['windows']
    print(f"  {n} subjects, {total_wins} windows, shape: {list(dummy.shape)}")
    print()

    train_args = {
        'batch_size': args_ns.batch_size, 'lr': args_ns.lr, 'wd': args_ns.wd,
        'epochs': args_ns.epochs, 'patience': args_ns.patience,
        'beta': args_ns.beta, 'alpha': args_ns.alpha,
        'recon_weight': args_ns.recon_weight,
        'beta_anneal': args_ns.beta_anneal, 'warmup': args_ns.warmup,
        'show_epoch': args_ns.show_epoch, 'k': args_ns.k,
        'targets': targets, 'augment': args_ns.augment,
    }

    if args_ns.split == 'los':
        run_loso(subjects, train_args)
    else:
        run_gkf(subjects, train_args)

    print("\n" + "=" * 60)
    print("  Baseline: XGB 19 features OA+OC → NPLAN Spear=+0.529 (p=0.001)")
    print("=" * 60)


if __name__ == '__main__':
    main()
