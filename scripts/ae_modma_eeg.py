"""
MODMA EEG: Autoencoder pretraining on 16K windows + XGBoost on latent [16].
Self-supervised feature learning, then classical classification.
"""
import os, sys, glob, warnings, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import mne
from scipy.signal import welch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from xgboost import XGBClassifier

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
RANDOM_STATE = 42; SFREQ = 250; WINDOW_SEC = 2.0; OVERLAP = 0.5
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
N_CH = 64; N_BANDS = len(BANDS); FEAT_DIM = N_CH * N_BANDS
LATENT_DIM = 16; HIDDEN = [128, 64, 32]
MAX_WINDS = 300  # all windows per subject


def load_participants():
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1, on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 5, 6, 7, 8, 9]]
    p.columns = ['participant_id', 'gender', 'age', 'education', 'group', 'PHQ9', 'GAD7', 'PSQI']
    return p


def extract_bandpower_windows(data, sfreq):
    """Extract [n_w, 64ch, 5bands] from raw EEG."""
    ws = int(WINDOW_SEC * sfreq); stride = int(ws * (1 - OVERLAP))
    n_w = (data.shape[1] - ws) // stride + 1
    if n_w < 1: return None
    n_use = min(N_CH, data.shape[0])
    windows = np.lib.stride_tricks.sliding_window_view(data[:n_use], ws, axis=1)[:, ::stride].transpose(1,0,2)
    windows = windows[:n_w].astype(np.float32)
    bp = np.zeros((n_w, n_use, N_BANDS), dtype=np.float32)
    for ci in range(n_use):
        for bi, (lo, hi) in enumerate(BANDS.values()):
            f, psd = welch(windows[:, ci, :], fs=sfreq, nperseg=ws, noverlap=ws//2, axis=1)
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0: bp[:, ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)
    bp = bp.reshape(n_w, -1)  # [n_w, 320]
    # Z-score per window
    bp = (bp - bp.mean(axis=0, keepdims=True)) / (bp.std(axis=0, keepdims=True) + 1e-8)
    return bp.astype(np.float32)


def load_all_windows():
    """Load all EEG windows + labels. Returns X [n_w, 320], y [n_w], sub_ids [n_w]"""
    participants = load_participants()
    sub_to_group = dict(zip(participants['participant_id'], participants['group']))
    sub_dirs = sorted(glob.glob(os.path.join(EEG_DIR, 'sub-*')))

    all_X, all_y, all_sub_ids = [], [], []
    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        g = sub_to_group.get(sub_id)
        if g not in ('MDD', 'HC'): continue
        edf_files = glob.glob(os.path.join(sd, 'eeg', '*Resting-state*eeg.EDF'))
        if not edf_files: edf_files = glob.glob(os.path.join(sd, 'eeg', '*.EDF'))
        if not edf_files: continue
        try:
            raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
        except: continue
        data = raw.get_data()
        if data.shape[0] < N_CH: continue
        bp = extract_bandpower_windows(data, int(raw.info['sfreq']))
        if bp is None: continue
        # Subsample to MAX_WINDS
        if bp.shape[0] > MAX_WINDS:
            rng = np.random.RandomState(42)
            idx = rng.choice(bp.shape[0], MAX_WINDS, replace=False)
            bp = bp[idx]
        label = 1 if g == 'MDD' else 0
        all_X.append(bp); all_y.extend([label] * len(bp)); all_sub_ids.extend([sub_id] * len(bp))

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int32)
    subs = np.array(all_sub_ids)
    return X, y, subs


class AutoEncoder(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, latent=LATENT_DIM, hidden=HIDDEN):
        super().__init__()
        layers = []; prev = in_dim
        for h in hidden: layers.extend([nn.Linear(prev, h), nn.ReLU()]); prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent)
        self.fc_logvar = nn.Linear(prev, latent)
        layers = []; prev = latent
        for h in reversed(hidden): layers.extend([nn.Linear(prev, h), nn.ReLU()]); prev = h
        layers.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Linear(latent, 1)

    def encode(self, x):
        h = self.encoder(x); return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar); return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparam(mu, logvar)
        return self.decoder(z), mu, logvar, self.head(mu).squeeze(-1)

    def get_latent(self, x):
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu.cpu().numpy()


def train_ae(X, y, args):
    """Train autoencoder on all windows. Return latents per subject."""
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=args['batch_size'], shuffle=True)

    model = AutoEncoder().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  AE params: {n:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args['epochs'])

    for ep in range(1, args['epochs'] + 1):
        model.train(); total = 0.0; n_batch = 0
        for Xb, yb in loader:
            Xb = Xb.to(device)
            opt.zero_grad()
            recon, mu, logvar, pred = model(Xb)
            mse = F.mse_loss(recon, Xb)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Xb.size(0)
            loss = mse + 0.1 * kl
            loss.backward(); opt.step()
            total += loss.item(); n_batch += 1
        sched.step()
        if ep == 1 or ep % args['show_epoch'] == 0 or ep == args['epochs']:
            print(f"  E{ep:4d}/{args['epochs']} loss={total/n_batch:.4f}")

    return model


def evaluate_latent(model, X_all, y_all, subs_all, n_folds=5):
    """LOSO-like: for each fold, train XGBoost on training latents, test on held-out."""
    unique_subs = np.unique(subs_all)
    sub_labels = np.array([y_all[subs_all == s][0] for s in unique_subs])

    # Extract latents per subject
    latents = model.get_latent(torch.from_numpy(X_all).to(device))
    sub_latents = np.array([latents[subs_all == s].mean(axis=0) for s in unique_subs])

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_baccs, fold_accs = [], []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(unique_subs)), sub_labels, groups=unique_subs)):
        Xtr, Xte = sub_latents[tvi], sub_latents[tei]
        ytr, yte = sub_labels[tvi], sub_labels[tei]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

        xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                            random_state=RANDOM_STATE, objective='binary:logistic',
                            eval_metric='logloss', verbosity=0, n_jobs=-1)
        xgb.fit(Xtr_s, ytr)
        yp = xgb.predict(Xte_s)
        fold_baccs.append(balanced_accuracy_score(yte, yp))
        fold_accs.append(accuracy_score(yte, yp))
        print(f"  Fold {fi+1}: bacc={fold_baccs[-1]:.3f} acc={fold_accs[-1]:.3f}")

    return float(np.mean(fold_baccs)), float(np.mean(fold_accs))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--show-epoch', type=int, default=20)
    args_ns = parser.parse_args()

    print(f"Device: {device}")
    print(f"MODMA EEG: Autoencoder pretraining on {MAX_WINDS} windows/subject")
    print(f"Input: {FEAT_DIM} (64ch x 5bands) -> latent [{LATENT_DIM}] -> XGBoost")
    print(f"AE epochs: {args_ns.epochs}")

    print("\nLoading EEG windows + band power...")
    X, y, subs = load_all_windows()
    total_wins = X.shape[0]; n_sub = len(np.unique(subs))
    print(f"  Windows: {total_wins}, Subjects: {n_sub}")
    print(f"  MDD: {np.sum(y==1)}, HC: {np.sum(y==0)}")

    ae_args = {'batch_size': args_ns.batch_size, 'lr': args_ns.lr, 'wd': args_ns.wd,
               'epochs': args_ns.epochs, 'show_epoch': args_ns.show_epoch}

    print(f"\nPhase 1: Pretrain Autoencoder on {total_wins} windows (unsupervised)")
    model = train_ae(X, y, ae_args)

    print(f"\nPhase 2: XGBoost on latent [{LATENT_DIM}] vectors (5-fold SGKF)")
    bacc, acc = evaluate_latent(model, X, y, subs, n_folds=5)

    print(f"\nResult: bacc={bacc:.3f} acc={acc:.3f}")
    print(f"Baseline classical EEG: XGBoost v3 rich bacc=0.577")
    improvement = bacc - 0.577
    print(f"Delta: {improvement:+.3f}")
    if bacc > 0.577:
        print("IMPROVEMENT OVER CLASSICAL!")


if __name__ == '__main__': main()
