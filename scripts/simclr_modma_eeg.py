"""
MODMA EEG: SimCLR self-supervised + XGBoost classifier.
Contrastive learning on 16K windows (pairwise augmentations).
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import mne
from scipy.signal import welch

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from xgboost import XGBClassifier

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
RANDOM_STATE = 42; SFREQ = 250; WINDOW_SEC = 2.0; OVERLAP = 0.5
BANDS = [(0.5,4),(4,8),(8,13),(13,30),(30,50)]
N_CH = 64; FEAT_DIM = N_CH * len(BANDS); LATENT_DIM = 32; MAX_WIND = 300


def load_participants():
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1, on_bad_lines='skip', engine='python')
    p = p[[0,2,3,5,6,7,8,9]]
    p.columns = ['pid','gender','age','edu','group','PHQ9','GAD7','PSQI']
    return p


def extract_bandpower_windows(data, sfreq):
    ws = int(WINDOW_SEC * sfreq); stride = int(ws * (1 - OVERLAP))
    n_w = (data.shape[1] - ws) // stride + 1
    if n_w < 1: return None
    n_use = min(N_CH, data.shape[0])
    win = np.lib.stride_tricks.sliding_window_view(data[:n_use], ws, axis=1)[:, ::stride].transpose(1,0,2)[:n_w]
    bp = np.zeros((n_w, n_use, len(BANDS)), dtype=np.float32)
    for ci in range(n_use):
        for bi, (lo, hi) in enumerate(BANDS):
            f, psd = welch(win[:, ci, :], fs=sfreq, nperseg=ws, noverlap=ws//2, axis=1)
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0: bp[:, ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)
    bp = bp.reshape(n_w, -1)
    return bp.astype(np.float32)


def load_all_windows():
    participants = load_participants()
    sg = dict(zip(participants['pid'], participants['group']))
    sub_dirs = sorted(glob.glob(os.path.join(EEG_DIR, 'sub-*')))
    all_X, all_y, all_subs = [], [], []
    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        g = sg.get(sub_id)
        if g not in ('MDD','HC'): continue
        edf = glob.glob(os.path.join(sd,'eeg','*Resting-state*eeg.EDF'))
        if not edf: edf = glob.glob(os.path.join(sd,'eeg','*.EDF'))
        if not edf: continue
        try: raw = mne.io.read_raw_edf(edf[0], preload=True, verbose=False)
        except: continue
        data = raw.get_data()
        if data.shape[0] < N_CH: continue
        bp = extract_bandpower_windows(data, int(raw.info['sfreq']))
        if bp is None: continue
        if bp.shape[0] > MAX_WIND:
            rng = np.random.RandomState(42)
            bp = bp[rng.choice(bp.shape[0], MAX_WIND, replace=False)]
        label = 1 if g == 'MDD' else 0
        all_X.append(bp); all_y.extend([label]*len(bp)); all_subs.extend([sub_id]*len(bp))
    X = np.concatenate(all_X).astype(np.float32)
    # Global z-score per feature
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, np.array(all_y), np.array(all_subs)


class SimCLR(nn.Module):
    """Encoder + projector for contrastive learning."""
    def __init__(self, in_dim=FEAT_DIM, latent=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
    def forward(self, x): return self.projector(self.encoder(x))
    def encode(self, x): return self.encoder(x)


def augment(x, noise=0.05, drop_p=0.1):
    """Augment a batch of window features."""
    x_aug = x + torch.randn_like(x) * noise
    # Randomly zero out some channels (channel dropout)
    B, D = x.shape
    mask = (torch.rand(B, D, device=x.device) > drop_p).float()
    return x_aug * mask


def nt_xent_loss(z1, z2, temperature=0.07):
    """NT-Xent loss for SimCLR."""
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    sim_i_j = torch.diag(sim, z1.size(0))
    sim_j_i = torch.diag(sim, -z1.size(0))
    pos = torch.cat([sim_i_j, sim_j_i], dim=0)
    mask = torch.eye(z.size(0), device=z.device).bool()
    neg = sim[~mask].view(z.size(0), -1)
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    # Simple: use log-softmax over all
    log_softmax = F.log_softmax(sim, dim=1)
    nll = -torch.diag(log_softmax, z1.size(0))[:z1.size(0)].mean() - \
          torch.diag(log_softmax, -z1.size(0))[:z1.size(0)].mean()
    return nll


def train_simclr(X, args):
    """Train SimCLR on all windows. Return encoder."""
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
    loader = torch.utils.data.DataLoader(ds, batch_size=args['batch_size'], shuffle=True)

    model = SimCLR().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  SimCLR params: {n:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args['epochs'])

    for ep in range(1, args['epochs'] + 1):
        model.train(); total = 0.0; n_b = 0
        for (Xb,) in loader:
            Xb = Xb.to(device)
            X1 = augment(Xb); X2 = augment(Xb)
            z1 = model(X1); z2 = model(X2)
            loss = nt_xent_loss(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n_b += 1
        sched.step()
        if ep == 1 or ep % args['show_epoch'] == 0 or ep == args['epochs']:
            print(f"  E{ep:4d}/{args['epochs']} loss={total/n_b:.4f}")

    return model


def eval_classifier(model, X_all, y_all, subs_all, n_folds=5):
    """Extract latents, train XGBoost on subject-level vectors."""
    unique_subs = np.unique(subs_all)
    sub_labs = np.array([y_all[subs_all == s][0] for s in unique_subs])

    # Extract latents per windows, average per subject
    model.eval()
    with torch.no_grad():
        latents = model.encode(torch.from_numpy(X_all).to(device)).cpu().numpy()
    sub_lats = np.array([latents[subs_all == s].mean(axis=0) for s in unique_subs])

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    baccs, accs = [], []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(unique_subs)), sub_labs, groups=unique_subs)):
        Xtr, Xte = sub_lats[tvi], sub_lats[tei]
        ytr, yte = sub_labs[tvi], sub_labs[tei]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                            random_state=RANDOM_STATE, objective='binary:logistic',
                            eval_metric='logloss', verbosity=0, n_jobs=-1)
        xgb.fit(Xtr_s, ytr)
        yp = xgb.predict(Xte_s)
        baccs.append(balanced_accuracy_score(yte, yp))
        accs.append(accuracy_score(yte, yp))
        print(f"  Fold {fi+1}: bacc={baccs[-1]:.3f} acc={accs[-1]:.3f}")

    return float(np.mean(baccs)), float(np.mean(accs))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--show-epoch', type=int, default=20)
    args_ns = p.parse_args()

    print(f"Device: {device}")
    print(f"MODMA EEG: SimCLR contrastive pretraining")
    print(f"Input: {FEAT_DIM} -> latent [{LATENT_DIM}] -> XGBoost")

    X, y, subs = load_all_windows()
    total = X.shape[0]; n_sub = len(np.unique(subs))
    print(f"\n  Windows: {total}, Subjects: {n_sub}")
    print(f"  MDD: {np.sum(y==1)}, HC: {np.sum(y==0)}")

    ae_args = {'batch_size': args_ns.batch_size, 'lr': args_ns.lr, 'wd': args_ns.wd,
               'epochs': args_ns.epochs, 'show_epoch': args_ns.show_epoch}

    print(f"\nPhase 1: SimCLR pretraining on {total} windows (self-supervised)")
    model = train_simclr(X, ae_args)

    print(f"\nPhase 2: XGBoost on latent [{LATENT_DIM}] vectors (5-fold SGKF)")
    bacc, acc = eval_classifier(model, X, y, subs, n_folds=5)

    print(f"\nResult: bacc={bacc:.3f} acc={acc:.3f}")
    print(f"Baseline classical EEG: XGBoost v3 rich bacc=0.577")
    print(f"Delta: {bacc-0.577:+.3f}")


if __name__ == '__main__': main()
