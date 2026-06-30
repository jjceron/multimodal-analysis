"""
MODMA EEG DL v3: Global Average Pool + Small MLP on [64ch x 5bands] 3D tensor.
Spatial GAP + Spectral GAP + concat → MLP. 
No flatten → preserves ch/band structure. ~1K params.
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.insert(0, '.')
from src.utils.training_logger import log_header, log_epoch, log_fold_test, log_summary

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

EEG_PATH = 'data/processed/modma_eeg_features.npz'
PARTICIPANTS_PATH = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state/participants.tsv'
RANDOM_STATE = 42; N_CH = 64; N_BANDS = 5


def load_data():
    eeg = np.load(EEG_PATH, allow_pickle=True)
    subs = eeg['subjects']
    X = eeg['X'].astype(np.float32)
    if X.ndim == 3: X = X.reshape(X.shape[0], N_CH, N_BANDS)
    elif X.ndim == 2: X = X.reshape(X.shape[0], N_CH, N_BANDS)
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1, on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 5, 6, 7, 8, 9]]
    p.columns = ['participant_id', 'gender', 'age', 'education', 'group', 'PHQ9', 'GAD7', 'PSQI']
    sg = dict(zip(p['participant_id'], p['group']))
    y = []; vs = []
    for s in subs:
        g = sg.get(s)
        if g in ('MDD', 'HC'): y.append(1 if g == 'MDD' else 0); vs.append(s)
    return X[:len(vs)], np.array(y), np.array(vs)


class SpatialSpectralGAP(nn.Module):
    """GAP over channels → [B, Bands], GAP over bands → [B, Ch], concat → MLP."""
    def __init__(self, n_ch=N_CH, n_bands=N_BANDS, hidden=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ch + n_bands, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        z_ch = x.mean(dim=2)  # GAP across bands → [B, 64] per-channel avg power
        z_bn = x.mean(dim=1)  # GAP across channels → [B, 5] per-band avg power
        z = torch.cat([z_ch, z_bn], dim=-1)  # [B, 69]
        return self.net(z).squeeze(-1)


def train_one_fold(X_train, y_train, X_val, y_val, X_test, y_test, args):
    t_mean = X_train.mean(axis=(0, 1), keepdims=True)
    t_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_n = (X_train - t_mean) / t_std
    X_val_n = (X_val - t_mean) / t_std
    X_test_n = (X_test - t_mean) / t_std

    model = SpatialSpectralGAP(dropout=args['dropout']).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  GAPNet params: {n:,}")

    Xt = torch.from_numpy(X_train_n).float().to(device); yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_val_n).float().to(device); yv = torch.from_numpy(y_val).float().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    pw = torch.tensor([np.sum(y_train==0) / max(np.sum(y_train==1), 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_vl = float('inf'); best_st = None; pat = 0
    log_header('clas')

    for ep in range(1, args['epochs'] + 1):
        model.train(); optimizer.zero_grad()
        pred = model(Xt); loss = crit(pred, yt)
        loss.backward(); optimizer.step(); scheduler.step()
        tr_p = torch.sigmoid(pred).detach().cpu().numpy()
        tr_m = _bin(yt.cpu().numpy(), (tr_p >= 0.5).astype(int))

        model.eval()
        with torch.no_grad():
            vl_p = model(Xv); vl_l = crit(vl_p, yv)
            vl_pp = torch.sigmoid(vl_p).cpu().numpy()
            vl_m = _bin(yv.cpu().numpy(), (vl_pp >= 0.5).astype(int))

        if vl_l < best_vl: best_vl = vl_l; best_st = model.state_dict(); pat = 0
        else: pat += 1

        show = args.get('show_epoch', 30)
        if ep == 1 or ep % show == 0 or pat == 0 or ep == args['epochs']:
            log_epoch(ep, loss.item(), vl_l.item(), tr_m, vl_m, pat, 'clas')
        if pat >= args['patience']: break

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_test_n).float().to(device)
        tp = (torch.sigmoid(model(Xte)).cpu().numpy() >= 0.5).astype(int)
    test_m = log_fold_test(y_test, tp, 'clas')
    return y_test, tp, test_m


def _bin(t, p):
    return {'acc':float(accuracy_score(t,p)), 'bacc':float(balanced_accuracy_score(t,p)),
            'f1':float(f1_score(t,p,zero_division=0)),
            'sens':float(f1_score(t,p,pos_label=1,zero_division=0)),
            'spec':float(f1_score(t,p,pos_label=0,zero_division=0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--show-epoch', type=int, default=30)
    parser.add_argument('--quick', action='store_true')
    args_ns = parser.parse_args()
    if args_ns.quick: args_ns.epochs = 5; args_ns.patience = 3; args_ns.k = 3

    print(f"Device: {device}")
    print(f"MODMA EEG DL v3: Spatial GAP + Spectral GAP -> MLP ({N_CH+N_BANDS}->16->8->1)")

    X, y, subs = load_data()
    print(f"  Subjects: {len(subs)} (MDD: {np.sum(y==1)}, HC: {np.sum(y==0)})")
    print(f"  Shape: {X.shape} (sub x ch x bands)")

    train_args = {'lr': args_ns.lr, 'wd': args_ns.wd, 'epochs': args_ns.epochs,
                  'patience': args_ns.patience, 'dropout': args_ns.dropout,
                  'show_epoch': args_ns.show_epoch}
    skf = StratifiedGroupKFold(n_splits=args_ns.k, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(subs)), y, groups=subs)):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+fi)
        tv = y[tvi]; ti, vi = next(inner.split(np.zeros(len(tv)), tv, groups=subs[tvi]))
        Xtr, Xte = X[tvi], X[tei]; ytr, yte = y[tvi], y[tei]
        print(f"\nFold {fi+1}/{args_ns.k}: train={len(ti)} val={len(vi)} test={len(tei)}")
        _, _, fm = train_one_fold(Xtr[ti], ytr[ti], Xtr[vi], ytr[vi], Xte, yte, train_args)
        fold_metrics.append(fm)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    log_summary(fold_metrics, n_folds=args_ns.k, mode='clas', split_type='gkf')
    print(f"\nBaseline classical EEG: XGBoost v3 bacc=0.577")


if __name__ == '__main__': main()
