"""
MODMA EEG DL Benchmark — unified script for classic deep learning models.

Usage:
    python scripts/benchmark_dl_final.py --model shallowconvnet --epochs 100
    python scripts/benchmark_dl_final.py --model deepconvnet --epochs 100
    python scripts/benchmark_dl_final.py --model cnnlstm --epochs 100

Models: shallowconvnet | deepconvnet | eegnet | cnnlstm
Input: raw EEG windows [64ch, 500samples], 200 windows/subject
Split: 5-fold StratifiedGroupKFold (subject-aware, no leakage)
"""
import os, sys, glob, warnings, json, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import mne
from scipy.signal import welch

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
PARTICIPANTS_PATH = f'{EEG_DIR}/participants.tsv'
RANDOM_STATE = 42
SFREQ = 250
WINDOW_SEC = 2.0
OVERLAP = 0.5
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
N_CH = 64
N_WIN = 500

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score


def load_participants():
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                     on_bad_lines='skip', engine='python')
    p = p[[0, 2, 3, 5, 6, 7, 8, 9]]
    p.columns = ['pid', 'gender', 'age', 'edu', 'group', 'PHQ9', 'GAD7', 'PSQI']
    return p


def extract_bandpower_windows(data, sfreq):
    ws = int(WINDOW_SEC * sfreq)
    stride = int(ws * (1 - OVERLAP))
    n_w = (data.shape[1] - ws) // stride + 1
    if n_w < 1:
        return None, None
    n_use = min(N_CH, data.shape[0])
    win = np.lib.stride_tricks.sliding_window_view(
        data[:n_use], ws, axis=1)[:, ::stride].transpose(1, 0, 2)
    win = win[:n_w].astype(np.float32)
    raw_windows = win.copy()
    bp = np.zeros((n_w, n_use, len(BANDS)), dtype=np.float32)
    for ci in range(n_use):
        for bi, (lo, hi) in enumerate(BANDS):
            f, psd = welch(win[:, ci, :], fs=sfreq, nperseg=ws,
                           noverlap=ws // 2, axis=1)
            mask = (f >= lo) & (f <= hi)
            if mask.sum() > 0:
                bp[:, ci, bi] = np.trapezoid(psd[:, mask], f[mask], axis=1)
    return raw_windows, bp.reshape(n_w, -1).astype(np.float32)


def load_all_subjects():
    participants = load_participants()
    sg = dict(zip(participants['pid'], participants['group']))
    sub_dirs = sorted(glob.glob(os.path.join(EEG_DIR, 'sub-*')))
    subjects = {}
    for sd in sub_dirs:
        sub_id = os.path.basename(sd)
        g = sg.get(sub_id)
        if g not in ('MDD', 'HC'):
            continue
        edf = glob.glob(os.path.join(sd, 'eeg', '*Resting-state*eeg.EDF'))
        if not edf:
            edf = glob.glob(os.path.join(sd, 'eeg', '*.EDF'))
        if not edf:
            continue
        try:
            raw = mne.io.read_raw_edf(edf[0], preload=True, verbose=False)
        except Exception:
            continue
        data = raw.get_data()
        if data.shape[0] < N_CH:
            continue
        raw_w, bp_w = extract_bandpower_windows(data, int(raw.info['sfreq']))
        if raw_w is None:
            continue
        if raw_w.shape[0] > 200:
            rng = np.random.RandomState(RANDOM_STATE)
            idx = rng.choice(raw_w.shape[0], 200, replace=False)
            raw_w = raw_w[idx]
            bp_w = bp_w[idx]
        subjects[sub_id] = {
            'raw_windows': raw_w.astype(np.float32),
            'bp_windows': bp_w.astype(np.float32),
            'label': 1 if g == 'MDD' else 0,
        }
    return subjects


def build_model_factory(model_name):
    if model_name == 'shallowconvnet':
        from src.models.shallowconvnet import ShallowConvNet

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = ShallowConvNet(n_channels=N_CH, n_classes=1,
                                             n_samples=N_WIN, dropout=0.5)
            def forward(self, x):
                return self.model(x).squeeze(-1)

        return lambda: Wrapper(), 'ShallowConvNet'

    elif model_name == 'deepconvnet':
        from src.models.deepconvnet import DeepConvNet

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = DeepConvNet(n_channels=N_CH, n_classes=1,
                                          n_samples=N_WIN, dropout=0.5)
            def forward(self, x):
                return self.model(x).squeeze(-1)

        return lambda: Wrapper(), 'DeepConvNet'

    elif model_name == 'eegnet':
        from src.models.eegnet import EEGNet

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = EEGNet(n_channels=N_CH, n_classes=1,
                                     F1=8, D=2, F2=16,
                                     temporal_kern=31, separable_kern=15,
                                     pool1=4, pool2=4, dropout=0.5,
                                     meanmax_alpha=0.0)
            def forward(self, x):
                logits, _ = self.model(x)
                return logits.squeeze(-1)

        return lambda: Wrapper(), 'EEGNet'

    elif model_name == 'cnnlstm':
        from src.models.cnn_lstm import CNNLSTM

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = CNNLSTM(n_channels=N_CH, n_classes=1,
                                      n_samples=N_WIN, dropout=0.5)
            def forward(self, x):
                return self.model(x).squeeze(-1)

        return lambda: Wrapper(), 'CNN-LSTM'

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose: shallowconvnet, deepconvnet, eegnet, cnnlstm")


def train_convnet(subjects, model_factory, model_name, args):
    cods = sorted(subjects.keys())
    labels = np.array([subjects[c]['label'] for c in cods])
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_baccs = []

    print(f"\n  {model_name} training (5-fold SGKF, {args.epochs} epochs):")
    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(cods)), labels, groups=cods)):
        train_ids = [cods[i] for i in tvi]
        test_ids = [cods[i] for i in tei]

        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + fi)
        tv_labs = labels[tvi]
        tr_i, vl_i = next(inner.split(np.zeros(len(tvi)), tv_labs, groups=np.array(train_ids)))
        train_subj = [train_ids[i] for i in tr_i]
        val_subj = [train_ids[i] for i in vl_i]

        tr_X = np.concatenate([subjects[c]['raw_windows'] for c in train_subj], axis=0)
        tr_y = np.concatenate([np.full(len(subjects[c]['raw_windows']), subjects[c]['label'])
                                for c in train_subj], axis=0)
        vl_X = np.concatenate([subjects[c]['raw_windows'] for c in val_subj], axis=0)
        vl_y = np.concatenate([np.full(len(subjects[c]['raw_windows']), subjects[c]['label'])
                                for c in val_subj], axis=0)

        tr_mean = tr_X.mean(axis=(1, 2), keepdims=True)
        tr_std = tr_X.std(axis=(1, 2), keepdims=True) + 1e-8
        tr_X = (tr_X - tr_mean) / tr_std
        vl_mean = vl_X.mean(axis=(1, 2), keepdims=True)
        vl_std = vl_X.std(axis=(1, 2), keepdims=True) + 1e-8
        vl_X = (vl_X - vl_mean) / vl_std

        tr_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(tr_X).float(), torch.from_numpy(tr_y).float())
        vl_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(vl_X).float(), torch.from_numpy(vl_y).float())
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        vl_loader = torch.utils.data.DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False)

        model = model_factory().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if fi == 0:
            print(f"    Params: {n_params:,}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        pos_w = torch.tensor([np.sum(tr_y == 0) / max(np.sum(tr_y == 1), 1)]).to(device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_vl = float('inf')
        best_st = None
        patience = 0
        for ep in range(1, args.epochs + 1):
            model.train()
            tr_loss = 0.0
            tr_n = 0
            for X, y in tr_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(X), y)
                loss.backward()
                opt.step()
                tr_loss += loss.item() * X.size(0)
                tr_n += X.size(0)
            tr_loss /= tr_n
            sched.step()

            model.eval()
            vl_loss = 0.0
            vl_n = 0
            with torch.no_grad():
                for X, y in vl_loader:
                    X, y = X.to(device), y.to(device)
                    loss = crit(model(X), y)
                    vl_loss += loss.item() * X.size(0)
                    vl_n += X.size(0)
            vl_loss /= vl_n

            if vl_loss < best_vl:
                best_vl = vl_loss
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= args.patience:
                break

        model.load_state_dict(best_st)
        model.eval()
        test_true, test_pred = [], []
        with torch.no_grad():
            for c in test_ids:
                w = subjects[c]['raw_windows']
                w_n = (w - w.mean(axis=(1, 2), keepdims=True)) / (w.std(axis=(1, 2), keepdims=True) + 1e-8)
                preds = torch.sigmoid(model(torch.from_numpy(w_n).float().to(device)))
                test_pred.append(preds.cpu().numpy().mean())
                test_true.append(subjects[c]['label'])
        bacc = balanced_accuracy_score(test_true, (np.array(test_pred) >= 0.5).astype(int))
        fold_baccs.append(bacc)
        print(f"    Fold {fi + 1}: bacc={bacc:.3f}  (best vl={best_vl:.4f}, ep={ep - patience})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_bacc = float(np.mean(fold_baccs))
    std_bacc = float(np.std(fold_baccs))
    print(f"  {model_name}: bacc={mean_bacc:.3f} +/- {std_bacc:.3f}")
    return mean_bacc, std_bacc, fold_baccs


def main():
    p = argparse.ArgumentParser(description='MODMA EEG DL Benchmark')
    p.add_argument('--model', required=True,
                   choices=['shallowconvnet', 'deepconvnet', 'eegnet', 'cnnlstm'],
                   help='Model architecture to train')
    p.add_argument('--batch-size', type=int, default=32,
                   help='Batch size (default: 32, safe for 6GB GPU)')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    p.add_argument('--epochs', type=int, default=100, help='Max epochs')
    p.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    args = p.parse_args()

    print(f"Device: {device}")
    print(f"{'='*60}")
    print(f"  MODMA EEG DL BENCHMARK — {args.model.upper()}")
    print(f"  Batch={args.batch_size}  LR={args.lr}  Epochs={args.epochs}  Patience={args.patience}")
    print(f"{'='*60}")

    print("\nLoading EEG raw windows from EDFs...")
    subjects = load_all_subjects()
    cods = sorted(subjects.keys())
    labels = np.array([subjects[c]['label'] for c in cods])
    total_wins = sum(len(subjects[c]['raw_windows']) for c in cods)
    print(f"  Subjects: {len(subjects)} (MDD={np.sum(labels == 1)}, HC={np.sum(labels == 0)})")
    print(f"  Windows/subject: ~{total_wins // len(subjects)}")
    print(f"  Raw window shape: {subjects[cods[0]]['raw_windows'].shape[1:]}")

    model_factory, display_name = build_model_factory(args.model)
    mean_bacc, std_bacc, fold_baccs = train_convnet(subjects, model_factory, display_name, args)

    result = {
        'model': args.model,
        'display_name': display_name,
        'bacc_mean': mean_bacc,
        'bacc_std': std_bacc,
        'fold_baccs': fold_baccs,
        'n_subjects': len(subjects),
        'n_mdd': int(np.sum(labels == 1)),
        'n_hc': int(np.sum(labels == 0)),
        'device': str(device),
        'batch_size': args.batch_size,
        'lr': args.lr,
        'wd': args.wd,
        'epochs': args.epochs,
        'patience': args.patience,
    }

    os.makedirs('results', exist_ok=True)
    out_path = f'results/modma_dl_{args.model}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"  bacc_mean={mean_bacc:.4f}, bacc_std={std_bacc:.4f}")
    print(f"  Folds: {fold_baccs}")


if __name__ == '__main__':
    main()
