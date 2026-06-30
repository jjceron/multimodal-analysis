"""
Deep Learning: Spectrogram [18,33,15] + CNN 2D → Barratt regression
LOSO + inner GroupShuffleSplit. Uses training_logger for consistent output.
"""
import glob, os, re, sys, argparse, warnings
import numpy as np; import pandas as pd
import torch; import torch.nn as nn; import torch.nn.functional as F
from scipy import stats
from scipy.signal import spectrogram
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from src.utils.training_logger import (
    regression_metrics, log_header, log_epoch, log_fold_test, log_summary,
)

torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
RANDOM_STATE = 42; SFREQ = 250; WINDOW_SEC = 2.0; OVERLAP = 0.5

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
TARGETS = ['COG','MOT','MOT_V4','NPLAN','COG_V1']

# Spectrogram parameters
NPERSEG = 64; NOVERLAP = 32


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
                if isinstance(l,bytes): ch_names.append(l.decode('utf-8','replace').strip('\x00'))
                elif hasattr(l,'tobytes'): ch_names.append(l.tobytes().decode('utf-8','replace').strip('\x00'))
                else: ch_names.append(str(l))
        data=np.fromfile(fdt,dtype=np.float32).reshape(nch,pnts,order='F')
        return data,ch_names,sr


def compute_spectrograms(data_18, sfreq):
    """Compute spectrogram of FULL signal [18, n_total], then extract windows.
    Returns: [n_windows, 18, n_freq, n_time] where shape is DETERMINISTIC."""
    ws = int(WINDOW_SEC * sfreq)            # window_samples
    stride = int(ws * (1 - OVERLAP))        # stride_samples
    n_total = data_18.shape[1]
    n_w = (n_total - ws) // stride + 1

    # Spectrogram of the FULL signal → deterministic time bins
    n_freq = NPERSEG // 2 + 1               # 33
    n_time_full = (n_total - NPERSEG) // (NPERSEG - NOVERLAP) + 1  # full signal time bins
    n_time_window = (ws - NPERSEG) // (NPERSEG - NOVERLAP) + 1     # per-window time bins

    full_spec = np.zeros((len(CH_18), n_freq, n_time_full), dtype=np.float32)
    for ch in range(len(CH_18)):
        _, _, Sxx = spectrogram(data_18[ch], fs=sfreq, nperseg=NPERSEG, noverlap=NOVERLAP)
        full_spec[ch] = Sxx.astype(np.float32)[:, :n_time_full]

    # Extract overlapping windows in time dimension
    spec_stride = int(stride * (n_time_full / n_total))  # ≈ stride / (nperseg - noverlap)
    if spec_stride < 1:
        spec_stride = 1
    specs = np.zeros((n_w, len(CH_18), n_freq, n_time_window), dtype=np.float32)
    for wi in range(n_w):
        t_start = wi * spec_stride
        t_end = t_start + n_time_window
        if t_end > n_time_full:
            t_end = n_time_full
            t_start = max(0, t_end - n_time_window)
        specs[wi] = full_spec[:, :, t_start:t_end]

    # Log-scale
    return np.log1p(np.maximum(specs, 0))


def subject_aggregate(preds_window, trues_window, cods, subjects, target_key):
    """Aggregate per-window predictions into per-subject."""
    true_s, pred_s = [], []
    offset = 0
    for cod in cods:
        nw = len(subjects[cod]['windows'])
        pred_s.append(np.mean(preds_window[offset:offset + nw]))
        sub_trues = np.array(trues_window[offset:offset + nw])
        true_s.append(sub_trues[0] if len(sub_trues) > 0 else subjects[cod][target_key])
        offset += nw
    return np.array(true_s), np.array(pred_s)


def load_oc_subjects():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1'] = meta[['3.','6.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    raw_specs = {}

    for fpath in all_files:
        bn = os.path.basename(fpath)
        if '_OA.set' in fpath: continue
        cod = re.sub(r'_OC\.set$','',bn)
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
        specs = compute_spectrograms(data_18, sfreq)
        raw_specs[cod] = specs

    # Pad all spectrograms to the same shape (sfreq varies: 250 vs 256 Hz)
    max_freq = max(s.shape[2] for s in raw_specs.values()) if raw_specs else 33
    max_time = max(s.shape[3] for s in raw_specs.values()) if raw_specs else 14
    for cod in raw_specs:
        s = raw_specs[cod]
        pad_f = max_freq - s.shape[2]
        pad_t = max_time - s.shape[3]
        if pad_f > 0 or pad_t > 0:
            s = np.pad(s, ((0,0),(0,0),(0,pad_f),(0,pad_t)), mode='edge')
            raw_specs[cod] = s

    subjects = {}
    for cod, s in raw_specs.items():
        if cod not in meta.index: continue
        entry = {'windows': s}  # [n_w, 18, n_freq, n_time]
        for t in ['COG','MOT','MOT_V4','NPLAN','COG_V1']:
            v = meta.loc[cod, t]
            entry[t] = float(v) if not pd.isna(v) else None
        subjects[cod] = entry
    return subjects


class SpectrogramDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = windows  # [N, 18, F, T]
        self.y = labels   # [N]
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class SpectroCNN(nn.Module):
    """Small 2D CNN for spectrogram [18, n_freq, n_time] → regression.
    Global Avg Pool to minimize params: ~14K total."""
    def __init__(self, in_channels=18, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)


def train_one_fold(train_cods, val_cods, test_cods, subjects, target, args):
    tr_X = np.concatenate([subjects[c]['windows'] for c in train_cods], axis=0)
    tr_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c][target]) for c in train_cods], axis=0)
    vl_X = np.concatenate([subjects[c]['windows'] for c in val_cods], axis=0) if val_cods else None
    vl_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c][target]) for c in val_cods], axis=0) if val_cods else None

    train_ds = SpectrogramDataset(tr_X, tr_y)
    val_ds = SpectrogramDataset(vl_X, vl_y) if vl_y is not None else None
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False) if val_ds else None

    model = SpectroCNN(in_channels=18, dropout=args['dropout']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    log_header('regr')

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        tr_preds, tr_trues = [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * X.size(0)
            tr_count += X.size(0)
            tr_preds.extend(pred.detach().cpu().numpy())
            tr_trues.extend(y.cpu().numpy())
        tr_loss = tr_loss_sum / tr_count
        tr_true_s, tr_pred_s = subject_aggregate(tr_preds, tr_trues, train_cods, subjects, target)
        tr_m = regression_metrics(tr_true_s, tr_pred_s)

        if val_loader is not None:
            model.eval()
            vl_loss_sum, vl_count = 0.0, 0
            vl_preds, vl_trues = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    loss = criterion(pred, y)
                    vl_loss_sum += loss.item() * X.size(0)
                    vl_count += X.size(0)
                    vl_preds.extend(pred.cpu().numpy())
                    vl_trues.extend(y.cpu().numpy())
            vl_loss = vl_loss_sum / vl_count
            vl_true_s, vl_pred_s = subject_aggregate(vl_preds, vl_trues, val_cods, subjects, target)
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
            for chunk in wins_t.chunk(4, dim=0):
                preds.append(model(chunk).cpu().numpy())
            test_pred.append(np.concatenate(preds).mean())
            test_true.append(subjects[cod][target])

    test_m = log_fold_test(np.array(test_true), np.array(test_pred), 'regr')
    return test_true, test_pred, test_m


def run_loso(subjects, args):
    cods = sorted(subjects.keys())

    for target in TARGETS:
        valid_cods = [c for c in cods if subjects[c].get(target) is not None and not (isinstance(subjects[c][target], float) and np.isnan(subjects[c][target]))]
        nv = len(valid_cods)
        yv = np.array([subjects[c][target] for c in valid_cods])
        print(f"\n{'='*60}")
        print(f"  LOSO SpectroCNN — {target}: n={nv} range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print(f"{'='*60}")

        fold_metrics = []
        for ti, test_cod in enumerate(valid_cods):
            train_val_cods = [c for i,c in enumerate(valid_cods) if i != ti]
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


def run_gkf(subjects, args):
    cods = sorted(subjects.keys())

    for target in TARGETS:
        valid_cods = [c for c in cods if subjects[c].get(target) is not None and not (isinstance(subjects[c][target], float) and np.isnan(subjects[c][target]))]
        nv = len(valid_cods)
        yv = np.array([subjects[c][target] for c in valid_cods])
        # Binarize for StratifiedGroupKFold (requires discrete labels)
        labels = np.array([0 if subjects[c][target] <= np.median(yv) else 1 for c in valid_cods])

        print(f"\n{'='*60}")
        print(f"  GKF k={args['k']} SpectroCNN — {target}: n={nv} range=[{yv.min():.0f},{yv.max():.0f}] mean={yv.mean():.1f} std={yv.std():.1f}")
        print(f"{'='*60}")

        outer_gkf = StratifiedGroupKFold(n_splits=args['k'], shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = []

        for fold_id, (train_val_idx, test_idx) in enumerate(outer_gkf.split(np.zeros(nv), labels, groups=valid_cods)):
            train_val_cods = [valid_cods[i] for i in train_val_idx]
            test_cods = [valid_cods[i] for i in test_idx]

            # Inner split: StratifiedGroupKFold (same pattern as create_dataloaders in modma_db.py)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['los','gkf'], default='los')
    p.add_argument('--k', type=int, default=5, help='k for GKF outer folds')
    p.add_argument('--inner-k', type=int, default=5, help='k for GKF inner folds (unused, uses GroupShuffleSplit)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--show-epoch', type=int, default=5)
    p.add_argument('--quick', action='store_true')
    return p.parse_args()


def main():
    args_ns = parse_args()
    print(f"Device: {device}")
    print(f"Input: OC-only spectrograms [18, ~33, ~15] (log1p)")

    if args_ns.quick:
        args_ns.epochs = 2; args_ns.patience = 1
        if args_ns.split == 'gkf': args_ns.k = 3

    print("\nLoading OC-only EEG + computing spectrograms...")
    subjects = load_oc_subjects()
    n = len(subjects)
    total_wins = sum(len(s['windows']) for s in subjects.values())
    dummy_spec = subjects[next(iter(subjects))]['windows'][0]
    print(f"  {n} subjects, {total_wins} windows, spectrogram shape: {list(dummy_spec.shape)}")
    model_tmp = SpectroCNN(18, 0.2)
    n_model_params = sum(p.numel() for p in model_tmp.parameters())
    print(f"  Model: SpectroCNN (2x Conv2d + GAP + Linear(32→1), {n_model_params:,} params)")

    train_args = {
        'batch_size': args_ns.batch_size,
        'lr': args_ns.lr,
        'wd': args_ns.wd,
        'epochs': args_ns.epochs,
        'patience': args_ns.patience,
        'dropout': args_ns.dropout,
        'show_epoch': args_ns.show_epoch,
        'k': args_ns.k,
        'inner_k': args_ns.inner_k,
    }

    if args_ns.split == 'gkf':
        run_gkf(subjects, train_args)
    else:
        run_loso(subjects, train_args)


if __name__ == '__main__':
    main()
