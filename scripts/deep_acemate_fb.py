"""
Deep Learning: Filter bank + augmentation → window-level regression
OC-only EEG → COG (MSE). No MeanPool across windows.
"""
import glob, os, re, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.eegnet import EEGNet
from src.models.augmentations import GaussianNoise, ChannelDropout, TimeMasking

torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
RANDOM_STATE = 42;  SFREQ = 250;  WINDOW_SEC = 2.0;  OVERLAP = 0.5
BAND_RANGES = [(0.5,4),(4,8),(8,13),(13,30),(30,50)]

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


# ── EEG loading + filter bank ──────────────────────────────────────────
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
                if isinstance(l,bytes): ch_names.append(l.decode('utf-8',errors='replace').strip('\x00'))
                elif hasattr(l,'tobytes'): ch_names.append(l.tobytes().decode('utf-8',errors='replace').strip('\x00'))
                else: ch_names.append(str(l))
        data=np.fromfile(fdt,dtype=np.float32).reshape(nch,pnts,order='F')
        return data,ch_names,sr


def apply_filter_bank(data_18, sfreq):
    filtered = []
    for lo, hi in BAND_RANGES:
        sos = butter(4, [lo, hi], btype='band', fs=sfreq, output='sos')
        for ch in range(data_18.shape[0]):
            filtered.append(sosfiltfilt(sos, data_18[ch]))
    return np.array(filtered, dtype=np.float32)


def load_oc_subjects():
    meta = pd.read_excel(META_PATH)
    meta = meta.set_index('Cod')
    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    ws = int(WINDOW_SEC * SFREQ);  stride = int(ws * (1 - OVERLAP))
    fb_data = {}

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
        data_90 = apply_filter_bank(data_18, sfreq)
        data_90 = (data_90 - data_90.mean(axis=1, keepdims=True)) / (data_90.std(axis=1, keepdims=True) + 1e-10)
        n_w = (data_90.shape[1] - ws) // stride + 1
        windows = np.lib.stride_tricks.sliding_window_view(data_90, ws, axis=1)[:, ::stride].transpose(1,0,2)
        windows = windows[:n_w].astype(np.float32)
        fb_data[cod] = windows

    subjects = {}
    for cod, w in fb_data.items():
        if cod not in meta.index: continue
        cog = meta.loc[cod, 'COG']
        if pd.isna(cog): continue
        subjects[cod] = {'windows': w, 'cog': float(cog)}
    return subjects


# ── Dataset: window-level ──────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, all_windows, all_labels):
        self.X = all_windows
        self.y = all_labels
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


# ── Model: EEGNet per window, aggregate=True ──────────────────────────
class WindowRegressor(nn.Module):
    def __init__(self, n_channels=90, dropout=0.5):
        super().__init__()
        self.eegnet = EEGNet(n_channels=n_channels, n_classes=1, F1=8, D=2, F2=16,
                             temporal_kern=63, separable_kern=15, pool1=8, pool2=8,
                             dropout=dropout, meanmax_alpha=0.0)
    def forward(self, x):
        logits, _ = self.eegnet(x)
        return logits.squeeze(-1)


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)


def subject_metrics(windows, model, n_chunks=4):
    """Predict all windows for a subject, return mean prediction."""
    model.eval()
    chunks = torch.from_numpy(windows).chunk(n_chunks, dim=0)
    preds = []
    with torch.no_grad():
        for chunk in chunks:
            preds.append(model(chunk.to(device)).cpu().numpy())
    return np.concatenate(preds).mean()


def subject_regression_metrics(true, pred):
    """Compute MAE, R2, NRMSE, Spearman r, Pearson r at subject level."""
    t = np.array(true, dtype=float); p = np.array(pred, dtype=float)
    mae = mean_absolute_error(t, p) if len(t) > 1 else float('nan')
    r2 = r2_score(t, p) if len(t) > 1 else float('nan')
    nrmse = np.sqrt(np.mean((t - p)**2)) / (t.std() + 1e-10) if len(t) > 1 else float('nan')
    sr, sp = stats.spearmanr(t, p) if len(t) > 2 else (float('nan'), 1.0)
    pr, pp = stats.pearsonr(t, p) if len(t) > 2 else (float('nan'), 1.0)
    return {'mae': mae, 'r2': r2, 'nrmse': nrmse, 'spear': sr, 'pear': pr, 'spear_p': sp, 'pear_p': pp}


def aggregate_subject_predictions(preds_window, trues_window, cods, subjects):
    """Aggregate per-window predictions into per-subject predictions."""
    true_s, pred_s = [], []
    offset = 0
    for cod in cods:
        nw = len(subjects[cod]['windows'])
        pred_s.append(np.mean(preds_window[offset:offset+nw]))
        sub_trues = np.array(trues_window[offset:offset+nw])
        true_s.append(sub_trues[0] if len(sub_trues) > 0 else subjects[cod]['cog'])
        offset += nw
    return np.array(true_s), np.array(pred_s)


def train_one_fold(train_cods, val_cods, test_cods, subjects, args):
    # Build window-level datasets
    tr_X = np.concatenate([subjects[c]['windows'] for c in train_cods], axis=0)
    tr_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c]['cog'])
                           for c in train_cods], axis=0)
    vl_X = np.concatenate([subjects[c]['windows'] for c in val_cods], axis=0) if val_cods else None
    vl_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c]['cog'])
                           for c in val_cods], axis=0) if val_cods else None

    train_ds = WindowDataset(tr_X, tr_y)
    val_ds = WindowDataset(vl_X, vl_y) if vl_y is not None else None
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False) if val_ds else None

    model = WindowRegressor(n_channels=90, dropout=args['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    criterion = nn.MSELoss()

    aug = nn.Sequential(GaussianNoise(snr=20.0), ChannelDropout(p=0.15),
                        TimeMasking(max_mask_ratio=0.15)) if args.get('augment') else None

    # Train-set COG median for binarization
    train_cogs = np.array([subjects[c]['cog'] for c in train_cods])
    threshold = np.median(train_cogs)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    hdr1 = f"  {'Epoch':>5s} | {'T_loss':>8s} {'V_loss':>8s} {'T_mae':>7s} {'V_mae':>7s}"
    hdr2 = f" | {'V_r2':>7s} {'V_spear':>7s} {'V_pear':>7s} {'V_nrmse':>8s} | pat"
    print(hdr1 + hdr2)

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        tr_loss_sum, tr_count = 0.0, 0
        tr_preds_all, tr_trues_all = [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if aug is not None:
                X = aug(X)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * X.size(0)
            tr_count += X.size(0)
            tr_preds_all.extend(pred.detach().cpu().numpy())
            tr_trues_all.extend(y.cpu().numpy())

        tr_loss = tr_loss_sum / tr_count
        tr_true_s, tr_pred_s = aggregate_subject_predictions(tr_preds_all, tr_trues_all, train_cods, subjects)
        tr_metrics = subject_regression_metrics(tr_true_s, tr_pred_s)

        # Validation
        if val_loader is not None:
            model.eval()
            vl_loss_sum, vl_count = 0.0, 0
            vl_preds_all, vl_trues_all = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    loss = criterion(pred, y)
                    vl_loss_sum += loss.item() * X.size(0)
                    vl_count += X.size(0)
                    vl_preds_all.extend(pred.cpu().numpy())
                    vl_trues_all.extend(y.cpu().numpy())
            vl_loss = vl_loss_sum / vl_count
            vl_true_s, vl_pred_s = aggregate_subject_predictions(vl_preds_all, vl_trues_all, val_cods, subjects)
            vl_metrics = subject_regression_metrics(vl_true_s, vl_pred_s)
        else:
            vl_loss = float('inf')
            vl_true_s, vl_pred_s = np.array([]), np.array([])
            vl_metrics = {'mae': 0.0, 'r2': 0.0, 'spear': 0.0, 'pear': 0.0}

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        show = args.get('show_epoch', 5)
        if epoch == 1 or epoch % show == 0 or patience == 0 or epoch == args['epochs']:
            print(f"  {epoch:5d} | {tr_loss:8.4f} {vl_loss:8.4f} "
                  f"{tr_metrics['mae']:7.3f} {vl_metrics['mae']:7.3f}"
                  f" | {vl_metrics['r2']:7.3f} {vl_metrics['spear']:7.3f} "
                  f"{vl_metrics['pear']:7.3f} {vl_metrics['nrmse']:8.3f} | {patience:2d}")

        if patience >= args['patience']:
            break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_true, test_pred = [], []
    for cod in test_cods:
        windows = subjects[cod]['windows']
        p = subject_metrics(windows, model)
        test_true.append(subjects[cod]['cog'])
        test_pred.append(p)

    test_true = np.array(test_true); test_pred = np.array(test_pred)
    test_m = subject_regression_metrics(test_true, test_pred)

    print(f"  >>> test: mae={test_m['mae']:.3f} r2={test_m['r2']:+.3f} "
          f"spear={test_m['spear']:+.3f} pear={test_m['pear']:+.3f} "
          f"nrmse={test_m['nrmse']:.3f}")
    return test_true, test_pred


def run_loso(subjects, args):
    cods = sorted(subjects.keys())
    n = len(cods)
    cogs = np.array([subjects[c]['cog'] for c in cods])
    print(f"\n{'='*60}")
    print(f"  LOSO (FB+Aug, OC-only, COG regr, WINDOW-level): {n} subj")
    print(f"  COG: range=[{cogs.min():.0f},{cogs.max():.0f}] mean={cogs.mean():.1f} std={cogs.std():.1f}")
    print(f"{'='*60}")

    all_true, all_pred = [], []
    for ti, test_cod in enumerate(cods):
        train_val_cods = [c for i,c in enumerate(cods) if i != ti]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+ti)
        tr_i, vl_i = next(gss.split(train_val_cods, groups=train_val_cods))
        train_cods = [train_val_cods[i] for i in tr_i]
        val_cods = [train_val_cods[i] for i in vl_i]

        set_seed(RANDOM_STATE + ti)
        print(f"\n[{ti+1:2d}/{n}] Test={test_cod}  train={len(train_cods)} val={len(val_cods)}")
        print(f"  train windows: {sum(len(subjects[c]['windows']) for c in train_cods)}")
        test_true, test_pred = train_one_fold(train_cods, val_cods, [test_cod], subjects, args)
        all_true.extend(test_true); all_pred.extend(test_pred)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_true = np.array(all_true); all_pred = np.array(all_pred)
    m = subject_regression_metrics(all_true, all_pred)

    print(f"\n{'='*60}")
    print(f"  LOSO FINAL ({n} folds)")
    print(f"  mae={m['mae']:.3f}  r2={m['r2']:+.3f}  nrmse={m['nrmse']:.3f}  "
          f"spear={m['spear']:+.3f} (p={m['spear_p']:.3f})  "
          f"pear={m['pear']:+.3f} (p={m['pear_p']:.3f})")
    print(f"{'='*60}")


def run_gkf(subjects, args):
    cods = sorted(subjects.keys())
    n = len(cods)
    cogs = np.array([subjects[c]['cog'] for c in cods])
    # Binarize COG for StratifiedGroupKFold (requires discrete labels)
    cog_median = np.median(cogs)
    labels = np.array([0 if subjects[c]['cog'] <= cog_median else 1 for c in cods])

    print(f"\n{'='*60}")
    print(f"  GKF k={args['k']} inner={args['inner_k']} (FB+Aug, OC-only, COG regr, WINDOW-level)")
    print(f"  {n} subj, COG: range=[{cogs.min():.0f},{cogs.max():.0f}] mean={cogs.mean():.1f} std={cogs.std():.1f}")
    print(f"{'='*60}")

    outer_gkf = StratifiedGroupKFold(n_splits=args['k'], shuffle=True, random_state=RANDOM_STATE)
    all_test_true, all_test_pred = [], []
    fold_metrics = []

    for fold_id, (train_val_idx, test_idx) in enumerate(outer_gkf.split(np.zeros(n), labels, groups=cods)):
        train_val_cods = [cods[i] for i in train_val_idx]
        test_cods = [cods[i] for i in test_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+fold_id)
        tr_i, vl_i = next(gss.split(train_val_cods, groups=train_val_cods))
        train_cods = [train_val_cods[i] for i in tr_i]
        val_cods = [train_val_cods[i] for i in vl_i]

        print(f"\nFold {fold_id+1}/{args['k']}: train={len(train_cods)} val={len(val_cods)} test={len(test_cods)}")
        print(f"  train windows: {sum(len(subjects[c]['windows']) for c in train_cods)}")

        set_seed(RANDOM_STATE + fold_id)
        test_true, test_pred = train_one_fold(train_cods, val_cods, test_cods, subjects, args)
        all_test_true.extend(test_true); all_test_pred.extend(test_pred)
        fold_m = subject_regression_metrics(test_true, test_pred)
        fold_metrics.append(fold_m)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_true = np.array(all_test_true); all_pred = np.array(all_test_pred)
    all_m = subject_regression_metrics(all_true, all_pred)

    fold_keys = ['r2', 'mae', 'spear', 'pear', 'nrmse']
    fold_arrays = {k: np.array([m[k] for m in fold_metrics]) for k in fold_keys}

    nfolds = args['k']
    w_k, w_m, w_f = 7, 8, 10  # widths: metric, mean±std, fold

    print(f"\n{'='*60}")
    print(f"  GKF RESULT ({nfolds} folds)")

    # Table 1: mean ± std
    print(f"  {'':>{w_k}} | {'mean':>{w_m}} {'±':>2s} {'std':>{w_m-3}}")
    print(f"  {'':->{w_k}}-+-{'-'*w_m}---{'-'*w_m}")
    for k in fold_keys:
        mn, sd = np.mean(fold_arrays[k]), np.std(fold_arrays[k])
        print(f"  {k:>{w_k}} | {mn:{w_m}.3f} ± {sd:>{w_m-3}.3f}")
    print()

    # Table 2: per-fold values
    fold_hdr = ' | '.join([f"{k:>{w_k}}" for k in fold_keys])
    fold_sep = '-+-'.join(['-'*w_k for _ in fold_keys])
    print(f"  {'Fold':>{w_k}} | {fold_hdr}")
    print(f"  {'':->{w_k}}-+-{fold_sep}")
    for fi in range(nfolds):
        vals = ' | '.join([f"{fold_arrays[k][fi]:{w_k}.3f}" for k in fold_keys])
        print(f"  {fi+1:>{w_k}} | {vals}")

    print(f"{'='*60}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['los', 'gkf'], default='los')
    p.add_argument('--k', type=int, default=5, help='k for GKF outer folds')
    p.add_argument('--inner-k', type=int, default=5, help='not used (GroupShuffleSplit 80/20)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--show-epoch', type=int, default=5)
    p.add_argument('--augment', action='store_true', default=True)
    p.add_argument('--no-augment', dest='augment', action='store_false')
    p.add_argument('--quick', action='store_true')
    return p.parse_args()


def main():
    args_ns = parse_args()
    print(f"Device: {device}")
    print(f"Split: {args_ns.split}")
    print(f"Filter bank: [18ch x5bands] -> [90, 500]")
    print(f"Augment: {'ON' if args_ns.augment else 'OFF'}")
    print(f"Model: EEGNet(90ch, n_classes=1) per-window regression")

    if args_ns.quick:
        args_ns.epochs = 5; args_ns.patience = 2
        if args_ns.split == 'gkf': args_ns.k = 3

    print("\nLoading OC-only EEG + filter bank...")
    subjects = load_oc_subjects()
    cogs = np.array([s['cog'] for s in subjects.values()])
    total_wins = sum(len(s['windows']) for s in subjects.values())
    print(f"  Loaded {len(subjects)} subjects, {total_wins} total windows")
    print(f"  COG: range=[{cogs.min():.0f},{cogs.max():.0f}] mean={cogs.mean():.1f} std={cogs.std():.1f}")

    train_args = {
        'batch_size': args_ns.batch_size,
        'lr': args_ns.lr,
        'wd': args_ns.wd,
        'epochs': args_ns.epochs,
        'patience': args_ns.patience,
        'dropout': args_ns.dropout,
        'show_epoch': args_ns.show_epoch,
        'augment': args_ns.augment,
        'k': args_ns.k,
        'inner_k': args_ns.inner_k,
    }

    if args_ns.split == 'gkf':
        run_gkf(subjects, train_args)
    else:
        run_loso(subjects, train_args)


if __name__ == '__main__':
    main()
