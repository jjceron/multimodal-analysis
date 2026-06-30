"""
Deep Learning regression: EEGNet + MeanPool → MOT_V4, MOT, TOTAL
LOSO evaluation with per-subject predictions
"""
import glob, os, re, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.eegnet import EEGNet

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

EEG_DIR = "data/raw/acemate/eeg_speech/eeg_not_locch"
META_PATH = "data/raw/acemate/eeg_speech/metadata.xlsx"
RANDOM_STATE = 42
SFREQ = 250
WINDOW_SEC = 2.0
OVERLAP = 0.5

CHANNEL_18 = ['FP1','F7','T7','P7','F3','C3','P3','O1','FZ','PZ',
              'FP2','F8','T8','P8','F4','C4','P4','O2']

CDMS_MAP = {n:n for n in CHANNEL_18}
CSO_MAP = {'FP1-Cz':'FP1','FP2-Cz':'FP2','FZ-Cz':'FZ','PZ-Cz':'PZ',
    'F3-Cz':'F3','F4-Cz':'F4','F7-Cz':'F7','F8-Cz':'F8',
    'C3-Cz':'C3','C4-Cz':'C4','T3-Cz':'T7','T4-Cz':'T8','T5-Cz':'P7','T6-Cz':'P8',
    'P3-Cz':'P3','P4-Cz':'P4','O1-Cz':'O1','O2-Cz':'O2'}
BIOSEMI_MAP = {'A1':'FP1','A5':'F3','A7':'F7','A12':'C3','A15':'T7','A18':'P3',
    'A21':'P7','A23':'O1','B1':'FP2','B3':'FZ','B5':'F4','B7':'F8',
    'B12':'C4','B15':'T8','B18':'P4','B21':'P8','B24':'O2','B27':'PZ'}


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


def load_subjects():
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4'] = meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    ws = int(WINDOW_SEC * SFREQ)
    stride = int(ws * (1 - OVERLAP))
    raw = {}

    for fpath in all_files:
        bn = os.path.basename(fpath)
        cod = re.sub(r'_(OA|OC)\.set$','',bn)
        cond = 'OA' if '_OA.set' in fpath else 'OC'
        try:
            data, ch_names, sfreq = read_eeg(fpath)
        except:
            continue
        nch = data.shape[0]
        if nch == 32: cm = CDMS_MAP
        elif nch == 19: cm = CSO_MAP
        elif nch >= 137:
            cm = BIOSEMI_MAP
            keep = [i for i,n in enumerate(ch_names) if n in BIOSEMI_MAP]
            data = data[keep]; ch_names = [ch_names[i] for i in keep]
        else: continue

        sel = []
        for ch in CHANNEL_18:
            found = False
            for on, tn in cm.items():
                if tn == ch and on in ch_names:
                    sel.append(ch_names.index(on)); found = True; break
            if not found and ch in ch_names:
                sel.append(ch_names.index(ch))
        if len(sel) < 18: continue
        data_18 = data[sel]
        data_18 = (data_18 - data_18.mean(axis=1, keepdims=True)) / (data_18.std(axis=1, keepdims=True) + 1e-10)

        n_w = (data_18.shape[1] - ws) // stride + 1
        windows = np.lib.stride_tricks.sliding_window_view(data_18, ws, axis=1)[:, ::stride].transpose(1,0,2)
        windows = windows[:n_w].astype(np.float32)
        raw.setdefault(cod,{})[cond] = windows

    subjects = {}
    for cod, conds in raw.items():
        oa = conds.get('OA', None)
        oc = conds.get('OC', None)
        all_w = []
        if oa is not None: all_w.append(oa)
        if oc is not None: all_w.append(oc)
        if not all_w: continue
        windows = np.concatenate(all_w, axis=0)
        # Subsample max 200 windows to speed up training
        if windows.shape[0] > 200:
            rng = np.random.RandomState(42)
            idx = rng.choice(windows.shape[0], 200, replace=False)
            windows = windows[idx]
        subjects[cod] = {
            'windows': windows,
            'MOT_V4': meta.loc[cod, 'MOT_V4'] if cod in meta.index else None,
            'MOT': meta.loc[cod, 'MOT'] if cod in meta.index else None,
            'TOTAL': meta.loc[cod, 'TOTAL'] if cod in meta.index else None,
        }
    return subjects


class SubjectRegDataset(Dataset):
    def __init__(self, cods, subjects, target):
        self.cods = cods
        self.subjects = subjects
        self.target = target

    def __len__(self):
        return len(self.cods)

    def __getitem__(self, idx):
        cod = self.cods[idx]
        d = self.subjects[cod]
        return cod, torch.from_numpy(d['windows']), d[self.target]


def collate_reg(batch):
    names, wins, labs = zip(*batch)
    labs = torch.tensor(labs, dtype=torch.float32)
    max_w = max(w.shape[0] for w in wins)
    C, T = wins[0].shape[1], wins[0].shape[2]
    padded = torch.zeros(len(batch), max_w, C, T)
    mask = torch.zeros(len(batch), max_w, dtype=torch.bool)
    for i, w in enumerate(wins):
        n = w.shape[0]
        padded[i, :n] = w
        mask[i, :n] = True
    return names, padded, labs, mask


class MeanPoolModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, mask=None):
        B, W, C, T = x.shape
        logits, _ = self.backbone(x.view(B * W, C, T))
        window_out = logits.view(B, W, -1)
        if mask is not None:
            window_out = window_out.masked_fill(~mask.unsqueeze(-1), 0.0)
            out = window_out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            out = window_out.mean(dim=1)
        return out.squeeze(-1)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_fold(train_cods, val_cods, test_cod, subjects, target, args):
    train_ds = SubjectRegDataset(train_cods, subjects, target)
    val_ds = SubjectRegDataset(val_cods, subjects, target)
    test_ds = SubjectRegDataset([test_cod], subjects, target)

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_reg)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_reg)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_reg)

    backbone = EEGNet(
        n_channels=18, n_classes=1, F1=8, D=2, F2=16,
        temporal_kern=63, separable_kern=15, pool1=8, pool2=8,
        dropout=args['dropout'], meanmax_alpha=0.0,
    )
    model = MeanPoolModel(backbone).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience = 0
    y_train_mean = np.mean([subjects[c][target] for c in train_cods])

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        tr_loss = 0.0
        for _, X, y, mask in train_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(X, mask)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(train_cods)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for _, X, y, mask in val_loader:
                X, y, mask = X.to(device), y.to(device), mask.to(device)
                pred = model(X, mask)
                loss = criterion(pred, y)
                vl_loss += loss.item() * X.size(0)
        vl_loss /= len(val_cods)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        show = args.get('show_epoch', 5)
        if epoch == 1 or epoch % show == 0 or patience == 0 or epoch == args['epochs']:
            tr_rmse = np.sqrt(tr_loss)
            vl_rmse = np.sqrt(vl_loss)
            print(f"  Ep {epoch:3d} | tr_loss={tr_loss:.4f} (rmse={tr_rmse:.2f}) | "
                  f"vl_loss={vl_loss:.4f} (rmse={vl_rmse:.2f}) | pat={patience:2d}")

        if patience >= args['patience']:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for _, X, y, mask in test_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            pred = model(X, mask)
            true_val = y.item()
            pred_val = pred.item()
    return true_val, pred_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--show-epoch', type=int, default=5)
    parser.add_argument('--quick', action='store_true')
    args_ns = parser.parse_args()

    if args_ns.quick:
        args_ns.epochs = 5
        args_ns.patience = 2

    print("="*70)
    print("  ACEMATE DEEP REGRESSION - EEGNet + MeanPool, LOSO")
    print("="*70+"\n")

    subjects = load_subjects()
    cods = sorted(subjects.keys())
    n = len(cods)
    print(f"Sujetos: {n}")
    n_wins = [subjects[c]['windows'].shape[0] for c in cods]
    print(f"Ventanas: min={min(n_wins)}, max={max(n_wins)}, mean={np.mean(n_wins):.0f}")
    print()

    args = {
        'batch_size': args_ns.batch_size,
        'lr': args_ns.lr,
        'wd': args_ns.wd,
        'epochs': args_ns.epochs,
        'patience': args_ns.patience,
        'dropout': args_ns.dropout,
        'show_epoch': args_ns.show_epoch,
    }

    for target in ['MOT_V4', 'MOT', 'TOTAL']:
        # Filter subjects with valid target
        valid_cods = [c for c in cods if subjects[c].get(target) is not None and not (isinstance(subjects[c][target], float) and np.isnan(subjects[c][target]))]
        nv = len(valid_cods)
        y_all = np.array([subjects[c][target] for c in valid_cods])
        print(f"{'='*70}")
        print(f"  REGRESION: {target}  (n={nv}, range=[{y_all.min():.0f}, {y_all.max():.0f}], "
              f"mean={y_all.mean():.1f}, std={y_all.std():.1f})")
        print(f"{'='*70}\n")

        all_true, all_pred = [], []
        for ti, test_cod in enumerate(valid_cods):
            train_val_cods = [c for i,c in enumerate(valid_cods) if i != ti]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+ti)
            tr_i, vl_i = next(gss.split(train_val_cods, groups=train_val_cods))
            train_cods = [train_val_cods[i] for i in tr_i]
            val_cods = [train_val_cods[i] for i in vl_i]

            set_seed(RANDOM_STATE + ti)
            print(f"[{ti+1:2d}/{nv}] Test: {test_cod:30s}  "
                  f"train={len(train_cods)} val={len(val_cods)}", end='', flush=True)

            true_val, pred_val = train_one_fold(train_cods, val_cods, test_cod, subjects, target, args)
            all_true.append(true_val)
            all_pred.append(pred_val)
            print(f"  true={true_val:.1f}  pred={pred_val:.1f}  "
                  f"err={abs(pred_val-true_val):.1f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Metrics
        r2 = r2_score(all_true, all_pred)
        mae = mean_absolute_error(all_true, all_pred)
        pearson_r, pearson_p = stats.pearsonr(all_true, all_pred)
        spearman_r, spearman_p = stats.spearmanr(all_true, all_pred)
        pred_mean = np.mean(all_pred)
        pred_std = np.std(all_pred)

        # Baseline: predict mean
        y_mean = np.mean(all_true)
        r2_baseline = r2_score(all_true, [y_mean] * len(all_true))
        mae_baseline = mean_absolute_error(all_true, [y_mean] * len(all_true))

        print(f"\n  >>> {target} RESULTS:")
        print(f"      R2      = {r2:+.4f}  (baseline={r2_baseline:.4f})")
        print(f"      MAE     = {mae:.3f}  (baseline={mae_baseline:.3f})")
        print(f"      Pearson = {pearson_r:+.4f}  (p={pearson_p:.4f})")
        print(f"      Spearman= {spearman_r:+.4f}  (p={spearman_p:.4f})")
        print(f"      Pred distribution: mean={pred_mean:.1f}, std={pred_std:.1f} "
              f"(true std={np.std(all_true):.1f})")
        print()

    print(f"{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
