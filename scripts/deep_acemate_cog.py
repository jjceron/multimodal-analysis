"""
Deep Learning OC-only → COG classification (median split)
EEGNet + MeanPool, LOSO / GKF
Based on finding: OC delta in central/frontal channels predicts cognitive impulsivity
"""
import glob, os, re, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.eegnet import EEGNet

torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def load_oc_subjects():
    """Load ONLY OC (eyes-closed) EEG. Returns dict: cod -> {windows, cog}"""
    meta = pd.read_excel(META_PATH)
    meta = meta.set_index('Cod')

    all_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    ws = int(WINDOW_SEC * SFREQ)
    stride = int(ws * (1 - OVERLAP))

    oc_windows = {}
    for fpath in all_files:
        bn = os.path.basename(fpath)
        if '_OA.set' in fpath:
            continue  # Only OC
        cod = re.sub(r'_OC\.set$','',bn)
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
        oc_windows[cod] = windows

    # Match with metadata and build labels
    subjects = {}
    for cod, w in oc_windows.items():
        if cod not in meta.index: continue
        cog = meta.loc[cod, 'COG']
        if pd.isna(cog): continue
        subjects[cod] = {'windows': w, 'cog': float(cog)}
    return subjects


def build_cog_labels(subjects):
    """Binary labels via median split of COG."""
    cods = sorted(subjects.keys())
    scores = np.array([subjects[c]['cog'] for c in cods])
    median = np.median(scores)

    labeled = {}
    for c in cods:
        v = subjects[c]['cog']
        labeled[c] = {
            'windows': subjects[c]['windows'],
            'label': 0 if v <= median else 1,
        }
    return labeled, median


class SubjectDataset(Dataset):
    def __init__(self, cods, subjects):
        self.cods = cods
        self.subjects = subjects
    def __len__(self): return len(self.cods)
    def __getitem__(self, idx):
        cod = self.cods[idx]
        d = self.subjects[cod]
        return cod, torch.from_numpy(d['windows']), d['label']


def collate_subject_windows(batch):
    names, wins, labs = zip(*batch)
    labs = torch.tensor(labs, dtype=torch.long)
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
        logits = logits.view(B, W, -1)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
            out = logits.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            out = logits.mean(dim=1)
        return out


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_fold(train_cods, val_cods, test_cods, subjects, args):
    train_ds = SubjectDataset(train_cods, subjects)
    val_ds = SubjectDataset(val_cods, subjects)
    test_ds = SubjectDataset(test_cods, subjects)

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                               collate_fn=collate_subject_windows)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False,
                             collate_fn=collate_subject_windows)
    test_loader = DataLoader(test_ds, batch_size=args['batch_size'], shuffle=False,
                              collate_fn=collate_subject_windows)

    backbone = EEGNet(
        n_channels=18, n_classes=2, F1=8, D=2, F2=16,
        temporal_kern=63, separable_kern=15, pool1=8, pool2=8,
        dropout=args['dropout'], meanmax_alpha=0.0,
    )
    model = MeanPoolModel(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for _, X, y, mask in train_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(X, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * X.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += X.size(0)
        tr_loss /= len(train_cods)
        tr_acc = tr_correct / max(1, tr_total)

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for _, X, y, mask in val_loader:
                X, y, mask = X.to(device), y.to(device), mask.to(device)
                logits = model(X, mask)
                loss = criterion(logits, y)
                vl_loss += loss.item() * X.size(0)
                vl_correct += (logits.argmax(1) == y).sum().item()
                vl_total += X.size(0)
        vl_loss /= len(val_cods)
        vl_acc = vl_correct / max(1, vl_total)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        show = args.get('show_epoch', 5)
        if epoch == 1 or epoch % show == 0 or patience == 0 or epoch == args['epochs']:
            print(f"  E{epoch:3d} | tr_acc={tr_acc:.3f} tr_loss={tr_loss:.4f} | "
                  f"vl_acc={vl_acc:.3f} vl_loss={vl_loss:.4f} | pat={patience}")

        if patience >= args['patience']:
            break

    model.load_state_dict(best_state)
    model.eval()
    test_true, test_pred = [], []
    with torch.no_grad():
        for _, X, y, mask in test_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            logits = model(X, mask)
            test_pred.extend(logits.argmax(1).cpu().tolist())
            test_true.extend(y.cpu().tolist())
    return test_true, test_pred


def run_loso(subjects, args):
    cods = sorted(subjects.keys())
    n = len(cods)
    n_high = sum(s['label'] for s in subjects.values())
    print(f"\n{'='*60}")
    print(f"  LOSO (OC-only, COG median split): {n} subjects ({n_high} high, {n - n_high} low)")
    print(f"{'='*60}")

    all_true, all_pred = [], []
    for ti, test_cod in enumerate(cods):
        train_val_cods = [c for i,c in enumerate(cods) if i != ti]
        train_val_labels = [subjects[c]['label'] for c in train_val_cods]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+ti)
        tr_i, vl_i = next(gss.split(train_val_cods, train_val_labels, groups=train_val_cods))
        train_cods = [train_val_cods[i] for i in tr_i]
        val_cods = [train_val_cods[i] for i in vl_i]

        set_seed(RANDOM_STATE + ti)
        test_true, test_pred = train_one_fold(train_cods, val_cods, [test_cod], subjects, args)
        all_true.extend(test_true); all_pred.extend(test_pred)
        print(f"\n[{ti+1:2d}/{n}] {test_cod}: true={test_true[0]} pred={test_pred[0]} "
              f"{'OK' if test_true[0]==test_pred[0] else 'WRONG'}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc = accuracy_score(all_true, all_pred)
    bal = balanced_accuracy_score(all_true, all_pred)
    correct = sum(np.array(all_true) == np.array(all_pred))
    print(f"\n{'='*60}")
    print(f"  LOSO RESULT (OC-only, COG median)")
    print(f"  Acc={acc:.3f}  BalAcc={bal:.3f}  Correct={correct}/{n}")
    print(f"{'='*60}")
    return acc, bal


def run_gkf(subjects, args):
    cods = sorted(subjects.keys())
    n = len(cods)
    labels = np.array([subjects[c]['label'] for c in cods])
    n_high = sum(labels)
    print(f"\n{'='*60}")
    print(f"  GKF k={args['k']} inner={args['inner_k']} (OC-only, COG median): "
          f"{n} subjects ({n_high} high, {n - n_high} low)")
    print(f"{'='*60}")

    outer_gkf = StratifiedGroupKFold(n_splits=args['k'], shuffle=True, random_state=RANDOM_STATE)
    all_test_true, all_test_pred = [], []
    fold_accs, fold_bals = [], []

    for fold_id, (train_val_idx, test_idx) in enumerate(outer_gkf.split(np.zeros(n), labels, groups=cods)):
        train_val_cods = [cods[i] for i in train_val_idx]
        train_val_labels = labels[train_val_idx]
        test_cods = [cods[i] for i in test_idx]

        inner_gkf = StratifiedGroupKFold(n_splits=args['inner_k'], shuffle=True, random_state=RANDOM_STATE)
        tr_i, vl_i = next(inner_gkf.split(
            np.zeros(len(train_val_cods)), train_val_labels, groups=train_val_cods
        ))
        train_cods = [train_val_cods[i] for i in tr_i]
        val_cods = [train_val_cods[i] for i in vl_i]

        print(f"\nFold {fold_id+1}/{args['k']}: train={len(train_cods)} val={len(val_cods)} test={len(test_cods)}")
        set_seed(RANDOM_STATE + fold_id)
        test_true, test_pred = train_one_fold(train_cods, val_cods, test_cods, subjects, args)
        all_test_true.extend(test_true); all_test_pred.extend(test_pred)
        fold_acc = accuracy_score(test_true, test_pred)
        fold_bal = balanced_accuracy_score(test_true, test_pred)
        fold_accs.append(fold_acc); fold_bals.append(fold_bal)
        print(f"  Fold acc={fold_acc:.3f} bal={fold_bal:.3f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc = accuracy_score(all_test_true, all_test_pred)
    bal = balanced_accuracy_score(all_test_true, all_test_pred)
    print(f"\n{'='*60}")
    print(f"  GKF RESULT (OC-only, COG median)")
    print(f"  Acc={acc:.3f}  BalAcc={bal:.3f}")
    print(f"  Per-fold acc: {[f'{a:.3f}' for a in fold_accs]}  "
          f"mean={np.mean(fold_accs):.3f} +- {np.std(fold_accs):.3f}")
    print(f"  Per-fold bal: {[f'{a:.3f}' for a in fold_bals]}  "
          f"mean={np.mean(fold_bals):.3f} +- {np.std(fold_bals):.3f}")
    print(f"{'='*60}")
    return acc, bal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['los', 'gkf'], default='los')
    p.add_argument('--k', type=int, default=5, help='k for GKF outer folds')
    p.add_argument('--inner-k', type=int, default=5, help='k for GKF inner folds')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--show-epoch', type=int, default=5)
    p.add_argument('--quick', action='store_true', help='5 epochs, pat=2, k=3')
    return p.parse_args()


def main():
    args_ns = parse_args()
    print(f"Device: {device}")
    print(f"Split: {args_ns.split} | Quick: {args_ns.quick}")
    print(f"Input: OC-only (eyes closed)\n")

    if args_ns.quick:
        args_ns.epochs = 5
        args_ns.patience = 2
        if args_ns.split == 'gkf':
            args_ns.k = 3
            args_ns.inner_k = 2

    subjects = load_oc_subjects()
    print(f"Loaded OC EEG: {len(subjects)} subjects")

    labeled, median = build_cog_labels(subjects)
    n_high = sum(s['label'] for s in labeled.values())
    print(f"COG median split: median={median:.1f}, "
          f"{len(labeled)} total ({n_high} high, {len(labeled)-n_high} low)")

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

    if args_ns.split == 'los':
        run_loso(labeled, train_args)
    else:
        run_gkf(labeled, train_args)


if __name__ == '__main__':
    main()
