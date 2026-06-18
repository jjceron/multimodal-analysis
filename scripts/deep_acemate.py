"""
Deep learning baseline for ACEMATE — Classification only (Tipo)
EEGNet + MeanPooling across windows, LOSO evaluation
"""
import glob, os, re, warnings, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.eegnet import EEGNet

warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

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


def load_all_subjects():
    """Load all EEG files, select 18 channels, z-score, extract windows.
    Returns dict: cod -> {windows, label}
    """
    meta = pd.read_excel(META_PATH)
    meta['MOT_V4']=meta[['8.','13.','16.','21.','23.']].sum(axis=1)
    meta['COG_V1']=meta[['3.','6.']].sum(axis=1)
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
        # Select 18 channels
        sel = []
        for ch in CHANNEL_18:
            found = False
            for on, tn in cm.items():
                if tn == ch and on in ch_names:
                    sel.append(ch_names.index(on))
                    found = True; break
            if not found:
                # try direct match
                if ch in ch_names:
                    sel.append(ch_names.index(ch))
        if len(sel) < 18:
            continue
        data_18 = data[sel]
        # Z-score per channel
        data_18 = (data_18 - data_18.mean(axis=1, keepdims=True)) / (data_18.std(axis=1, keepdims=True) + 1e-10)
        # Extract windows
        n_w = (data_18.shape[1] - ws) // stride + 1
        windows = np.lib.stride_tricks.sliding_window_view(data_18, ws, axis=1)[:, ::stride].transpose(1,0,2)
        windows = windows[:n_w].astype(np.float32)
        raw.setdefault(cod,{})[cond] = windows

    # Build per-subject data
    subjects = {}
    for cod, conds in raw.items():
        oa = conds.get('OA', None)
        oc = conds.get('OC', None)
        all_w = []
        if oa is not None: all_w.append(oa)
        if oc is not None: all_w.append(oc)
        if not all_w: continue
        windows = np.concatenate(all_w, axis=0)
        label = meta.loc[cod, 'Tipo'] if cod in meta.index else None
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue
        subjects[cod] = {
            'windows': windows,
            'label': 0 if label == 'low_imp' else 1,
        }
    return subjects


class SubjectDataset(Dataset):
    def __init__(self, cods, subjects):
        self.cods = cods
        self.subjects = subjects
    def __len__(self):
        return len(self.cods)
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


def train_one_fold(train_cods, val_cods, test_cod, subjects, args):
    train_ds = SubjectDataset(train_cods, subjects)
    val_ds = SubjectDataset(val_cods, subjects)
    test_ds = SubjectDataset([test_cod], subjects)

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                               collate_fn=collate_subject_windows)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False,
                             collate_fn=collate_subject_windows)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
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
        tr_acc = tr_correct / tr_total

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
        vl_acc = vl_correct / vl_total

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        if epoch == 1 or epoch % 25 == 0 or patience == 0:
            print(f"  Ep {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                  f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f} | pat={patience:2d}")

        if patience >= args['patience']:
            break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for _, X, y, mask in test_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            logits = model(X, mask)
            pred = logits.argmax(1).item()
            true = y.item()
            correct = int(pred == true)
    return true, pred, correct


def main():
    print("="*70)
    print("  ACEMATE DEEP - EEGNet + MeanPool, LOSO (Tipo)")
    print("="*70+"\n")

    subjects = load_all_subjects()
    cods = sorted(subjects.keys())
    n = len(cods)
    labels = [subjects[c]['label'] for c in cods]
    print(f"Sujetos: {n} ({sum(labels)} high, {len(labels)-sum(labels)} low)")
    n_wins = [subjects[c]['windows'].shape[0] for c in cods]
    print(f"Ventanas: min={min(n_wins)}, max={max(n_wins)}, mean={np.mean(n_wins):.0f}")
    print()

    args = {
        'batch_size': 4,
        'lr': 1e-3,
        'wd': 1e-4,
        'epochs': 100,
        'patience': 15,
        'dropout': 0.5,
    }

    all_true, all_pred = [], []
    for ti, test_cod in enumerate(cods):
        train_val_cods = [c for i,c in enumerate(cods) if i != ti]
        train_val_labels = [labels[i] for i,c in enumerate(cods) if i != ti]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+ti)
        tr_i, vl_i = next(gss.split(train_val_cods, train_val_labels, groups=train_val_cods))
        train_cods = [train_val_cods[i] for i in tr_i]
        val_cods = [train_val_cods[i] for i in vl_i]

        set_seed(RANDOM_STATE + ti)
        print(f"[{ti+1:2d}/{n}] Test: {test_cod}  "
              f"train={len(train_cods)} val={len(val_cods)}", flush=True)

        true, pred, correct = train_one_fold(train_cods, val_cods, test_cod, subjects, args)
        all_true.append(true); all_pred.append(pred)
        print(f"  >>> {test_cod}: true={'high' if true else 'low'} "
              f"pred={'high' if pred else 'low'} {'CORRECT' if correct else 'WRONG'}")
        print()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc = accuracy_score(all_true, all_pred)
    bal = balanced_accuracy_score(all_true, all_pred)
    print(f"\n{'='*70}")
    print(f"  RESULTADO FINAL (LOSO, {n} folds)")
    print(f"{'='*70}")
    print(f"  Correctos: {sum(np.array(all_true)==np.array(all_pred))}/{n}")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Bal Acc:   {bal:.3f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
