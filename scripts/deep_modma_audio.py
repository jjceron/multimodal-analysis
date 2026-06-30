"""
MODMA Audio: Temporal Attention Segments (TAS).
Attention across audio files -> subject embedding. ~500 params.
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd, glob
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.io.wavfile as wav
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from src.utils.training_logger import log_header, log_epoch, log_fold_test, log_summary

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

AUDIO_DIR = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015'
AUDIO_XLSX = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx'
RANDOM_STATE = 42


def extract_features(wav_path):
    """Extract 15 spectral features from a wav file."""
    try:
        sr, audio = wav.read(wav_path)
    except: return None
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32: audio = audio.astype(np.float32) / 2147483648.0
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=512, noverlap=256)
    feat = np.zeros(15, dtype=np.float32)
    feat[0] = len(audio) / max(sr, 1)
    feat[1] = np.mean(Sxx); feat[2] = np.std(Sxx)
    feat[3] = np.max(Sxx); feat[4] = np.sum(Sxx**2)
    bands = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
    for bi, (lo, hi) in enumerate(bands.values()):
        mask = (f >= lo) & (f <= hi)
        feat[5+bi] = np.mean(Sxx[mask, :]) if mask.sum() > 0 else 0.0
    feat[10] = np.sqrt(np.mean(audio**2))
    feat[11] = np.mean(np.abs(np.diff(np.sign(audio))) > 0)
    feat[12] = np.sum(f * Sxx.mean(axis=1)) / (Sxx.mean(axis=1).sum() + 1e-10)
    feat[13] = np.sqrt(np.sum((f - feat[12])**2 * Sxx.mean(axis=1)) / (Sxx.mean(axis=1).sum() + 1e-10))
    feat[14] = int(sr)
    return feat


def load_data():
    """Load per-subject: [n_files, 15] features + label."""
    df = pd.read_excel(AUDIO_XLSX)
    excel_types = df['type'].tolist()
    sub_dirs = sorted(glob.glob(os.path.join(AUDIO_DIR, '020*')))

    subjects = {}
    for sd, tp in zip(sub_dirs, excel_types):
        sub_id = os.path.basename(sd)
        wavs = sorted(glob.glob(os.path.join(sd, '*.wav')))
        feats = []
        for wf in wavs:
            f = extract_features(wf)
            if f is not None: feats.append(f)
        if not feats: continue
        subjects[sub_id] = {
            'windows': np.stack(feats, axis=0).astype(np.float32),
            'label': 1 if tp == 'MDD' else 0
        }
    return subjects


class TemporalAttention(nn.Module):
    """Self-attention across audio files -> subject embedding."""
    def __init__(self, in_dim=15, d_model=8, dropout=0.3):
        super().__init__()
        self.project = nn.Linear(in_dim, d_model)
        self.attn_key = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, N_files, 15] or [B, 15] (single file)
        if x.dim() == 2: x = x.unsqueeze(1)  # [B, 1, 15]
        B, N, D = x.shape
        h = self.project(x)  # [B, N, D']
        scores = self.attn_key(h)  # [B, N, 1]
        alpha = F.softmax(scores, dim=1)  # [B, N, 1]
        z = (alpha * h).sum(dim=1)  # [B, D']
        z = self.dropout(z)
        return self.classifier(z).squeeze(-1)


def train_one_fold(train_dict, val_dict, test_dict, args):
    train_subjects_dict = train_dict
    # Collapse subjects to [N_train, N_files_per_subj, 15]
    train_X = np.concatenate([train_subjects_dict[c]['windows'] for c in train_subjects_dict], axis=0)
    train_y = np.concatenate([np.full(len(train_dict[c]['windows']), train_dict[c]['label'])
                              for c in train_dict], axis=0)
    val_cods = list(val_dict.keys())
    test_cods = list(test_dict.keys())

    # Standardize per fold
    t_mean = train_X.mean(axis=0, keepdims=True); t_std = train_X.std(axis=0, keepdims=True) + 1e-8
    train_X = (train_X - t_mean) / t_std

    # Build val set
    max_files = max(len(train_dict[c]['windows']) for c in train_dict)
    max_files = max(max_files, max(len(val_dict.get(c, {}).get('windows', [])) for c in val_cods))
    max_files = max(max_files, max(len(test_dict.get(c, {}).get('windows', [])) for c in test_cods))

    # Train in file-level batches
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(train_X).float(), torch.from_numpy(train_y).float())
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=min(args['batch_size'], len(train_X)),
                                                shuffle=True)

    model = TemporalAttention(in_dim=15, d_model=8, dropout=args['dropout']).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  TAS params: {n:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    pw = torch.tensor([np.sum(train_y==0)/max(np.sum(train_y==1), 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val_loss = float('inf'); best_state = None; patience = 0
    log_header('clas')

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        tr_loss, tr_count = 0.0, 0; tr_p, tr_t = [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)  # per-file
            loss = crit(pred, y)
            loss.backward(); optimizer.step()
            tr_loss += loss.item() * X.size(0); tr_count += X.size(0)
            tr_p.extend(torch.sigmoid(pred).detach().cpu().numpy())
            tr_t.extend(y.cpu().numpy())
        tr_loss /= tr_count; scheduler.step()
        tr_m = _bin(np.array(tr_t), (np.array(tr_p) >= 0.5).astype(int))

        # Val: aggregate per subject
        model.eval()
        vl_p_all, vl_t_all = [], []
        vl_loss_sum, vl_count = 0.0, 0
        with torch.no_grad():
            for cod in val_cods:
                w = val_dict[cod]['windows']
                w_n = (w - t_mean) / t_std
                preds = torch.sigmoid(model(torch.from_numpy(w_n).float().to(device))).cpu().numpy()
                vl_p_all.append(preds.mean())
                vl_t_all.append(val_dict[cod]['label'])
                w_n = (w - t_mean) / t_std
                preds = torch.sigmoid(model(torch.from_numpy(w_n).float().to(device))).cpu().numpy()
                vl_p_all.append(preds.mean())
                vl_t_all.append(val_dict[cod]['label'])
            vl_p = np.array(vl_p_all); vl_t = np.array(vl_t_all)
            loss_val = F.binary_cross_entropy(torch.from_numpy(vl_p).float(), torch.from_numpy(vl_t).float())
            vl_loss = loss_val.item()
        vl_m = _bin(vl_t, (vl_p >= 0.5).astype(int))

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss; best_state = model.state_dict(); patience = 0
        else: patience += 1

        show = args.get('show_epoch', 20)
        if epoch == 1 or epoch % show == 0 or patience == 0 or epoch == args['epochs']:
            log_epoch(epoch, tr_loss, vl_loss, tr_m, vl_m, patience, 'clas')

        if patience >= args['patience']: break

    # Test
    model.load_state_dict(best_state); model.eval()
    test_p, test_t = [], []
    with torch.no_grad():
        for cod in test_cods:
            w = test_dict[cod]['windows']
            w_n = (w - t_mean) / t_std
            preds = torch.sigmoid(model(torch.from_numpy(w_n).float().to(device))).cpu().numpy()
            test_p.append(preds.mean()); test_t.append(test_dict[cod]['label'])
    tp = np.array(test_p); tt = np.array(test_t)
    test_m = log_fold_test(tt, (tp >= 0.5).astype(int), 'clas')
    return tt, (tp >= 0.5).astype(int), test_m


def _bin(t, p):
    return {'acc':float(accuracy_score(t,p)), 'bacc':float(balanced_accuracy_score(t,p)),
            'f1':float(f1_score(t,p,zero_division=0)),
            'sens':float(f1_score(t,p,pos_label=1,zero_division=0)),
            'spec':float(f1_score(t,p,pos_label=0,zero_division=0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--show-epoch', type=int, default=30)
    parser.add_argument('--quick', action='store_true')
    args_ns = parser.parse_args()

    if args_ns.quick: args_ns.epochs = 5; args_ns.patience = 3; args_ns.k = 3

    print(f"Device: {device}")
    print(f"MODMA Audio: Temporal Attention over Segments (TAS)")

    subjects = load_data()
    cods = sorted(subjects.keys())
    labels = np.array([subjects[c]['label'] for c in cods])
    print(f"  Subjects: {len(subjects)} (MDD: {np.sum(labels==1)}, HC: {np.sum(labels==0)})")
    total = sum(len(s['windows']) for s in subjects.values())
    print(f"  Total files: {total} (~{total//len(subjects)} per subject)")

    train_args = {'batch_size': args_ns.batch_size, 'lr': args_ns.lr, 'wd': args_ns.wd,
                  'epochs': args_ns.epochs, 'patience': args_ns.patience,
                  'dropout': args_ns.dropout, 'show_epoch': args_ns.show_epoch}
    n_folds = args_ns.k
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold_id, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(cods)), labels, groups=cods)):
        tv_cods = [cods[i] for i in train_val_idx]
        test_cods = [cods[i] for i in test_idx]

        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+fold_id)
        tv_labels = labels[train_val_idx]
        tr_i, vl_i = next(inner.split(np.zeros(len(tv_cods)), tv_labels, groups=tv_cods))

        train_dict = {c: subjects[c] for c in tv_cods}
        val_dict = {tv_cods[i]: subjects[tv_cods[i]] for i in vl_i}
        test_dict = {c: subjects[c] for c in test_cods}

        print(f"\nFold {fold_id+1}/{n_folds}: train={len(tr_i)} val={len(vl_i)} test={len(test_idx)}")
        _, _, fm = train_one_fold(train_dict, val_dict, test_dict, train_args)
        fold_metrics.append(fm)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    log_summary(fold_metrics, n_folds=n_folds, mode='clas', split_type='gkf')
    print(f"\nBaseline classical audio: RF d=10 bacc=0.572")


if __name__ == '__main__': main()
