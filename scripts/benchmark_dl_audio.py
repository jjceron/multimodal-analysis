"""
MODMA Audio DL Benchmark — classic DL architectures on mel-spectrograms.

Usage:
    python scripts/benchmark_dl_audio.py --model shallowconvnet --epochs 100
    python scripts/benchmark_dl_audio.py --model deepconvnet --epochs 100
    python scripts/benchmark_dl_audio.py --model eegnet --epochs 100
    python scripts/benchmark_dl_audio.py --model cnnlstm --epochs 100

Input: mel-spectrogram windows [64 mel, 200 frames] per 2s segment
Split: 5-fold StratifiedGroupKFold (subject-aware, no leakage)
"""
import os, sys, glob, warnings, json, argparse
import numpy as np, pandas as pd
import scipy.io.wavfile as wav
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

AUDIO_DIR = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015'
AUDIO_XLSX = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx'
RANDOM_STATE = 42
SR_TARGET = 16000
WINDOW_SEC = 2.0
OVERLAP = 0.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
N_TIMESTEPS = 200

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score


def compute_mel_spectrogram(wav_path):
    try:
        sr, audio = wav.read(wav_path)
    except Exception:
        return None
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != SR_TARGET:
        resampler = torchaudio.transforms.Resample(sr, SR_TARGET)
        waveform = resampler(waveform)
    waveform = waveform[0]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR_TARGET, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0, f_min=20, f_max=8000)(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
    return mel_db.numpy().astype(np.float32)


def extract_mel_windows(mel_spec):
    if mel_spec.shape[1] < N_TIMESTEPS:
        return None
    stride = int(N_TIMESTEPS * (1 - OVERLAP))
    n_w = (mel_spec.shape[1] - N_TIMESTEPS) // stride + 1
    if n_w < 1:
        return None
    win = np.lib.stride_tricks.sliding_window_view(mel_spec, N_TIMESTEPS, axis=1)
    win = win[:, ::stride].transpose(1, 0, 2)[:n_w]
    return win.astype(np.float32)


def load_all_audio_subjects():
    df = pd.read_excel(AUDIO_XLSX).sort_values('subject id')
    labels_all = df['type'].tolist()
    y_binary = [1 if lbl == 'MDD' else 0 for lbl in labels_all]

    sub_dirs = sorted(glob.glob(os.path.join(AUDIO_DIR, '020*')))
    subjects = {}
    n_processed = 0
    for li, (sd, lbl) in enumerate(zip(sub_dirs, y_binary)):
        sub_id = os.path.basename(sd)
        wav_files = sorted(glob.glob(os.path.join(sd, '*.wav')))
        if not wav_files:
            continue
        all_windows = []
        for wf in wav_files:
            mel = compute_mel_spectrogram(wf)
            if mel is None:
                continue
            windows = extract_mel_windows(mel)
            if windows is not None and windows.shape[0] > 0:
                all_windows.append(windows)
        if not all_windows:
            continue
        all_windows = np.concatenate(all_windows, axis=0)
        if all_windows.shape[0] > 200:
            rng = np.random.RandomState(RANDOM_STATE)
            idx = rng.choice(all_windows.shape[0], 200, replace=False)
            all_windows = all_windows[idx]
        subjects[sub_id] = {
            'windows': all_windows,
            'label': lbl,
        }
        n_processed += 1
    print(f"  Processed {n_processed} subjects")
    print(f"  Window shape: {list(subjects.values())[0]['windows'].shape[1:]}")
    return subjects


def build_model_factory(model_name, n_ch, n_samp):
    if model_name == 'shallowconvnet':
        from src.models.shallowconvnet import ShallowConvNet
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = ShallowConvNet(n_channels=n_ch, n_classes=1,
                                         n_samples=n_samp, dropout=0.5)
            def forward(self, x): return self.m(x).squeeze(-1)
        return lambda: Wrap(), 'ShallowConvNet'

    elif model_name == 'deepconvnet':
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.blk1 = nn.Sequential(nn.Conv2d(1, 16, (1, 5)), nn.BatchNorm2d(16),
                                           nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout2d(0.25))
                self.blk2 = nn.Sequential(nn.Conv2d(16, 32, (n_ch, 1)), nn.BatchNorm2d(32),
                                           nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout2d(0.25))
                self.blk3 = nn.Sequential(nn.Conv2d(32, 64, (1, 5)), nn.BatchNorm2d(64),
                                           nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout2d(0.5))
                self.blk4 = nn.Sequential(nn.Conv2d(64, 128, (1, 5)), nn.BatchNorm2d(128),
                                           nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout2d(0.5))
                dummy = torch.randn(1, 1, n_ch, n_samp)
                x = self.blk1(dummy); x = self.blk2(x)
                x = self.blk3(x); x = self.blk4(x)
                self.fc = nn.Linear(int(x.numel()), 1)
            def forward(self, x):
                if x.dim() == 3: x = x.unsqueeze(1)
                x = self.blk1(x); x = self.blk2(x)
                x = self.blk3(x); x = self.blk4(x)
                return self.fc(x.flatten(1)).squeeze(-1)
        return lambda: Wrap(), 'DeepConvNet'

    elif model_name == 'eegnet':
        from src.models.eegnet import EEGNet
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = EEGNet(n_channels=n_ch, n_classes=1,
                                 F1=8, D=2, F2=16,
                                 temporal_kern=31, separable_kern=15,
                                 pool1=4, pool2=4, dropout=0.5,
                                 meanmax_alpha=0.0)
            def forward(self, x):
                logits, _ = self.m(x)
                return logits.squeeze(-1)
        return lambda: Wrap(), 'EEGNet'

    elif model_name == 'cnnlstm':
        from src.models.cnn_lstm import CNNLSTM
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = CNNLSTM(n_channels=n_ch, n_classes=1,
                                  n_samples=n_samp, dropout=0.5)
            def forward(self, x): return self.m(x).squeeze(-1)
        return lambda: Wrap(), 'CNN-LSTM'

    raise ValueError(f"Unknown model: {model_name}")


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

        tr_X = np.concatenate([subjects[c]['windows'] for c in train_subj], axis=0)
        tr_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c]['label'])
                                for c in train_subj], axis=0)
        vl_X = np.concatenate([subjects[c]['windows'] for c in val_subj], axis=0)
        vl_y = np.concatenate([np.full(len(subjects[c]['windows']), subjects[c]['label'])
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
            tr_loss = 0.0; tr_n = 0
            for X, y in tr_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(X), y)
                loss.backward(); opt.step()
                tr_loss += loss.item() * X.size(0); tr_n += X.size(0)
            tr_loss /= tr_n; sched.step()

            model.eval()
            vl_loss = 0.0; vl_n = 0
            with torch.no_grad():
                for X, y in vl_loader:
                    X, y = X.to(device), y.to(device)
                    loss = crit(model(X), y)
                    vl_loss += loss.item() * X.size(0); vl_n += X.size(0)
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
                w = subjects[c]['windows']
                w_n = (w - w.mean(axis=(1, 2), keepdims=True)) / (w.std(axis=(1, 2), keepdims=True) + 1e-8)
                preds = torch.sigmoid(model(torch.from_numpy(w_n).float().to(device)))
                test_pred.append(preds.cpu().numpy().mean())
                test_true.append(subjects[c]['label'])
        bacc = balanced_accuracy_score(test_true, (np.array(test_pred) >= 0.5).astype(int))
        fold_baccs.append(bacc)
        print(f"    Fold {fi + 1}: bacc={bacc:.3f}  (vl={best_vl:.4f}, ep={ep - patience})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_bacc = float(np.mean(fold_baccs))
    std_bacc = float(np.std(fold_baccs))
    print(f"  {model_name}: bacc={mean_bacc:.3f} +/- {std_bacc:.3f}")
    return mean_bacc, std_bacc, fold_baccs


def main():
    p = argparse.ArgumentParser(description='MODMA Audio DL Benchmark')
    p.add_argument('--model', required=True,
                   choices=['shallowconvnet', 'deepconvnet', 'eegnet', 'cnnlstm'],
                   help='Model architecture')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    args = p.parse_args()

    print(f"Device: {device}")
    print(f"{'='*60}")
    print(f"  MODMA AUDIO DL BENCHMARK — {args.model.upper()}")
    print(f"  Mel-spec: {N_MELS} mel x {N_TIMESTEPS} frames @ {SR_TARGET}Hz")
    print(f"  Batch={args.batch_size}  LR={args.lr}  Epochs={args.epochs}  Patience={args.patience}")
    print(f"{'='*60}")

    print("\nLoading audio, computing mel-spectrograms...")
    subjects = load_all_audio_subjects()
    cods = sorted(subjects.keys())
    labels = np.array([subjects[c]['label'] for c in cods])
    total_wins = sum(len(subjects[c]['windows']) for c in cods)
    print(f"  Subjects: {len(subjects)} (MDD={np.sum(labels == 1)}, HC={np.sum(labels == 0)})")
    print(f"  Windows/subject: ~{total_wins // len(subjects)}")

    model_factory, display_name = build_model_factory(args.model, N_MELS, N_TIMESTEPS)
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
        'input': f'mel-spec {N_MELS}x{N_TIMESTEPS} @ {SR_TARGET}Hz',
        'device': str(device),
        'batch_size': args.batch_size,
        'lr': args.lr, 'wd': args.wd,
        'epochs': args.epochs, 'patience': args.patience,
    }

    os.makedirs('results', exist_ok=True)
    out_path = f'results/modma_audio_dl_{args.model}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"  bacc_mean={mean_bacc:.4f}, bacc_std={std_bacc:.4f}")
    print(f"  Folds: {fold_baccs}")


if __name__ == '__main__':
    main()
