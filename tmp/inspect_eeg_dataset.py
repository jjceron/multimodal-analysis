from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_eeg import EEGDataset, create_kfold_dataloaders

CONDITION = "closed"
BATCH_SIZE = 32
K = 10

with open("configs/preprocessing.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

eeg_cfg = cfg["eeg"]

dataset = EEGDataset(condition=CONDITION)
summary = dataset.get_summary_dataframe()

summary["n_trials"] = summary["selected_shape"].apply(lambda x: int(x[0]))

n_channels = int(summary["n_channels"].iloc[0])
n_samples = int(summary["n_samples"].iloc[0])
total_trials = int(summary["n_trials"].sum())

print()
print("EEG DATASET SUMMARY")
print(f"Condition: {CONDITION}")
print(f"Window duration: {eeg_cfg['window_duration']} s")
print(f"Overlap: {eeg_cfg['overlap']}")
print(f"Target fs: {eeg_cfg['target_fs']} Hz")
print(f"Bandpass: {eeg_cfg['lowcut']} - {eeg_cfg['highcut']} Hz")
print(f"Channels: {n_channels}")
print(f"Samples per trial: {n_samples}")
print(f"Total subjects: {len(dataset)}")
print(f"Total trials/epochs: {total_trials}")

print()
print("Trials per subject")
print(summary[["subject_id", "label_name", "selected_shape", "n_trials"]].to_string(index=False))

print()
print("Trials per subject statistics")
print(f"Min:    {summary['n_trials'].min()}")
print(f"Median: {summary['n_trials'].median()}")
print(f"Max:    {summary['n_trials'].max()}")

print()
print("Building folds...")
folds = create_kfold_dataloaders(
    dataset,
    k=K,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print()
print("FOLD SUMMARY")
for fold_idx, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
    print()
    print(f"Fold {fold_idx}")
    print(f"  Train trials: {len(train_loader.dataset)} | X shape: {tuple(train_loader.dataset.X.shape)} | batches/epoch: {len(train_loader)}")
    print(f"  Val trials:   {len(val_loader.dataset)} | X shape: {tuple(val_loader.dataset.X.shape)} | batches/epoch: {len(val_loader)}")
    print(f"  Test trials:  {len(test_loader.dataset)} | X shape: {tuple(test_loader.dataset.X.shape)} | batches/epoch: {len(test_loader)}")
