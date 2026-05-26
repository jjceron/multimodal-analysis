from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src.datasets.mdd_db import MDDDataset, create_dataloaders, parse_optional_float
from src.models.eegnet import EEGNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def standardize_eeg(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def split_info(loader):
    ds = loader.dataset
    names = list(ds.names)
    subjects = list(getattr(ds, "subjects", ds.names))
    labels = [int(v) for v in ds.y.detach().cpu().tolist()]
    return names, subjects, labels


def overlap(a, b):
    return sorted(set(a) & set(b))


def label_text(labels):
    c = Counter(labels)
    return f"H={c.get(0, 0):02d} MDD={c.get(1, 0):02d}"


def check_model_forward(args, loader, n_channels: int, n_classes: int, device):
    model = EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        temporal_kern=args.temporal_kern,
        separable_kern=args.separable_kern,
        pool1=args.pool1,
        pool2=args.pool2,
        dropout=args.dropout,
        meanmax_alpha=args.meanmax_alpha,
        pp_as="tensor",
        aggregate=args.aggregate,
        norm=args.norm,
    ).to(device)

    model.eval()

    names, x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)

    if args.standardize:
        x = standardize_eeg(x)

    with torch.no_grad():
        logits, logits_time = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        probs = torch.softmax(logits, dim=-1)

    errors = []
    expected_t = x.shape[-1] // (args.pool1 * args.pool2)

    if logits.shape != (x.shape[0], n_classes):
        errors.append(f"logits shape inválido: {tuple(logits.shape)}")

    if logits_time.shape != (x.shape[0], expected_t, n_classes):
        errors.append(
            f"logits_time inválido: {tuple(logits_time.shape)}; esperado {(x.shape[0], expected_t, n_classes)}"
        )

    if not torch.isfinite(x).all():
        errors.append("batch X tiene NaN/Inf después de estandarizar")

    if not torch.isfinite(logits).all():
        errors.append("logits tiene NaN/Inf")

    if not torch.isfinite(loss):
        errors.append("loss tiene NaN/Inf")

    prob_sum_error = (probs.sum(dim=-1) - 1).abs().max().item()
    if prob_sum_error > 1e-5:
        errors.append(f"softmax no suma 1; max_error={prob_sum_error:.2e}")

    print(
        f"  forward: X={tuple(x.shape)} logits_time={tuple(logits_time.shape)} "
        f"logits={tuple(logits.shape)} loss={float(loss.item()):.4f} norm={model.norm}"
    )

    return errors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "data/raw/mdd_db"))
    parser.add_argument("--condition", type=str, required=True, choices=["EC", "EO"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lowcut", type=parse_optional_float, default=0.5)
    parser.add_argument("--highcut", type=parse_optional_float, default=60.0)
    parser.add_argument("--notch", type=parse_optional_float, default=50.0)
    parser.add_argument("--target-fs", type=parse_optional_float, default=None)
    parser.add_argument("--duration-sec", type=parse_optional_float, default=None)
    parser.add_argument("--channel-strategy", type=str, default="common", choices=["common", "all"])

    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--temporal-kern", type=int, default=63)
    parser.add_argument("--separable-kern", type=int, default=15)
    parser.add_argument("--pool1", type=int, default=8)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--meanmax-alpha", type=float, default=0.0)
    parser.add_argument("--aggregate", type=parse_bool, default=True)
    parser.add_argument("--norm", type=str, default="auto", choices=["auto", "batch", "group"])
    parser.add_argument("--standardize", type=parse_bool, default=True)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    errors = []
    warnings = []

    print("\nMDD tensor audit")
    print("=" * 72)
    print(f"condition={args.condition} | pp_as=tensor | device={device}")
    print(
        f"preproc: lowcut={args.lowcut} highcut={args.highcut} notch={args.notch} "
        f"target_fs={args.target_fs} duration_sec={args.duration_sec}"
    )
    print(
        f"model: EEGNet F1={args.F1} D={args.D} F2={args.F2} "
        f"pool={args.pool1}x{args.pool2} aggregate={args.aggregate} "
        f"meanmax_alpha={args.meanmax_alpha} norm={args.norm}"
    )

    dataset = MDDDataset(
        root=args.root,
        condition=args.condition,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as="tensor",
        channel_strategy=args.channel_strategy,
    )

    labels = [int(s["label"].item()) for s in dataset.samples]
    names = [s["name"] for s in dataset.samples]
    subjects = [s["subject"] for s in dataset.samples]
    shapes = [tuple(s["eeg"].shape) for s in dataset.samples]

    label_by_subject = defaultdict(set)
    for subject, label in zip(subjects, labels):
        label_by_subject[subject].add(label)

    conflicts = {s: y for s, y in label_by_subject.items() if len(y) > 1}
    if conflicts:
        errors.append(f"sujetos con etiquetas contradictorias: {conflicts}")

    if len(set(shapes)) != 1:
        errors.append(f"tensor mode con shapes múltiples: {Counter(shapes)}")

    if Counter(labels).get(0, 0) == 0 or Counter(labels).get(1, 0) == 0:
        errors.append(f"falta una clase en dataset: {dict(Counter(labels))}")

    x_all = dataset.X if hasattr(dataset, "X") else torch.stack([s["eeg"] for s in dataset.samples])
    if not torch.isfinite(x_all).all():
        errors.append("dataset.X contiene NaN/Inf")

    std_per_sample_channel = x_all.std(dim=-1, unbiased=False)
    if (std_per_sample_channel <= 1e-8).any():
        warnings.append("hay canales con std casi cero en alguna muestra")

    c, t = shapes[0]
    print("\nDataset")
    print(f"  samples={len(dataset)} unique_subjects={len(set(subjects))}")
    print(f"  class_balance={dict(Counter(labels))}  [0=H, 1=MDD]")
    print(f"  C={c} T={t} approx_sec@256={t / 256:.2f}")
    print(f"  shape_unique={dict(Counter(shapes))}")
    print(f"  duplicate_subjects={sum(v > 1 for v in Counter(subjects).values())}")

    folds = create_dataloaders(
        dataset=dataset,
        k_folder=args.k,
        batch_size=args.batch_size,
        shuffle=True,
        split_seed=args.split_seed,
        inner_split=args.inner_splits,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nSplits + forward checks")
    test_name_counter = Counter()
    test_subject_to_folds = defaultdict(set)

    for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
        split = {}
        for split_name, loader in [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]:
            split[split_name] = split_info(loader)

        train_names, train_subjects, train_labels = split["train"]
        val_names, val_subjects, val_labels = split["val"]
        test_names, test_subjects, test_labels = split["test"]

        checks = {
            "subj train/val": overlap(train_subjects, val_subjects),
            "subj train/test": overlap(train_subjects, test_subjects),
            "subj val/test": overlap(val_subjects, test_subjects),
            "name train/val": overlap(train_names, val_names),
            "name train/test": overlap(train_names, test_names),
            "name val/test": overlap(val_names, test_names),
        }

        for key, values in checks.items():
            if values:
                errors.append(f"fold {fold_id}: fuga {key}: {values[:5]}")

        for n in test_names:
            test_name_counter[n] += 1
        for s in test_subjects:
            test_subject_to_folds[s].add(fold_id)

        for split_name, (_, _, split_labels) in split.items():
            lc = Counter(split_labels)
            if lc.get(0, 0) == 0 or lc.get(1, 0) == 0:
                warnings.append(f"fold {fold_id} {split_name}: una clase ausente {dict(lc)}")

        print(
            f"fold {fold_id:02d}: "
            f"train N={len(train_names):02d} {label_text(train_labels)} | "
            f"val N={len(val_names):02d} {label_text(val_labels)} | "
            f"test N={len(test_names):02d} {label_text(test_labels)} | "
            f"overlap=0"
        )

        if fold_id == 1:
            errors.extend(check_model_forward(args, train_loader, c, 2, device))

    missing = sorted(set(names) - set(test_name_counter))
    repeated = sorted(n for n, count in test_name_counter.items() if count != 1)
    multi_test_subjects = {
        s: sorted(fs) for s, fs in test_subject_to_folds.items() if len(fs) > 1
    }

    if missing:
        errors.append(f"muestras ausentes en outer test: {missing[:5]}")
    if repeated:
        errors.append(f"muestras repetidas en outer test: {repeated[:5]}")
    if multi_test_subjects:
        errors.append(f"sujetos en múltiples folds test: {dict(list(multi_test_subjects.items())[:5])}")

    print("\nOuter test coverage")
    print(f"  dataset_samples={len(names)}")
    print(f"  test_counted={sum(test_name_counter.values())}")
    print(f"  test_unique={len(test_name_counter)}")
    print(f"  missing={len(missing)} repeated={len(repeated)} subjects_multi_test={len(multi_test_subjects)}")

    if warnings:
        print("\nWARN")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nOK: tensor results are methodologically consistent for this split/config.")


if __name__ == "__main__":
    main()
