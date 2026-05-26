from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

from src.datasets.mdd_db import MDDDataset, create_dataloaders, parse_optional_float


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/mdd_db/validation"


def get_split_info(loader) -> dict:
    dataset = loader.dataset

    names = list(dataset.names)
    subjects = list(getattr(dataset, "subjects", dataset.names))
    labels = [int(v) for v in dataset.y.detach().cpu().tolist()]

    if isinstance(dataset.X, torch.Tensor):
        shapes = [tuple(dataset.X[i].shape) for i in range(dataset.X.shape[0])]
    else:
        shapes = [tuple(x.shape) for x in dataset.X]

    return {
        "names": names,
        "subjects": subjects,
        "labels": labels,
        "shapes": shapes,
        "label_count": Counter(labels),
        "subject_count": Counter(subjects),
        "name_count": Counter(names),
    }


def overlap(a, b):
    return sorted(set(a) & set(b))


def shape_text(shapes: list[tuple[int, int]]) -> str:
    counter = Counter(shapes)

    if len(counter) == 1:
        (c, t), n = next(iter(counter.items()))
        return f"C={c} | T={t} | N={n}"

    channels = sorted({s[0] for s in shapes})
    lengths = [s[1] for s in shapes]

    return f"C={channels} | T={min(lengths)}-{max(lengths)} | shape_count={len(counter)}"


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

    parser.add_argument("--pp-as", type=str, default="tensor", choices=["tensor", "list"])
    parser.add_argument("--channel-strategy", type=str, default="common", choices=["common", "all"])
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_ROOT))

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MDDDataset(
        root=args.root,
        condition=args.condition,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
        pp_as=args.pp_as,
        channel_strategy=args.channel_strategy,
    )

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

    errors = []
    warnings = []
    summary_rows = []

    dataset_names = [sample["name"] for sample in dataset.samples]
    dataset_subjects = [sample["subject"] for sample in dataset.samples]
    dataset_labels = [int(sample["label"].item()) for sample in dataset.samples]

    labels_by_subject = defaultdict(set)
    for subject, label in zip(dataset_subjects, dataset_labels):
        labels_by_subject[subject].add(label)

    conflicting_subject_labels = {
        subject: labels
        for subject, labels in labels_by_subject.items()
        if len(labels) > 1
    }

    if conflicting_subject_labels:
        errors.append(f"Subjects with conflicting labels: {conflicting_subject_labels}")

    global_test_name_counter = Counter()
    global_test_subject_to_folds = defaultdict(set)

    print("\nMDD dataloader leakage validation")
    print("=" * 80)
    print(f"Condition: {args.condition}")
    print(f"pp_as: {args.pp_as}")
    print(f"k: {args.k}")
    print(f"inner_splits: {args.inner_splits}")
    print(f"split_seed: {args.split_seed}")
    print(f"batch_size: {args.batch_size}")
    print(f"Samples: {len(dataset)}")
    print(f"Unique subjects: {len(set(dataset_subjects))}")
    print(f"Class balance: {dict(Counter(dataset_labels))}")

    for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
        split_infos = {
            "train": get_split_info(train_loader),
            "val": get_split_info(val_loader),
            "test": get_split_info(test_loader),
        }

        train = split_infos["train"]
        val = split_infos["val"]
        test = split_infos["test"]

        subject_overlaps = {
            "train_val": overlap(train["subjects"], val["subjects"]),
            "train_test": overlap(train["subjects"], test["subjects"]),
            "val_test": overlap(val["subjects"], test["subjects"]),
        }

        name_overlaps = {
            "train_val": overlap(train["names"], val["names"]),
            "train_test": overlap(train["names"], test["names"]),
            "val_test": overlap(val["names"], test["names"]),
        }

        for key, values in subject_overlaps.items():
            if values:
                errors.append(
                    f"Fold {fold_id:02d}: subject leakage {key}: {values[:10]}"
                )

        for key, values in name_overlaps.items():
            if values:
                errors.append(
                    f"Fold {fold_id:02d}: file/name leakage {key}: {values[:10]}"
                )

        for name in test["names"]:
            global_test_name_counter[name] += 1

        for subject in test["subjects"]:
            global_test_subject_to_folds[subject].add(fold_id)

        for split_name, info in split_infos.items():
            label_count = info["label_count"]
            n_h = label_count.get(0, 0)
            n_mdd = label_count.get(1, 0)

            if n_h == 0 or n_mdd == 0:
                warnings.append(
                    f"Fold {fold_id:02d} {split_name}: missing class. "
                    f"H={n_h}, MDD={n_mdd}"
                )

            summary_rows.append(
                {
                    "fold": fold_id,
                    "split": split_name,
                    "n": len(info["names"]),
                    "unique_subjects": len(set(info["subjects"])),
                    "H": n_h,
                    "MDD": n_mdd,
                    "batches": {
                        "train": len(train_loader),
                        "val": len(val_loader),
                        "test": len(test_loader),
                    }[split_name],
                    "shape_text": shape_text(info["shapes"]),
                    "subject_overlap_train_val": len(subject_overlaps["train_val"]),
                    "subject_overlap_train_test": len(subject_overlaps["train_test"]),
                    "subject_overlap_val_test": len(subject_overlaps["val_test"]),
                    "name_overlap_train_val": len(name_overlaps["train_val"]),
                    "name_overlap_train_test": len(name_overlaps["train_test"]),
                    "name_overlap_val_test": len(name_overlaps["val_test"]),
                }
            )

        print(f"\nFold {fold_id:02d}")
        for split_name in ["train", "val", "test"]:
            info = split_infos[split_name]
            labels = info["label_count"]

            print(
                f"  {split_name:5s} | "
                f"N={len(info['names']):3d} | "
                f"subjects={len(set(info['subjects'])):3d} | "
                f"H={labels.get(0, 0):3d} | "
                f"MDD={labels.get(1, 0):3d} | "
                f"{shape_text(info['shapes'])}"
            )

        print(
            "  overlaps | "
            f"subjects train/val={len(subject_overlaps['train_val'])}, "
            f"train/test={len(subject_overlaps['train_test'])}, "
            f"val/test={len(subject_overlaps['val_test'])} | "
            f"names train/val={len(name_overlaps['train_val'])}, "
            f"train/test={len(name_overlaps['train_test'])}, "
            f"val/test={len(name_overlaps['val_test'])}"
        )

    missing_from_test = sorted(set(dataset_names) - set(global_test_name_counter))
    repeated_in_test = sorted(
        name for name, count in global_test_name_counter.items()
        if count != 1
    )

    if missing_from_test:
        errors.append(
            f"{len(missing_from_test)} samples never appear in outer test folds: "
            f"{missing_from_test[:10]}"
        )

    if repeated_in_test:
        errors.append(
            f"{len(repeated_in_test)} samples appear != 1 times in outer test folds: "
            f"{repeated_in_test[:10]}"
        )

    subjects_in_multiple_test_folds = {
        subject: sorted(folds_seen)
        for subject, folds_seen in global_test_subject_to_folds.items()
        if len(folds_seen) > 1
    }

    if subjects_in_multiple_test_folds:
        errors.append(
            "Subjects appear in multiple outer test folds: "
            f"{dict(list(subjects_in_multiple_test_folds.items())[:10])}"
        )

    summary = pd.DataFrame(summary_rows)
    csv_path = out_dir / f"mdd_dataloaders_{args.condition}_{args.pp_as}_split-{args.split_seed}.csv"
    summary.to_csv(csv_path, index=False)

    print("\nOuter test coverage:")
    print(f"  Dataset samples: {len(dataset_names)}")
    print(f"  Test samples counted: {sum(global_test_name_counter.values())}")
    print(f"  Unique test samples: {len(global_test_name_counter)}")
    print(f"  Missing from test: {len(missing_from_test)}")
    print(f"  Repeated in test: {len(repeated_in_test)}")
    print(f"  Subjects in multiple test folds: {len(subjects_in_multiple_test_folds)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\nSaved table: {csv_path}")
        sys.exit(1)

    print("\nOK: no subject/file leakage detected in dataloaders.")
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()