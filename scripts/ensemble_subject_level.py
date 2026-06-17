from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "modma_db"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model folder names, e.g. CSPLDA/window EEGNet/window_v2")
    parser.add_argument("--version-name", type=str, default="ensemble",
                        help="Name for ensemble output (just logs, no dir)")
    return parser.parse_args()


def main():
    args = parse_args()

    all_preds: dict[int, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"preds": [], "true": None})
    )
    n_models = len(args.models)

    for model_path in args.models:
        path = MODELS_DIR / model_path / "predictions.csv"
        if not path.exists():
            print(f"[SKIP] predictions.csv not found: {path}")
            continue

        df = pd.read_csv(path)
        test_df = df[df["split"] == "test"]

        for _, row in test_df.iterrows():
            subject = row["subject"]
            fold = row["fold"]
            all_preds[fold][subject]["preds"].append(row["pred_label"])
            all_preds[fold][subject]["true"] = row["true_label"]

    if not all_preds:
        print("No predictions found. Check --models paths.")
        return

    print(f"\nEnsemble of {n_models} models\n")

    fold_results = []
    for fold in sorted(all_preds.keys()):
        subjects = all_preds[fold]
        y_true, y_ensemble = [], []

        for subject, data in subjects.items():
            preds = data["preds"]
            true_label = data["true"]
            if len(preds) != n_models:
                print(f"  [WARN] Fold {fold}, {subject}: only {len(preds)}/{n_models} preds")
                continue

            majority = max(set(preds), key=preds.count)
            y_true.append(true_label)
            y_ensemble.append(majority)

        acc = accuracy_score(y_true, y_ensemble)
        bacc = balanced_accuracy_score(y_true, y_ensemble)
        f1 = f1_score(y_true, y_ensemble, average="macro")

        fold_results.append({
            "fold": fold,
            "n_subjects": len(y_true),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1,
        })
        print(f"  Fold {fold}: acc={acc:.4f} bal_acc={bacc:.4f} f1={f1:.4f} ({len(y_true)} subjects)")

    df = pd.DataFrame(fold_results)
    print(f"\n  Ensemble overall ({n_models} models):")
    print(f"    mean_accuracy:          {df['accuracy'].mean():.4f} +/- {df['accuracy'].std():.4f}")
    print(f"    mean_balanced_accuracy: {df['balanced_accuracy'].mean():.4f} +/- {df['balanced_accuracy'].std():.4f}")
    print(f"    mean_f1_macro:          {df['f1_macro'].mean():.4f} +/- {df['f1_macro'].std():.4f}")


if __name__ == "__main__":
    main()
