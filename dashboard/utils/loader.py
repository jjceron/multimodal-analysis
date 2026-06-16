from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "models"


def _base_dir(dataset: str) -> Path:
    return OUTPUT_ROOT / dataset


def list_models(dataset: str = "modma_db") -> list[str]:
    base = _base_dir(dataset)
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("."))


def list_versions(dataset: str = "modma_db", model: str | None = None) -> list[str]:
    if not model:
        return []
    model_dir = _base_dir(dataset) / model
    if not model_dir.exists():
        return []
    return sorted(d.name for d in model_dir.iterdir() if d.is_dir())


def get_experiment_dir(dataset: str, model: str, version: str) -> Path:
    return _base_dir(dataset) / model / version


def load_config(dataset: str = "modma_db", model: str | None = None, version: str | None = None) -> dict:
    if not model or not version:
        return {}
    path = get_experiment_dir(dataset, model, version) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_overall_metrics(dataset: str = "modma_db", model: str | None = None, version: str | None = None) -> dict:
    if not model or not version:
        return {}
    path = get_experiment_dir(dataset, model, version) / "overall_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df.iloc[0].to_dict()
    return {}


def load_fold_metrics(dataset: str = "modma_db", model: str | None = None, version: str | None = None) -> pd.DataFrame:
    if not model or not version:
        return pd.DataFrame()
    path = get_experiment_dir(dataset, model, version) / "fold_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_predictions(dataset: str = "modma_db", model: str | None = None, version: str | None = None) -> pd.DataFrame:
    if not model or not version:
        return pd.DataFrame()
    path = get_experiment_dir(dataset, model, version) / "predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_results(dataset: str = "modma_db", model: str | None = None, version: str | None = None) -> dict:
    if not model or not version:
        return {}
    path = get_experiment_dir(dataset, model, version) / "results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_all_experiments_summary(dataset: str = "modma_db") -> pd.DataFrame:
    rows = []
    for model in list_models(dataset):
        for version in list_versions(dataset, model):
            cfg = load_config(dataset, model, version)
            overall = load_overall_metrics(dataset, model, version)
            df_folds = load_fold_metrics(dataset, model, version)

            val_acc = df_folds["val_accuracy"].mean() if "val_accuracy" in df_folds.columns and not df_folds.empty else None
            val_bal = df_folds["val_balanced_accuracy"].mean() if "val_balanced_accuracy" in df_folds.columns and not df_folds.empty else None
            val_f1 = df_folds["val_f1_macro"].mean() if "val_f1_macro" in df_folds.columns and not df_folds.empty else None

            rows.append({
                "model": model,
                "version": version,
                "test_accuracy": overall.get("mean_accuracy"),
                "test_accuracy_std": overall.get("std_accuracy"),
                "test_balanced_accuracy": overall.get("mean_balanced_accuracy"),
                "test_balanced_accuracy_std": overall.get("std_balanced_accuracy"),
                "test_f1_macro": overall.get("mean_f1_macro"),
                "test_f1_macro_std": overall.get("std_f1_macro"),
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bal,
                "val_f1_macro": val_f1,
                "duration_sec": cfg.get("duration_sec", "?"),
                "batch_size": cfg.get("batch_size", "?"),
                "weight_decay": cfg.get("weight_decay", 0),
                "lr_scheduler": cfg.get("lr_scheduler", False),
                "n_epochs": cfg.get("epochs", "?"),
            })
    return pd.DataFrame(rows)
