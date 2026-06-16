from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models" / "modma_db"


def list_models() -> list[str]:
    if not BASE_DIR.exists():
        return []
    return sorted(d.name for d in BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))


def list_versions(model: str) -> list[str]:
    model_dir = BASE_DIR / model
    if not model_dir.exists():
        return []
    return sorted(d.name for d in model_dir.iterdir() if d.is_dir())


def get_experiment_dir(model: str, version: str) -> Path:
    return BASE_DIR / model / version


def load_config(model: str, version: str) -> dict:
    path = get_experiment_dir(model, version) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_overall_metrics(model: str, version: str) -> dict:
    path = get_experiment_dir(model, version) / "overall_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df.iloc[0].to_dict()
    return {}


def load_fold_metrics(model: str, version: str) -> pd.DataFrame:
    path = get_experiment_dir(model, version) / "fold_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_predictions(model: str, version: str) -> pd.DataFrame:
    path = get_experiment_dir(model, version) / "predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_results(model: str, version: str) -> dict:
    path = get_experiment_dir(model, version) / "results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_all_experiments_summary() -> pd.DataFrame:
    rows = []
    for model in list_models():
        for version in list_versions(model):
            cfg = load_config(model, version)
            overall = load_overall_metrics(model, version)

            rows.append({
                "model": model,
                "version": version,
                "experiment": f"{model}/{version}",
                "accuracy": overall.get("mean_accuracy"),
                "accuracy_std": overall.get("std_accuracy"),
                "balanced_accuracy": overall.get("mean_balanced_accuracy"),
                "balanced_accuracy_std": overall.get("std_balanced_accuracy"),
                "f1_macro": overall.get("mean_f1_macro"),
                "f1_macro_std": overall.get("std_f1_macro"),
                "model_params": f"F1={cfg.get('F1', '?')} D={cfg.get('D', '?')} F2={cfg.get('F2', '?')}",
                "duration_sec": cfg.get("duration_sec", "?"),
                "batch_size": cfg.get("batch_size", "?"),
                "weight_decay": cfg.get("weight_decay", 0),
                "lr_scheduler": cfg.get("lr_scheduler", False),
                "n_epochs": cfg.get("epochs", "?"),
            })
    return pd.DataFrame(rows)
