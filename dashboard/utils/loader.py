from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models" / "modma_db"


def list_experiments() -> list[str]:
    if not BASE_DIR.exists():
        return []
    return sorted(d.name for d in BASE_DIR.iterdir() if d.is_dir())


def get_experiment_dir(experiment: str) -> Path:
    return BASE_DIR / experiment


def load_config(experiment: str) -> dict:
    path = get_experiment_dir(experiment) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_overall_metrics(experiment: str) -> dict:
    path = get_experiment_dir(experiment) / "overall_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df.iloc[0].to_dict()
    return {}


def load_fold_metrics(experiment: str) -> pd.DataFrame:
    path = get_experiment_dir(experiment) / "fold_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_predictions(experiment: str) -> pd.DataFrame:
    path = get_experiment_dir(experiment) / "predictions.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df
    return pd.DataFrame()


def load_results(experiment: str) -> dict:
    path = get_experiment_dir(experiment) / "results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_all_experiments_summary() -> pd.DataFrame:
    rows = []
    for exp in list_experiments():
        cfg = load_config(exp)
        overall = load_overall_metrics(exp)

        rows.append({
            "experiment": exp,
            "accuracy": overall.get("mean_accuracy"),
            "accuracy_std": overall.get("std_accuracy"),
            "balanced_accuracy": overall.get("mean_balanced_accuracy"),
            "balanced_accuracy_std": overall.get("std_balanced_accuracy"),
            "f1_macro": overall.get("mean_f1_macro"),
            "f1_macro_std": overall.get("std_f1_macro"),
            "model": f"F1={cfg.get('F1', '?')} D={cfg.get('D', '?')} F2={cfg.get('F2', '?')}",
            "duration_sec": cfg.get("duration_sec", "?"),
            "batch_size": cfg.get("batch_size", "?"),
            "weight_decay": cfg.get("weight_decay", 0),
            "lr_scheduler": cfg.get("lr_scheduler", False),
            "n_epochs": cfg.get("epochs", "?"),
        })
    return pd.DataFrame(rows)
