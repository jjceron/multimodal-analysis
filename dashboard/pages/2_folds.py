import streamlit as st
import pandas as pd
from utils.loader import (
    list_experiments,
    load_fold_metrics,
    load_predictions,
    load_config,
)
from utils.plots import plot_training_curves, plot_dual_confusion_matrix
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Folds", page_icon="🔬", layout="wide")
render_sidebar()

st.title("🔬 Fold-Level Analysis")

experiments = list_experiments()

if not experiments:
    st.warning("No experiments found.")
    st.stop()

selected = st.selectbox(
    "Experiment",
    options=experiments,
    index=len(experiments) - 1,
    key="folds_exp_selector",
)

df_folds = load_fold_metrics(selected)
df_pred = load_predictions(selected)
cfg = load_config(selected)

if df_folds.empty:
    st.warning("No fold metrics available.")
    st.stop()

available_folds = sorted(df_folds["fold"].unique())
selected_fold = st.selectbox("Fold", options=available_folds, index=len(available_folds) - 1)

fold_metrics_row = df_folds[df_folds["fold"] == selected_fold].iloc[0]

st.markdown(f"### Fold {selected_fold:02d} Metrics")
cols = st.columns(4)
for i, c in enumerate(fold_metrics_row.index):
    if c in ("fold", "best_epoch", "n_epochs"):
        continue
    cols[i % 4].metric(
        c.replace("_", " ").title(),
        f"{fold_metrics_row[c]:.4f}" if isinstance(fold_metrics_row[c], float) else fold_metrics_row[c],
    )

fold_pred = df_pred[df_pred["fold"] == selected_fold]

if not fold_pred.empty:
    val_pred = fold_pred[fold_pred["split"] == "val"]
    test_pred = fold_pred[fold_pred["split"] == "test"]

    if not val_pred.empty and not test_pred.empty:
        st.markdown(f"### Confusion Matrices — Fold {selected_fold:02d}")
        fig = plot_dual_confusion_matrix(
            y_true_val=val_pred["true_label"].tolist(),
            y_pred_val=val_pred["pred_label"].tolist(),
            y_true_test=test_pred["true_label"].tolist(),
            y_pred_test=test_pred["pred_label"].tolist(),
            class_names=["HC", "MDD"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Predictions Table")
    st.dataframe(fold_pred.drop(columns=["fold"]), use_container_width=True, hide_index=True)

if cfg:
    st.markdown("### Model Configuration for This Experiment")
    st.json(cfg)
