from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import EEGNet
from utils.loader import (
    load_config,
    load_fold_metrics,
    load_predictions,
    load_results,
    list_experiments,
)
from utils.plots import (
    plot_dual_confusion_matrix,
    plot_fold_bars,
    plot_training_curves_combined,
)
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Experiments", page_icon="📊", layout="wide")
render_sidebar()

st.title("Experiment Analysis")

experiments = list_experiments()

if not experiments:
    st.warning("No experiments found.")
    st.stop()

selected = st.selectbox(
    "Experiment",
    options=experiments,
    index=len(experiments) - 1,
    key="exp_page_selector",
)

cfg = load_config(selected)
df_folds = load_fold_metrics(selected)
results = load_results(selected)
df_preds = load_predictions(selected)

tab_all, tab_per, tab_struct = st.tabs(["All Folds", "Per Fold", "Structure"])

with tab_all:
    if results:
        overall = results.get("overall", {})
        if overall:
            st.markdown("##### Overall Metrics")
            cols = st.columns(3)
            cols[0].metric(
                "Mean Accuracy",
                f"{overall.get('mean_accuracy', 0):.2%}",
                delta=f"±{overall.get('std_accuracy', 0):.2%}",
            )
            cols[1].metric(
                "Mean Balanced Acc",
                f"{overall.get('mean_balanced_accuracy', 0):.2%}",
                delta=f"±{overall.get('std_balanced_accuracy', 0):.2%}",
            )
            cols[2].metric(
                "Mean F1-macro",
                f"{overall.get('mean_f1_macro', 0):.4f}",
                delta=f"±{overall.get('std_f1_macro', 0):.4f}",
            )

    exp_data = {}
    for exp in experiments:
        r = load_results(exp)
        fd = r.get("fold_data", [])
        if fd:
            exp_data[exp] = {"fold_data": fd}

    if exp_data:
        st.markdown("##### Training Curves (average across folds)")
        fig = plot_training_curves_combined(exp_data)
        st.plotly_chart(fig, use_container_width=True)

    if not df_folds.empty:
        st.markdown("##### Fold Comparison")
        metric = st.selectbox(
            "Metric",
            options=[c for c in df_folds.columns if c not in ("fold", "best_epoch", "n_epochs")],
            index=0,
            key="tab_all_metric",
        )
        fig = plot_fold_bars(df_folds, metric=metric)
        st.plotly_chart(fig, use_container_width=True)

    if not df_preds.empty:
        st.markdown("##### Overall Confusion Matrix")
        yt_val = df_preds[df_preds["split"] == "val"]["true_label"].tolist()
        yp_val = df_preds[df_preds["split"] == "val"]["pred_label"].tolist()
        yt_test = df_preds[df_preds["split"] == "test"]["true_label"].tolist()
        yp_test = df_preds[df_preds["split"] == "test"]["pred_label"].tolist()
        if yt_val and yt_test:
            fig = plot_dual_confusion_matrix(yt_val, yp_val, yt_test, yp_test)
            st.plotly_chart(fig, use_container_width=True)

with tab_per:
    if df_folds.empty:
        st.info("No per-fold metrics available.")
    else:
        fold_ids = sorted(df_folds["fold"].unique())
        selected_fold = st.selectbox("Fold", options=fold_ids, key="tab_per_fold")

        row = df_folds[df_folds["fold"] == selected_fold].iloc[0]

        st.markdown("##### Metrics")
        mc = st.columns(5)
        mc[0].metric("Best Epoch", int(row["best_epoch"]))
        mc[1].metric("Total Epochs", int(row["n_epochs"]))
        mc[2].metric("Val Accuracy", f"{row['val_accuracy']:.2%}")
        mc[3].metric("Test Accuracy", f"{row['test_accuracy']:.2%}")
        mc[4].metric("Test F1-macro", f"{row['test_f1_macro']:.4f}")

        if not df_preds.empty:
            st.markdown("##### Confusion Matrix")
            fold_preds = df_preds[df_preds["fold"] == selected_fold]
            yt_v = fold_preds[fold_preds["split"] == "val"]["true_label"].tolist()
            yp_v = fold_preds[fold_preds["split"] == "val"]["pred_label"].tolist()
            yt_t = fold_preds[fold_preds["split"] == "test"]["true_label"].tolist()
            yp_t = fold_preds[fold_preds["split"] == "test"]["pred_label"].tolist()
            if yt_v and yt_t:
                fig = plot_dual_confusion_matrix(yt_v, yp_v, yt_t, yp_t)
                st.plotly_chart(fig, use_container_width=True)

with tab_struct:
    if not cfg:
        st.info("No configuration available to build the model.")
    else:
        F1 = cfg.get("F1", 8)
        D = cfg.get("D", 2)
        F2 = cfg.get("F2", 16)
        n_ch = cfg.get("n_channels", 128)
        n_cls = cfg.get("n_classes", 2)
        dr = cfg.get("dropout", 0.5)
        ma = cfg.get("meanmax_alpha", 0.5)

        duration = cfg.get("duration_sec", 120.0)
        fs = cfg.get("target_fs", None) or 250
        T = int(duration * fs)

        @st.cache_resource
        def build_model():
            return EEGNet(
                n_channels=n_ch, n_classes=n_cls,
                F1=F1, D=D, F2=F2,
                dropout=dr, meanmax_alpha=ma,
                aggregate=True,
            )

        model = build_model()

        st.markdown("##### Parameter Summary")
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cols = st.columns(3)
        cols[0].metric("Total", f"{total_p:,}")
        cols[1].metric("Trainable", f"{train_p:,} ({train_p / total_p * 100:.1f}%)")
        cols[2].metric("Non-Trainable", f"{total_p - train_p:,}")

        st.markdown("##### Layer Summary")
        try:
            from torchinfo import summary

            res = summary(
                model,
                input_size=(1, n_ch, T),
                device="cpu",
                verbose=0,
                col_names=("input_size", "output_size", "num_params", "trainable"),
                depth=5,
            )

            rows = []
            for layer in res.summary_list:
                if layer.is_leaf_layer:
                    rows.append({
                        "Layer": layer.class_name,
                        "Input Shape": str(layer.input_size) if layer.input_size else "",
                        "Output Shape": str(layer.output_size) if layer.output_size else "",
                        "Params": layer.num_params,
                    })
            st.dataframe(
                pd.DataFrame(rows),
                column_config={"Params": st.column_config.NumberColumn(format="%d")},
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Raw torchinfo output"):
                st.code(str(res), language="text")

        except Exception as e:
            st.text(f"Summary unavailable: {e}")
