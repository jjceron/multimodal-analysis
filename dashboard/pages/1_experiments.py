from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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
    plot_fold_training_curves,
)
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Training", page_icon="📊", layout="wide")
render_sidebar()

st.title("Training")

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

tab_struct, tab_folds = st.tabs(["Structure", "Folds"])

with tab_struct:
    if not cfg:
        st.info("No configuration available.")
        st.stop()

    F1 = cfg.get("F1", 8)
    D = cfg.get("D", 2)
    F2 = cfg.get("F2", 16)
    n_ch = cfg.get("n_channels", 128)
    n_cls = cfg.get("n_classes", 2)
    dr = cfg.get("dropout", 0.5)
    ma = cfg.get("meanmax_alpha", 0.5)
    p1 = cfg.get("pool1", 8)
    p2 = cfg.get("pool2", 8)
    duration = cfg.get("duration_sec", 120.0)
    fs = cfg.get("target_fs", None) or 250
    T = int(duration * fs)

    @st.cache_resource
    def build_model():
        return EEGNet(
            n_channels=n_ch, n_classes=n_cls,
            F1=F1, D=D, F2=F2,
            dropout=dr, meanmax_alpha=ma, aggregate=True,
        )

    model = build_model()
    T_pool1 = T // p1
    T_total = T_pool1 // p2

    col_diag, col_info = st.columns([2, 1], gap="large")

    with col_diag:
        st.markdown("##### End-to-End Pipeline")

        pipes = [
            ("Raw EEG", f"128 ch × {T} samples\n({duration}s @ {fs}Hz)", "#636efa",
             "Bandpass [0.5–60] Hz\nNotch 50 Hz\nAverage Ref"),
            ("Temporal Conv", f"Conv2d 1 → {F1}\nkernel=(1,63)\nBatchNorm", "#00cc96",
             f"→ ({F1}, 128, {T})"),
            ("Depthwise Spatial", f"DepthConv {F1} → {F1*D}\nkernel=(128,1) groups={F1}", "#ef553b",
             f"→ ({F1*D}, 1, {T_pool1})\nELU + AvgPool + Dropout"),
            ("Separable Conv", f"DepthConv {F1*D} → {F1*D}\nPointwise {F1*D} → {F2}", "#ab63fa",
             f"→ ({F2}, 1, {T_total})\nELU + AvgPool + Dropout"),
            ("Classifier", f"Conv2d {F2} → {n_cls}\nkernel=(1,1)", "#ffa15a",
             f"→ ({T_total}, {n_cls})"),
            ("Aggregation", f"Mean-Max Pooling\nα = {ma}", "#19d3f3",
             f"→ ({n_cls})"),
            ("Output", f"HC vs MDD\n({n_cls} classes)", "#e6ab02",
             f"logits → softmax"),
        ]

        fig = go.Figure()
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1, 6.5]),
            height=580,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        y_positions = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]

        for i, (name, inner, color, out) in enumerate(pipes):
            yc = y_positions[i]
            bw, bh = 1.4, 0.45

            fig.add_shape(type="rect", x0=-bw/2, x1=bw/2, y0=yc-bh/2, y1=yc+bh/2,
                          line=dict(color=color, width=2), fillcolor=color, opacity=0.12)

            fig.add_annotation(x=-bw/2-0.08, y=yc, xanchor="right",
                               text=f"<b>{name}</b>", font=dict(size=11, color=color),
                               showarrow=False)
            fig.add_annotation(x=0, y=yc, text=inner.replace("\n", "<br>"),
                               font=dict(size=9, color="#666"), showarrow=False)
            fig.add_annotation(x=bw/2+0.08, y=yc, xanchor="left",
                               text=f"<span style='color:#999'>{out.replace(chr(10),'<br>')}</span>",
                               showarrow=False)

            if i < len(pipes) - 1:
                ny = y_positions[i + 1]
                fig.add_annotation(
                    x=0, y=(yc - bh/2 + ny + 0.225)/2,
                    ax=0, ay=yc - bh/2, axref="x", ayref="y",
                    xref="x", yref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#bbb",
                )

        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        st.markdown("##### Model Info")
        st.markdown(f"**Architecture:** EEGNet")
        st.markdown(f"**F1:** {F1}  |  **D:** {D}  |  **F2:** {F2}")
        st.markdown(f"**Dropout:** {dr}  |  **α:** {ma}")
        st.markdown(f"**Parameters:**")
        st.markdown(f"Total: **{total_p:,}**  |  Trainable: **{train_p:,}**")
        st.markdown(f"**Input:** {n_ch} ch × {T} samples ({duration}s)")
        st.markdown(f"**Output:** {n_cls} classes")

        if results and results.get("overall"):
            o = results["overall"]
            st.divider()
            st.markdown("##### Results")
            st.markdown(f"**Accuracy:** {o.get('mean_accuracy', 0):.2%} ± {o.get('std_accuracy', 0):.2%}")
            st.markdown(f"**Balanced Acc:** {o.get('mean_balanced_accuracy', 0):.2%} ± {o.get('std_balanced_accuracy', 0):.2%}")
            st.markdown(f"**F1-macro:** {o.get('mean_f1_macro', 0):.4f} ± {o.get('std_f1_macro', 0):.4f}")

    st.divider()
    st.markdown("##### Model Summary")

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

        total_p2 = sum(p.numel() for p in model.parameters())
        train_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)

        lines = []
        lines.append("=" * 95)
        lines.append(f"{'Layer (type:depth-idx)':<40} {'Input Shape':<22} {'Output Shape':<22} {'Param #':<10}")
        lines.append("=" * 95)

        for layer in res.summary_list:
            indent = "  " * layer.depth if hasattr(layer, 'depth') else ""
            prefix = f"{indent}{layer.class_name}"

            in_str = str(layer.input_size) if layer.input_size and layer.is_leaf_layer else ""
            out_str = str(layer.output_size) if layer.output_size else ""
            param_str = str(layer.num_params) if layer.is_leaf_layer else ""

            if layer.is_leaf_layer:
                lines.append(f"{prefix:<40} {in_str:<22} {out_str:<22} {param_str:<10}")
            else:
                lines.append(f"{prefix:<40}")

        lines.append("-" * 95)
        lines.append(f"{'Total params':<40} {total_p2:<10}")
        lines.append(f"{'Trainable params':<40} {train_p2:<10}")
        lines.append(f"{'Non-trainable params':<40} {total_p2 - train_p2:<10}")
        lines.append("-" * 95)

        st.code("\n".join(lines), language="text")

    except Exception as e:
        st.text(f"Summary unavailable: {e}")

with tab_folds:
    if not results or not results.get("fold_data"):
        st.info("No fold training data available for this experiment.")
    else:
        fold_data = results["fold_data"]

        st.markdown("##### Training Curves per Fold")
        fig_loss, fig_acc = plot_fold_training_curves(fold_data)
        st.plotly_chart(fig_loss, use_container_width=True)
        st.plotly_chart(fig_acc, use_container_width=True)

        st.divider()
        st.markdown("##### Per-Fold Confusion Matrices")

        for fd in fold_data:
            fid = fd["fold_id"]
            fold_preds = df_preds[df_preds["fold"] == fid]
            yt_v = fold_preds[fold_preds["split"] == "val"]["true_label"].tolist()
            yp_v = fold_preds[fold_preds["split"] == "val"]["pred_label"].tolist()
            yt_t = fold_preds[fold_preds["split"] == "test"]["true_label"].tolist()
            yp_t = fold_preds[fold_preds["split"] == "test"]["pred_label"].tolist()

            if yt_v and yt_t:
                st.markdown(f"**Fold {fid:02d}**")
                fig = plot_dual_confusion_matrix(yt_v, yp_v, yt_t, yp_t)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("##### Summary")

        if not df_folds.empty:
            avail = [c for c in ["fold", "val_accuracy", "val_balanced_accuracy", "val_f1_macro",
                                 "test_accuracy", "test_balanced_accuracy", "test_f1_macro"]
                     if c in df_folds.columns]

            mean_vals = df_folds[[c for c in avail if c != "fold"]].mean()
            std_vals = df_folds[[c for c in avail if c != "fold"]].std()

            highlight_lines = []
            highlight_lines.append("**Mean ± Std across folds**")
            highlight_lines.append("")
            for m in ["val_accuracy", "val_balanced_accuracy", "test_accuracy",
                      "test_balanced_accuracy", "val_f1_macro", "test_f1_macro"]:
                if m in mean_vals.index:
                    highlight_lines.append(
                        f"- **{m.replace('_', ' ').title()}:** "
                        f"{mean_vals[m]:.2%} ± {std_vals[m]:.2%}"
                    )

        col_tbl, col_cm = st.columns([1, 1], gap="large")

        with col_tbl:
            st.markdown("**Per-Fold Metrics**")
            if not df_folds.empty:
                st.dataframe(
                    df_folds[avail],
                    use_container_width=True,
                    hide_index=True,
                    column_config={c: st.column_config.NumberColumn(format="%.4f")
                                   for c in avail if c != "fold"},
                )

            st.markdown("\n".join(highlight_lines))

        with col_cm:
            st.markdown("**Global Confusion Matrix**")
            if not df_preds.empty:
                yt_v_all = df_preds[df_preds["split"] == "val"]["true_label"].tolist()
                yp_v_all = df_preds[df_preds["split"] == "val"]["pred_label"].tolist()
                yt_t_all = df_preds[df_preds["split"] == "test"]["true_label"].tolist()
                yp_t_all = df_preds[df_preds["split"] == "test"]["pred_label"].tolist()
                if yt_v_all and yt_t_all:
                    fig = plot_dual_confusion_matrix(yt_v_all, yp_v_all, yt_t_all, yp_t_all)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
