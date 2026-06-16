from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.loader import (
    load_config,
    load_fold_metrics,
    load_predictions,
    load_results,
)
from utils.plots import (
    plot_dual_confusion_matrix,
    plot_fold_training_curves,
)
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Training", page_icon="📊", layout="wide")
render_sidebar()

st.title("Training")

sel_m = st.session_state.get("selected_model")
sel_v = st.session_state.get("selected_version")

if not sel_m or not sel_v:
    st.warning("Select a model and version from the sidebar.")
    st.stop()

cfg = load_config(sel_m, sel_v)
df_folds = load_fold_metrics(sel_m, sel_v)
results = load_results(sel_m, sel_v)
df_preds = load_predictions(sel_m, sel_v)

tab_struct, tab_folds = st.tabs(["Structure", "Folds"])

with tab_struct:
    if not cfg:
        st.info("No configuration available.")
        st.stop()

    n_ch = cfg.get("n_channels", 128)
    n_cls = cfg.get("n_classes", 2)
    duration = cfg.get("duration_sec", 120.0)
    fs = cfg.get("target_fs", None) or 250
    T = int(duration * fs)

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.models import EEGNet

        model = EEGNet(
            n_channels=n_ch, n_classes=n_cls,
            F1=cfg.get("F1", 8), D=cfg.get("D", 2), F2=cfg.get("F2", 16),
            dropout=cfg.get("dropout", 0.5), meanmax_alpha=cfg.get("meanmax_alpha", 0.5),
            aggregate=True,
        )
    except Exception:
        st.info(f"Cannot build model for {sel_m}")
        st.stop()

    col_diag, col_info = st.columns([2, 1], gap="large")

    with col_diag:
        st.markdown("##### End-to-End Pipeline")
        pipes = [
            ("Raw EEG", f"{n_ch} ch × {T} samples\n({duration}s @ {fs}Hz)", "#636efa",
             "Bandpass [0.5–60] Hz\nNotch 50 Hz\nAverage Ref"),
            ("Temporal Conv", f"Conv2d 1 → {cfg.get('F1',8)}\nkernel=(1,63)\nBatchNorm", "#00cc96",
             f"→ ({cfg.get('F1',8)}, {n_ch}, {T})"),
            ("Depthwise Spatial", f"DepthConv {cfg.get('F1',8)} → {cfg.get('F1',8)*cfg.get('D',2)}\n"
             f"kernel=({n_ch},1) groups={cfg.get('F1',8)}", "#ef553b",
             f"→ ({cfg.get('F1',8)*cfg.get('D',2)}, 1, {T // cfg.get('pool1',8)})\nELU + AvgPool + Dropout"),
            ("Separable Conv", f"DepthConv → Pointwise\n{cfg.get('F1',8)*cfg.get('D',2)} → {cfg.get('F2',16)}", "#ab63fa",
             f"→ ({cfg.get('F2',16)}, 1, {T // cfg.get('pool1',8) // cfg.get('pool2',8)})\nELU + AvgPool + Dropout"),
            ("Classifier", f"Conv2d {cfg.get('F2',16)} → {n_cls}\nkernel=(1,1)", "#ffa15a",
             f"→ logits per time-step"),
            ("Aggregation", f"Mean-Max Pooling\nα = {cfg.get('meanmax_alpha',0.5)}", "#19d3f3",
             f"→ ({n_cls})"),
            ("Output", f"HC vs MDD\n({n_cls} classes)", "#e6ab02",
             f"logits → softmax"),
        ]

        fig = go.Figure()
        fig.update_layout(showlegend=False, xaxis=dict(visible=False, range=[-1.5, 1.5]),
                          yaxis=dict(visible=False, range=[-1, 6.5]), height=580,
                          margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        y_positions = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        for i, (name, inner, color, out) in enumerate(pipes):
            yc = y_positions[i]
            bw, bh = 1.4, 0.45
            fig.add_shape(type="rect", x0=-bw/2, x1=bw/2, y0=yc-bh/2, y1=yc+bh/2,
                          line=dict(color=color, width=2), fillcolor=color, opacity=0.12)
            fig.add_annotation(x=-bw/2-0.08, y=yc, xanchor="right", text=f"<b>{name}</b>",
                               font=dict(size=11, color=color), showarrow=False)
            fig.add_annotation(x=0, y=yc, text=inner.replace("\n", "<br>"),
                               font=dict(size=9, color="#666"), showarrow=False)
            fig.add_annotation(x=bw/2+0.08, y=yc, xanchor="left",
                               text=f"<span style='color:#999'>{out.replace(chr(10),'<br>')}</span>",
                               showarrow=False)
            if i < len(pipes) - 1:
                ny = y_positions[i + 1]
                fig.add_annotation(x=0, y=(yc-bh/2 + ny+0.225)/2, ax=0, ay=yc-bh/2,
                                   axref="x", ayref="y", xref="x", yref="y",
                                   showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#bbb")
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.markdown("##### Model Info")
        st.markdown(f"**Architecture:** {sel_m}")
        st.markdown(f"**Version:** {sel_v}")
        st.markdown(f"**Parameters:** Total **{total_p:,}** | Trainable **{train_p:,}**")
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
        shapes: dict = {}
        hooks = []

        def make_hook(path: str):
            def hook_fn(m, inp, out):
                in_shape = tuple(inp[0].shape) if isinstance(inp, (list, tuple)) and inp[0] is not None else ()
                out_shape = tuple(out.shape) if hasattr(out, "shape") else ()
                p = sum(p.numel() for p in m.parameters())
                shapes[path] = {"in": in_shape, "out": out_shape, "params": p}
            return hook_fn

        for name, m in model.named_modules():
            if name:
                hooks.append(m.register_forward_hook(make_hook(name)))

        dummy = torch.zeros(1, n_ch, T)
        with torch.no_grad():
            logits, logits_time = model(dummy)

        for h in hooks:
            h.remove()

        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        lines = []
        sep = "=" * 95
        lines.append(sep)
        lines.append(f"{'Module':<35} {'Input Shape':<28} {'Output Shape':<28} {'Params':<8}")
        lines.append(sep)

        input_shape = f"(1, {n_ch}, {T})"
        lines.append(f"{'Input':<35} {'—':<28} {input_shape:<28} {'0':<8}")
        lines.append(f"{'  └── Raw EEG signal':<35} {'—':<28} {input_shape:<28} {'0':<8}")
        lines.append("")

        parent_map: dict[str, list[str]] = {}
        all_paths = sorted(shapes.keys(), key=lambda x: (x.count("."), x))
        for path in all_paths:
            if "." in path:
                parent = path.rsplit(".", 1)[0]
                parent_map.setdefault(parent, []).append(path)

        root_children = parent_map.get(sel_m.lower() if sel_m.lower() in parent_map else "", [])
        if not root_children:
            root_children = [p for p in all_paths if "." not in p]

        for path in all_paths:
            if "." in path or path == sel_m.lower():
                continue
            s = shapes.get(path, {})
            if not s:
                continue
            label = path.replace("_", " ").title()
            lines.append(f"{label:<35} {str(s.get('in', ())):<28} {str(s.get('out', ())):<28} {s.get('params', 0):<8}")
            children = parent_map.get(path, [])
            for child in children:
                cs = shapes.get(child, {})
                if cs:
                    child_label = child.split(".")[-1]
                    child_type = type(dict(model.named_modules()).get(child, nn.Module)).__name__
                    display = f"{child_label}"
                    lines.append(
                        f"  ├── {display:<33} {str(cs.get('in', ())):<28} {str(cs.get('out', ())):<28} {cs.get('params', 0):<8}"
                    )
            lines.append("")

        classif_shape = shapes.get("classifier", {}).get("out", None)
        if classif_shape:
            B, C, _, T_last = classif_shape
            logits_per_time = (B, T_last, C)
            lines.append(f"{'Logits (per time-step)':<35} {str(classif_shape):<28} {str(logits_per_time):<28} {'0':<8}")
            lines.append("")

        agg_out = tuple(logits.shape) if logits is not None else None
        if agg_out:
            logits_time_shape = tuple(logits_time.shape) if logits_time is not None else None
            inp = logits_per_time if classif_shape else logits_time_shape
            lines.append(f"{'Aggregation':<35} {str(inp):<28} {str(agg_out):<28} {'0':<8}")
            lines.append("")
            lines.append(f"{'Output':<35} {str(agg_out):<28} {str(agg_out):<28} {'0':<8}")

        lines.append("")
        lines.append(sep)
        lines.append(f"{'Total parameters':<35} {total_p:<8}")
        lines.append(f"{'Trainable parameters':<35} {train_p:<8}")
        lines.append(f"{'Non-trainable parameters':<35} {total_p - train_p:<8}")
        lines.append(sep)

        st.code("\n".join(lines), language="text")

    except Exception as e:
        st.code(f"Summary unavailable: {e}")

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
