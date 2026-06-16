from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import EEGNet
from utils.loader import load_config

st.set_page_config(page_title="Architecture", layout="wide")
st.title("Model Architecture")

cfg = load_config(st.session_state.get("selected_experiment", ""))

if not cfg:
    st.info("Select an experiment from the sidebar to view its architecture.")
    st.stop()

F1 = cfg.get("F1", 8)
D = cfg.get("D", 2)
F2 = cfg.get("F2", 16)
n_channels = cfg.get("n_channels", 128)
n_classes = cfg.get("n_classes", 2)
temporal_kern = cfg.get("temporal_kern", 63)
separable_kern = cfg.get("separable_kern", 15)
pool1 = cfg.get("pool1", 8)
pool2 = cfg.get("pool2", 8)
dropout = cfg.get("dropout", 0.5)
meanmax_alpha = cfg.get("meanmax_alpha", 0.5)
duration_sec = cfg.get("duration_sec", 120.0)
target_fs = cfg.get("target_fs", None)
fs = target_fs if target_fs is not None else 250
T = int(duration_sec * fs)


@st.cache_resource
def build_model(
    n_channels_: int,
    n_classes_: int,
    F1_: int,
    D_: int,
    F2_: int,
    temporal_kern_: int,
    separable_kern_: int,
    pool1_: int,
    pool2_: int,
    dropout_: float,
    meanmax_alpha_: float,
) -> EEGNet:
    return EEGNet(
        n_channels=n_channels_,
        n_classes=n_classes_,
        F1=F1_,
        D=D_,
        F2=F2_,
        temporal_kern=temporal_kern_,
        separable_kern=separable_kern_,
        pool1=pool1_,
        pool2=pool2_,
        dropout=dropout_,
        meanmax_alpha=meanmax_alpha_,
        aggregate=True,
    )


model = build_model(
    n_channels, n_classes, F1, D, F2,
    temporal_kern, separable_kern, pool1, pool2,
    dropout, meanmax_alpha,
)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    T_pool1 = T // pool1
    T_total = T_pool1 // pool2

    blocks = [
        {
            "name": "Input",
            "layers": f"Shape: (B, {n_channels}, {T})",
            "color": "#636efa",
            "shape": f"(B, {n_channels}, {T})",
        },
        {
            "name": "Temporal Block",
            "layers": (
                f"Conv2d(1→{F1}, (1,{temporal_kern}))\n"
                f"BatchNorm2d({F1})"
            ),
            "color": "#00cc96",
            "shape": f"(B, {F1}, {n_channels}, {T})",
        },
        {
            "name": "Spatial Block",
            "layers": (
                f"DepthConv({F1}→{F1*D}, ({n_channels},1))\n"
                f"BatchNorm2d({F1*D})\n"
                f"ELU + AvgPool(1,{pool1}) + Dropout"
            ),
            "color": "#ef553b",
            "shape": f"(B, {F1*D}, 1, {T_pool1})",
        },
        {
            "name": "Separable Block",
            "layers": (
                f"DepthConv({F1*D}→{F1*D}, (1,{separable_kern}))\n"
                f"PointConv({F1*D}→{F2}, (1,1))\n"
                f"BatchNorm2d({F2})\n"
                f"ELU + AvgPool(1,{pool2}) + Dropout"
            ),
            "color": "#ab63fa",
            "shape": f"(B, {F2}, 1, {T_total})",
        },
        {
            "name": "Classifier",
            "layers": (
                f"Conv2d({F2}→{n_classes}, (1,1))\n"
                f"Squeeze + Permute"
            ),
            "color": "#ffa15a",
            "shape": f"(B, {T_total}, {n_classes})",
        },
        {
            "name": "Aggregation",
            "layers": (
                f"Mean-Max Pooling\n"
                f"(α = {meanmax_alpha:.2f})"
            ),
            "color": "#19d3f3",
            "shape": f"(B, {n_classes})",
        },
    ]

    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=f"<b>EEGNet</b> | F1={F1} D={D} F2={F2} | {duration_sec}s @ {fs}Hz",
            font=dict(size=16),
        ),
        showlegend=False,
        xaxis=dict(visible=False, range=[-1, 1]),
        yaxis=dict(visible=False, range=[-1, 6.5]),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    y_spacing = 1.0
    start_y = 6.0
    box_w = 1.2
    box_h = 0.55

    for i, block in enumerate(blocks):
        y_center = start_y - i * y_spacing
        y_bottom = y_center - box_h / 2
        y_top = y_center + box_h / 2

        fig.add_shape(
            type="rect",
            x0=-box_w / 2,
            x1=box_w / 2,
            y0=y_bottom,
            y1=y_top,
            line=dict(color=block["color"], width=2),
            fillcolor=block["color"],
            opacity=0.15,
        )

        fig.add_annotation(
            x=-box_w / 2 - 0.05,
            y=y_center,
            xanchor="right",
            text=f"<b>{block['name']}</b>",
            font=dict(size=12, color=block["color"]),
            showarrow=False,
        )

        fig.add_annotation(
            x=box_w / 2 + 0.05,
            y=y_center,
            xanchor="left",
            text=block["layers"].replace("\n", "<br>"),
            font=dict(size=10, color="#888"),
            showarrow=False,
        )

        fig.add_annotation(
            x=0,
            y=y_bottom - 0.12,
            text=block["shape"],
            font=dict(size=9, color="#aaa"),
            showarrow=False,
        )

        if i < len(blocks) - 1:
            y_next = start_y - (i + 1) * y_spacing
            y_arrow_bottom = y_bottom - 0.02
            y_arrow_top = y_next + box_h / 2 + 0.02

            fig.add_annotation(
                x=0,
                y=(y_arrow_bottom + y_arrow_top) / 2,
                ax=0,
                ay=y_arrow_bottom,
                axref="x",
                ayref="y",
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor="#555",
            )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    st.subheader("Parameter Summary")

    st.markdown(
        f"""
    | Metric | Value |
    |---|---|
    | **Total parameters** | {total_params:,} |
    | **Trainable** | {trainable_params:,} ({trainable_params / total_params * 100:.1f}%) |
    | **Non-trainable** | {non_trainable_params:,} ({non_trainable_params / total_params * 100:.1f}%) |
    """
    )

    st.subheader("Hyperparameters")
    st.json(
        {
            "F1 (temporal filters)": F1,
            "D (depth multiplier)": D,
            "F2 (separable filters)": F2,
            "Temporal kernel": temporal_kern,
            "Separable kernel": separable_kern,
            "Pool 1": pool1,
            "Pool 2": pool2,
            "Dropout": dropout,
            "Mean-Max α": meanmax_alpha,
            "Duration (s)": duration_sec,
            "Sampling rate (Hz)": fs,
            "Channels": n_channels,
            "Classes": n_classes,
        },
        expanded=True,
    )

st.divider()
st.subheader("Layer-by-Layer Breakdown")

try:
    from torchinfo import summary as torch_summary

    summary_result = torch_summary(
        model,
        input_size=(1, n_channels, T),
        device="cpu",
        verbose=0,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=5,
    )

    rows = []
    for layer in summary_result.summary_list:
        if layer.is_leaf_layer:
            rows.append(
                {
                    "Layer": layer.class_name,
                    "Depth": layer.depth,
                    "Input Shape": str(layer.input_size) if layer.input_size else "",
                    "Output Shape": str(layer.output_size) if layer.output_size else "",
                    "Params": layer.num_params,
                    "Trainable": "Yes" if layer.trainable else "No",
                }
            )

    df_layers = pd.DataFrame(rows)
    st.dataframe(
        df_layers,
        column_config={
            "Params": st.column_config.NumberColumn(format="%d"),
        },
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Show raw torchinfo output"):
        st.code(str(summary_result), language="text")

except Exception as e:
    st.text(f"Torchinfo summary unavailable: {e}")

st.caption(
    f"Model instantiated with config from: **{st.session_state.get('selected_experiment', '')}**"
)
