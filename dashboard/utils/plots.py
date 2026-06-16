from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CLASS_COLORS = {0: "#4A90D9", 1: "#E74C3C"}
CLASS_NAMES = ["HC", "MDD"]


def plot_training_curves(
    fold_id: int,
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: list[float] | None = None,
    val_metrics: list[float] | None = None,
    metric_name: str = "Accuracy",
) -> go.Figure:
    epochs = list(range(1, len(train_losses) + 1))
    best_epoch = int(np.argmin(val_losses)) + 1

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Loss Curves — Fold {fold_id:02d}",
                        f"{metric_name} Curves — Fold {fold_id:02d}"),
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=train_losses, mode="lines+markers",
                   name="Train", line=dict(color="#4A90D9"), marker=dict(size=4)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_losses, mode="lines+markers",
                   name="Val", line=dict(color="#E74C3C"), marker=dict(size=4)),
        row=1, col=1,
    )
    fig.add_vline(x=best_epoch, line=dict(color="gray", dash="dash", width=1),
                  row=1, col=1)
    fig.add_annotation(
        x=best_epoch, y=val_losses[best_epoch - 1],
        text=f"Best @ {best_epoch} ({val_losses[best_epoch - 1]:.4f})",
        showarrow=True, arrowhead=1, row=1, col=1,
    )

    if train_metrics is not None and val_metrics is not None:
        fig.add_trace(
            go.Scatter(x=epochs, y=train_metrics, mode="lines+markers",
                       name="Train", line=dict(color="#4A90D9"), marker=dict(size=4),
                       showlegend=False),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_metrics, mode="lines+markers",
                       name="Val", line=dict(color="#E74C3C"), marker=dict(size=4),
                       showlegend=False),
            row=1, col=2,
        )
        fig.add_vline(x=best_epoch, line=dict(color="gray", dash="dash", width=1),
                      row=1, col=2)
        fig.add_annotation(
            x=best_epoch, y=val_metrics[best_epoch - 1],
            text=f"Best @ {best_epoch} ({val_metrics[best_epoch - 1]:.4f})",
            showarrow=True, arrowhead=1, row=1, col=2,
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text=metric_name, row=1, col=2)
    fig.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40))

    return fig


def plot_dual_confusion_matrix(
    y_true_val: list[int],
    y_pred_val: list[int],
    y_true_test: list[int],
    y_pred_test: list[int],
    class_names: list[str] | None = None,
) -> go.Figure:
    if class_names is None:
        class_names = CLASS_NAMES

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Validation", "Test"),
        horizontal_spacing=0.2,
    )

    for col_idx, (yt, yp, title) in enumerate([
        (y_true_val, y_pred_val, "Validation"),
        (y_true_test, y_pred_test, "Test"),
    ], start=1):
        cm = confusion_matrix(yt, yp)
        cm_perc = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

        z_text = [[f"{cm[i, j]}<br>({cm_perc[i, j]:.1f}%)"
                   for j in range(cm.shape[1])]
                  for i in range(cm.shape[0])]

        fig.add_trace(
            go.Heatmap(
                z=cm_perc,
                x=class_names,
                y=class_names,
                text=z_text,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=(col_idx == 2),
                colorbar=dict(title="%", x=1.02),
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
            ),
            row=1, col=col_idx,
        )

    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_yaxes(title_text="True", row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=1, col=2)
    fig.update_yaxes(title_text="True", row=1, col=2)
    fig.update_layout(height=400, margin=dict(l=40, r=60, t=40, b=40))

    return fig


def plot_metrics_comparison(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for metric, color, label in [
        ("accuracy", "#4A90D9", "Accuracy"),
        ("balanced_accuracy", "#2ECC71", "Balanced Acc"),
        ("f1_macro", "#E74C3C", "F1-macro"),
    ]:
        mean_col = metric
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            continue

        fig.add_trace(go.Bar(
            name=label,
            x=df["experiment"],
            y=df[mean_col],
            error_y=dict(type="data", array=df[std_col], visible=True),
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Experiment",
        yaxis_title="Score",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=80),
    )

    return fig


def plot_training_curves_combined(
    experiments_data: dict[str, dict],
    metric_name: str = "Accuracy",
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Loss", metric_name),
        vertical_spacing=0.12,
    )

    colors = px.colors.qualitative.Plotly

    for exp_idx, (exp_name, data) in enumerate(experiments_data.items()):
        fold_data = data.get("fold_data", [])
        if not fold_data:
            continue

        max_len = max(len(fd["train_losses"]) for fd in fold_data)
        train_losses_sum = np.zeros(max_len)
        val_losses_sum = np.zeros(max_len)
        train_accs_sum = np.zeros(max_len)
        val_accs_sum = np.zeros(max_len)
        n_folds = len(fold_data)

        for fd in fold_data:
            tl = np.array(fd["train_losses"])
            vl = np.array(fd["val_losses"])
            ta = np.array(fd["train_accs"])
            va = np.array(fd["val_accs"])
            train_losses_sum[:len(tl)] += tl
            val_losses_sum[:len(vl)] += vl
            train_accs_sum[:len(ta)] += ta
            val_accs_sum[:len(va)] += va

        train_losses_avg = train_losses_sum / n_folds
        val_losses_avg = val_losses_sum / n_folds
        train_accs_avg = train_accs_sum / n_folds
        val_accs_avg = val_accs_sum / n_folds
        epochs = list(range(1, max_len + 1))
        color = colors[exp_idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=epochs, y=train_losses_avg, mode="lines",
            name=f"{exp_name} (train)", line=dict(color=color, width=2),
            legendgroup=exp_name,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=val_losses_avg, mode="lines",
            name=f"{exp_name} (val)", line=dict(color=color, width=2, dash="dash"),
            legendgroup=exp_name,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=train_accs_avg, mode="lines",
            name=f"{exp_name} (train)", line=dict(color=color, width=2),
            legendgroup=exp_name, showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=val_accs_avg, mode="lines",
            name=f"{exp_name} (val)", line=dict(color=color, width=2, dash="dash"),
            legendgroup=exp_name, showlegend=False,
        ), row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text=metric_name, row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=50, r=50, t=40, b=40))
    return fig


def plot_fold_bars(df: pd.DataFrame, metric: str = "test_accuracy") -> go.Figure:
    df = df.copy()
    df["fold_label"] = df["fold"].apply(lambda x: f"Fold {x}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["fold_label"],
        y=df[metric],
        marker_color="#4A90D9",
        text=df[metric].round(4),
        textposition="outside",
    ))

    mean_val = df[metric].mean()
    fig.add_hline(
        y=mean_val,
        line=dict(color="#E74C3C", dash="dash", width=2),
        annotation_text=f"Mean: {mean_val:.4f}",
    )

    fig.update_layout(
        xaxis_title="Fold",
        yaxis_title=metric.replace("_", " ").title(),
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
    )

    return fig
