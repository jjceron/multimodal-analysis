import streamlit as st
from utils.loader import (
    load_config,
    load_fold_metrics,
    load_predictions,
    load_results,
    list_experiments,
)
from utils.plots import plot_training_curves, plot_dual_confusion_matrix, plot_fold_bars
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Experiments", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 Experiment Analysis")

experiments = list_experiments()

if not experiments:
    st.warning("No experiments found.")
    st.stop()

selected = st.selectbox(
    "Select experiment",
    options=experiments,
    index=len(experiments) - 1,
    key="exp_page_selector",
)

cfg = load_config(selected)
df_folds = load_fold_metrics(selected)
results = load_results(selected)

if cfg:
    st.markdown("### Configuration")
    cols = st.columns(5)
    cols[0].metric("Model", f"F1={cfg.get('F1')} D={cfg.get('D')} F2={cfg.get('F2')}")
    cols[1].metric("Duration", f"{cfg.get('duration_sec', '?')}s")
    cols[2].metric("Batch", cfg.get("batch_size"))
    cols[3].metric("Weight Decay", cfg.get("weight_decay", 0))
    cols[4].metric("LR Scheduler", "Yes" if cfg.get("lr_scheduler") else "No")

if results:
    overall = results.get("overall", {})
    if overall:
        st.markdown("### Overall Metrics")
        cols = st.columns(3)
        cols[0].metric("Mean Accuracy", f"{overall.get('mean_accuracy', 0):.2%}",
                       delta=f"±{overall.get('std_accuracy', 0):.2%}")
        cols[1].metric("Mean Balanced Acc", f"{overall.get('mean_balanced_accuracy', 0):.2%}",
                       delta=f"±{overall.get('std_balanced_accuracy', 0):.2%}")
        cols[2].metric("Mean F1-macro", f"{overall.get('mean_f1_macro', 0):.4f}",
                       delta=f"±{overall.get('std_f1_macro', 0):.4f}")

if not df_folds.empty:
    st.markdown("### Per-Fold Metrics")
    st.dataframe(df_folds, use_container_width=True, hide_index=True)

    st.markdown("### Fold Bar Chart")
    metric = st.selectbox("Metric", options=[c for c in df_folds.columns
                                             if c not in ("fold", "best_epoch", "n_epochs")],
                          index=0, key="exp_bar_metric")
    fig = plot_fold_bars(df_folds, metric=metric)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Training Curves per Fold")
    train_data = results.get("fold_data", [])

    if train_data:
        for fold_data in train_data:
            fold_id = fold_data.get("fold_id", 0)
            fig = plot_training_curves(
                fold_id=fold_id,
                train_losses=fold_data.get("train_losses", []),
                val_losses=fold_data.get("val_losses", []),
                train_metrics=fold_data.get("train_accs", []),
                val_metrics=fold_data.get("val_accs", []),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Training curves data not available in results.json. "
                "Run a new benchmark with full logging to see curves here.")
