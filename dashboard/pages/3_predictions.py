import streamlit as st
import pandas as pd
from utils.loader import list_experiments, load_predictions, load_config
from utils.plots import plot_dual_confusion_matrix
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")
render_sidebar()

st.title("🎯 Predictions Analysis")

experiments = list_experiments()

if not experiments:
    st.warning("No experiments found.")
    st.stop()

selected = st.selectbox(
    "Experiment",
    options=experiments,
    index=len(experiments) - 1,
    key="pred_exp_selector",
)

df_pred = load_predictions(selected)

if df_pred.empty:
    st.warning("No predictions available.")
    st.stop()

st.markdown(f"**Total predictions:** {len(df_pred)}")
st.markdown(f"**Subjects:** {df_pred['subject'].nunique()}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Global Confusion Matrix")
    val_all = df_pred[df_pred["split"] == "val"]
    test_all = df_pred[df_pred["split"] == "test"]

    if not val_all.empty and not test_all.empty:
        fig = plot_dual_confusion_matrix(
            y_true_val=val_all["true_label"].tolist(),
            y_pred_val=val_all["pred_label"].tolist(),
            y_true_test=test_all["true_label"].tolist(),
            y_pred_test=test_all["pred_label"].tolist(),
            class_names=["HC", "MDD"],
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Correct vs Incorrect")
    df_pred["correct"] = df_pred["true_label"] == df_pred["pred_label"]
    correct_count = df_pred["correct"].sum()
    total = len(df_pred)
    st.metric("Correct", f"{correct_count}/{total}", delta=f"{correct_count/total:.2%}")

    errors = df_pred[~df_pred["correct"]]
    if not errors.empty:
        st.markdown("### Misclassified Subjects")
        st.dataframe(
            errors[["subject", "fold", "split", "true_label", "pred_label"]]
            .rename(columns={"true_label": "True", "pred_label": "Pred"}),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.markdown("### All Predictions Table")
view_mode = st.radio("View", ["All", "Errors Only", "By Split"], horizontal=True)

if view_mode == "Errors Only":
    display_df = df_pred[~df_pred["correct"]]
elif view_mode == "By Split":
    split = st.selectbox("Split", options=df_pred["split"].unique())
    display_df = df_pred[df_pred["split"] == split]
else:
    display_df = df_pred

st.dataframe(
    display_df.drop(columns=["correct"], errors="ignore"),
    use_container_width=True,
    hide_index=True,
)
