import streamlit as st
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")
render_sidebar()

st.title("Predictions Analysis")

st.info(
    "🚧 **Predictions page — coming soon.**\n\n"
    "This space will show per-subject classification results, including:\n\n"
    "- **Subject-level predictions** — HC vs MDD across all folds\n"
    "- **Confidence calibration** — reliability diagrams for predicted probabilities\n"
    "- **Error analysis** — identify frequently misclassified subjects with raw signal inspection\n"
    "- **Cross-version comparison** — compare predictions from different aggregation strategies\n"
    "- **Temporal attention** — time windows driving classification decisions\n\n"
    "Results will appear here once the prediction pipeline is integrated."
)
