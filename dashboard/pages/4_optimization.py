import streamlit as st
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Optimization", page_icon="⚙️", layout="wide")
render_sidebar()

st.title("⚙️ Hyperparameter Optimization")

st.info(
    "🚧 **Optimization page — coming soon.**\n\n"
    "This space will integrate **Optuna** hyperparameter search results, "
    "including:\n\n"
    "- **Parallel coordinate plots** of hyperparameter trials\n"
    "- **Importance analysis** of each hyperparameter\n"
    "- **Best trial** details and configuration\n"
    "- **Comparison** of optimized vs baseline models\n\n"
    "Once Optuna tuning is implemented in `src/tuning/`, results will appear here."
)
