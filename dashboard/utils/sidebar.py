from __future__ import annotations

import streamlit as st
from utils.loader import list_models, list_versions

DATASET_MAP = {"MODMA": "modma_db"}


def render_sidebar() -> None:
    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.title("Neuro Signals DL")

    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "modma_db"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "selected_version" not in st.session_state:
        st.session_state.selected_version = None

    display_opts = list(DATASET_MAP.keys())
    current_display = next((k for k, v in DATASET_MAP.items()
                            if v == st.session_state.selected_dataset), display_opts[-1])
    selected_display = st.sidebar.selectbox(
        "Dataset",
        options=display_opts,
        index=display_opts.index(current_display),
        key="dataset_selector",
    )
    dataset = DATASET_MAP[selected_display]

    models = list_models(dataset)
    sel_m = st.sidebar.selectbox(
        "Model",
        options=models if models else ["No models"],
        index=len(models) - 1 if models else 0,
        key="model_selector",
    )

    versions = list_versions(dataset, sel_m) if sel_m and sel_m != "No models" else []
    sel_v = st.sidebar.selectbox(
        "Version",
        options=versions if versions else ["No versions"],
        index=len(versions) - 1 if versions else 0,
        key="version_selector",
    )

    changed = (
        dataset != st.session_state.selected_dataset
        or sel_m != st.session_state.selected_model
        or sel_v != st.session_state.selected_version
    )

    st.session_state.selected_dataset = dataset
    if sel_m and sel_m != "No models":
        st.session_state.selected_model = sel_m
    if sel_v and sel_v != "No versions":
        st.session_state.selected_version = sel_v

    if changed:
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.page_link("app.py", label="Overview", icon="🏠")
    st.sidebar.page_link("pages/1_experiments.py", label="Training", icon="📊")
    st.sidebar.page_link("pages/3_predictions.py", label="Predictions", icon="🎯")
    st.sidebar.page_link("pages/4_optimization.py", label="Optimization", icon="⚙️")
    st.sidebar.page_link("pages/5_interpretability.py", label="Interpretability", icon="🔍")
