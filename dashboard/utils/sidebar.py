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
    has_models = bool(models)
    sel_m = st.sidebar.selectbox(
        "Model",
        options=models if has_models else ["No models"],
        index=len(models) - 1 if has_models else 0,
        key="model_selector",
    )

    has_valid_m = sel_m and sel_m != "No models"
    versions = list_versions(dataset, sel_m) if has_valid_m else []
    has_versions = bool(versions)
    sel_v = st.sidebar.selectbox(
        "Version",
        options=versions if has_versions else ["No versions"],
        index=len(versions) - 1 if has_versions else 0,
        key="version_selector",
    )

    has_valid_v = sel_v and sel_v != "No versions"

    changed = (
        dataset != st.session_state.selected_dataset
        or (has_valid_m and sel_m != st.session_state.selected_model)
        or (has_valid_v and sel_v != st.session_state.selected_version)
    )

    st.session_state.selected_dataset = dataset
    if has_valid_m:
        st.session_state.selected_model = sel_m
    if has_valid_v:
        st.session_state.selected_version = sel_v

    if changed and (has_valid_m or has_valid_v):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.page_link("app.py", label="Overview", icon="🏠")
    st.sidebar.page_link("pages/1_experiments.py", label="Training", icon="📊")
    st.sidebar.page_link("pages/3_predictions.py", label="Predictions", icon="🎯")
    st.sidebar.page_link("pages/4_optimization.py", label="Optimization", icon="⚙️")
    st.sidebar.page_link("pages/5_interpretability.py", label="Interpretability", icon="🔍")
