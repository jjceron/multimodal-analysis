from __future__ import annotations

import streamlit as st
from utils.loader import list_models, list_versions


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

    models = list_models()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = models[-1] if models else None
    if "selected_version" not in st.session_state:
        st.session_state.selected_version = None

    sel_model = st.sidebar.selectbox(
        "Model",
        options=models if models else ["No models"],
        index=len(models) - 1 if models else 0,
        key="model_selector",
    )

    versions = list_versions(sel_model) if sel_model and sel_model != "No models" else []
    sel_version = st.sidebar.selectbox(
        "Version",
        options=versions if versions else ["No versions"],
        index=len(versions) - 1 if versions else 0,
        key="version_selector",
    )

    changed = (
        sel_model != st.session_state.selected_model
        or sel_version != st.session_state.selected_version
    )

    if sel_model and sel_model != "No models":
        st.session_state.selected_model = sel_model
    if sel_version and sel_version != "No versions":
        st.session_state.selected_version = sel_version

    if changed:
        st.cache_data.clear()
        st.rerun()

    n_models = len(models)
    n_versions = sum(len(list_versions(m)) for m in models)
    st.sidebar.success(f"**{n_models} models / {n_versions} versions**")
    st.sidebar.markdown("---")

    st.sidebar.page_link("app.py", label="Overview", icon="🏠")
    st.sidebar.page_link("pages/1_experiments.py", label="Training", icon="📊")
    st.sidebar.page_link("pages/3_predictions.py", label="Predictions", icon="🎯")
    st.sidebar.page_link("pages/4_optimization.py", label="Optimization", icon="⚙️")
    st.sidebar.page_link("pages/5_interpretability.py", label="Interpretability", icon="🔍")
