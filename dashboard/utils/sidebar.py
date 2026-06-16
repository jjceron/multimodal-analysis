from __future__ import annotations

import streamlit as st
from utils.loader import list_experiments


def render_sidebar() -> None:
    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.title("MultimodalAnalysis")

    experiments = list_experiments()

    if "selected_experiment" not in st.session_state:
        if experiments:
            st.session_state.selected_experiment = experiments[-1]
        else:
            st.session_state.selected_experiment = None

    selected = st.sidebar.selectbox(
        "Experiment",
        options=experiments if experiments else ["No experiments"],
        index=len(experiments) - 1 if experiments else 0,
        key="experiment_selector",
    )

    if selected != st.session_state.selected_experiment:
        st.session_state.selected_experiment = selected
        st.cache_data.clear()
        st.rerun()

    st.sidebar.success(f"**{len(experiments)}** experiments available")
    st.sidebar.markdown("---")

    st.sidebar.page_link("app.py", label="Overview", icon="🏠")
    st.sidebar.page_link("pages/1_experiments.py", label="Experiments", icon="📊")
    st.sidebar.page_link("pages/2_folds.py", label="Folds", icon="🔬")
    st.sidebar.page_link("pages/3_predictions.py", label="Predictions", icon="🎯")
    st.sidebar.page_link("pages/4_optimization.py", label="Optimization", icon="⚙️")
    st.sidebar.page_link("pages/5_interpretability.py", label="Interpretability", icon="🔍")
