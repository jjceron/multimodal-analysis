from __future__ import annotations

import streamlit as st
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Predictions", page_icon="", layout="wide")
render_sidebar()

st.title("Predictions Analysis")

st.markdown("""
### Future Plans

- **Subject-level predictions** — visualize per-subject classification results (HC vs MDD) across all folds
- **Confidence calibration** — reliability diagrams showing how well predicted probabilities match true outcomes
- **Error analysis** — identify subjects or EEG segments most frequently misclassified, with raw signal inspection
- **Cross-version comparison** — compare predictions from different aggregation strategies side by side
- **Temporal attention** — highlight which time windows drive decisions (requires saliency/attention mechanisms)
- **Export reports** — generate PDF summaries of prediction results per model version
""")
