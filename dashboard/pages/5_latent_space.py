import streamlit as st
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Latent Space", page_icon="🌌", layout="wide")
render_sidebar()

st.title("🌌 Latent Space Visualization")

st.info(
    "🚧 **Latent Space page — coming soon.**\n\n"
    "This space will visualize the **feature representations** learned by the model, "
    "including:\n\n"
    "- **t-SNE / UMAP projections** of EEG embeddings\n"
    "- **Feature relevance** analysis (e.g., channel importance)\n"
    "- **Separability** of classes in latent space\n"
    "- **Attention maps** for Transformer-based models\n\n"
    "Once feature extraction utilities are implemented in `src/utils/`, "
    "visualizations will appear here."
)
