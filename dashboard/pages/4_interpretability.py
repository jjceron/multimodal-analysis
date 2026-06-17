import streamlit as st
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Interpretability", page_icon="🔍", layout="wide")
render_sidebar()

tab_latent, tab_explain = st.tabs(["Latent Space", "Feature Attribution"])

with tab_latent:
    st.title("Latent Space Visualization")

    st.info(
        "🚧 **Latent Space — coming soon.**\n\n"
        "This tab will visualize the feature representations learned by the model, "
        "including:\n\n"
        "- **t-SNE / UMAP projections** of EEG embeddings\n"
        "- **Feature relevance** analysis (e.g., channel importance)\n"
        "- **Separability** of classes in latent space\n"
        "- **Attention maps** for Transformer-based models"
    )

with tab_explain:
    st.title("Feature Attribution")

    st.info(
        "🚧 **Feature Attribution — coming soon.**\n\n"
        "This tab will show:\n\n"
        "- **Saliency maps** / Grad-CAM for EEG signals\n"
        "- **Channel-wise importance** scores\n"
        "- **Temporal regions** driving classification decisions\n"
        "- **Occlusion sensitivity** analysis"
    )
