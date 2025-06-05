import streamlit as st
import os
from transformers import pipeline

# ✅ Use safe writable directories for model caching
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

# ✅ Load summarizer model to /tmp and cache it
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", cache_dir="/tmp")

# ✅ Page config
st.set_page_config(
    page_title="GenAI Summarizer",
    layout="wide",
    page_icon="🧠"
)

# ✅ Custom CSS for a clean dark UI
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0e1117;
            color: #f0f2f6;
        }
        .block-container { padding: 3rem 2rem 2rem 2rem; }
        h1.title { font-size: 3.2rem; color: #F63366; text-align: center; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; font-size: 1.1rem; color: #999; margin-bottom: 2.5rem; }
        .summary-box {
            background-color: #1e222d;
            border: 1px solid #333;
            padding: 1.2rem 1.5rem;
            border-radius: 10px;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 0 8px rgba(255,255,255,0.03);
        }
        textarea { font-size: 1.05rem !important; padding: 1rem; }
        .stButton > button {
            background-color: #F63366;
            color: white;
            border-radius: 8px;
            font-size: 1.1rem;
            padding: 0.6rem 1.2rem;
            margin-top: 1rem;
            border: none;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #F63366 !important;
            opacity: 0.9;
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Title and instructions
st.markdown("<h1 class='title'>🧠 GenAI Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Paste any article text and get a sharp, concise summary powered by DistilBART.</div>", unsafe_allow_html=True)

# ✅ Load model
with st.spinner("🔄 Initializing AI engine..."):
    summarizer = load_model()

# ✅ Text input
article_text = st.text_area("", height=300, placeholder="Paste your full article here...")

# ✅ Button and result
if st.button("🚀 Summarize"):
    if article_text.strip():
        with st.spinner("Summarizing..."):
            try:
                summary = summarizer(article_text, max_length=250, min_length=80, do_sample=False)
                st.markdown("### Summary")
                st.markdown(f"<div class='summary-box'>{summary[0]['summary_text']}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please paste article text before summarizing.")
