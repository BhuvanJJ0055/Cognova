"""
Task 2 - app.py (Standalone Streamlit UI)
Author: Bhuvan J J

This is the standalone Streamlit application for the Medical Q&A Chatbot (Task 2).
It features:
1. A modern, premium medical interface with customized HSL color styling (glassmorphism details, professional layout).
2. Real-time extraction and color-coded tag rendering of medical entities (symptoms, diseases, treatments).
3. Retrieval-augmented medical question answering, showing confidence match scores.
4. Dataset statistics visualization, listing total Q&A pairs and unique diseases.
5. An interactive button to automatically download the full 16,407 Q&A pair dataset from GitHub and rebuild the index.
6. A professional medical disclaimer.
"""

import os
import streamlit as st
import pandas as pd

# Import local modules
from build_index import MedicalRetriever, INDEX_SAVE_PATH, DEFAULT_CSV_PATH, SAMPLE_CSV_PATH
from entity_recognition import MedicalEntityRecognizer
from data_loader import download_and_extract_medquad, build_dataframe_from_xml

# Page Config
st.set_page_config(
    page_title="Cognova Medical Advisor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich HSL colors, premium modern feel)
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(30, 60, 114, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .header-box h1 {
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-box p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }

    /* Tag Styling */
    .pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin: 0.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .pill-disease {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .pill-symptom {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .pill-treatment {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Chat bubbles styling */
    .chat-bubble {
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    
    .user-bubble {
        background-color: #f0f4f8;
        border-left: 5px solid #2a5298;
    }
    
    .bot-bubble {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #10b981;
    }
    
    .similarity-badge {
        float: right;
        background-color: #10b981;
        color: white;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        font-weight: bold;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Variables
if "index_rebuilt" not in st.session_state:
    st.session_state.index_rebuilt = False

# Helper: check dataset existence
csv_candidates = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "medical_qa.csv"),
    DEFAULT_CSV_PATH,
    SAMPLE_CSV_PATH
]
csv_path = SAMPLE_CSV_PATH
for path in csv_candidates:
    if os.path.exists(path):
        csv_path = path
        break

has_full_data = os.path.exists(DEFAULT_CSV_PATH) or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "medical_qa.csv"))

# Load database info
df_stats = pd.read_csv(csv_path)
total_qa_pairs = len(df_stats)
unique_focus = df_stats["focus"].nunique() if "focus" in df_stats.columns else df_stats["question"].nunique()

# Initialize Retriever and Entity Recognizer
@st.cache_resource
def get_retriever():
    return MedicalRetriever(fallback_csv_path=csv_path)

@st.cache_resource
def get_recognizer():
    return MedicalEntityRecognizer(dataset_csv=csv_path)

retriever = get_retriever()
recognizer = get_recognizer()


# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=96)
    st.title("Advisor Controls")
    
    st.subheader("Dataset Statistics")
    st.metric(label="Active Dataset", value="Full MedQuAD" if has_full_data else "Demo Sample")
    st.metric(label="Total Q&A Pairs", value=f"{total_qa_pairs:,}")
    st.metric(label="Unique Focus Topics", value=f"{unique_focus:,}")
    
    st.subheader("Search Parameters")
    conf_threshold = st.slider(
        "Similarity Cutoff Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="Higher values filter out irrelevant answers. Lower values retrieve more items."
    )
    
    max_results = st.slider(
        "Max Retrieved Results",
        min_value=1,
        max_value=5,
        value=3
    )

    st.markdown("---")
    st.subheader("Data Ingestion")
    
    if not has_full_data:
        st.info("Currently running on a lightweight sample (~40 pairs). Click below to download the full 16,407 Q&A pairs from GitHub and rebuild the index (~24MB).")
        if st.button("Download & Build Full Index"):
            with st.spinner("Downloading MedQuAD repository from GitHub and parsing XMLs (takes about 45s)..."):
                try:
                    # Run download and extraction
                    download_and_extract_medquad()
                    build_dataframe_from_xml()
                    
                    # Force rebuild retrieval index
                    temp_retriever = MedicalRetriever(index_path=INDEX_SAVE_PATH, fallback_csv_path=DEFAULT_CSV_PATH)
                    temp_retriever.build_and_save_index(DEFAULT_CSV_PATH)
                    
                    st.success("Successfully unzipped and indexed 16,407 QA pairs! Please reload the app to apply changes.")
                    st.session_state.index_rebuilt = True
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error rebuilding index: {ex}")
    else:
        st.success("Full MedQuAD Dataset Ingested and Indexed!")
        if st.button("Force Rebuild Index"):
            with st.spinner("Rebuilding TF-IDF representation..."):
                retriever.build_and_save_index(DEFAULT_CSV_PATH)
                st.success("Index rebuilt successfully!")
                st.rerun()


# Main Application Page
st.markdown("""
    <div class="header-box">
        <h1>⚕️ Cognova Medical Q&A Advisor</h1>
        <p>Intelligent, fast semantic question-answering powered by the MedQuAD Dataset</p>
    </div>
""", unsafe_allow_html=True)

# Medical Disclaimer Box
st.warning(
    "⚠️ **MEDICAL DISCLAIMER:** This chatbot is an automated information retrieval demonstration using the MedQuAD dataset. "
    "It is intended solely for educational and research purposes. It **does not** provide medical advice, diagnosis, or treatment. "
    "Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition."
)

st.markdown("### Ask a Medical Question")
st.write("Type a question below. The advisor will recognize entities (diseases, symptoms, treatments) and retrieve relevant QA entries from MedQuAD.")

user_query = st.text_input(
    "Your Query:",
    placeholder="e.g., What are the symptoms of Lupus? Or how is Diabetes treated?",
    key="medical_query_input"
)

if user_query:
    # 1. Run Entity Recognition
    entities = recognizer.extract_entities(user_query)
    
    # Render entities beautifully
    st.markdown("#### 🔍 Recognized Medical Entities:")
    has_entities = False
    
    disease_html = ""
    for d in entities["diseases"]:
        disease_html += f'<span class="pill pill-disease">🦠 Disease: {d.title()}</span>'
        has_entities = True
        
    symptom_html = ""
    for s in entities["symptoms"]:
        symptom_html += f'<span class="pill pill-symptom">⚠️ Symptom: {s.title()}</span>'
        has_entities = True
        
    treatment_html = ""
    for t in entities["treatments"]:
        treatment_html += f'<span class="pill pill-treatment">💊 Treatment: {t.title()}</span>'
        has_entities = True

    if has_entities:
        st.markdown(f"<div>{disease_html}{symptom_html}{treatment_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown("ℹ️ *No common diseases, symptoms, or treatments detected in query. Performing generic keyword retrieval.*")

    st.markdown("---")
    
    # 2. Run Retrieval Indexing
    with st.spinner("Retrieving matching knowledge..."):
        results = retriever.retrieve(user_query, threshold=conf_threshold, top_k=max_results)
        
    if results:
        st.markdown(f"### 📋 Retrieved Answers (Top {len(results)} matches):")
        for i, match in enumerate(results):
            score_pct = int(match["similarity_score"] * 100)
            
            with st.container():
                st.markdown(f"""
                <div class="chat-bubble bot-bubble">
                    <span class="similarity-badge">{score_pct}% Match Score</span>
                    <h5 style="margin: 0; color: #2a5298;"><b>Topic Focus:</b> {match['focus']} ({match['question_type'].upper()})</h5>
                    <p style="margin-top: 0.5rem; color: #555;"><b>Question:</b> <i>{match['question']}</i></p>
                    <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e2e8f0;">
                    <p style="margin: 0.5rem 0 0 0; color: #2D3748; line-height: 1.6;"><b>Answer:</b> {match['answer']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("### ❌ No Matches Found")
        st.error(
            f"No medical answers in our index matched your question above the similarity threshold of **{conf_threshold}**. "
            "Try lowering the threshold in the sidebar slider or phrasing your question differently."
        )

# Add sample questions helper
st.markdown("---")
st.markdown("### 💡 Try these sample queries:")
cols = st.columns(3)
with cols[0]:
    if st.button("What is Asthma?"):
        st.info("Paste this into the box above!")
with cols[1]:
    if st.button("How is Lupus treated?"):
        st.info("Paste this into the box above!")
with cols[2]:
    if st.button("What causes high blood pressure?"):
        st.info("Paste this into the box above!")
