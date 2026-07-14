"""
Cognova Unified Hub - app.py (Root Streamlit Application)
Author: Bhuvan J J

This is the unified entry point for the Cognova ecosystem.
It integrates:
1. Task 1 - Sentiment-Aware Customer Support Agent (VADER / Supervised ML)
2. Task 2 - MedQuAD-based Medical Q&A Advisor (TF-IDF Retrieval + Entity Recognition)

Cross-Feature Integration:
In Medical Q&A mode, the chatbot analyzes the user's sentiment. If the user sounds 
worried or in pain (detected as 'upset' or highly negative), the system dynamically 
displays an empathetic, comforting de-escalation message before retrieving the medical information.
"""

import sys
import os
import re
# Load local .env variables manually if exists
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Add the directories to python path to avoid import errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Task1_Sentiment_Chatbot"))
sys.path.append(os.path.join(BASE_DIR, "Task2_Medical_QA_Chatbot"))
sys.path.append(os.path.join(BASE_DIR, "Task3_ArXiv_CS_Chatbot"))
sys.path.append(os.path.join(BASE_DIR, "Task4_Multimodal_Assistant"))

# Import Task 4 Modules
from Task4_Multimodal_Assistant.multimodal_agent import MultimodalAgent

# Import Task 3 Modules
from Task3_ArXiv_CS_Chatbot.arxiv_loader import get_local_papers, search_arxiv_api, import_papers_to_local, CSV_PATH as ARXIV_CSV_PATH
from Task3_ArXiv_CS_Chatbot.build_arxiv_index import ArXivRetriever, INDEX_PATH as ARXIV_INDEX_PATH
from Task3_ArXiv_CS_Chatbot.nlp_utils import extract_concepts as extract_arxiv_concepts, summarize_text as summarize_arxiv_text
from Task3_ArXiv_CS_Chatbot.llm_explainer import generate_explanation as generate_arxiv_explanation

# Import Task 1 Modules
from Task1_Sentiment_Chatbot.chatbot_v2 import SupportChatbot, score_mood_vader, tag_intent, LOG_PATH

# Import Task 2 Modules
from Task2_Medical_QA_Chatbot.build_index import MedicalRetriever, INDEX_SAVE_PATH, DEFAULT_CSV_PATH, SAMPLE_CSV_PATH
from Task2_Medical_QA_Chatbot.entity_recognition import MedicalEntityRecognizer
from Task2_Medical_QA_Chatbot.data_loader import download_and_extract_medquad, build_dataframe_from_xml
from Task2_Medical_QA_Chatbot.knowledge_updater import KnowledgeUpdater, start_background_updater

# Page Config
st.set_page_config(
    page_title="Cognova AI Ecosystem",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Unified CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant Title Cards */
    .cognova-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .cognova-header h1 {
        font-weight: 700;
        font-size: 2.8rem;
        margin: 0;
    }
    
    .cognova-header p {
        font-size: 1.1rem;
        opacity: 0.85;
        margin: 0.5rem 0 0 0;
    }
    
    /* Sentiment Badges */
    .mood-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: bold;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    
    .mood-happy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .mood-calm {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .mood-upset {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Entity Pills */
    .pill {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        margin: 0.15rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .pill-disease { background-color: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2); }
    .pill-symptom { background-color: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.2); }
    .pill-treatment { background-color: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }

    /* Custom Chat Container */
    .message-container {
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #2d3748;
    }
    
    .user-msg {
        background-color: #f7fafc;
        border-left: 4px solid #4a5568;
    }
    
    .assistant-msg {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
    }
</style>
""", unsafe_allow_html=True)

# Setup Knowledge Updater and Scheduler
@st.cache_resource
def get_updater():
    up = KnowledgeUpdater()
    # Copy pristine sample CSV backup for restore features
    base_sample_path = os.path.join(up.data_dir, "sample_medquad_qa.csv")
    pristine_path = os.path.join(up.data_dir, "pristine_sample_medquad_qa.csv")
    if os.path.exists(base_sample_path) and not os.path.exists(pristine_path):
        import shutil
        try:
            shutil.copy(base_sample_path, pristine_path)
        except Exception as e:
            print(f"[Warning] Failed to backup pristine sample: {e}")
            
    # Start the background sync thread
    start_background_updater(up)
    return up

updater = get_updater()
csv_path = updater.get_active_csv_path()
has_full_data = os.path.exists(DEFAULT_CSV_PATH) or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Task2_Medical_QA_Chatbot", "data", "medical_qa.csv"))

# Dynamic Reloading Check (monitoring index file mtime on disk)
index_file_path = INDEX_SAVE_PATH
current_mtime = os.path.getmtime(index_file_path) if os.path.exists(index_file_path) else 0.0

if "index_mtime" not in st.session_state:
    st.session_state.index_mtime = current_mtime
    st.session_state.reload_count = 0

if current_mtime > st.session_state.index_mtime:
    st.session_state.index_mtime = current_mtime
    st.session_state.reload_count += 1

if "arxiv_chat_history" not in st.session_state:
    st.session_state.arxiv_chat_history = []

if "arxiv_last_retrieved" not in st.session_state:
    st.session_state.arxiv_last_retrieved = []

if "multimodal_chat_history" not in st.session_state:
    st.session_state.multimodal_chat_history = []

if "multimodal_last_retrieved" not in st.session_state:
    st.session_state.multimodal_last_retrieved = []

if "multimodal_visual_analysis" not in st.session_state:
    st.session_state.multimodal_visual_analysis = {}

if "multimodal_routing_notes" not in st.session_state:
    st.session_state.multimodal_routing_notes = ""

@st.cache_resource
def get_multimodal_agent():
    return MultimodalAgent()

multimodal_agent = get_multimodal_agent()

# Load database info dynamically (reflects background updates)
df_stats = pd.read_csv(csv_path)
total_qa_pairs = len(df_stats)
unique_focus = df_stats["focus"].nunique() if "focus" in df_stats.columns else df_stats["question"].nunique()

# Initialize Retriever and Entity Recognizer (Reload-Aware)
@st.cache_resource
def get_retriever(reload_count=0):
    return MedicalRetriever(fallback_csv_path=csv_path)

@st.cache_resource
def get_recognizer(reload_count=0):
    return MedicalEntityRecognizer(dataset_csv=csv_path)

# Default variables to satisfy static analysis linters
support_bot = None
conf_threshold = 0.15
max_results = 3
arxiv_conf_threshold = 0.08
arxiv_max_results = 3
arxiv_llm_option = "Local Fallback (Deterministic)"
arxiv_hf_token = ""
arxiv_gemini_key = ""
multimodal_gemini_key = ""
multimodal_consistency_threshold = 0.35

retriever = get_retriever(reload_count=st.session_state.reload_count)
recognizer = get_recognizer(reload_count=st.session_state.reload_count)


# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/artificial-intelligence.png", width=100)
    st.title("Cognova Controller")
    st.write("Unified workspace of AI models.")
    
    app_mode = st.radio(
        "Select Application Module:",
        ["💬 Customer Support Agent", "⚕️ Medical Q&A Advisor", "📚 ArXiv Scientific Expert", "📸 Multi-Modal Agent"]
    )
    
    st.markdown("---")
    
    if app_mode == "💬 Customer Support Agent":
        st.subheader("Support Agent Config")
        model_selection = st.selectbox(
            "Sentiment Classifier Model",
            ["Supervised ML (Logistic Regression)", "VADER Lexicon Scorer"]
        )
        
        # Load Support Chatbot
        model_code = "ml" if "Supervised" in model_selection else "vader"
        @st.cache_resource
        def load_support_bot(mtype):
            # Hack because SupportChatbot expects the relative dir structure of Task1
            return SupportChatbot(model_type=mtype)
        
        support_bot = load_support_bot(model_code)
        
        # Reset chat history helper
        if st.button("Clear Chat Logs"):
            if os.path.exists(LOG_PATH):
                os.remove(LOG_PATH)
            st.success("Audit logs cleared!")
            
    elif app_mode == "⚕️ Medical Q&A Advisor":
        st.subheader("Medical Q&A Config")
        st.write(f"**Index scope:** {'Full Dataset' if has_full_data else 'Sample subset'}")
        st.write(f"**Indexed Pairs:** {total_qa_pairs:,}")
        
        conf_threshold = st.slider(
            "Similarity Confidence Cutoff",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.05
        )
        
        max_results = st.slider(
            "Max Matches",
            min_value=1,
            max_value=5,
            value=3
        )
        
        if not has_full_data:
            if st.button("Import Full Dataset (~24MB)"):
                with st.spinner("Downloading from GitHub and indexing..."):
                    try:
                        download_and_extract_medquad()
                        build_dataframe_from_xml()
                        ret = MedicalRetriever(index_path=INDEX_SAVE_PATH, fallback_csv_path=DEFAULT_CSV_PATH)
                        ret.build_and_save_index(DEFAULT_CSV_PATH)
                        st.success("Successfully loaded 16,407 pairs! Reloading...")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Ingestion failed: {ex}")
    elif app_mode == "📚 ArXiv Scientific Expert":
        st.subheader("ArXiv Scientific Expert Config")
        df_arxiv_stats = get_local_papers()
        st.write(f"**Indexed Papers:** {len(df_arxiv_stats)}")
        st.write(f"**Domains Covered:** {df_arxiv_stats['primary_category'].nunique()} fields")
        
        arxiv_conf_threshold = st.slider(
            "ArXiv Similarity Cutoff",
            min_value=0.0,
            max_value=0.5,
            value=0.08,
            step=0.02,
            key="arxiv_sidebar_cutoff"
        )
        
        arxiv_max_results = st.slider(
            "ArXiv Max Reference Papers",
            min_value=1,
            max_value=5,
            value=3,
            key="arxiv_sidebar_max"
        )
        
        arxiv_llm_option = st.selectbox(
            "ArXiv Explanation Engine",
            ["Local Fallback (Deterministic)", "Hugging Face Inference API", "Google Gemini API"],
            key="arxiv_sidebar_llm"
        )
        
        arxiv_hf_token = ""
        arxiv_gemini_key = os.environ.get("GEMINI_API_KEY", "")
        
        if arxiv_llm_option == "Hugging Face Inference API":
            arxiv_hf_token = st.text_input("HF API Token", type="password", value=os.environ.get("HF_API_TOKEN", ""), help="Input Hugging Face Bearer Token", key="arxiv_sidebar_hf_tok")
        elif arxiv_llm_option == "Google Gemini API":
            arxiv_gemini_key = st.text_input("Gemini API Key", type="password", value=arxiv_gemini_key, help="Input Google Gemini API Key", key="arxiv_sidebar_gem_key")
            
        if st.button("Clear ArXiv Chat Memory", key="arxiv_clear_mem"):
            st.session_state.arxiv_chat_history = []
            st.toast("ArXiv Chat history cleared!")
            
        st.markdown("---")
        st.subheader("Ingest Kaggle Dataset")
        st.write("Extract and index papers from the Kaggle metadata snapshot.")
        
        arxiv_kaggle_limit = st.number_input(
            "Max papers to import",
            min_value=100,
            max_value=50000,
            value=15000,
            step=500,
            key="arxiv_kaggle_limit"
        )
        
        if st.button("Import Kaggle Dataset", key="arxiv_kaggle_btn"):
            progress_bar = st.progress(0.0)
            kaggle_status_placeholder = st.empty()
            
            def update_progress(val, text):
                progress_bar.progress(val)
                kaggle_status_placeholder.text(text)
                
            try:
                from Task3_ArXiv_CS_Chatbot.data_processor_arxiv import process_kaggle_dataset
                with st.spinner("Processing Kaggle snapshot..."):
                    count = process_kaggle_dataset(limit=arxiv_kaggle_limit, progress_callback=update_progress)
                st.success(f"Successfully indexed {count:,} papers!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
    else:
        st.subheader("📸 Multi-Modal Config")
        st.write("Reason across text and image inputs with agentic routing and factual verification.")
        
        multimodal_consistency_threshold = st.slider(
            "Factual Consistency Cutoff",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="If the generated response's similarity with retrieved evidence is below this threshold, a warning will be displayed."
        )
        
        multimodal_gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Provide a Google Gemini API Key to enable live multimodal analysis. If left blank, local offline heuristics will run.",
            key="multimodal_sidebar_gem_key"
        )
        
        if st.button("Clear Multimodal Chat History", key="multimodal_clear_mem"):
            st.session_state.multimodal_chat_history = []
            st.session_state.multimodal_last_retrieved = []
            st.session_state.multimodal_routing_notes = ""
            st.session_state.multimodal_visual_analysis = {}
            st.toast("Multimodal chat history cleared!")

        st.markdown("---")
        st.subheader("🧪 Automated Diagnostics")
        st.write("Run the unit test suite inside the application server.")
        if st.button("Run Multimodal Unit Tests", key="multimodal_run_tests_btn"):
            import unittest
            from Task4_Multimodal_Assistant.test_multimodal import TestMultimodalAgent
            
            # Load test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(TestMultimodalAgent)
            
            # Run test suite and capture output
            from io import StringIO
            stream = StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2)
            result = runner.run(suite)
            
            # Display results
            st.code(stream.getvalue())
            if result.wasSuccessful():
                st.success("All tests passed successfully!")
            else:
                st.error(f"Test suite failed with {len(result.failures)} failures and {len(result.errors)} errors.")


# Header Card
st.markdown(f"""
    <div class="cognova-header">
        <h1>Cognova Integrated Assistant</h1>
        <p>Current Module: {app_mode.split(' ', 1)[1]}</p>
    </div>
""", unsafe_allow_html=True)


# --- RENDER MODULES ---

if app_mode == "💬 Customer Support Agent":
    st.write(
        "This agent monitors customer expressions and sentiments dynamically, adjusting the reply tone "
        "to de-escalate frustration or welcome positive feedback. Intent categories are routed automatically."
    )
    
    # Text input
    customer_input = st.text_input(
        "Enter customer support request:",
        placeholder="e.g., My order #5432 has not arrived yet, and I am extremely angry."
    )
    
    if customer_input and support_bot is not None:
        # Score sentiment and intent
        mood, compound = support_bot.score_mood(customer_input)
        intent = tag_intent(customer_input)
        
        # Select response tone
        reply = support_bot.reply_to(customer_input)
        
        # Display analytics columns
        col1, col2, col3 = st.columns(3)
        with col1:
            mood_style = "mood-happy" if mood == "happy" else ("mood-upset" if mood == "upset" else "mood-calm")
            st.markdown(f"**Detected Sentiment:** <span class='mood-badge {mood_style}'>{mood}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Confidence Score:** `{compound:.3f}`")
        with col3:
            st.markdown(f"**Detected Intent:** `{intent.upper()}`")
            
        # Display response bubble
        st.markdown(f"""
        <div class="message-container user-msg">
            <b>Customer:</b> "{customer_input}"
        </div>
        <div class="message-container assistant-msg">
            <b>Empathetic Response ({support_bot.model_type.upper()} Mode):</b><br>
            {reply}
        </div>
        """, unsafe_allow_html=True)
        
    # Render historic logs
    st.markdown("---")
    st.subheader("📜 Support Audit Trail (Stored in chat_history.csv)")
    if os.path.exists(LOG_PATH):
        df_logs = pd.read_csv(LOG_PATH)
        st.dataframe(df_logs.tail(10), use_container_width=True)
    else:
        st.info("No conversations logged yet. Type a message above to write to the audit trail.")


elif app_mode == "⚕️ Medical Q&A Advisor":
    # Warning disclaimer
    st.warning(
        "⚠️ **RESEARCH DEMONSTRATION:** This retrieval bot uses the MedQuAD dataset. "
        "It does not provide clinical consultations. Seek a physician's advice for active health issues."
    )
    
    # Tabs Navigation
    tab_chat, tab_kb = st.tabs(["💬 Advisor Chat", "⚙️ Knowledge Management Hub"])

    with tab_chat:
        st.write("Input a query to retrieve matching Q&A answers from the MedQuAD index.")
        
        med_query = st.text_input(
            "Describe symptoms or search treatments:",
            placeholder="e.g., What are the symptoms of pneumonia? Or how is gout treated?"
        )
        
        if med_query:
            # --- Task 1 + Task 2 Integration ---
            # Analyze user sentiment using VADER from Task 1
            analyzer_bot = SupportChatbot(model_type="vader")
            mood, score = analyzer_bot.score_mood(med_query)
            
            # If user sounds anxious/upset, display a de-escalating disclaimer
            if mood == "upset":
                st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.08); border-left: 5px solid #ef4444; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
                    <h5 style="color: #ef4444; margin: 0; font-weight: bold;">❤️ Empathetic Note</h5>
                    <p style="margin: 0.5rem 0 0 0; color: #7f1d1d; font-size: 0.95rem;">
                        We detect you might be experiencing discomfort, pain, or anxiety in your question. 
                        Please take a deep breath. While we search our medical records for your query, 
                        remember that health issues can be stressful. We recommend talking to a physician if symptoms persist.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # 1. Entity Recognition
            entities = recognizer.extract_entities(med_query)
            st.markdown("#### 🔍 Identified Entities:")
            
            has_e = False
            disease_html = ""
            for d in entities["diseases"]:
                disease_html += f'<span class="pill pill-disease">🦠 Disease: {d.title()}</span>'
                has_e = True
                
            symptom_html = ""
            for s in entities["symptoms"]:
                symptom_html += f'<span class="pill pill-symptom">⚠️ Symptom: {s.title()}</span>'
                has_e = True
                
            treatment_html = ""
            for t in entities["treatments"]:
                treatment_html += f'<span class="pill pill-treatment">💊 Treatment: {t.title()}</span>'
                has_e = True
    
            if has_e:
                st.markdown(f"<div>{disease_html}{symptom_html}{treatment_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("*No standard entities detected in query. Searching retrieval index directly.*")
                
            st.markdown("---")
            
            # 2. Retrieval Search
            with st.spinner("Searching MedQuAD knowledge index..."):
                hits = retriever.retrieve(med_query, threshold=conf_threshold, top_k=max_results)
                
            if hits:
                st.markdown(f"### 📋 Retrieved Information (Top {len(hits)} matches):")
                for idx, match in enumerate(hits):
                    match_score = int(match["similarity_score"] * 100)
                    st.markdown(f"""
                    <div class="message-container assistant-msg" style="background-color: white; border: 1px solid #e2e8f0; border-left: 4px solid #10b981;">
                        <span class="similarity-badge" style="background-color: #10b981;">{match_score}% Confidence</span>
                        <h5 style="margin: 0; color: #2c5364;"><b>Focus Topic:</b> {match['focus']} | <b>Type:</b> {match['question_type'].upper()}</h5>
                        <p style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;"><b>Indexed Question:</b> <i>{match['question']}</i></p>
                        <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #edf2f7;">
                        <p style="margin: 0.5rem 0 0 0; color: #2d3748; line-height: 1.6;"><b>Answer:</b> {match['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(
                    f"No matching medical entries could be retrieved above the similarity threshold of **{conf_threshold}**. "
                    "Try broadening your terms or lowering the similarity threshold in the sidebar."
                )
                
        # Recommended Searches Helper
        st.markdown("---")
        st.markdown("### 💡 Recommended Searches:")
        cols = st.columns(3)
        with cols[0]:
            if st.button("What is Asthma?"): st.info("Type this into the box above!")
        with cols[1]:
            if st.button("How is depression treated?"): st.info("Type this into the box above!")
        with cols[2]:
            if st.button("What causes pneumonia?"): st.info("Type this into the box above!")


    with tab_kb:
        st.markdown("### 📊 Dynamic Knowledge Base Dashboard")
        st.write("Manage active sources, customize intervals, view sync history, or manually ingest medical knowledge in real-time.")

        # Load configurations
        config = updater.load_config()
        history = []
        last_sync = "Never"
        last_count = 0
        if os.path.exists(updater.log_path):
            try:
                with open(updater.log_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if history:
                        last_run = history[-1]
                        last_sync = last_run.get("timestamp", "").split(".")[0].replace("T", " ")
                        last_count = last_run.get("added_count", 0)
            except Exception:
                pass

        # Status columns
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            status_text = "🟢 Active" if config.get("scheduler_active", True) else "🔴 Paused"
            st.metric(label="Scheduler Status", value=status_text)
        with c2:
            st.metric(label="Sync Interval", value=f"{config.get('sync_interval_seconds', 60)} sec")
        with c3:
            st.metric(label="Total QA Database Size", value=f"{total_qa_pairs:,}")
        with c4:
            st.metric(label="Last Auto-Sync Records", value=f"{last_count} pairs")

        st.markdown(f"⏱️ **Last sync attempt:** `{last_sync}`")
        st.markdown("---")

        # Layout for Scheduler and Addition
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("⚙️ Scheduler Controls")
            
            active_toggle = st.toggle("Enable Background Sync Thread", value=config.get("scheduler_active", True), key="root_sched_toggle")
            interval_input = st.number_input("Sync Checking Interval (seconds)", min_value=10, max_value=86400, value=config.get("sync_interval_seconds", 60), step=10, key="root_sched_interval")
            
            if active_toggle != config.get("scheduler_active") or interval_input != config.get("sync_interval_seconds"):
                config["scheduler_active"] = active_toggle
                config["sync_interval_seconds"] = interval_input
                updater.save_config(config)
                st.toast("Configuration Saved!")
                st.rerun()

            if st.button("🔄 Sync Configured Sources Now", type="secondary", use_container_width=True, key="root_sync_now"):
                with st.spinner("Syncing all folders and URL sources..."):
                    added = updater.sync_all_sources()
                    if added > 0:
                        st.success(f"Sync complete! Added {added} new Q&A pairs.")
                        st.session_state.reload_count += 1
                        st.rerun()
                    else:
                        st.info("Sync complete. No new updates found.")

            st.markdown("#### 📂 Active Sources Configuration")
            sources_list = config.get("sources", [])
            if sources_list:
                for idx, source in enumerate(sources_list):
                    sc1, sc2, sc3 = st.columns([1, 3, 1])
                    with sc1:
                        st.write(f"`{source.get('type').upper()}`")
                    with sc2:
                        st.write(f"*{source.get('path') or source.get('url')}*")
                    with sc3:
                        if st.button("🗑️ Remove", key=f"root_del_src_{idx}", use_container_width=True):
                            updater.remove_source(idx)
                            st.toast("Source removed.")
                            st.rerun()
            else:
                st.warning("No active sync sources configured.")

            # Expandable Add Source Form
            with st.expander("➕ Add New Ingestion Source"):
                new_src_type = st.selectbox("Source Type", ["Local Folder", "Remote URL"], key="root_new_src_type")
                new_src_loc = st.text_input("Path / URL Address", placeholder="e.g. Task2_Medical_QA_Chatbot/data/pending_updates or https://myweb/data.json", key="root_new_src_loc")
                if st.button("Save Ingestion Source", key="root_save_src"):
                    if new_src_loc:
                        stype = "folder" if new_src_type == "Local Folder" else "url"
                        if updater.add_source(stype, new_src_loc):
                            st.success(f"Source saved successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to add source. Make sure it isn't a duplicate.")
                    else:
                        st.error("Please enter a valid path or URL.")

        with col_right:
            st.subheader("✍️ Manual Knowledge Ingestion")
            st.write("Directly inject a new medical Q&A pair. The chatbot will index and retrieve it immediately.")
            
            m_focus = st.text_input("Focus Condition / Topic", placeholder="e.g. COVID-19 or Hypertension", key="root_m_focus")
            m_qtype = st.text_input("Question Type / Category", placeholder="e.g. symptoms, prevention, treatment", key="root_m_qtype")
            m_q = st.text_input("Question String", placeholder="e.g. What are the symptoms of COVID-19?", key="root_m_q")
            m_a = st.text_area("Answer Text", placeholder="e.g. The typical symptoms include fever, cough, fatigue...", height=120, key="root_m_a")
            
            if st.button("Add Q&A Entry to Database", type="primary", use_container_width=True, key="root_m_submit"):
                if m_q and m_a:
                    success, msg = updater.ingest_manual(m_focus, m_qtype, m_q, m_a)
                    if success:
                        st.success(msg)
                        st.session_state.reload_count += 1
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("Required fields (Question and Answer) are missing.")

        st.markdown("---")

        # Explorer and history tabs/sections
        st.subheader("📋 Custom Added Records Explorer")
        df_custom = updater.load_custom_updates()
        
        if not df_custom.empty:
            st.write(f"Displaying **{len(df_custom)}** custom knowledge base extensions:")
            
            search_filter = st.text_input("🔍 Search Custom Knowledge Base", placeholder="Type keywords to filter...", key="root_search_custom")
            if search_filter:
                df_filtered = df_custom[
                    df_custom["focus"].str.contains(search_filter, case=False, na=False) |
                    df_custom["question"].str.contains(search_filter, case=False, na=False) |
                    df_custom["answer"].str.contains(search_filter, case=False, na=False)
                ]
            else:
                df_filtered = df_custom
                
            st.dataframe(df_filtered, use_container_width=True)
            
            if st.button("🗑️ Restore Original Database (Delete All Custom entries)", type="primary", use_container_width=True, key="root_restore_db"):
                with st.spinner("Wiping custom entries and rebuilding index..."):
                    if updater.wipe_custom_knowledge():
                        st.success("Custom knowledge database successfully wiped!")
                        st.session_state.reload_count += 1
                        st.rerun()
                    else:
                        st.error("Restoring base database failed.")
        else:
            st.info("No custom Q&A records have been ingested. Add folder sources or use the manual entry form above to ingest new information.")

        st.subheader("📜 Sync History Logs")
        if history:
            # Show logs reversed (most recent first)
            log_df = pd.DataFrame(history[::-1])
            # Capitalize headers
            log_df.columns = [col.upper() for col in log_df.columns]
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No sync history logged yet.")

elif app_mode == "📚 ArXiv Scientific Expert":
    # Initialize retriever
    arxiv_retriever = ArXivRetriever()
    df_arxiv_papers = get_local_papers()
    
    st.markdown("""
    <div style="background-color: rgba(75, 63, 114, 0.05); border-left: 5px solid #4b3f72; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <h5 style="color: #4b3f72; margin: 0; font-weight: bold;">📚 Scientific Advisor Instructions</h5>
        <p style="margin: 0.5rem 0 0 0; color: #2d3748; font-size: 0.95rem;">
            Ask questions about Machine Learning, NLP, or Deep Learning. The expert will extract key terms 
            and retrieve reference papers to generate context-specific, educational explanations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_arxiv_chat, tab_arxiv_explore, tab_arxiv_viz = st.tabs([
        "💬 Expert Advisor Chat", 
        "🔍 Search & Ingest Papers", 
        "📊 Category Similarity Map"
    ])
    
    with tab_arxiv_chat:
        # Display Chat History
        for msg in st.session_state.arxiv_chat_history:
            bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
            speaker = "Student" if msg["role"] == "user" else "ArXiv Expert"
            st.markdown(f"""
            <div class="chat-bubble {bubble_class}">
                <b>{speaker}:</b><br>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
            
        # Text input
        arxiv_user_query = st.text_input("Ask a research question or follow-up:", key="arxiv_root_chat_query")
        
        if st.button("Send Query", type="primary", key="arxiv_root_send"):
            if arxiv_user_query:
                # Add to history
                st.session_state.arxiv_chat_history.append({"role": "user", "content": arxiv_user_query})
                
                # Retrieve references
                with st.spinner("Searching indexing vector space..."):
                    hits = arxiv_retriever.retrieve(arxiv_user_query, threshold=arxiv_conf_threshold, top_k=arxiv_max_results)
                    st.session_state.arxiv_last_retrieved = hits
                    
                # Generate explanation
                with st.spinner("Generating scientific explanation..."):
                    ans = generate_arxiv_explanation(
                        query=arxiv_user_query,
                        retrieved_papers=hits,
                        chat_history=st.session_state.arxiv_chat_history[:-1],
                        hf_token=arxiv_hf_token if arxiv_llm_option == "Hugging Face Inference API" else None,
                        gemini_key=arxiv_gemini_key if arxiv_llm_option == "Google Gemini API" else None
                    )
                    
                st.session_state.arxiv_chat_history.append({"role": "assistant", "content": ans})
                st.rerun()
                
        # Sidebar display of references
        if st.session_state.arxiv_last_retrieved:
            st.markdown("---")
            st.markdown("### 🔍 Relevant Reference Context")
            
            # Extract concepts
            last_q = st.session_state.arxiv_chat_history[-2]["content"] if len(st.session_state.arxiv_chat_history) >= 2 else ""
            concepts = extract_arxiv_concepts(last_q) if last_q else []
            
            if concepts:
                st.write("**Extracted Technical Concepts:**")
                pills_html = "".join([f'<span class="concept-pill">⚙️ {c.title()}</span>' for c in concepts])
                st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
                
            st.write(f"Matched **{len(st.session_state.arxiv_last_retrieved)}** research papers in local index:")
            
            for idx, hit in enumerate(st.session_state.arxiv_last_retrieved):
                score_pct = int(hit["similarity_score"] * 100)
                
                with st.expander(f"📚 {hit['title']} ({score_pct}% Match Score)", expanded=True if idx == 0 else False):
                    st.write(f"**Authors**: {hit['authors']}")
                    st.write(f"**Published**: {hit['published']} | **Category**: {hit['primary_category']}")
                    st.write(f"**Abstract**: {hit['summary']}")
                    st.markdown(f"[Link to full paper]({hit['url']})")
                    
                    # Extractive summary
                    summary = summarize_arxiv_text(hit["summary"], num_sentences=2)
                    st.info(f"💡 **Extracted Key Points:** {summary}")
                    
    with tab_arxiv_explore:
        st.subheader("Manage Research Database")
        st.write("Browse current papers or fetch new publications dynamically from the live arXiv API.")
        
        col_l, col_r = st.columns([1, 1])
        
        with col_l:
            st.markdown("#### 📂 Local Scientific Catalog")
            search_kw = st.text_input("Filter catalog by keyword:", placeholder="e.g. BERT or residual", key="arxiv_root_search_local_kw")
            
            df_display = df_arxiv_papers.copy()
            if search_kw:
                df_display = df_display[
                    df_display["title"].str.contains(search_kw, case=False, na=False) |
                    df_display["summary"].str.contains(search_kw, case=False, na=False) |
                    df_display["authors"].str.contains(search_kw, case=False, na=False)
                ]
                
            st.write(f"Displaying **{len(df_display)}** papers:")
            st.dataframe(
                df_display[["id", "title", "authors", "primary_category", "published"]],
                use_container_width=True,
                key="arxiv_root_local_df"
            )
            
        with col_r:
            st.markdown("#### 🌐 Fetch Fresh Papers from arXiv API")
            api_query = st.text_input("arXiv API Query string:", placeholder="e.g. attention model or category:cs.LG", key="arxiv_root_query_api_inp")
            max_num = st.number_input("Max papers to fetch:", min_value=1, max_value=20, value=5, key="arxiv_root_query_api_max")
            
            # Use session state to cache fetched papers across button presses
            if "fetched_arxiv_papers" not in st.session_state:
                st.session_state.fetched_arxiv_papers = []
                
            if st.button("Search arXiv API", key="arxiv_root_search_api_btn"):
                if api_query:
                    with st.spinner("Querying arXiv server..."):
                        fetched = search_arxiv_api(api_query, max_results=max_num)
                        st.session_state.fetched_arxiv_papers = fetched
                else:
                    st.error("Please enter a query string.")
                    
            if st.session_state.fetched_arxiv_papers:
                st.success(f"Retrieved {len(st.session_state.fetched_arxiv_papers)} papers matching search.")
                
                # Show fetched papers list
                for idx, paper in enumerate(st.session_state.fetched_arxiv_papers):
                    st.markdown(f"**{idx+1}. {paper['title']}**")
                    st.write(f"Authors: {paper['authors']} | Published: {paper['published']} | Category: {paper['primary_category']}")
                    st.markdown("---")
                    
                # Add to local catalog button
                if st.button("Import Fetched Papers into Local DB", key="arxiv_root_import_btn"):
                    added = import_papers_to_local(st.session_state.fetched_arxiv_papers)
                    st.success(f"Import complete! Added {added} new unique papers. Rebuilding index...")
                    # Force index rebuild
                    arxiv_retriever.build_and_save_index(ARXIV_CSV_PATH)
                    st.session_state.fetched_arxiv_papers = []
                    st.rerun()

    with tab_arxiv_viz:
        st.subheader("📊 Document Semantic Map (PCA Projection)")
        st.write(
            "This scatter plot projects the TF-IDF representation of our papers' abstracts "
            "into 2D space using Principal Component Analysis (PCA). Points that cluster "
            "closer together share similar semantic vocabulary and themes."
        )
        
        # Run PCA
        if arxiv_retriever.vectorizer is not None and arxiv_retriever.tfidf_matrix is not None:
            try:
                with st.spinner("Computing dimensional projection..."):
                    dense_matrix = arxiv_retriever.tfidf_matrix.toarray()
                    n_papers = dense_matrix.shape[0]
                    n_components = min(2, n_papers)
                    
                    if n_components >= 2:
                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(dense_matrix)
                        
                        df_plot = pd.DataFrame(arxiv_retriever.metadata)
                        df_plot["x"] = coords[:, 0]
                        df_plot["y"] = coords[:, 1]
                        
                        # Matplotlib plot
                        plt.figure(figsize=(10, 6))
                        sns.set_theme(style="whitegrid")
                        
                        categories = df_plot["primary_category"].unique()
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        palette = sns.color_palette("Set2", len(categories))
                        color_map = dict(zip(categories, palette))
                        
                        for cat in categories:
                            df_cat = df_plot[df_plot["primary_category"] == cat]
                            ax.scatter(
                                df_cat["x"], df_cat["y"], 
                                label=cat, 
                                s=100, 
                                alpha=0.8,
                                edgecolors='w',
                                color=color_map[cat]
                            )
                            
                        # Annotate top 8 papers
                        for idx, row in df_plot.head(8).iterrows():
                            short_title = row['title'][:20] + "..." if len(row['title']) > 20 else row['title']
                            ax.annotate(
                                short_title, 
                                (row['x'], row['y']),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=8,
                                alpha=0.7
                            )
                            
                        ax.set_title("Paper Abstract Clusters (PCA projection)", fontsize=12, fontweight="bold", pad=15)
                        ax.set_xlabel("Component 1 (Variance)", fontsize=10)
                        ax.set_ylabel("Component 2 (Variance)", fontsize=10)
                        ax.legend(title="Primary Category", bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    else:
                        st.info("Add at least 3 papers to generate a 2D PCA cluster map.")
            except Exception as e:
                st.error(f"Failed to generate visualization: {e}")
        else:
            st.warning("Retriever index not loaded. Cannot run PCA visualization.")

else: # Multi-Modal Agent
    st.markdown("""
    <div style="background-color: rgba(15, 32, 39, 0.05); border-left: 5px solid #0f2027; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <h5 style="color: #0f2027; margin: 0; font-weight: bold;">📸 Multi-Modal AI Assistant Instructions</h5>
        <p style="margin: 0.5rem 0 0 0; color: #2d3748; font-size: 0.95rem;">
            Upload an image (scan, chart, receipt, etc.) and ask a question. The assistant automatically routes 
            the input to the relevant index, retrieves evidence, generates a grounded response, and checks for hallucinations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_chat = st.columns([1, 2])

    with col_upload:
        st.subheader("🖼️ Ingest Visual Input")
        uploaded_file = st.file_uploader(
            "Upload Image (PNG, JPG, JPEG)", 
            type=["png", "jpg", "jpeg"],
            key="multimodal_file_uploader"
        )
        
        if uploaded_file is not None:
            # 1. Display Image
            st.image(uploaded_file, caption="Uploaded Visual Input", use_container_width=True)
            
            # 2. Extract properties
            props = multimodal_agent.parse_image_properties(uploaded_file)
            st.markdown("#### 📊 Image Properties")
            if "error" not in props:
                st.write(f"- **Format:** `{props['format']}` | **Mode:** `{props['mode']}`")
                st.write(f"- **Dimensions:** `{props['width']}x{props['height']}` pixels")
                st.write(f"- **Megapixels:** `{props['megapixels']}` MP")
            else:
                st.error(props["error"])

            # 3. Dynamic Ambiguity Override options
            st.markdown("#### 🛠️ Ambiguity Override Options")
            st.write("If the agent misroutes the image domain, you can manually override it here:")
            override_domain = st.selectbox(
                "Manual Domain Override",
                ["No Override (Let Agent Decide)", "Medical", "Scientific/CS", "Customer Support", "General"],
                key="multimodal_domain_override"
            )
            
            # Store in session state
            st.session_state.multimodal_domain_override_val = override_domain
        else:
            st.info("Upload an image file to start the multimodal session.")

    with col_chat:
        st.subheader("💬 Conversations with Evidence Checks")
        
        # Display past chats
        for idx, chat in enumerate(st.session_state.multimodal_chat_history):
            # User bubble
            st.markdown(f"""
            <div class="message-container user-msg">
                <b>User:</b> "{chat['prompt']}" <br>
                <small style="color: #718096;">Uploaded file: <code>{chat['filename']}</code> ({chat['properties'].get('width')}x{chat['properties'].get('height')})</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Routing card
            domain_color = "#3182ce" if chat['domain'] == "scientific" else ("#10b981" if chat['domain'] == "medical" else "#ef4444")
            st.markdown(f"""
            <div style="background-color: #f7fafc; border-left: 4px solid {domain_color}; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.75rem;">
                <span class="pill" style="background-color: {domain_color}22; color: {domain_color}; font-weight: bold; text-transform: uppercase;">routed: {chat['domain']}</span>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #4a5568;"><b>Visual Findings:</b> {chat['visual_desc']}</p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: #718096; font-style: italic;">{chat['routing_notes']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # AI assistant explanation
            st.markdown(f"""
            <div class="message-container assistant-msg">
                <b>AI Assistant:</b><br>
                {chat['response']}
            </div>
            """, unsafe_allow_html=True)

            # Factual Validation banner
            score_pct = int(chat['consistency_score'] * 100)
            if chat['consistency_score'] >= 0.70:
                banner_color = "rgba(16, 185, 129, 0.1)"
                border_color = "#10b981"
                text_color = "#064e3b"
                status_lbl = "✅ High Factual Consistency"
            elif chat['consistency_score'] >= 0.35:
                banner_color = "rgba(245, 158, 11, 0.1)"
                border_color = "#f59e0b"
                text_color = "#78350f"
                status_lbl = "⚠️ Moderate Factual Consistency"
            else:
                banner_color = "rgba(239, 68, 68, 0.1)"
                border_color = "#ef4444"
                text_color = "#7f1d1d"
                status_lbl = "🚨 Low Factual Consistency (Potential Hallucination Warning)"

            st.markdown(f"""
            <div style="background-color: {banner_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; color: {text_color};">
                <h6 style="margin: 0; font-weight: bold; color: {border_color};">{status_lbl} - Score: {score_pct}%</h6>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; line-height: 1.4;">
                    <b>Aligned Terms (Evidence-backed):</b> {", ".join(chat['aligned_keywords']) if chat['aligned_keywords'] else 'None'}<br>
                    <span style="color: #ef4444;"><b>Missing/Extra Terms (Unsupported):</b> {", ".join(chat['missing_keywords']) if chat['missing_keywords'] else 'None'}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # References details
            if chat['context']:
                with st.expander("📚 Show Matched Verifiable Reference Documents", expanded=False):
                    for r_idx, ref in enumerate(chat['context']):
                        st.markdown(f"**Document [{r_idx+1}]**: {ref['title']}")
                        st.write(ref["text"])
                        st.caption(f"Source: {ref['source']}")
                        st.markdown("---")

        # Chat inputs
        st.write("---")
        m_prompt = st.text_input(
            "Enter query or comment on the image:",
            placeholder="e.g. Is there any evidence of pneumonia in this X-ray? or Explain this PCA cluster plot.",
            key="multimodal_query_input_text"
        )
        
        # Ambiguity resolving warning
        if uploaded_file is not None and m_prompt:
            props = multimodal_agent.parse_image_properties(uploaded_file)
            clarifications = multimodal_agent.check_ambiguity(m_prompt, uploaded_file.name, props)
            if clarifications:
                st.warning("⚠️ **Ambiguity Detected in Query / Image Combination:**")
                for q in clarifications:
                    st.write(f"- {q}")
                st.info("💡 *Tip: Try describing details of the image or type a longer, more specific question.*")

        if st.button("Send Multimodal Query", type="primary", key="multimodal_send_query_btn"):
            if uploaded_file is None:
                st.error("Please upload an image file first.")
            elif not m_prompt:
                st.error("Please enter a question or query about the image.")
            else:
                with st.spinner("Analyzing image and routing query..."):
                    # Extract properties
                    props = multimodal_agent.parse_image_properties(uploaded_file)
                    filename = uploaded_file.name
                    
                    # 1. Run visual analysis
                    if multimodal_gemini_key:
                        # Call Gemini Multimodal API to get detailed description
                        desc_prompt = (
                            "Extract all relevant details from this image. If it is a medical scan, describe anatomical observations. "
                            "If it is a technical chart or PCA scatter plot, detail clusters, trends, and axes. "
                            "If it is a document, screenshot, or receipt, perform OCR to extract text and details. "
                            "Write a detailed summary of key facts."
                        )
                        visual_desc = multimodal_agent.query_gemini_multimodal(uploaded_file, desc_prompt, multimodal_gemini_key)
                        
                        # Set routed domain by classifying visual desc
                        desc_lower = visual_desc.lower() + " " + m_prompt.lower()
                        if any(w in desc_lower for w in ["xray", "medical", "scan", "mri", "symptom", "pneumonia", "gout", "disease", "clinical"]):
                            routed_domain = "medical"
                        elif any(w in desc_lower for w in ["chart", "plot", "graph", "pca", "vector", "attention", "transformer", "arxiv", "paper", "concept"]):
                            routed_domain = "scientific"
                        elif any(w in desc_lower for w in ["receipt", "invoice", "ticket", "bill", "order", "support", "angry", "upset"]):
                            routed_domain = "support"
                        else:
                            routed_domain = "general"
                            
                        visual_analysis = {
                            "routed_domain": routed_domain,
                            "description": visual_desc,
                            "detected_entities": re.findall(r'\b[a-zA-Z]{5,20}\b', m_prompt)
                        }
                    else:
                        # Call local offline visual analysis heuristics
                        visual_analysis = multimodal_agent.run_local_visual_fallback(filename, props, m_prompt)
                    
                    # Apply manual override if selected
                    override_val = st.session_state.get("multimodal_domain_override_val", "No Override (Let Agent Decide)")
                    if override_val != "No Override (Let Agent Decide)":
                        if override_val == "Medical":
                            visual_analysis["routed_domain"] = "medical"
                        elif override_val == "Scientific/CS":
                            visual_analysis["routed_domain"] = "scientific"
                        elif override_val == "Customer Support":
                            visual_analysis["routed_domain"] = "support"
                        else:
                            visual_analysis["routed_domain"] = "general"

                    # 2. Agentic Routing & Retrieve evidence
                    domain, context, routing_notes = multimodal_agent.agentic_route_and_retrieve(m_prompt, visual_analysis)
                    
                    # 3. Generate final response
                    response = multimodal_agent.generate_response(
                        prompt=m_prompt,
                        visual_desc=visual_analysis["description"],
                        context=context,
                        gemini_key=multimodal_gemini_key if multimodal_gemini_key else None
                    )
                    
                    # 4. Run Factual Validation (Hallucination check)
                    score, aligned, missing = multimodal_agent.check_factual_consistency(response, context)
                    
                    # Add to session history
                    st.session_state.multimodal_chat_history.append({
                        "prompt": m_prompt,
                        "filename": filename,
                        "properties": props,
                        "domain": domain,
                        "visual_desc": visual_analysis["description"],
                        "routing_notes": routing_notes,
                        "response": response,
                        "context": context,
                        "consistency_score": score,
                        "aligned_keywords": aligned,
                        "missing_keywords": missing
                    })
                    
                    st.rerun()

