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
import json
import streamlit as st
import pandas as pd

# Import local modules
from build_index import MedicalRetriever, INDEX_SAVE_PATH, DEFAULT_CSV_PATH, SAMPLE_CSV_PATH
from entity_recognition import MedicalEntityRecognizer
from data_loader import download_and_extract_medquad, build_dataframe_from_xml
from knowledge_updater import KnowledgeUpdater, start_background_updater

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
has_full_data = os.path.exists(DEFAULT_CSV_PATH) or os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "medical_qa.csv"))

# Dynamic Reloading Check (monitoring index file mtime on disk)
index_file_path = INDEX_SAVE_PATH
current_mtime = os.path.getmtime(index_file_path) if os.path.exists(index_file_path) else 0.0

if "index_mtime" not in st.session_state:
    st.session_state.index_mtime = current_mtime
    st.session_state.reload_count = 0

if current_mtime > st.session_state.index_mtime:
    st.session_state.index_mtime = current_mtime
    st.session_state.reload_count += 1

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

retriever = get_retriever(reload_count=st.session_state.reload_count)
recognizer = get_recognizer(reload_count=st.session_state.reload_count)


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

# Tabs Navigation
tab_chat, tab_kb = st.tabs(["💬 Advisor Chat", "⚙️ Knowledge Management Hub"])

with tab_chat:
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
        
        active_toggle = st.toggle("Enable Background Sync Thread", value=config.get("scheduler_active", True))
        interval_input = st.number_input("Sync Checking Interval (seconds)", min_value=10, max_value=86400, value=config.get("sync_interval_seconds", 60), step=10)
        
        if active_toggle != config.get("scheduler_active") or interval_input != config.get("sync_interval_seconds"):
            config["scheduler_active"] = active_toggle
            config["sync_interval_seconds"] = interval_input
            updater.save_config(config)
            st.toast("Configuration Saved!")
            st.rerun()

        if st.button("🔄 Sync Configured Sources Now", type="secondary", use_container_width=True):
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
                    if st.button("🗑️ Remove", key=f"del_src_{idx}", use_container_width=True):
                        updater.remove_source(idx)
                        st.toast("Source removed.")
                        st.rerun()
        else:
            st.warning("No active sync sources configured.")

        # Expandable Add Source Form
        with st.expander("➕ Add New Ingestion Source"):
            new_src_type = st.selectbox("Source Type", ["Local Folder", "Remote URL"])
            new_src_loc = st.text_input("Path / URL Address", placeholder="e.g. data/pending_updates or https://myweb/data.json")
            if st.button("Save Ingestion Source"):
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
        
        m_focus = st.text_input("Focus Condition / Topic", placeholder="e.g. COVID-19 or Hypertension")
        m_qtype = st.text_input("Question Type / Category", placeholder="e.g. symptoms, prevention, treatment")
        m_q = st.text_input("Question String", placeholder="e.g. What are the symptoms of COVID-19?")
        m_a = st.text_area("Answer Text", placeholder="e.g. The typical symptoms include fever, cough, fatigue...", height=120)
        
        if st.button("Add Q&A Entry to Database", type="primary", use_container_width=True):
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
        
        search_filter = st.text_input("🔍 Search Custom Knowledge Base", placeholder="Type keywords to filter...")
        if search_filter:
            df_filtered = df_custom[
                df_custom["focus"].str.contains(search_filter, case=False, na=False) |
                df_custom["question"].str.contains(search_filter, case=False, na=False) |
                df_custom["answer"].str.contains(search_filter, case=False, na=False)
            ]
        else:
            df_filtered = df_custom
            
        st.dataframe(df_filtered, use_container_width=True)
        
        if st.button("🗑️ Restore Original Database (Delete All Custom entries)", type="primary", use_container_width=True):
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

