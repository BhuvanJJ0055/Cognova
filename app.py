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
import streamlit as st
import pandas as pd

# Add the directories to python path to avoid import errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Task1_Sentiment_Chatbot"))
sys.path.append(os.path.join(BASE_DIR, "Task2_Medical_QA_Chatbot"))

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

retriever = get_retriever(reload_count=st.session_state.reload_count)
recognizer = get_recognizer(reload_count=st.session_state.reload_count)


# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/artificial-intelligence.png", width=100)
    st.title("Cognova Controller")
    st.write("Unified workspace of AI models.")
    
    app_mode = st.radio(
        "Select Application Module:",
        ["💬 Customer Support Agent", "⚕️ Medical Q&A Advisor"]
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
            
    else:
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


else: # Medical Q&A Advisor
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
