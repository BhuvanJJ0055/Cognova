"""
Cognova Unified Hub - app.py (Root Streamlit Application & Base Sentiment Chatbot)
Author: Bhuvan J J

This is the main entry point for Cognova and houses Task 1: Sentiment-Aware Support Chatbot.
"""

import sys
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import Task 1 Sentiment Module
try:
    from src.modules.sentiment import SupportChatbot, score_mood_vader, tag_intent
except ImportError:
    from modules.sentiment import SupportChatbot, score_mood_vader, tag_intent

st.set_page_config(
    page_title="Cognova AI Ecosystem",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .cognova-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sentiment-badge-happy {
        background-color: #059669;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .sentiment-badge-upset {
        background-color: #dc2626;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .sentiment-badge-calm {
        background-color: #2563eb;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cognova-header">
    <h1>🤖 Cognova GenAI Assistant Platform</h1>
    <p>Unified Enterprise Ecosystem • Task 1 Sentiment Support • RAG Retrieval • Multimodal • Multilingual</p>
</div>
""", unsafe_allow_html=True)

# Initialize Chatbot State
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Hello! I am your Sentiment-Aware Customer Support Assistant. How can I help you today?", "mood": "calm", "score": 0.0, "intent": "greeting"}
    ]

@st.cache_resource
def get_bot():
    return SupportChatbot(model_type="vader")

bot = get_bot()

tab1, tab2 = st.tabs(["💬 Task 1: Sentiment Support Chatbot", "📊 Platform Architecture & Modules"])

with tab1:
    st.subheader("💬 Interactive Customer Support Chatbot")
    st.caption("Analyzes real-time user sentiment (Happy, Upset, Calm) and automatically adjusts response tone.")

    # Render Chat History
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "user":
                mood = msg.get("mood", "calm")
                score = msg.get("score", 0.0)
                intent = msg.get("intent", "other")
                badge_class = f"sentiment-badge-{mood}"
                st.markdown(f"**Detected Mood**: <span class='{badge_class}'>{mood.upper()} ({score:+.2f})</span> | **Intent**: `{intent}`", unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input("Type your message here (e.g. 'My order is broken and I want a refund!' or 'Thanks for the great service')...")

    if user_input:
        mood, compound = bot.score_mood(user_input)
        try:
            intent = tag_intent(user_input, mood)
        except TypeError:
            intent = tag_intent(user_input)
        reply = bot.reply_to(user_input)

        st.session_state["chat_messages"].append({
            "role": "user",
            "content": user_input,
            "mood": mood,
            "score": compound,
            "intent": intent
        })
        st.session_state["chat_messages"].append({
            "role": "assistant",
            "content": reply
        })

        st.rerun()

with tab2:
    st.subheader("⚡ Ecosystem Modules Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>💬 Task 1. Sentiment-Aware Support Agent</h4>
            <p>VADER + ML sentiment classifier, plain question neutral override, intent tagging, and adaptive reply synthesis.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>🩺 Task 2. Medical Q&A Advisor (Page 1)</h4>
            <p>MedQuAD TF-IDF retrieval, medical entity recognition (diseases, symptoms, treatments), and empathetic sentiment de-escalation.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>🔄 Task 3. Dynamic Knowledge Base Expansion</h4>
            <p>Incoming document watcher, MD5 hash deduplication, and RAG index updating without server restarts.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>📚 Task 4. ArXiv CS Expert Assistant (Page 2)</h4>
            <p>Semantic research paper search, live ArXiv API integration, key AI concept extraction, paper summarization, and LLM explanations.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>👁️ Task 5. Multimodal Vision Assistant (Page 3)</h4>
            <p>Gemini Vision processing for architecture diagrams, technical charts, document OCR, and visual Q&A inspection.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>🌐 Task 6. Multilingual Chat Assistant (Page 4)</h4>
            <p>Auto-language detection, code-switching support (Hinglish/Kanglish), prompt translation, and cross-lingual RAG querying.</p>
        </div>
        """, unsafe_allow_html=True)
