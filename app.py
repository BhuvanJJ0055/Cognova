"""
Cognova GenAI Platform - Main Application Entry Point
Author: Bhuvan J J

Renders Sentiment Support Chatbot interface and platform architecture overview.
"""

import sys
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.sentiment import SupportChatbot, tag_intent, score_mood_vader

st.set_page_config(
    page_title="Cognova AI Ecosystem",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .cognova-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .cognova-header h1 {
        color: #38bdf8;
        font-weight: 700;
        font-size: 2.3rem;
        margin-bottom: 0.5rem;
    }
    .cognova-header p {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    .sentiment-badge-happy {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .sentiment-badge-upset {
        background-color: #ef4444;
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
    <h1>🤖 Cognova GenAI Platform</h1>
    <p>Enterprise AI Ecosystem • Sentiment Chatbot • Medical QA • ArXiv Expert • Multimodal • Multilingual</p>
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

tab1, tab2 = st.tabs(["💬 Sentiment Support Chatbot", "📊 Platform Architecture & Modules"])

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
            <h4>💬 Sentiment Support Chatbot</h4>
            <p>VADER + ML sentiment classifier, plain question neutral override, 7 intent categories, and adaptive reply synthesis.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>🩺 Medical QA Advisor</h4>
            <p>MedQuAD TF-IDF retrieval, medical entity recognition (diseases, symptoms, treatments), and empathetic sentiment de-escalation.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>🔄 Dynamic Knowledge Base Expansion</h4>
            <p>Incoming document watcher, MD5 hash deduplication, and RAG index updating without server restarts.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>📚 ArXiv Research Expert</h4>
            <p>Semantic research paper search, live ArXiv API integration, key AI concept extraction, paper summarization, and LLM explanations.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>👁️ Multimodal Vision Assistant</h4>
            <p>Gemini Vision processing for architecture diagrams, technical charts, document OCR, and visual Q&A inspection.</p>
        </div>
        <div style='background-color:#1e293b; padding:1.2rem; border-radius:10px; margin-bottom:1rem; border:1px solid #334155; color:white;'>
            <h4>🌐 Multilingual Chat Assistant</h4>
            <p>Auto-language detection, code-switching support (Hinglish/Kanglish), prompt translation, and cross-lingual RAG querying.</p>
        </div>
        """, unsafe_allow_html=True)
