"""
Page 4 - Multilingual Chat Assistant
Author: Bhuvan J J
"""

import sys
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.modules.multilingual import MultilingualAgent, SUPPORTED_LANGS
except ImportError:
    from modules.multilingual import MultilingualAgent, SUPPORTED_LANGS

st.set_page_config(page_title="Multilingual Chat", page_icon="🌐", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #8E2DE2, #4A00E0); padding: 1.8rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.5rem;">
    <h1>🌐 Multilingual & Cross-Lingual Assistant</h1>
    <p>English • Hindi (हिन्दी / Hinglish) • Kannada (ಕನ್ನಡ / Kanglish) • Spanish • French • Auto Language Detection & Pivot Translation</p>
</div>
""", unsafe_allow_html=True)

# Instantiate agent fresh to bypass stale resource caching
agent = MultilingualAgent()
agent.initialize_retrievers()

if "multilingual_chat" not in st.session_state:
    st.session_state.multilingual_chat = []

api_key_input = st.sidebar.text_input("🔑 Gemini API Key (Optional for Full Translation):", type="password")

user_input = st.text_input(
    "Enter your message in any supported language or code-switched text (EN, HI, KN, ES, FR):",
    placeholder="e.g. Madhumeha (diabetes) ke symptoms kya hain? /¿Cuáles son los síntomas de la diabetes?"
)

if st.button("Send Message", type="primary"):
    if user_input.strip():
        with st.spinner("Detecting language and processing cross-lingual context..."):
            detection = agent.detect_and_translate(user_input, api_key=api_key_input)
            
            # Metric badges
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Primary Language", f"{detection['language_name']} ({detection['primary_language'].upper()})")
            with col2:
                st.metric("Code-Switched / Mixed", "Yes (Hinglish/Kanglish)" if detection["is_mixed"] else "No")
            with col3:
                st.metric("Detected Pipeline Languages", ", ".join(detection["detected_languages"]))

            st.markdown(f"🔀 **Pivot English Representation**: *\"{detection['translated_query']}\"*")
            st.divider()

            # Query underlying retrievers using English pivot
            if agent.medical_retriever:
                results = agent.medical_retriever.retrieve(detection["translated_query"], top_k=2)
                if results:
                    st.subheader("💡 Knowledge Base Response (Cross-Lingual Match):")
                    for idx, r in enumerate(results, 1):
                        with st.expander(f"Result #{idx}: {r['question']} (Match: {r['similarity']*100:.1f}%)", expanded=(idx==1)):
                            st.markdown(f"**Focus Area**: `{r['focus']}` | **Category**: `{r['question_type']}`")
                            st.markdown(f"**Answer**:\n{r['answer']}")
                            
                        # Save to conversation memory
                        st.session_state.multilingual_chat.append({
                            "user_input": user_input,
                            "lang": detection['primary_language'],
                            "translated": detection['translated_query'],
                            "answer": r['answer']
                        })
                else:
                    st.info("Query processed cleanly. No exact matching entry found in retriever.")
    else:
        st.error("Please enter a message.")

# Render Conversational Memory across language switches
if st.session_state.multilingual_chat:
    st.divider()
    st.subheader("📜 Cross-Lingual Conversational History")
    for msg in reversed(st.session_state.multilingual_chat):
        st.markdown(f"🌐 **User ({msg['lang'].upper()})**: {msg['user_input']}")
        st.markdown(f"🤖 **Assistant**: {msg['answer'][:180]}...")
        st.markdown("---")
