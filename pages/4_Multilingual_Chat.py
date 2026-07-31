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
    <p>English • Hindi • Kannada • Spanish • French • German • Code-Switching & Cross-Lingual Context Preservation</p>
</div>
""", unsafe_allow_html=True)

# Instantiate agent
agent = MultilingualAgent()
agent.initialize_retrievers()

if "multilingual_chat" not in st.session_state:
    st.session_state.multilingual_chat = []

env_key = os.environ.get("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input("🔑 Gemini API Key:", value=env_key, type="password", help="Pre-loaded from .env")

user_input = st.text_input(
    "Enter your message in any supported language or code-switched text (EN, HI, KN, ES, FR, DE):",
    placeholder="e.g. Madhumeha ke symptoms kya hain? / ¿Cuáles son los tratamientos de la diabetes? / Nange headache ide"
)

if st.button("Send Message", type="primary"):
    if user_input.strip():
        with st.spinner("Detecting language, resolving cross-lingual context, and checking ambiguity..."):
            # Pass existing chat history for multi-turn pronoun resolution across language switches!
            detection = agent.detect_and_translate(user_input, chat_history=st.session_state.multilingual_chat, api_key=api_key_input)
            
            # Metric badges
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Primary Language", f"{detection['language_name']} ({detection['primary_language'].upper()})")
            with col2:
                st.metric("Code-Switched / Mixed", "Yes (Code-Switched)" if detection["is_mixed"] else "No")
            with col3:
                st.metric("Detected Pipeline Languages", ", ".join([l.upper() for l in detection["detected_languages"]]))

            st.markdown(f"🔀 **Pivot English Representation (Context-Resolved)**: *\"{detection['translated_query']}\"*")
            
            # Ambiguity Check Alert
            if detection.get("is_ambiguous"):
                st.warning(f"⚠️ **Ambiguous Query Detected**: {detection.get('clarification_question')}")

            st.divider()

            # Query underlying retrievers using English pivot
            results = []
            if agent.medical_retriever:
                results = agent.medical_retriever.retrieve(detection["translated_query"], top_k=2)

            # Generate grounded target-language response & calculate factual overlap
            gen_output = agent.generate_response(
                user_prompt=user_input,
                translated_query=detection["translated_query"],
                lang_info=detection,
                context_docs=results,
                chat_history=st.session_state.multilingual_chat,
                api_key=api_key_input
            )

            score, aligned, missing = agent.check_factual_consistency(gen_output.get("response_english", ""), results)

            # Render Assistant Output
            st.subheader("🤖 Assistant Response (Target Language & Grounded):")
            st.markdown(f"> {gen_output.get('response')}")

            # Groundedness & Consistency Badge
            score_col1, score_col2 = st.columns([1, 3])
            with score_col1:
                st.metric("Factual Overlap Score", f"{score * 100:.1f}%")
            with score_col2:
                if aligned:
                    st.success(f"Aligned Evidence Tokens: `{', '.join(aligned[:8])}`")
                if missing:
                    st.caption(f"Unmatched Tokens: `{', '.join(missing[:5])}`")

            # Show reference documents expander
            if results:
                with st.expander("📚 Source Reference Documents (Cross-Lingual Match)", expanded=False):
                    for idx, r in enumerate(results, 1):
                        st.markdown(f"**Document #{idx}**: {r.get('question', 'Knowledge Entry')} (Similarity: {r.get('similarity', r.get('score', 0.8))*100:.1f}%)")
                        st.markdown(f"*Answer*: {r.get('answer', r.get('text', ''))}")
                        st.divider()

            # Save turn into session state memory
            st.session_state.multilingual_chat.append({
                "user_input": user_input,
                "prompt": user_input,
                "lang": detection['primary_language'],
                "translated": detection['translated_query'],
                "response": gen_output.get('response'),
                "answer": gen_output.get('response'),
                "score": score
            })

    else:
        st.error("Please enter a message.")

# Render Conversational Memory across language switches
if st.session_state.multilingual_chat:
    st.divider()
    st.subheader("📜 Cross-Lingual Conversational History")
    for turn_idx, msg in enumerate(reversed(st.session_state.multilingual_chat), 1):
        st.markdown(f"🌐 **Turn User [{msg['lang'].upper()}]**: {msg['user_input']}")
        st.markdown(f"🤖 **Assistant**: {msg['response']}")
        st.markdown(f"*Pivot Query*: `{msg['translated']}`")
        st.markdown("---")
