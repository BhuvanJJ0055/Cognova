"""
Page 3 - Multimodal Vision & Document Assistant
Author: Bhuvan J J
"""

import sys
import os
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.modules.multimodal import MultimodalAssistant
except ImportError:
    from modules.multimodal import MultimodalAssistant

st.set_page_config(page_title="Multimodal Assistant", page_icon="👁️", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #11998e, #38ef7d); padding: 1.8rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.5rem;">
    <h1>👁️ Multimodal Vision & Document Assistant</h1>
    <p>Image Analysis • Technical Diagram Inspection • Ambiguity Handling • Evidence Verification Pass</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Vision API Key configuration
st.sidebar.title("⚙️ Vision Model Settings")
api_key_input = st.sidebar.text_input("🔑 Gemini / Vision API Key (Optional):", type="password", help="Enter a Gemini API Key to enable 100% full AI Vision scene description.")

# Instantiate Assistant fresh to bypass stale resource caching
assistant = MultimodalAssistant(api_key=api_key_input)

if "multimodal_chat" not in st.session_state:
    st.session_state.multimodal_chat = []

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 Upload Visual Content")
    uploaded_file = st.file_uploader("Upload Image or Document (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Visual Input", use_container_width=True)
        
        # Safe visual evidence extraction
        if hasattr(assistant, "extract_visual_evidence"):
            evidence = assistant.extract_visual_evidence(image)
        else:
            w, h = image.size
            evidence = {"width": w, "height": h, "format": image.format or "PNG", "orientation": "Landscape" if w > h else "Portrait", "color_mode": image.mode}

        st.info(
            f"📊 **Visual Evidence Extracted**: "
            f"**Resolution**: `{evidence['width']}x{evidence['height']} px` | "
            f"**Format**: `{evidence['format']}` | "
            f"**Orientation**: `{evidence['orientation']}` | "
            f"**Mode**: `{evidence['color_mode']}`"
        )

with col_right:
    st.subheader("💬 Multimodal Reasoning & Chat")
    user_prompt = st.text_area(
        "Enter question or prompt about the image:",
        placeholder="e.g. Describe the key elements in this image, extract text, or explain the architecture diagram.",
        height=100
    )
    
    if st.button("Analyze Image & Verify Evidence", type="primary"):
        if uploaded_file is None:
            st.error("Please upload an image first.")
        elif not user_prompt.strip():
            st.error("Please enter a question or prompt.")
        else:
            image = Image.open(uploaded_file)
            with st.spinner("Executing multimodal reasoning and evidence verification pass..."):
                try:
                    res = assistant.analyze_image(image, user_prompt, api_key_override=api_key_input)
                except TypeError:
                    res = assistant.analyze_image(image, user_prompt)
                
                # Append to conversation memory
                st.session_state.multimodal_chat.append({"role": "user", "content": user_prompt})
                st.session_state.multimodal_chat.append({"role": "assistant", "content": res["response"]})

                if res.get("is_ambiguous"):
                    st.warning(res["response"])
                else:
                    st.success("### ✅ Multimodal Reasoning Result:")
                    st.markdown(res["response"])

    # Render Conversation History
    if st.session_state.multimodal_chat:
        st.divider()
        st.subheader("📜 Conversational Context Memory")
        for msg in reversed(st.session_state.multimodal_chat):
            if msg["role"] == "user":
                st.markdown(f"👤 **User**: {msg['content']}")
            else:
                st.markdown(f"🤖 **Assistant**: {msg['content']}")
