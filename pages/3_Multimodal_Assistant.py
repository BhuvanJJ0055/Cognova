"""
Page 3 - Multimodal Vision & Document Assistant
Author: Bhuvan J J
"""

import sys
import os
import io
import base64
import requests
import streamlit as st
from PIL import Image

# --- Env Loading ---
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_file):
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass

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

# --- Sidebar ---
st.sidebar.title("⚙️ Vision Model Settings")
env_key = os.environ.get("GEMINI_API_KEY", "").strip()
api_key_input = st.sidebar.text_input(
    "🔑 Gemini API Key:",
    value=env_key,
    type="password",
    help="Pre-loaded from .env. Must be a valid Gemini API key (e.g. starting with AQ. or AIzaSy) for full AI Vision."
)
clean_key = api_key_input.strip() if api_key_input and api_key_input.strip() else None

# Show key status in sidebar
if clean_key and len(clean_key) > 10:
    st.sidebar.success(f"✅ API Key loaded: `{clean_key[:12]}...`")
else:
    st.sidebar.warning("⚠️ No API Key — local analysis only.")

assistant = MultimodalAssistant(api_key=clean_key)

if "multimodal_chat" not in st.session_state:
    st.session_state.multimodal_chat = []

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📷 Upload Visual Content")
    uploaded_file = st.file_uploader("Upload Image or Document (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Visual Input", use_container_width=True)

        if hasattr(assistant, "extract_visual_evidence"):
            evidence = assistant.extract_visual_evidence(image)
        else:
            w, h = image.size
            evidence = {
                "width": w, "height": h,
                "format": image.format or "PNG",
                "orientation": "Landscape" if w > h else "Portrait",
                "color_mode": image.mode,
                "aspect_ratio": round(w / h, 2),
                "mean_brightness": 128.0,
                "ocr_text": ""
            }

        st.info(
            f"📊 **Visual Evidence Extracted**: "
            f"**Resolution**: `{evidence['width']}x{evidence['height']} px` | "
            f"**Format**: `{evidence.get('format', 'PNG')}` | "
            f"**Orientation**: `{evidence.get('orientation', '—')}` | "
            f"**Mode**: `{evidence.get('color_mode', 'RGB')}`"
        )

with col_right:
    st.subheader("💬 Multimodal Reasoning & Chat")
    user_prompt = st.text_area(
        "Enter question or prompt about the image:",
        placeholder="e.g. Extract all text from the image | Explain the image | Describe the diagram",
        height=100
    )

    if st.button("🔍 Analyze Image & Verify Evidence", type="primary"):
        if uploaded_file is None:
            st.error("Please upload an image first.")
        elif not user_prompt.strip():
            st.error("Please enter a question or prompt.")
        else:
            image = Image.open(uploaded_file)
            with st.spinner("⚙️ Running Gemini Vision AI & multimodal reasoning..."):
                try:
                    res = assistant.analyze_image(
                        image,
                        user_prompt,
                        api_key_override=clean_key,
                        chat_history=st.session_state.multimodal_chat
                    )
                except Exception as e:
                    res = {
                        "response": f"❌ **Analysis Error**: `{e}`",
                        "is_ambiguous": False,
                        "confidence": 0.0,
                        "visual_evidence": {},
                        "api_debug": str(e)
                    }

                # Show API debug info if present
                if res.get("api_debug"):
                    with st.expander("🔧 API Debug Info (click to expand)", expanded=True):
                        st.code(res["api_debug"], language="text")

                st.session_state.multimodal_chat.append({"role": "user", "content": user_prompt})
                st.session_state.multimodal_chat.append({"role": "assistant", "content": res["response"]})

                if res.get("is_ambiguous"):
                    st.warning(res["response"])
                else:
                    raw_conf = res.get("confidence", 0.0)
                    try:
                        conf = float(raw_conf) if isinstance(raw_conf, (int, float, str)) else 0.0
                    except (ValueError, TypeError):
                        conf = 0.0
                    conf_label = "🟢 High" if conf >= 0.9 else ("🟡 Medium (Local Analysis)" if conf >= 0.7 else "🔴 Low")
                    st.success(f"✅ Multimodal Reasoning Result — Confidence: {conf_label}")
                    st.markdown(res["response"])

    if st.session_state.multimodal_chat:
        st.divider()
        st.subheader("📜 Conversational Context Memory")
        for msg in reversed(st.session_state.multimodal_chat):
            if msg["role"] == "user":
                st.markdown(f"👤 **User**: {msg['content']}")
            else:
                st.markdown(f"🤖 **Assistant**: {msg['content']}")
