# Cognova 🤖 — Multi-Agent Generative AI Platform

> **Enterprise Multi-Page AI Ecosystem** integrating Sentiment-Aware Support, Domain-Specific Healthcare RAG (NIH MedQuAD), ArXiv Academic Research Summarization, Multimodal Computer Vision with Local Fallback, and Multilingual Code-Switching Chat.

---

## 🌟 Executive Overview

**Cognova** is an end-to-end, multi-agent Generative AI ecosystem designed to solve specialized domain challenges in natural language processing and computer vision. Built with Python and Streamlit, Cognova unifies five domain-tailored AI assistants under a single multi-page web hub.

Rather than relying purely on external cloud services, Cognova is built with **resilience and fault-tolerance** at its core:
*   **Hybrid RAG Architecture**: Uses TF-IDF cosine similarity search for fast, deterministic retrieval over NIH MedQuAD healthcare data and ArXiv computer science literature.
*   **Resilient Multimodal Vision**: Combines Google Gemini 2.0 / 1.5 Cloud Vision with a local open-source fallback engine (PIL edge-contrast scanning, Tesseract OCR, and EasyOCR) to guarantee uninterrupted service even when cloud quotas are exhausted.
*   **Cross-Lingual Context Retention**: Supports 6+ languages and mixed code-switched dialects (Hinglish, Kanglish, etc.) with multi-turn continuity retention and automated factual overlap guardrails.

---

## 🧩 Core Modules & Features

```
                                  +-----------------------+
                                  |   Cognova Hub UI      |
                                  |       (app.py)        |
                                  +-----------+-----------+
                                              |
        +------------------+------------------+------------------+------------------+
        |                  |                  |                  |                  |
        v                  v                  v                  v                  v
+---------------+  +---------------+  +---------------+  +---------------+  +---------------+
| 💬 Sentiment  |  | 🩺 Medical QA |  | 🔬 ArXiv CS   |  | 👁️ Multimodal |  | 🌐 Multilingual|
| Support Bot   |  | Advisor (RAG) |  | Assistant     |  | Vision & OCR  |  | Chat & Guard  |
+---------------+  +---------------+  +---------------+  +---------------+  +---------------+
```

### 1. 💬 Sentiment-Aware Support Chatbot (`app.py`)
- **Real-Time Mood Analysis**: Classifies customer sentiment into **Happy**, **Upset**, or **Calm** using a dual VADER + Scikit-Learn classification model.
- **Dynamic Response Adaptation**: Automatically adjusts response tone and vocabulary depending on customer emotional state (e.g. empathetic de-escalation for distressed users).
- **Intent Tagging**: Automatically detects intent categories (Billing, Troubleshooting, Product Inquiry, Greeting).

### 2. 🩺 MedQuAD Medical Q&A Advisor (`pages/1_Medical_QA.py`)
- **NIH MedQuAD RAG Engine**: Queries 47,000+ verified medical Q&A pairs from the National Institutes of Health.
- **Medical Entity Recognition (NER)**: Extracts Diseases, Symptoms, and Treatments from queries.
- **Dynamic Knowledge Base Updater**: Allows users to ingest new medical document files (`.txt`, `.json`, `.md`) dynamically into the RAG index without restarting the application.

### 3. 🔬 ArXiv CS Research Assistant (`pages/2_ArXiv_Expert.py`)
- **Live Paper Indexing & Search**: Fetches computer science research papers from the live ArXiv API and local index.
- **Key Concept Summarizer**: Synthesizes complex academic papers into key takeaways, technical methodology, and core contributions.
- **Interactive Query Engine**: Answers technical questions grounded in indexed literature.

### 4. 👁️ Multimodal Vision & Document Assistant (`pages/3_Multimodal_Assistant.py`)
- **Cloud & Local Hybrid Inspection**: Powered by Gemini 2.0 / 1.5 Flash Vision.
- **Fault-Tolerant Local Fallback**: When cloud API quotas (HTTP 429) occur, Cognova automatically fails over to a local open-source vision engine that calculates edge density, spatial region layout, and OCR text extraction.
- **Evidence Verification Pass**: Validates visual output against physical metadata (resolution, aspect ratio, color space, average luminance).

### 5. 🌐 Multilingual Conversational Assistant (`pages/4_Multilingual_Chat.py`)
- **6+ Language & Code-Switching Support**: Seamlessly processes English, Hindi, Kannada, Spanish, French, German, and mixed inputs (Hinglish/Kanglish).
- **Multi-Turn Continuity Retention**: Carries context and pronoun references across language switches (e.g. user asks in English and follows up in Hindi using pronouns).
- **Cross-Lingual Guardrail Engine**: Translates queries to English for index retrieval, then verifies the generated response against source documents using token overlap scoring.

---

## 📁 Repository Structure

```
Cognova/
├── app.py                             # Main Streamlit hub entry point & Sentiment Support UI
├── requirements.txt                   # Production Python dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Version control exclusions (.env, venv, caches)
├── .env                               # Local environment variables (GEMINI_API_KEY)
│
├── pages/                             # Streamlit Multi-Page Directory
│   ├── 1_Medical_QA.py                # Medical Advisor & Knowledge Base Updater UI
│   ├── 2_ArXiv_Expert.py              # ArXiv CS Research Assistant UI
│   ├── 3_Multimodal_Assistant.py      # Multimodal Vision & Document OCR UI
│   └── 4_Multilingual_Chat.py         # Multilingual Code-Switching & Translation UI
│
├── src/                               # Core Source Code Package
│   ├── core/
│   │   └── rag_pipeline.py            # Reusable TF-IDF & Cosine Similarity RAG engine
│   └── modules/
│       ├── sentiment.py               # VADER & ML Sentiment Classifier
│       ├── medical_qa.py              # MedQuAD retriever & Medical NER
│       ├── kb_updater.py              # Zero-downtime Knowledge Base updater
│       ├── arxiv_expert.py            # ArXiv API & paper indexing module
│       ├── multimodal.py              # Vision processing & local fallback OCR engine
│       └── multilingual.py            # Language detector, context retention & guardrail
│
├── scripts/                           # Testing & Maintenance CLI Scripts
│   ├── build_general_index.py         # CLI index builder script
│   ├── discover_gemini_models.py      # Gemini API key & model discovery diagnostic
│   ├── test_all_bots.py               # Comprehensive platform test suite
│   └── test_ocr_install.py            # OCR installation diagnostic script
│
└── reports/                           # Project Artifacts & Reports
    ├── Internship_Report.md           # Technical internship documentation
    └── screenshots/                   # Verification evidence images
```

---

## ⚙️ Local Installation & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/BhuvanJJ0055/Cognova.git
cd Cognova
```

### 2. Set Up Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv_new
.\.venv_new\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_google_gemini_api_key_here"
```
*(Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey) by choosing "Create API key in new project").*

---

## 🚀 Launching the Application

Run the Streamlit multi-page web hub:

```bash
streamlit run app.py
```

The application will open automatically at **`http://localhost:8501`**.

---

## 🧪 Testing & Verification

Cognova includes a suite of automated diagnostic scripts:

- **Run Full System Test Suite**:
  ```bash
  python scripts/test_all_bots.py
  ```
- **Test Gemini API Key & Discover Available Models**:
  ```bash
  python scripts/discover_gemini_models.py
  ```
- **Verify Local OCR Engines (Tesseract / EasyOCR)**:
  ```bash
  python scripts/test_ocr_install.py
  ```

---

## 👤 Author & Credits

- **Author**: Bhuvan J J  
- **Repository**: [`Cognova`](https://github.com/BhuvanJJ0055/Cognova)  
- **License**: MIT License  
