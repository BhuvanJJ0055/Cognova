# Cognova - Multi-Agent GenAI Platform

**Author**: Bhuvan J J  
**Repository**: `genai-chatbot-internship` / `Cognova`  

Cognova is an enterprise-grade, multi-page Generative AI ecosystem integrating Sentiment Analysis, MedQuAD Medical Retrieval Q&A, ArXiv CS Research Assistant, Multimodal Vision Processing, and Multilingual Translation.

---

## 📁 Project Structure

```
genai-chatbot-internship/
├── README.md                          ✅ System documentation
├── requirements.txt                   ✅ Dependencies list
├── .gitignore                         ✅ Version control rules
├── app.py                             ✅ Main Streamlit hub entry point
├── pages/
│   ├── 1_Medical_QA.py                ✅ Medical Q&A Advisor & KB updater
│   ├── 2_ArXiv_Expert.py              ✅ ArXiv Research Assistant & paper summarizer
│   ├── 3_Multimodal_Assistant.py      ✅ Vision & Document Analysis UI
│   └── 4_Multilingual_Chat.py         ✅ Multilingual translation & chat UI
├── src/
│   ├── __init__.py                    ✅ Source package init
│   ├── core/
│   │   ├── __init__.py                ✅ Core package init
│   │   └── rag_pipeline.py            ✅ Reusable TF-IDF & Cosine Similarity RAG engine
│   └── modules/
│       ├── __init__.py                ✅ Modules package init
│       ├── sentiment.py               ✅ VADER & ML Sentiment Classifier
│       ├── medical_qa.py              ✅ MedQuAD retriever & Medical NER
│       ├── kb_updater.py              ✅ Knowledge base auto-updater
│       ├── arxiv_expert.py            ✅ ArXiv API & paper indexing module
│       └── multilingual.py            ✅ Multi-language parser & code-switcher
├── scripts/
│   └── build_general_index.py         ✅ CLI index building script
├── data/
│   ├── general_docs/
│   │   └── sample_faq.txt             ✅ Sample FAQ text data
│   ├── kb_sources.json                ✅ KB source configuration
│   ├── incoming_docs/
│   │   └── README.txt                 ✅ Dynamic ingestion instructions
│   ├── medquad/                       ❌ Raw MedQuAD dataset (Git ignored)
│   ├── arxiv/                         ❌ Raw ArXiv dataset (Git ignored)
│   ├── general_kb_index/              ❌ Generated general index (Git ignored)
│   ├── medquad_index/                 ❌ Generated medical index (Git ignored)
│   └── arxiv_index/                   ❌ Generated ArXiv index (Git ignored)
├── reports/
│   ├── Internship_Report.md           ✅ Master internship report
│   └── screenshots/                   ✅ Evidence images
│       ├── task1_sentiment_negative.png
│       ├── task1_sentiment_positive.png
│       ├── task2_medical_qa.png
│       ├── task4_arxiv_search.png
│       ├── task5_multimodal.png
│       └── task6_multilingual.png
└── venv/                              ❌ Local virtual environment (Git ignored)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd genai-chatbot-internship
   ```

2. **Create and activate a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

---

## 🚀 Running the Application

Launch the Streamlit web hub:
```bash
streamlit run app.py
```

### CLI Script Execution
To build or rebuild the general knowledge base index manually:
```bash
python scripts/build_general_index.py
```

---

## 🌟 Modules & Features

1. **Medical Q&A Advisor (`pages/1_Medical_QA.py`)**:
   - MedQuAD TF-IDF retrieval.
   - Medical Entity Recognition for Diseases, Symptoms, and Treatments.
   - Empathetic de-escalation for distressed/upset users.
   - Knowledge Base updater.

2. **ArXiv Expert (`pages/2_ArXiv_Expert.py`)**:
   - Live ArXiv API & local paper indexing.
   - Key concept extraction & paper summarizer.
   - LLM paper explanation generation.

3. **Multimodal Assistant (`pages/3_Multimodal_Assistant.py`)**:
   - Image & technical diagram inspection.
   - Gemini 1.5 Flash vision processing.

4. **Multilingual Chat (`pages/4_Multilingual_Chat.py`)**:
   - Supports English, Hindi, Kannada, and code-switched text (Hinglish/Kanglish).
   - Dynamic translation & cross-lingual RAG retrieval.
