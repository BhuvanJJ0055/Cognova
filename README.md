# Cognova

An AI assistant that actually pays attention to *how* you're asking, not just *what* you're asking.

Most chatbots pick one trick and stick with it — either they answer questions from a knowledge base, or they detect sentiment, or they handle multiple languages. Cognova tries to do all of it at once, in the same conversation, without losing track of context when things change mid-chat.

This started as a set of internship tasks (sentiment detection, medical Q&A, dynamic knowledge updates, domain expertise, multi-modal reasoning, multilingual support) that I decided to build as one connected system instead of six separate scripts.

## What it does

- **Reads the room.** Detects whether you sound frustrated, happy, or neutral, and adjusts its response tone accordingly.
- **Knows its stuff.** Answers domain-specific questions using retrieval — right now that's medical Q&A (via MedQuAD) and research paper explanations (via arXiv), but the retrieval layer is built to plug into any dataset.
- **Keeps learning.** New information gets pulled in and embedded on a schedule, so the knowledge base doesn't go stale the day after you deploy it.
- **Understands images too.** You can show it something, not just tell it something, and it'll reason across both.
- **Switches languages without losing the plot.** Start a conversation in English, drop into Hindi halfway through, and it keeps the context intact instead of resetting.

## How it's put together

```
Input (text / image / language detection)
        ↓
Understanding (sentiment + entity/intent recognition)
        ↓
Knowledge layer (vector DB + scheduled re-ingestion)
        ↓
Generation (open-source LLM, language-aware output)
```

Nothing exotic here — it's mostly about getting these pieces to share one consistent state instead of running in isolation.

## Stack

- **UI:** Streamlit
- **Vector store:** FAISS / ChromaDB
- **LLM:** open-source (Llama / Mistral, swappable)
- **Sentiment:** transformer-based classifier
- **Language detection:** langdetect / fastText
- **Orchestration:** LangChain

None of this is locked in — swap any piece out as needed.

## Project Layout

```
Cognova/
├── Task1_Sentiment_Chatbot/    # Task 1: Sentiment-Aware Customer Support Agent
│   ├── data/                   # Sentiment training datasets
│   ├── chatbot_v2.py           # Core logic for VADER & Logistic Regression classifier
│   ├── test_accuracy.py        # Model validation and metrics
│   ├── internship_report.md    # Task 1 Internship Report
│   └── README.md               # Task 1 setup & methodology
├── Task2_Medical_QA_Chatbot/   # Task 2: MedQuAD-based Medical Advisor
│   ├── data/                   # MedQuAD datasets & serialized retriever indices
│   ├── data_loader.py          # Automatic NIH XML corpus downloader & parser
│   ├── build_index.py          # TF-IDF & Cosine Similarity search engine
│   ├── entity_recognition.py   # Regex-based medical entity tagging (symptoms, etc.)
│   ├── data_processor.py       # Data formatting and index build script
│   ├── test_entities.py        # NER tagger unit assertions
│   ├── test_pipeline.py        # Semantic retrieval unit assertions
│   ├── internship_report.md    # Task 2 Internship Report
│   └── README.md               # Task 2 setup & methodology
├── Task3_ArXiv_CS_Chatbot/     # Task 3: ArXiv Scientific Paper Expert Advisor
│   ├── data/                   # ArXiv CS database & serialized retriever index
│   ├── arxiv_loader.py         # Local loading & live API search/fetch
│   ├── build_arxiv_index.py    # TF-IDF Cosine Similarity retriever
│   ├── nlp_utils.py            # Extractive summarizer and Regex NER tagger
│   ├── llm_explainer.py        # LLM connector & offline fallback synthesis
│   ├── app.py                  # Standalone ArXiv advisor dashboard
│   ├── test_arxiv_pipeline.py  # Scientific retrieval & extraction tests
│   ├── internship_report.md    # Task 3 Internship Report
│   └── README.md               # Task 3 setup & methodology
├── app.py                      # Root Streamlit Application (Unified Hub)
├── requirements.txt            # Root dependencies
├── pyrefly.toml                # Project settings
└── README.md                   # This root documentation
```

## Running it Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BhuvanJJ0055/Cognova.git
   cd Cognova
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Datasets & Build Indexes:**
   To prepare the indices and run the test suites:
   * **Medical Q&A (Task 2)**:
     ```bash
     python Task2_Medical_QA_Chatbot/data_processor.py
     ```
   * **ArXiv CS Chatbot (Task 3)**:
     ```bash
     python Task3_ArXiv_CS_Chatbot/test_arxiv_pipeline.py
     ```

4. **Launch the Unified Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Internship Reports & Deliverables

* 📄 **Task 1 Report:** [Task1 Internship Report](file:///c:/Users/bhuva/Cognova/Task1_Sentiment_Chatbot/internship_report.md)
* 📄 **Task 2 Report:** [Task2 Internship Report](file:///c:/Users/bhuva/Cognova/Task2_Medical_QA_Chatbot/internship_report.md)
* 📄 **Task 3 Report:** [Task3 Internship Report](file:///c:/Users/bhuva/Cognova/Task3_ArXiv_CS_Chatbot/internship_report.md)

## License

MIT — use it, break it, learn from it.

## Author

Bhuvan J J

