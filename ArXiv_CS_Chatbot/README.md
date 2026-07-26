# ArXiv Scientific Paper Expert Advisor (Task 3 Add-on)

This module implements an expert chatbot focused on Computer Science & AI (Machine Learning, Deep Learning, Natural Language Processing) using scientific papers from the arXiv corpus. 

It provides semantic document retrieval, technical concept extraction (NER), extractive summarization, and interactive explanations powered by open-source LLMs (via Hugging Face API) or Google Gemini, with a robust local fallback engine.

---

## 1. Problem Statement
Keeping up with daily scientific literature in machine learning is a huge challenge. General chatbots often hallucinate technical details, fail to cite papers, and lack interactive context when explaining papers. This system solves this by:
1. **Targeted Semantic Search**: Querying a local vector space of seminal papers (e.g. Transformer, BERT, ResNet) with a direct link to search the live arXiv API.
2. **Technical Entity Recognition (NER)**: Identifying core technical terms in queries to display tag pills.
3. **Paper Summarization**: Creating concise extractive summaries of papers' abstracts to get core details in 1-2 sentences.
4. **Interactive Explanation & Follow-Ups**: Linking queries to Llama-3/Mistral/Gemini models or a template-based synthesis engine, maintaining rolling chat memory for conversational flow.

---

## 2. Directory Layout
```
Task3_ArXiv_CS_Chatbot/
├── data/
│   ├── arxiv_cs_papers.csv         # Database containing seminal paper metadata
│   └── (arxiv_retriever_index.joblib) # Serialized TF-IDF vectorizer and matrix
├── arxiv_loader.py                 # Core data ingestion, CSV loading, and arXiv API querying
├── build_arxiv_index.py            # TF-IDF index builder and Cosine Similarity retriever
├── nlp_utils.py                    # Extractive text summarization and keyword NER tagger
├── llm_explainer.py                # LLM API orchestrator (Hugging Face / Gemini) and local template explainer
├── app.py                          # Dedicated standalone Streamlit dashboard
├── test_arxiv_pipeline.py          # Automated pipeline assertions
├── internship_report.md            # Technical report of implementation details
└── README.md                       # This documentation
```

---

## 3. Installation & Usage

### Pre-requisites
Ensure root project dependencies are installed:
```bash
pip install -r requirements.txt
```

### Run Automated Tests
Verify semantic indexing, NER tagging, and extractive summaries:
```bash
python Task3_ArXiv_CS_Chatbot/test_arxiv_pipeline.py
```

### Launch the Standalone ArXiv Expert App
```bash
streamlit run Task3_ArXiv_CS_Chatbot/app.py
```

---

## 4. Methodology & Pipelines

### A. Document Retrieval
We use `scikit-learn`'s `TfidfVectorizer` to fit unigram/bigram features over a concatenated corpus of paper titles and abstracts. Cosine similarity is computed between user queries and the document matrix to retrieve the top papers above a threshold.

### B. Extractive Summarizer
To summarize abstracts, we compute word frequencies (excluding stop words) to score each sentence. Sentences containing highly frequent, meaningful words score higher. The top scoring sentences are returned in their original order.

### C. Technical Concept Tagging (NER)
A boundary-guarded (`\b`) regular expression scanner identifies standard CS/AI keywords in lowercase format, matching longer phrases first (e.g., *Self-Supervised Learning* is tagged before *Learning*) to prevent subword overlaps.

### D. LLM & Chat History
- **Hugging Face**: Connects to `meta-llama/Meta-Llama-3-8B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3` via the Serverless Inference API.
- **Gemini**: Connects to `gemini-1.5-flash` using `generativelanguage.googleapis.com`.
- **Local Fallback**: Synthesizes structured markdown definitions of matching keywords and extracts summary sentences of reference papers to generate answers without API calls.
- **Memory**: Stores queries and replies in Streamlit's `st.session_state.chat_history` to feed back into instructions for follow-up responses.
