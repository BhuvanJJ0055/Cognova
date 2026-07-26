# Internship Report: ArXiv Scientific Paper Expert Chatbot (Task 3 Add-on)
**Author**: Bhuvan J J  
**Workspace**: Cognova  
**Domain**: Artificial Intelligence / Natural Language Processing / Information Retrieval  
**Date**: July 13, 2026  

---

## 1. Introduction
This report documents the design, implementation, and evaluation of the **ArXiv Scientific Paper Expert Advisor (Task 3)**. The goal of this task is to extend the Cognova project with a chatbot specialized in Computer Science & AI, capable of:
1. Retrieval of semantically similar research papers using a local corpus and live queries to the arXiv API.
2. Tagging technical concepts using Named Entity Recognition (NER).
3. Compressing abstracts using extractive text summarization.
4. Explaining complex deep learning terms using an open-source LLM (Llama-3/Mistral via Hugging Face Serverless API) or Gemini, with conversational follow-up memory.
5. Providing PCA-based 2D clustering maps of papers to show semantic similarity.

---

## 2. Background & Architecture
When researchers explore a new field, they face dense papers and unfamiliar jargon. Standard search engines lack conversational contexts, while general generative LLMs hallucinate paper citations.

We designed a hybrid retrieval-augmented generation (RAG) pipeline:
*   **Dataset Ingestor**: Reads a local CSV database of seminal machine learning papers. Features a search tool that queries the official XML arXiv API in real-time, fetching and importing new papers dynamically.
*   **Retriever Matrix**: Uses TF-IDF representation and Cosine Similarity to find relevant papers. It indexes both paper titles and abstracts.
*   **NLP Text Extraction & Summarizer**: 
    *   **Extractive Summarization**: Tokenizes the abstract into sentences and scores each based on the frequencies of normalized words (excluding stop words). Returns the highest scoring sentences in chronological order.
    *   **Regex NER Tagger**: Automatically flags key concepts (e.g. *Transformer, CNN, LSTM, Adam Optimizer, Gradient Descent*) using boundary-guarded regular expressions.
*   **Explanation Orchestrator**: Feeds retrieved papers as context to the LLM. Integrates Hugging Face's Serverless API or Google Gemini API. Incorporates a smart local template synthesis engine as a fallback for offline deployment.
*   **Cluster Visualization**: Projects dense high-dimensional TF-IDF vectors into 2D coordinates using Principal Component Analysis (PCA) to plot domain groupings.

---

## 3. Learning Objectives
*   Master XML parsing and web query string formatting for third-party REST APIs (arXiv API).
*   Understand extractive summarization algorithms using word frequency and sentence tokenization.
*   Apply dimensionality reduction techniques (PCA) to visualize clusters of high-dimensional text vectors.
*   Integrate third-party open-source generative LLMs (via Hugging Face) and manage prompt templates for chat models.
*   Preserve conversation memory for rolling follow-up inquiries.

---

## 4. Activities and Tasks
1.  **Seminal Database Compilation**: Curated `arxiv_cs_papers.csv` containing popular papers across domains (NLP, Computer Vision, Optimization).
2.  **API Downloader & Ingestor**: Implemented `arxiv_loader.py` to parse arXiv API's Atom XML feeds into structured pandas records.
3.  **Search & Tagger Engines**: Wrote `build_arxiv_index.py` for indexing and `nlp_utils.py` for extractive summaries and concept regex matching.
4.  **Interactive LLM Orchestration**: Wrote `llm_explainer.py` to manage chat requests and maintain a conversational history list.
5.  **User Interfaces**: Created `app.py` displaying chat dialogs, searchable paper catalogs, and PCA scatter charts, and integrated it into the root `app.py`.
6.  **Automated Verification**: Written `test_arxiv_pipeline.py` to test searching, extraction, and summaries.

---

## 5. Skills & Competencies
*   **Information Retrieval**: TF-IDF, Cosine Similarity, vector space modeling.
*   **Advanced NLP**: Extractive Summarization, regex-based NER entity extraction.
*   **Generative AI**: API integration, prompt engineering, chat templates, session conversation memory.
*   **Data Visualization**: PCA dimensionality reduction, Matplotlib and Seaborn plotting.
*   **Software QA**: Isolated pipeline assertions, error fallback testing.

---

## 6. Challenges and Solutions

### Challenge 1: Hugging Face API rate limits and token keys
*   **Problem**: Serverless Hugging Face Inference API is rate-limited, and users may not have API keys.
*   **Solution**: Implemented a two-layer fallback. Users can input their own HF Token or Gemini API Key in the sidebar. If none are provided, the app falls back to a smart Local Fallback engine. This engine matches keywords to a static explanation dictionary and appends a key summary of retrieved papers, ensuring the chatbot remains highly functional offline.

### Challenge 2: Follow-up question contexts
*   **Problem**: In offline fallback mode, the chatbot has no active LLM to resolve pronouns in follow-up queries (e.g. "what is that?" or "summarise it").
*   **Solution**: Programmed the fallback engine to parse simple reference queries. If the user asks for a summary or explanation and papers were recently retrieved, it automatically maps the action to the top retrieved paper in memory, explaining it cleanly.

### Challenge 3: Visualizing paper similarity clusters
*   **Problem**: TF-IDF vectors have thousands of dimensions (vocabulary size), making plotting impossible.
*   **Solution**: Implemented a PCA component. Fitting PCA on the TF-IDF matrix reduces dimensions to 2 coordinates (`x` and `y`). Streamlit plots this using Seaborn, coloring dots by `primary_category` (e.g., `cs.CL`, `cs.LG`, `cs.CV`) to clearly display semantic clustering.

---

## 7. Outcomes & Conclusions
By implementing a dual semantic retriever, extractive summarizer, and LLM explanation layer, we built a comprehensive scientific chatbot. Integrating PCA clustering and live arXiv downloads shows how NLP tools can aid research exploration. Task 3 is successfully implemented, fully verified, and unified into the Cognova hub.
