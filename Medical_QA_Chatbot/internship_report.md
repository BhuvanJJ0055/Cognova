# Internship Report: Medical Q&A Chatbot & Dynamic Knowledge Base (Tasks 2 & 3)
**Author**: Bhuvan J J  
**Workspace**: Cognova  
**Domain**: Artificial Intelligence / Machine Learning / Natural Language Processing  
**Date**: July 9, 2026  

---

## 1. Introduction
This report documents the design, implementation, and evaluation of **Task 2 (Medical Q&A Chatbot)** and **Task 3 (Dynamic Knowledge Base Expansion)** under the Cognova workspace. 

The objective of these integrated tasks is to:
1. Create a specialized medical question-answering assistant utilizing the MedQuAD dataset, implementing a semantic retrieval mechanism to find relevant medical answers, extracting medical entities (diseases, symptoms, treatments) from user queries, and providing an interactive Streamlit user interface.
2. Build a robust system for dynamically expanding the chatbot's knowledge base over time. This includes mechanisms to poll directories and remote URLs for new CSV, JSON, or XML knowledge assets, filter duplicate questions, merge records with the active database, rebuild the sparse vector index, and hot-reload the changes in the Streamlit UI in real-time.

---

## 2. Background
In healthcare, patient-facing and research interfaces require rapid, credible, and up-to-date information. Relying on general generative LLMs risks hallucinations, which are unacceptable in clinical contexts. A retrieval-augmented generation (RAG) system restricts answers to a trusted corpus. However, medical knowledge is not static—new guidelines, drug approvals, and research emerge daily.

To address this, we developed:
1. **Semantic Sparse Indexing**: A retrieval engine utilizing TF-IDF representations and cosine similarity to match query sentences against thousands of validated medical records.
2. **Medical Entity Recognition (NER)**: A rule-based keyword extraction system protecting word boundaries (`\b`) to tag diseases, symptoms, and treatments.
3. **Dynamic Ingestion Engine**: A data pipeline that reads from local folders (CSV, JSON, XML) and web endpoints to continuously incorporate new facts, preventing data duplication through rigorous question deduplication.
4. **Hot-Reload Architecture**: A notification layer checking file modification timestamps (`mtime`) on disk so that running chatbot sessions immediately query new data without requiring server restarts.

---

## 3. Learning Objectives
The primary learning objectives achieved were:
- Master XML parsing using Python's `xml.etree.ElementTree` to flatten hierarchical data structures (like MedQuAD's XML format) into tabular formats.
- Master sparse vector retrieval baselines, understanding Term Frequency-Inverse Document Frequency (TF-IDF) mathematical models, cosine similarity matrix projections, and index persistence via `joblib`.
- Design regex-based Named Entity Recognition (NER) systems, using boundary markers (`\b`) to prevent false-positive subword matching.
- Learn concurrent programming in Python by launching background scheduler threads to poll external endpoints without blocking the main application.
- Master Streamlit cache-invalidation strategies, creating argument-dependent resource functions that refresh dynamically on disk updates.

---

## 4. Activities and Tasks
The implementation was completed through the following activities:
1. **Structured Data Ingestion**: Cleaned and compiled a 40-pair sample CSV (`sample_medquad_qa.csv`) and created an automated downloader (`data_loader.py`) to fetch the full 16,407-row MedQuAD dataset from GitHub on-demand.
2. **Tagger & Retriever Pipelines**: Wrote `build_index.py` for TF-IDF matrix compilation and `entity_recognition.py` to extract longest-matching medical entities using regular expressions.
3. **Dynamic Knowledge Updater (`knowledge_updater.py`)**: Built the core ingestion framework to manage:
   - Configuration files (`sources_config.json`) listing directories and URLs to poll.
   - Dynamic file parsers extracting question-answer pairs from CSV, JSON, and MedQuAD XML files.
   - Merging and deduplication, saving unique additions to `custom_updates.csv` and updating the master database.
   - Rebuilding the retrieval matrix on disk automatically on updates.
   - Sync logs recording history in `update_history.json`.
4. **Background Scheduler Thread**: Created a polling daemon that executes sync cycles periodically according to user-configured intervals.
5. **Interactive UI Tabs (`app.py` & root `app.py`)**: Split both standalone and unified entry-point interfaces into two tabs:
   - **💬 Advisor Chat**: The search panel with sentiment comforts, entity tag pills, and matching cards.
   - **⚙️ Knowledge Management Hub**: A comprehensive dashboard displaying scheduler health, active sources, direct Q&A manual entry, custom database explorer, logs, and database wipe controls.
6. **Automated Verification Suites**: Written and executed `test_pipeline.py`, `test_entities.py`, and `test_knowledge_update.py` to validate indexing, NER boundaries, folder synchronizations, deduplication, and database wipes.

---

## 5. Skills and Competencies
The technical competencies applied and developed during these tasks include:
- **Data Pipeline Engineering**: File system routing, remote resource downloading, archive unzipping, and parsing of diverse standard formats (CSV, JSON, XML).
- **Information Retrieval & NLP**: TF-IDF weighting, vector space modeling, cosine similarity ranking, index serialization, and pattern-based NER keyword matching.
- **Concurrent Programming**: Background daemon thread management, loop scheduling, safe file modification polling, and thread-safe data appending.
- **Frontend/UI Development**: Streamlit state preservation, dynamic widget updates, custom CSS styling with premium HSL medical colors, and cache resource management.
- **Software Quality Assurance**: Automated unit testing, testing in isolated sandboxes, assertion writing, and database state restoration.

---

## 6. Feedback and Evidence
The system was verified and demonstrated excellent performance across all domains:
- **Retrieval Accuracy**: Querying "What is Asthma?" returns the direct match with a `1.00` similarity score (100% Match Score). 
- **Entity Tagging Precision**: The system successfully ignores subwords (e.g. "pain" inside "painting" is skipped, while "joint pain" is extracted).
- **Dynamic Extension Verification**: Automated unit tests in `test_knowledge_update.py` executed successfully, asserting that:
  1. Direct manual injection appends records to the index and makes them searchable instantly.
  2. Duplicate questions are updated rather than double-inserted, confirming deduplication.
  3. Scan folder synchronizations parse CSV and XML, archive raw files into `processed_updates/`, merge the results, rebuild the index, and verify correct retrieval.
  4. Wiping custom knowledge restores the active CSV to its original, pristine base state.

---

## 7. Challenges and Solutions

### Challenge 1: Keeping the Streamlit UI in sync with background thread updates
- **Problem**: Streamlit runs sessions in separate threads. If a background sync thread updates the index file on disk, existing user tabs remain unaware and keep querying old cached indexes in memory.
- **Solution**: Implemented an **Index Modification-Time Monitor**. On every UI rerun, the session checks the modification timestamp (`mtime`) of `retriever_index.joblib` on disk. If the file is newer, the session increments a `reload_count` state variable. By passing this variable as a parameter to the cached `@st.cache_resource` load functions, Streamlit automatically invalidates the stale cache and hot-reloads the new index.

### Challenge 2: Accidental data loss during database wiping/restoring
- **Problem**: When a user clicks "Wipe Custom Knowledge" to delete dynamic additions, we must restore the active CSV to its original state. However, the database might be running on the Demo Sample or the full 16,407-row dataset, risking data loss or truncation.
- **Solution**: Implemented an automated backup system. On first startup, the app makes a backup of the original dataset (`pristine_sample_medquad_qa.csv`). Wiping restores this pristine backup for the sample mode, or triggers the XML parser recursively to rebuild the full dataset cleanly.

### Challenge 3: Path resolution across standalone and unified app modules
- **Problem**: Relative paths in imports failed when launching the app from the root directory rather than task folders.
- **Solution**: Standardized imports and resolved all directories dynamically using `os.path.dirname(os.path.abspath(__file__))` within `knowledge_updater.py`, making the path routing location-agnostic.

---

## 8. Outcomes and Impact
The outcomes of these integrated tasks are:
1. **Self-Expanding Healthcare Base**: The chatbot is no longer restricted to static launch data. It automatically incorporates new guidelines, clinical updates, or local hospital FAQs dynamically.
2. **Empathetic and Multi-Domain Interface**: The unified app binds VADER sentiment comforts with retrieval systems, showing how AI can comfort worried patients while delivering reliable, structured answers.
3. **No-Downtime Knowledge Hot-Swapping**: The Hot-Reload system allows medical database admins to drop new XML/CSV update files or input manual entries in the dashboard, and patients immediately receive the updated answers without any service downtime.

---

## 9. Conclusion
By building a self-contained TF-IDF retrieval index, designing a regex entity extractor, and implementing a thread-based background knowledge ingestion sync engine, we have created a scalable, production-ready medical Q&A advisor. The combination of structured data pipelines, thread-safe sync loops, and reload-aware caching shows how RAG systems can adapt dynamically over time while maintaining 100% hallucination-free medical compliance.
