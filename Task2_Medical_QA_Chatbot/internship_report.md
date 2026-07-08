# Internship Report: Medical Q&A Chatbot (Task 2)
**Author**: Bhuvan J J  
**Workspace**: Cognova  
**Domain**: Artificial Intelligence / Machine Learning / Natural Language Processing  
**Date**: July 7, 2026  

---

## 1. Introduction
This report documents the design, implementation, and evaluation of **Task 2: Medical Q&A Chatbot** under the Cognova workspace. The objective of this task is to create a specialized medical question-answering assistant utilizing the MedQuAD dataset, implementing a semantic retrieval mechanism to find relevant medical answers, extracting medical entities (diseases, symptoms, treatments) from user queries, and providing a modern, interactive Streamlit user interface.

---

## 2. Background
In healthcare, patients and students need quick, reliable, and evidence-based information. Relying on general-purpose generative models risks hallucinated facts, which can be dangerous in medical contexts. A retrieval-based medical question-answering system mitigates this by restricting answers to a trusted corpus.

For this implementation, the **MedQuAD (Medical Question Answering Dataset)** was utilized. MedQuAD contains 16,407 validated Q&A pairs originally annotated in XML files, sourced from credible NIH organizations. To build the chatbot, the following components were researched and implemented:
1. **XML Data Parsing**: Flattening hierarchical annotations into tabular format.
2. **TF-IDF + Cosine Similarity Indexing**: Creating a sparse retrieval engine capable of matching query text against thousands of indexed questions in real-time.
3. **Medical Entity Recognition (NER)**: Building a fast, lightweight dictionary-based entity tagger with regular expression word-boundary protections.

---

## 3. Learning Objectives
The primary learning objectives of this task were to:
- Learn XML parsing techniques in Python using the `xml.etree.ElementTree` module to extract structured data and attributes.
- Master Information Retrieval (IR) baselines, specifically TF-IDF vectorization and cosine similarity matching, and learn how to serialize retrieval indexes using `joblib`.
- Build a rule-based Named Entity Recognition (NER) system using regular expressions, understanding how word boundary assertions prevent false positive subword matches.
- Experience the integration of multiple distinct AI components (sentiment classifiers, semantic search engine, and NER taggers) into a unified application.
- Build interactive, data-driven web applications using Streamlit, applying CSS styling and curated palettes to create a professional user interface.

---

## 4. Activities and Tasks
The project was executed through the following activities:
1. **轻量级 Data Ingestion & Setup**: Created a representative subset (`sample_medquad_qa.csv`) containing 40 high-quality Q&A pairs spanning diverse diseases.
2. **Automated XML Loader (`data_loader.py`)**: Wrote a script that detects whether the local dataset folder exists, and if not, automatically downloads the master zip from the official MedQuAD GitHub repository, extracts it, scans for XML files, and parses the focus and question-answer details into a 16,407-row CSV.
3. **Retrieval Engine (`build_index.py`)**: Implemented a TF-IDF vectorizer fit on questions. Created a serialization pipeline that bundles the vectorizer, the TF-IDF matrix, and parallel metadata into a single `.joblib` index.
4. **Entity Recognition Tagger (`entity_recognition.py`)**: Built a recognizer that pulls disease names directly from the dataset's unique focus topics, compiles lists of common symptoms and treatments, and searches user text with word boundaries and descending length matching.
5. **Interactive UI (`app.py` & unified root `app.py`)**: Built the Streamlit interface with color-coded entity pills and confidence badges. In the unified app, when a user asks a medical question, their query is also run through the sentiment classifier from Task 1, triggering an empathetic note if the user sounds upset or in pain.
6. **Testing Suite**: Created automated tests (`test_entities.py` and `test_pipeline.py`) to run assertions on the codebase.

---

## 5. Skills and Competencies
The key technical skills developed and applied during this task include:
- **Data Engineering**: XML manipulation, file path management, remote ZIP downloading, and CSV consolidation.
- **Information Retrieval**: Term Frequency-Inverse Document Frequency, vector spaces, cosine similarity metrics, and index persistence.
- **Natural Language Processing (NLP)**: Tokenization, keyword extraction, regex word boundaries, and named entity rules.
- **Web App Development**: Streamlit state management, conditional rendering, custom HTML/CSS styling, and interactive analytics.
- **Software Quality Assurance**: Automated unit testing, assertion writing, and test parameter cleanup.

---

## 6. Feedback and Evidence
The implemented components were verified and achieved excellent performance:
- **Retrieval Accuracy**: Testing with specific queries (e.g. "What are the symptoms of asthma?") successfully returned the exact matched entry with a similarity of `1.000` (100% confidence). Partial searches like "how to treat diabetes" matched the correct diabetes treatment card with high similarity (~0.75+).
- **Entity Tagging Precision**: The boundary test asserted that subwords were ignored (e.g., "pain" inside "painting" was correctly skipped, while "headache" was tagged as a symptom). Multi-word disease names like "celiac disease" were successfully extracted as a single disease tag.
- **Cross-Feature Integration**: When testing queries like "I am in horrible pain and cannot breathe, does influenza cause fever?", the unified app successfully:
  1. Detected "upset" sentiment and displayed a warm, empathetic comforting note.
  2. Extracted "influenza" (disease), "pain", "fever", "cannot breathe" (symptoms).
  3. Retrieved the correct Influenza symptoms answer from MedQuAD.

All assertions were validated by running `test_entities.py` and `test_pipeline.py` successfully.

---

## 7. Challenges and Solutions

### Challenge 1: MedQuAD dataset size and Git storage limits
- **Problem**: The raw XML repository and final parsed data exceed 50MB, making it heavy to bundle or commit directly.
- **Solution**: Bundled a clean, 40-pair sample CSV for instant testing. Implemented an automatic downloader in `data_loader.py` and an "Import Full Dataset" button in Streamlit, which downloads, extracts, and indexes the entire dataset on-demand.

### Challenge 2: False positive subword matches in entity recognition
- **Problem**: Simple substring matching tagged generic words inside other words (e.g. matching the treatment "in" or the symptom "pain" inside "painting").
- **Solution**: Upgraded the matching pattern to use regular expressions with word boundary markers `\b` (e.g. `r'\b' + re.escape(term) + r'\b'`), ensuring only complete words and phrases are matched.

### Challenge 3: Path resolution when running the unified app
- **Problem**: Path imports failed when launching the app from the root directory due to files living in task-specific folders.
- **Solution**: Dynamically appended task folders to `sys.path` in the root `app.py`, ensuring all module imports resolve correctly regardless of from where the app is launched.

---

## 8. Outcomes and Impact
The outcomes of this task are:
1. **Reliable Medical Retrieval**: Semantically retrieves credible, physician-written Q&As from MedQuAD, avoiding LLM hallucinations.
2. **Intelligent Query Interpretation**: Color-coded entity tagging highlights key symptoms, diseases, and treatments, helping users see what terms drove the search.
3. **Unified User Experience**: The single Streamlit app binds the Customer Support Agent and Medical Advisor, highlighting a cohesive, multi-domain AI platform.
4. **Empathetic Medical Interface**: Blending sentiment analysis with medical search shows how AI can adapt to distressed patients, providing comfort alongside hard facts.

---

## 9. Conclusion
Task 2 has been successfully completed and integrated. By parsing the MedQuAD XML data, implementing a self-contained TF-IDF retrieval index, designing a precise entity recognizer, and building a sleek, unified Streamlit application, we have constructed a reliable, production-ready healthcare assistant. The cross-module integration (sentiment-aware medical search) provides a blueprint for human-centric AI interactions.
