# Internship Report: Multi-Modal AI Assistant (Task 4 Add-on)
**Author**: Bhuvan J J  
**Workspace**: Cognova  
**Domain**: Artificial Intelligence / Computer Vision / Multi-Modal Systems / Software Guardrails  
**Date**: July 14, 2026  

---

## 1. Introduction
This report documents the design, implementation, and evaluation of the **Multi-Modal AI Assistant (Task 4)**. The goal of this task is to extend the Cognova project with an intelligent system capable of:
1. Processing and reasoning over both text and image inputs (scans, charts, invoices).
2. Performing visual property and metadata extraction using `PIL`.
3. Integrating with the Google Gemini Multimodal REST API (`gemini-1.5-flash`) for live visual analysis.
4. Implementing an **Agentic Router** that classifies visual features and queries relevant domain indexes (NIH MedQuAD for medical scans, ArXiv CS for scientific plots, Support Chatbot for customer receipts).
5. Implementing an **Ambiguity Handler** that identifies vague queries or generic files and requests clarification.
6. Enforcing a **Factual Validator** that cross-checks LLM responses against retrieved evidence documents and flags potential hallucinations in real-time.

---

## 2. Background & Architecture
While typical AI models allow direct multimodal inputs, they act as "black boxes" that generate responses without factual verification, often leading to hallucinations. In specialized domains like medicine, research, or finance, this lack of check is unacceptable.

We designed a multi-stage **agentic reasoning architecture** rather than a simple model inference loop:
```
Visual Input (Image) + User Query
         ↓
PIL Metadata Extractor + Gemini / Heuristic Visual Analyzer
         ↓
Ambiguity Checker (Vague inputs intercepted here)
         ↓
Agentic Router (Routes to Medical, Scientific, Support, or General)
         ↓
Knowledge Retrieval (Queries domain index for factual documents)
         ↓
LLM Explanation Synthesizer (Generates grounded response)
         ↓
Factual Consistency Checker (Compares response vs retrieved documents)
         ↓
Output UI (Displays image, response, routing logs, and Consistency Score Banner)
```

### Core Components:
*   **Visual Property Extractor**: Uses PIL to parse properties like image format, mode, dimensions, aspect ratio, and megapixels.
*   **Visual Analyzer**: Sends the image payload (converted to a base64 inline block) via HTTP POST requests to Gemini. If offline, a heuristic visual processor analyzes the file name and properties to simulate OCR and feature tagging.
*   **Ambiguity Handler**: Intercepts queries where both the prompt and image name are generic (e.g. `image.png` + "explain"). Displays clarifying prompts or allows manual domain override.
*   **Agentic Router**: Inspects visual details to select the target index.
    *   *Medical*: Queries the MedQuAD TF-IDF index (Task 2).
    *   *Scientific*: Queries the ArXiv CS TF-IDF index (Task 3).
    *   *Support*: Queries the Customer Support sentiment classifier (Task 1).
*   **Factual Validator**: Computes a consistency score by calculating the intersection of non-stopwords/nouns in the response versus the retrieved documents. Banners show Green (>= 70%), Yellow (35%-70%), or Red (< 35% Hallucination Warning) status with details of missing terms.

---

## 3. Learning Objectives
*   Master REST API structure, JSON payload formatting, and base64 string compilation for vision models (Google Gemini API).
*   Implement agentic routing logic to construct cross-feature pipelines that reuse indices.
*   Design heuristic classifiers to simulate complex model outputs for fail-safe fallback modes.
*   Develop deterministic factual checkers (token overlap and keyword alignment) to serve as hallucination guardrails.
*   Build complex interactive layout components (image previews, color-coded consistency banners, and manual overrides) using Streamlit.

---

## 4. Activities and Tasks
1.  **Agentic Codebase Implementation**: Developed `multimodal_agent.py` under the `Task4_Multimodal_Assistant` directory, encapsulating vision parsing, routing, and checking.
2.  **Unit Tests**: Written `test_multimodal.py` verifying properties extraction, routing heuristics, ambiguity triggers, and consistency scores.
3.  **UI Development**: Modified `app.py` in the root workspace to integrate the `📸 Multi-Modal Agent` tab.
4.  **Verification**: Verified using mock files (chest scans, PCA charts, receipts) in local fallback mode and validated live Gemini API calls.

---

## 5. Skills & Competencies
*   **Multimodal Integration**: Image base64 formatting and REST API integration with Gemini.
*   **Computer Vision**: Image property extraction, manipulation, and PIL data management.
*   **Agentic Orchestration**: Rule-based routing, semantic index querying, and multi-turn conversational context management.
*   **AI Guardrails**: Factual alignment algorithms, stop-words removal, and keyword intersection scoring.
*   **Full-stack UI Engineering**: Designing modern Streamlit widgets, collapsible panels, custom HTML pills, and validation notification banners.

---

## 6. Challenges and Solutions

### Challenge 1: Offline development and API availability
*   **Problem**: Real-time Gemini API calls require active internet connections and keys.
*   **Solution**: Implemented a mock Heuristic Visual Fallback engine. It checks file names (e.g. `chest_xray.png`, `pca_plot.jpg`, `receipt_102.png`) and runs mock OCR/descriptions to simulate visual extraction, allowing full testing of the agentic routing and check pipeline.

### Challenge 2: Hallucination of domain facts
*   **Problem**: LLMs generate fluent text that may contain false claims not backed by the retrieved scientific or medical context.
*   **Solution**: Developed the `check_factual_consistency` tool. It checks what percentage of the key technical terms in the generated response match the retrieved documents. Any term introduced by the LLM that is not in the source text is highlighted as an "Unsupported term," and the user is warned with a warning card.

### Challenge 3: Blurry or ambiguous input images
*   **Problem**: A user might upload a low-resolution or generic image named `pic.png` with the prompt "what is this?", causing the LLM to speculate.
*   **Solution**: Created an Ambiguity Handler. It intercepts vague prompt-image pairs and displays a list of clarification prompts. Additionally, a manual override dropdown was added to the UI, allowing the user to guide the routing path.

---

## 7. Outcomes & Conclusions
By implementing a Multi-Modal AI Assistant, we successfully unified the Cognova ecosystem. The assistant shows how computer vision, semantic search, and generative models can be coordinated into a single agentic flow. The inclusion of ambiguity handling and factual validation guardrails ensures that the assistant produces verifiable, evidence-based responses.
