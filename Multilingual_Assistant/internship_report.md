# Internship Report: Multilingual Conversational Assistant (Task 5 Add-on)
**Author**: Bhuvan J J  
**Workspace**: Cognova  
**Domain**: Artificial Intelligence / Natural Language Processing / Multilingual Systems / Cross-Lingual Guardrails  
**Date**: July 15, 2026  

---

## 1. Introduction
This report documents the design, implementation, and evaluation of the **Multilingual Conversational Assistant (Task 5)**. The goal of this module is to extend the Cognova integrated chatbot to support seamless natural conversations in **English, Hindi, and Kannada** (as well as mixed code-switched inputs like Hinglish or Kanglish). 

Specifically, this task delivers:
1. **Automatic Language Identification**: Detecting the language of the user's query dynamically.
2. **Context and Continuity Retention**: Preserving the conversation context across language switches (e.g. asking in English and making a follow-up query in Kannada that uses pronouns referencing the past turn).
3. **Cross-Lingual Index Querying**: Automatically translating queries to English to perform highly precise TF-IDF retrieval on the English NIH MedQuAD and ArXiv CS databases.
4. **Target Language Synthesis**: Grounding the output response in the retrieved evidence and translating/generating it back to the detected target language.
5. **Cross-Lingual Hallucination Guardrail**: Enforcing factual validation checks by checking token overlaps of the English translation of the response against the retrieved source context.

---

## 2. Background & Architecture
Multilingual assistants in domain-specific tasks (like health and scientific literature) face a major challenge: domain databases are usually compiled in English. Querying an English index using scripts like Devanagari (Hindi) or Kannada results in poor retrieval quality.

To solve this, we implemented a **Translation-Augmented RAG (Retrieval-Augmented Generation) Architecture**:
```
User Query (English, Hindi, Kannada, or Mixed)
                    ↓
   Gemini API Language & Context Parser
(Identifies language, checks ambiguity, translates to English resolving history references)
                    ↓
          Retrieval Router
(Queries TF-IDF index for Medical or ArXiv based on user config)
                    ↓
          LLM Response Synthesizer
(Generates grounded response in target language + English translation for guardrail check)
                    ↓
       Factual Overlap Validator
(Compares English response translation against English source documents)
                    ↓
        Interactive Streamlit UI
(Displays chat history, language badge, consistency banner, references)
```

---

## 3. Learning Objectives
*   Understand cross-lingual reasoning pipelines and the importance of query translation in RAG workflows.
*   Model mixed-language inputs (code-switching/Hinglish/Kanglish) where words from multiple languages are interleaved in the same prompt.
*   Implement multi-turn context retention that carries translation references dynamically across switches.
*   Enforce consistency guardrails across languages by performing overlap verification on English translation counterparts of the generated response.
*   Build clean multi-widget interactive web interfaces in Streamlit to manage routing and show metrics.

---

## 4. Activities and Tasks
1.  **Component Design**: Developed `multilingual_agent.py` containing language parsing, translation logic, RAG querying, and consistency checks.
2.  **State Management**: Configured Streamlit session state keys (`multilingual_chat_history`, `multilingual_last_retrieved`) in `app.py`.
3.  **UI Layout integration**: Added config widgets, chat rendering, consistency banners, and diagnostics inside `app.py`.
4.  **Unit Tests**: Written unit tests in `test_multilingual.py` mocking Gemini requests to verify language parsing, mixed translations, and factual checks.
5.  **Manual Verification**: Successfully validated multi-lingual queries (Hindi, Kannada, English) and context preservation.

---

## 5. Skills & Competencies
*   **Multilingual NLP Orchestration**: Custom language parsing and alignment pipelines.
*   **Prompt Engineering**: Designing structured prompts that enforce JSON schemas for complex multi-task outputs.
*   **Cross-Lingual Information Retrieval**: Query translation mapping for semantic indexing.
*   **Validation & Guardrails**: English-counterpart overlap calculations for foreign-language outputs.
*   **Interactive Frontend Engineering**: Custom CSS bubbles, multi-column dashboard design, and collapsible reference tables.

---

## 6. Challenges and Solutions

### Challenge 1: Querying English TF-IDF Index using Hindi/Kannada Script
*   **Problem**: Querying an English-only TF-IDF index directly with "ಶ್ವಾಸಕೋಶದ ಉರಿಯೂತ" (pneumonia) yields a similarity score of zero.
*   **Solution**: The parser automatically translates the query into clear English before indexing. The English translation is used to perform the TF-IDF search, returning relevant English source texts.

### Challenge 2: Mixed-Language Queries (Code-switching)
*   **Problem**: Users frequently mix languages (e.g. *"nange chest pain ide, what is this?"*). Standard language detectors fail or classify this randomly.
*   **Solution**: Leveraged Gemini's zero-shot understanding to classify code-switched inputs, returning `is_mixed = True` and detecting the correct constituent languages.

### Challenge 3: Resolving Pronouns across Language Switches
*   **Problem**: If the user asks in English *"What is asthma?"* and follows up in Kannada *"ಇದರ ಚಿಕಿತ್ಸೆ ಏನು?"* ("What is its treatment?"), the word "ಇದರ" ("its") must resolve to "asthma" for retrieval to work.
*   **Solution**: The language parser takes the last 5 turns of conversation history into account. It resolves context references so the query translates to *"What is the treatment for asthma?"* rather than just *"What is its treatment?"*.

---

## 7. Outcomes & Conclusions
By implementing the Multilingual Assistant, we successfully delivered all internship deliverables. The module demonstrates that advanced language tasks can be handled securely and groundedly using modern generative AI and retrieval techniques. The user can switch between languages seamlessly without losing continuity, and all facts remain protected under a robust consistency guardrail.
