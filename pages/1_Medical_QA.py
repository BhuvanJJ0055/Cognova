"""
Page 1 - Medical Q&A Advisor & Sentiment De-escalation
Author: Bhuvan J J
"""

import sys
import os
import re
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.modules.medical_qa import MedicalRetriever, MedicalEntityRecognizer
    from src.modules.sentiment import score_mood_vader
    from src.modules.kb_updater import KnowledgeUpdater
except ImportError:
    from modules.medical_qa import MedicalRetriever, MedicalEntityRecognizer
    from modules.sentiment import score_mood_vader
    from modules.kb_updater import KnowledgeUpdater

st.set_page_config(page_title="Medical Q&A Advisor", page_icon="🩺", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding: 1.8rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.5rem;">
    <h1>🩺 Medical Q&A Advisor & Knowledge Base</h1>
    <p>MedQuAD Retrieval • Entity Extraction (Diseases, Symptoms, Treatments) • Empathetic De-escalation</p>
</div>
""", unsafe_allow_html=True)

# Comprehensive Lexicons for Live Page Entity Recognition
DISEASE_LEXICON = [
    "type 2 diabetes", "type 1 diabetes", "diabetes", "dengue fever", "dengue", "asthma", "lupus",
    "influenza", "flu", "hypertension", "high blood pressure", "cancer", "breast cancer",
    "prostate cancer", "depression", "arthritis", "hepatitis", "allergy", "migraine", "pneumonia", "stroke"
]

SYMPTOM_LEXICON = [
    "unexplained weight loss", "increased thirst", "frequent urination", "blurred vision",
    "slow-healing sores", "weight loss", "shortness of breath", "chest pain", "sore throat",
    "runny nose", "joint pain", "muscle pain", "thirst", "urination", "fatigue", "sores",
    "fever", "cough", "wheezing", "pain", "tiredness", "weakness", "headache", "dizziness",
    "nausea", "vomiting", "rash", "itching", "swelling", "stiffness", "insomnia",
    "early signs", "early sign", "signs", "sign", "symptoms", "symptom", "infections", "chills"
]

TREATMENT_LEXICON = [
    "lifestyle modifications", "physical activity", "blood sugar monitoring",
    "diabetes medications", "insulin therapy", "rescue inhalers", "inhaled corticosteroids",
    "metformin", "insulin", "albuterol", "corticosteroids", "medication", "medications",
    "therapy", "surgery", "vaccine", "antibiotic", "inhaler", "treatment", "treatments"
]


def _filter_subphrases(terms_list):
    generic = {"symptom", "symptoms", "treatment", "treatments", "sign", "signs"}
    filtered = [t for t in terms_list if t not in generic or len(terms_list) == 1]
    final_terms = []
    for term in filtered:
        if not any(other != term and term in other for other in filtered):
            final_terms.append(term)
    return list(dict.fromkeys(final_terms if final_terms else terms_list))


def extract_entities_live(text: str) -> dict:
    text_lower = text.lower()
    cleaned = re.sub(r'[^\w\s-]', ' ', text_lower)

    found_diseases = [d for d in DISEASE_LEXICON if d in text_lower or re.search(r'\b' + re.escape(d) + r'\b', cleaned)]
    found_symptoms = [s for s in SYMPTOM_LEXICON if s in text_lower or re.search(r'\b' + re.escape(s) + r'\b', cleaned)]
    found_treatments = [t for t in TREATMENT_LEXICON if t in text_lower or re.search(r'\b' + re.escape(t) + r'\b', cleaned)]

    return {
        "diseases": _filter_subphrases(list(dict.fromkeys(found_diseases))),
        "symptoms": _filter_subphrases(list(dict.fromkeys(found_symptoms))),
        "treatments": _filter_subphrases(list(dict.fromkeys(found_treatments)))
    }


def get_medical_retriever():
    retriever = MedicalRetriever()
    if not retriever.rag.metadata or len(retriever.rag.metadata) < 3:
        retriever.build_from_medquad_subfolder()
    return retriever

updater = KnowledgeUpdater()

tab1, tab2 = st.tabs(["💬 Ask Medical Question", "🔄 Knowledge Base Management"])

with tab1:
    user_query = st.text_input("Enter medical question or symptom details:", placeholder="e.g. What are the symptoms and treatments for Type 2 Diabetes?")
    
    if st.button("Search Medical QA", type="primary"):
        if user_query.strip():
            # 1. Sentiment check
            mood, compound = score_mood_vader(user_query)
            if mood == "upset":
                st.warning("💙 **Empathetic Support**: We hear that you may be feeling unwell or distressed. Please stay calm while we fetch relevant medical guidelines.")

            # 2. Retriever Execution & Relevance Thresholding
            retriever = get_medical_retriever()
            raw_results = retriever.retrieve(user_query, top_k=3)
            
            # Filter out zero-similarity noise matches (< 0.05 match score)
            valid_results = [r for r in raw_results if r.get('similarity', 0.0) > 0.05]

            # 3. Entity Recognition
            if valid_results:
                combined_context = user_query + " " + " ".join([r['answer'] for r in valid_results])
                entities = extract_entities_live(combined_context)
            else:
                entities = extract_entities_live(user_query)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Diseases**: {', '.join(entities['diseases']) if entities['diseases'] else 'None detected'}")
            with col2:
                st.markdown(f"**Symptoms**: {', '.join(entities['symptoms']) if entities['symptoms'] else 'None detected'}")
            with col3:
                st.markdown(f"**Treatments**: {', '.join(entities['treatments']) if entities['treatments'] else 'None detected'}")

            st.divider()

            if valid_results:
                for idx, res in enumerate(valid_results, 1):
                    with st.expander(f"Result #{idx}: {res['question']} (Match: {res['similarity']*100:.1f}%)", expanded=(idx==1)):
                        st.markdown(f"**Focus Area**: `{res['focus']}` | **Category**: `{res['question_type']}`")
                        st.markdown(f"**Answer**:\n{res['answer']}")
            else:
                st.info("ℹ️ **No matching entry found in Knowledge Base**. Switch to **Tab 2 ('Knowledge Base Management')** above to add Q&A guidelines for this condition.")
        else:
            st.error("Please enter a valid question.")

with tab2:
    st.subheader("Add Custom Q&A Pair to Knowledge Base")
    with st.form("add_qa_form"):
        q_input = st.text_input("Question", value="What is the treatment for Dengue Fever?")
        a_input = st.text_area("Answer", value="Dengue Fever treatment focuses on supportive care, hydration, fluid replacement, pain relievers like acetaminophen, and avoiding NSAIDs like aspirin or ibuprofen.")
        focus_input = st.text_input("Focus Area", value="Dengue Fever")
        qtype_input = st.text_input("Question Type", value="treatment")
        submitted = st.form_submit_button("Add to Knowledge Base")
        
        if submitted:
            if q_input.strip() and a_input.strip():
                # Update both KnowledgeUpdater and current retriever instance
                updater.add_qa_pair(q_input, a_input, focus_input, qtype_input)
                retriever = get_medical_retriever()
                retriever.rag.add_texts([f"Question: {q_input}\nAnswer: {a_input}"], metadata=[{
                    "question": q_input, "answer": a_input, "focus": focus_input, "question_type": qtype_input
                }])
                st.success("Successfully added Q&A pair! Knowledge Base updated. You can now search for this condition in Tab 1.")
            else:
                st.error("Both Question and Answer fields are required.")
