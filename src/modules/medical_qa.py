"""
Task 2 - Medical Q&A / MedQuAD Assistant Module
Author: Bhuvan J J

MedQuAD XML dataset parser, Medical Entity Recognition (Diseases, Symptoms, Treatments),
subfolder-targeted memory-efficient indexer, and medical safety disclaimer integration.
"""

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Optional

try:
    from src.core.rag_pipeline import RAGPipeline
except ImportError:
    from core.rag_pipeline import RAGPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MEDQUAD_DIR = os.path.join(DATA_DIR, "medquad")
INDEX_PATH = os.path.join(DATA_DIR, "medquad_index", "medquad_retriever_index.joblib")

MEDICAL_DISCLAIMER = (
    "\n\n⚠️ **Medical Disclaimer**: The information provided above is for educational and informational "
    "purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment."
)

SYMPTOM_TERMS = {
    "symptom", "symptoms", "sign", "signs", "early signs", "early sign", "fever", "cough", "wheezing",
    "shortness of breath", "difficulty breathing", "chest pain", "pain", "fatigue", "tiredness",
    "weakness", "headache", "dizziness", "nausea", "vomiting", "rash", "itching", "swelling",
    "joint pain", "muscle pain", "stiffness", "insomnia", "weight loss", "unexplained weight loss",
    "weight gain", "runny nose", "sore throat", "jaundice", "diarrhea", "chills", "thirst",
    "increased thirst", "urination", "frequent urination", "blurred vision", "sores", "slow-healing sores", "infections"
}

DISEASE_TERMS = {
    "diabetes", "type 2 diabetes", "type 1 diabetes", "asthma", "lupus", "influenza", "flu",
    "hypertension", "high blood pressure", "cancer", "breast cancer", "prostate cancer", "lung cancer",
    "depression", "arthritis", "hepatitis", "allergy", "migraine", "pneumonia", "celiac disease",
    "gout", "tuberculosis", "malaria", "anemia", "heart disease", "stroke"
}

TREATMENT_TERMS = {
    "treatment", "treatments", "cure", "medicine", "therapy", "surgery", "vaccine", "antibiotic",
    "inhaler", "corticosteroid", "medication", "chemotherapy", "radiation therapy", "immunotherapy",
    "transplant", "insulin", "prednisone", "aspirin", "ibuprofen", "acetaminophen", "physiotherapy", "dialysis"
}


def parse_medquad_xml_file(filepath: str) -> list:
    """Parses MedQuAD XML file extracting <QAPair> nodes (<Question>, <Answer>, <Focus>)."""
    qa_pairs = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        focus = root.findtext("Focus", default="General")
        
        for qa in root.findall(".//QAPair"):
            q_elem = qa.find("Question")
            a_elem = qa.find("Answer")
            if q_elem is not None and a_elem is not None:
                question = (q_elem.text or "").strip()
                answer = (a_elem.text or "").strip()
                qtype = qa.attrib.get("qtype", "general")
                if question and answer:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "focus": focus,
                        "question_type": qtype
                    })
    except Exception:
        pass
    return qa_pairs


def parse_medquad_folder(folder_path: str, max_files: int = 2000) -> pd.DataFrame:
    """Parses XML files inside a MedQuAD subfolder (e.g. data/medquad/1_CancerGov_QA)."""
    all_pairs = []
    if not os.path.exists(folder_path):
        return pd.DataFrame()

    file_count = 0
    for root_dir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xml"):
                fp = os.path.join(root_dir, file)
                pairs = parse_medquad_xml_file(fp)
                all_pairs.extend(pairs)
                file_count += 1
                if max_files and file_count >= max_files:
                    break
        if max_files and file_count >= max_files:
            break

    return pd.DataFrame(all_pairs)


class MedicalEntityRecognizer:
    """Extracts diseases, symptoms, and treatments from user queries."""

    def __init__(self):
        self.diseases = set(DISEASE_TERMS)
        self.symptoms = set(SYMPTOM_TERMS)
        self.treatments = set(TREATMENT_TERMS)

    def extract_entities(self, text: str) -> dict:
        text_lower = text.lower()
        cleaned = re.sub(r'[^\w\s-]', ' ', text_lower)

        found_diseases = [d for d in self.diseases if d in text_lower or re.search(r'\b' + re.escape(d) + r'\b', cleaned)]
        found_symptoms = [s for s in self.symptoms if s in text_lower or re.search(r'\b' + re.escape(s) + r'\b', cleaned)]
        found_treatments = [t for t in self.treatments if t in text_lower or re.search(r'\b' + re.escape(t) + r'\b', cleaned)]

        return {
            "diseases": sorted(list(set(found_diseases))),
            "symptoms": sorted(list(set(found_symptoms))),
            "treatments": sorted(list(set(found_treatments)))
        }


class MedicalRetriever:
    """MedQuAD retriever using subfolder indexing capability and RAGPipeline core."""

    def __init__(self, index_path=INDEX_PATH, fallback_csv_path=None):
        self.index_path = index_path
        self.fallback_csv_path = fallback_csv_path
        self.rag = RAGPipeline(
            index_path=index_path,
            text_fields=["question", "focus"],
            metadata_fields=["question", "answer", "focus", "question_type"]
        )
        self.build_from_medquad_subfolder()

    def build_from_medquad_subfolder(self, subfolder_path: Optional[str] = None, max_files: int = 1500):
        """Indexes MedQuAD subfolders and base medical Q&A entries."""
        target_path = subfolder_path or os.path.join(MEDQUAD_DIR, "1_CancerGov_QA")
        df = parse_medquad_folder(target_path, max_files=max_files)
        
        base_entries = [
            {
                "question": "What are the early signs and symptoms of Cancer?",
                "answer": "Cancer symptoms depend on the type and location, but common early general signs include unexplained weight loss, persistent fatigue, fever, skin changes (darkening or redness), and persistent unexplained pain.",
                "focus": "Cancer",
                "question_type": "symptoms"
            },
            {
                "question": "What are the symptoms of Cancer?",
                "answer": "General symptoms of Cancer include unexplained weight loss, fatigue, fever, skin changes, persistent pain, and unusual bleeding or discharge.",
                "focus": "Cancer",
                "question_type": "symptoms"
            },
            {
                "question": "What treatments are used for Cancer?",
                "answer": "Common Cancer treatments include surgery, chemotherapy, radiation therapy, immunotherapy, targeted therapy, and hormone therapy depending on stage and location.",
                "focus": "Cancer",
                "question_type": "treatment"
            },
            {
                "question": "What are the symptoms of Type 2 Diabetes?",
                "answer": "Common symptoms of Type 2 Diabetes include increased thirst, frequent urination, fatigue, blurred vision, slow-healing sores, and frequent infections.",
                "focus": "Type 2 Diabetes",
                "question_type": "symptoms"
            },
            {
                "question": "What treatments are used for Type 2 Diabetes?",
                "answer": "Treatments for Type 2 Diabetes include lifestyle modifications (healthy eating and regular physical activity), blood sugar monitoring, diabetes medications (such as metformin), and insulin therapy when needed.",
                "focus": "Type 2 Diabetes",
                "question_type": "treatment"
            },
            {
                "question": "What are the symptoms and treatments for Asthma?",
                "answer": "Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing. Treatments include quick-relief rescue inhalers (albuterol) and long-term control medications (inhaled corticosteroids).",
                "focus": "Asthma",
                "question_type": "treatment"
            },
            {
                "question": "What treatments are used for High Blood Pressure Hypertension?",
                "answer": "Hypertension treatments include dietary changes (low sodium), regular exercise, weight management, and anti-hypertensive medications.",
                "focus": "Hypertension",
                "question_type": "treatment"
            }
        ]

        if df.empty:
            df = pd.DataFrame(base_entries)
        else:
            base_df = pd.DataFrame(base_entries)
            df = pd.concat([df, base_df], ignore_index=True)
            
        return self.rag.build_from_dataframe(df, text_fields=["question", "focus"], metadata_fields=["question", "answer", "focus", "question_type"])

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.00) -> list:
        """Retrieves top-k medical Q&A entries matching query."""
        results = self.rag.query(query, top_k=top_k, threshold=0.00)
        formatted = []
        for res in results:
            formatted.append({
                "question": res.get("question", ""),
                "answer": res.get("answer", ""),
                "focus": res.get("focus", "General"),
                "question_type": res.get("question_type", "general"),
                "similarity": res.get("score", 0.0)
            })
        return formatted

    def answer_question(self, query: str) -> dict:
        ans_text, top_match, is_relevant, score = self.rag.answer(query, threshold=0.00)
        if is_relevant and top_match:
            full_response = f"**Focus**: `{top_match.get('focus', 'General')}`\n\n{ans_text}{MEDICAL_DISCLAIMER}"
        else:
            full_response = f"I could not locate specific medical guidelines for that query in the index.{MEDICAL_DISCLAIMER}"

        return {
            "answer": full_response,
            "top_match": top_match,
            "is_relevant": is_relevant,
            "score": score
        }
