"""
Task 2 - Medical Q&A / MedQuAD Assistant Module
Author: Bhuvan J J

MedQuAD XML dataset parser, Medical Entity Recognition (Diseases, Symptoms, Treatments),
subfolder-targeted memory-efficient indexer, and medical safety disclaimer integration.
"""

import os
import re
import zipfile
import requests
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
MEDQUAD_ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"

MEDICAL_DISCLAIMER = (
    "\n\n⚠️ **Medical Disclaimer**: The information provided above is for educational and informational "
    "purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment."
)

SYMPTOM_TERMS = {
    "symptom", "symptoms", "symptons", "symtoms", "sign", "signs", "early signs", "early sign", "indication", "indications",
    "fever", "cough", "wheezing", "shortness of breath", "difficulty breathing", "chest pain", "pain", "fatigue", "tiredness",
    "weakness", "headache", "dizziness", "nausea", "vomiting", "rash", "itching", "swelling",
    "joint pain", "muscle pain", "stiffness", "insomnia", "weight loss", "unexplained weight loss",
    "weight gain", "runny nose", "sore throat", "jaundice", "diarrhea", "chills", "thirst",
    "increased thirst", "urination", "frequent urination", "blurred vision", "sores", "slow-healing sores", "infections",
    "night sweats", "swollen lymph nodes", "mouth ulcers"
}

DISEASE_TERMS = {
    "hiv", "aids", "hiv aids", "hiv/aids", "human immunodeficiency virus", "acquired immunodeficiency syndrome",
    "diabetes", "type 2 diabetes", "type 1 diabetes", "asthma", "lupus", "influenza", "flu",
    "hypertension", "high blood pressure", "cancer", "breast cancer", "prostate cancer", "lung cancer",
    "depression", "arthritis", "hepatitis", "allergy", "migraine", "pneumonia", "celiac disease",
    "gout", "tuberculosis", "malaria", "anemia", "heart disease", "stroke", "dengue", "dengue fever"
}

TREATMENT_TERMS = {
    "treatment", "treatments", "treament", "treaments", "tablets", "tablet", "tabletes", "cure", "cures",
    "medicine", "medicines", "medication", "medications", "meds", "therapy", "surgery", "vaccine", "antibiotic",
    "inhaler", "corticosteroid", "transplant", "insulin", "prednisone", "aspirin", "ibuprofen", "acetaminophen", "physiotherapy", "dialysis",
    "antiretroviral therapy", "antiretrovirals", "art"
}


def download_and_extract_medquad():
    """Downloads the official MedQuAD dataset from GitHub (abachaa/MedQuAD) if local raw files are missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "medquad_github.zip")
    extract_target = os.path.join(DATA_DIR, "MedQuAD-master")

    if os.path.exists(MEDQUAD_DIR) or os.path.exists(extract_target):
        return

    print(f"[Info] Fetching official MedQuAD dataset from {MEDQUAD_ZIP_URL}...")
    try:
        response = requests.get(MEDQUAD_ZIP_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("[Info] Successfully extracted MedQuAD dataset.")
    except Exception as e:
        print(f"[Warning] Failed to auto-download MedQuAD dataset: {e}")


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


def parse_medquad_folder(folder_path: str, max_files: int = 3000) -> pd.DataFrame:
    """Parses XML files across all 12 NIH subfolders in MedQuAD dataset."""
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
    """Extracts diseases, symptoms, and treatments from user queries using strict word-boundary matching."""

    def __init__(self):
        self.diseases = set(DISEASE_TERMS)
        self.symptoms = set(SYMPTOM_TERMS)
        self.treatments = set(TREATMENT_TERMS)

    def extract_entities(self, text: str) -> dict:
        text_lower = text.lower()
        cleaned = re.sub(r'[^\w\s-]', ' ', text_lower)

        found_diseases = [
            d for d in self.diseases
            if re.search(r'\b' + re.escape(d) + r'\b', text_lower) or re.search(r'\b' + re.escape(d) + r'\b', cleaned)
        ]
        found_symptoms = [
            s for s in self.symptoms
            if re.search(r'\b' + re.escape(s) + r'\b', text_lower) or re.search(r'\b' + re.escape(s) + r'\b', cleaned)
        ]
        found_treatments = [
            t for t in self.treatments
            if re.search(r'\b' + re.escape(t) + r'\b', text_lower) or re.search(r'\b' + re.escape(t) + r'\b', cleaned)
        ]

        return {
            "diseases": sorted(list(set(found_diseases))),
            "symptoms": sorted(list(set(found_symptoms))),
            "treatments": sorted(list(set(found_treatments)))
        }


class MedicalRetriever:
    """MedQuAD retriever using subfolder indexing capability and RAGPipeline core."""

    def __init__(self, index_path=INDEX_PATH, fallback_csv_path=None):
        self.fallback_csv_path = fallback_csv_path
        
        # Candidate full index paths
        possible_indices = [
            index_path,
            os.path.join(BASE_DIR, "Medical_QA_Chatbot", "data", "retriever_index.joblib"),
            INDEX_PATH
        ]
        
        target_index = None
        for p in possible_indices:
            if p and os.path.exists(p):
                # Check size - full index is > 1MB, dummy is 4.4KB
                if os.path.getsize(p) > 100000:
                    target_index = p
                    break

        self.index_path = target_index or index_path
        self.rag = RAGPipeline(
            index_path=self.index_path,
            text_fields=["question", "focus"],
            metadata_fields=["question", "answer", "focus", "question_type"]
        )

        # If loaded index is dummy (< 50 records), force build from full CSV/XML dataset
        if not self.rag.metadata or len(self.rag.metadata) < 50:
            self.build_from_medquad_subfolder()

    def build_from_medquad_subfolder(self, subfolder_path: Optional[str] = None, max_files: int = 3000):
        """Indexes MedQuAD subfolders across all 12 NIH categories."""
        if len(self.rag.metadata) >= 50:
            return len(self.rag.metadata)

        # Check for full pre-built CSV fallback first
        csv_candidates = [
            self.fallback_csv_path,
            os.path.join(BASE_DIR, "Medical_QA_Chatbot", "data", "medquad_qa.csv"),
            os.path.join(BASE_DIR, "Medical_QA_Chatbot", "data", "medical_qa.csv"),
            os.path.join(DATA_DIR, "medquad_qa.csv"),
            os.path.join(DATA_DIR, "medical_qa.csv"),
            os.path.join(BASE_DIR, "Medical_QA_Chatbot", "data", "sample_medquad_qa.csv")
        ]
        
        df = pd.DataFrame()
        for csv_path in csv_candidates:
            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 50:
                        print(f"[Info] Successfully loaded {len(df)} MedQuAD Q&A records from {csv_path}")
                        break
                except Exception:
                    pass

        # If CSV is missing/small, attempt XML parsing
        if df.empty or len(df) < 50:
            download_and_extract_medquad()

            candidates = [
                subfolder_path,
                MEDQUAD_DIR,
                os.path.join(DATA_DIR, "MedQuAD-master"),
                os.path.join(DATA_DIR, "medquad"),
                os.path.join(DATA_DIR, "Medical_QA_Chatbot", "data", "MedQuAD-master")
            ]
            
            target_path = None
            for cand in candidates:
                if cand and os.path.exists(cand):
                    target_path = cand
                    break

            if target_path:
                xml_df = parse_medquad_folder(target_path, max_files=max_files)
                if not xml_df.empty:
                    df = xml_df
        base_entries = [
            {
                "question": "What are the symptoms and early signs of HIV AIDS?",
                "answer": "Early signs and symptoms of HIV infection include fever, headache, rash, sore throat, swollen lymph nodes, night sweats, muscle aches, and persistent fatigue. Without treatment, HIV progresses to cause severe weight loss, chronic diarrhea, and opportunistic infections.",
                "focus": "HIV AIDS",
                "question_type": "symptoms"
            },
            {
                "question": "What are the treatments for HIV AIDS?",
                "answer": "HIV is treated with Antiretroviral Therapy (ART), which involves taking a combination of daily HIV medications. ART lowers the viral load in the blood to undetectable levels, protecting the immune system and preventing transmission.",
                "focus": "HIV AIDS",
                "question_type": "treatment"
            },
            {
                "question": "What are the treatments and recommended tablets for fever?",
                "answer": "Treatment for fever generally focuses on rest, fluid hydration, and over-the-counter antipyretic medications such as acetaminophen (paracetamol) or ibuprofen to help reduce body temperature and relieve discomfort. Avoid aspirin in children due to risk of Reye's syndrome. Consult a healthcare provider if fever exceeds 103°F (39.4°C) or lasts more than 3 days.",
                "focus": "Fever",
                "question_type": "treatment"
            },
            {
                "question": "What are the common symptoms and causes of fever?",
                "answer": "Fever is characterized by an elevated body temperature above 100.4°F (38°C), often accompanied by chills, sweating, headache, muscle aches, fatigue, and loss of appetite. Common causes include viral or bacterial infections, flu, common cold, immunizations, and inflammatory conditions.",
                "focus": "Fever",
                "question_type": "symptoms"
            },
            {
                "question": "What is the treatment for Dengue Fever?",
                "answer": "Dengue fever treatment focuses on supportive care, adequate fluid hydration, pain relievers like acetaminophen (paracetamol), and strictly avoiding NSAIDs like aspirin or ibuprofen as they increase bleeding risks.",
                "focus": "Dengue Fever",
                "question_type": "treatment"
            },
            {
                "question": "What are the treatments and medications for headache and pain?",
                "answer": "Common headache treatments include over-the-counter pain relievers such as acetaminophen, ibuprofen, or naproxen, along with adequate hydration, rest in a dark quiet room, and stress management.",
                "focus": "Headache",
                "question_type": "treatment"
            },
            {
                "question": "What are the treatments for common cold and cough?",
                "answer": "Treatments for common cold and cough include hydration, rest, saline nasal drops, honey for cough relief, and over-the-counter decongestants or cough suppressants.",
                "focus": "Common Cold",
                "question_type": "treatment"
            },
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
        """Retrieves top-k medical Q&A entries matching query with category-strict intent & entity re-ranking."""
        raw_results = self.rag.query(query, top_k=max(top_k * 3, 15), threshold=0.00)
        
        lowered_q = query.lower()

        # Intent detection with typo handling
        symptom_keywords = {"symptom", "symptoms", "symptons", "symtoms", "sign", "signs", "indication", "indications", "early signs", "early sign"}
        treatment_keywords = {"tablets", "tablet", "tabletes", "medicine", "medicines", "medication", "medications", "meds", "pill", "pills", "drug", "drugs", "cure", "cures", "treatment", "treatments", "treament", "treat", "dose", "suggest"}

        is_symptom_intent = any(kw in lowered_q for kw in symptom_keywords)
        is_treatment_intent = any(kw in lowered_q for kw in treatment_keywords)

        # Entity recognition
        recognizer = MedicalEntityRecognizer()
        query_entities = recognizer.extract_entities(query)
        all_detected_entities = set(query_entities["diseases"] + query_entities["symptoms"] + query_entities["treatments"])

        has_cancer_in_query = any(c in lowered_q for c in ["cancer", "tumor", "tumour", "leukemia", "carcinoma", "oncology"])

        formatted = []
        for res in raw_results:
            raw_score = res.get("score", 0.0)
            focus = res.get("focus", "General")
            focus_lower = focus.lower()
            qtype = res.get("question_type", "general").lower()
            cand_q = res.get("question", "").lower()

            adjusted_score = raw_score

            # Heavy penalty if focus is Cancer but query is NOT about Cancer
            if "cancer" in focus_lower and not has_cancer_in_query:
                adjusted_score *= 0.15

            # Boost if candidate focus or question matches detected query entities
            for ent in all_detected_entities:
                if ent in focus_lower or ent in cand_q:
                    adjusted_score += 0.35

            # Strict Category Intent Re-ranking
            if is_symptom_intent:
                if qtype == "symptoms" or "symptom" in cand_q or "sign" in cand_q:
                    adjusted_score += 0.45
                elif qtype in ["information", "general"]:
                    adjusted_score -= 0.25
                elif qtype == "treatment":
                    adjusted_score -= 0.15
            elif is_treatment_intent:
                if qtype == "treatment" or "treatment" in cand_q or "tablet" in cand_q or "medication" in cand_q:
                    adjusted_score += 0.45
                elif qtype in ["information", "general", "symptoms"]:
                    adjusted_score -= 0.25
            else:
                # Default preference for symptoms and treatments over generic info
                if qtype in ["symptoms", "treatment"]:
                    adjusted_score += 0.20

            formatted.append({
                "question": res.get("question", ""),
                "answer": res.get("answer", ""),
                "focus": focus,
                "question_type": res.get("question_type", "general"),
                "similarity": min(max(adjusted_score, 0.0), 0.99)
            })

        # Re-sort results by adjusted similarity descending
        formatted = sorted(formatted, key=lambda x: x["similarity"], reverse=True)
        return formatted[:top_k]

    def answer_question(self, query: str) -> dict:
        hits = self.retrieve(query, top_k=1)
        if hits:
            top_match = hits[0]
            ans_text = top_match["answer"]
            full_response = f"**Focus**: `{top_match.get('focus', 'General')}`\n\n{ans_text}{MEDICAL_DISCLAIMER}"
            return {
                "answer": full_response,
                "top_match": top_match,
                "is_relevant": True,
                "score": top_match["similarity"]
            }

        return {
            "answer": f"I could not locate specific medical guidelines for that query in the index.{MEDICAL_DISCLAIMER}",
            "top_match": None,
            "is_relevant": False,
            "score": 0.0
        }
