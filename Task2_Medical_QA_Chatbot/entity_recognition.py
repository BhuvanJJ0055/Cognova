"""
Task 2 - entity_recognition.py
Author: Bhuvan J J

This script implements a lightweight, rule-based Medical Entity Recognizer.
It identifies three categories of entities in user text queries:
1. Diseases (e.g. diabetes, asthma, lupus, hypertension)
2. Symptoms (e.g. fever, joint pain, shortness of breath, cough, fatigue)
3. Treatments (e.g. inhaler, chemotherapy, insulin, antibiotics, vaccine, psychotherapy)

Methodology:
- For Diseases: We extract unique disease/focus names dynamically from the MedQuAD dataset if available,
  combining it with a robust baseline dictionary of common conditions.
- For Symptoms and Treatments: We compile a comprehensive dictionary of common medical keywords.
- Matching Mechanism: To ensure high accuracy, queries are normalized (lowercased, punctuation removed) 
  and matched using regular expressions with word boundaries (\\b).
- Multi-word entities are sorted by length in descending order, ensuring that longer, more specific 
  phrases (e.g., "type 2 diabetes", "joint pain") are matched before shorter, generic ones.
"""

import re
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "medquad_qa.csv")
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "sample_medquad_qa.csv")

# Baseline diseases if dataset is not loaded yet
STATIC_DISEASE_LIST = [
    "diabetes", "asthma", "lupus", "influenza", "flu", "hypertension", "high blood pressure",
    "cancer", "depression", "arthritis", "hepatitis", "allergy", "allergies", "migraine",
    "pneumonia", "celiac disease", "gout", "insomnia", "tuberculosis", "tb", "malaria", "anemia",
    "heart disease", "coronary artery disease", "stroke", "alzheimer's", "dementia", "parkinson's",
    "hiv", "aids", "cholera", "measles", "chickenpox", "mumps", "rubella", "ebola"
]

STATIC_SYMPTOM_LIST = [
    "fever", "cough", "wheezing", "shortness of breath", "difficulty breathing", "chest pain",
    "pain", "fatigue", "tiredness", "weakness", "headache", "dizziness", "nausea", "vomiting",
    "rash", "itching", "swelling", "joint pain", "joint stiffness", "muscle pain", "muscle ache",
    "stiffness", "insomnia", "sleeping too much", "weight loss", "weight gain", "runny nose",
    "stuffy nose", "sore throat", "dark urine", "jaundice", "diarrhea", "constipation", "bloating",
    "gas", "chills", "sweating", "night sweats", "loss of appetite", "mouth ulcers", "confusion",
    "memory loss", "numbness", "tingling", "hair loss", "shivering", "sneezing", "irritability",
    "dry eyes", "soreness", "cramps", "hives", "throbbing"
]

STATIC_TREATMENT_LIST = [
    "medicine", "therapy", "surgery", "vaccine", "vaccination", "antibiotic", "antibiotics",
    "inhaler", "corticosteroid", "corticosteroids", "medication", "medications", "drug", "drugs",
    "prescription", "chemotherapy", "radiation therapy", "immunotherapy", "transplant",
    "bone marrow transplant", "insulin", "prednisone", "hydroxychloroquine", "albuterol", "aspirin",
    "ibuprofen", "acetaminophen", "oseltamivir", "tamiflu", "allopurinol", "psychotherapy",
    "antidepressant", "antidepressants", "treatment", "treatments", "physiotherapy", "physical therapy",
    "operation", "dialysis", "supplements", "epinephrine", "epipen", "ointment"
]


class MedicalEntityRecognizer:
    """Detects and tags medical entities in user text queries."""

    def __init__(self, dataset_csv=None):
        self.diseases = set(STATIC_DISEASE_LIST)
        self.symptoms = set(STATIC_SYMPTOM_LIST)
        self.treatments = set(STATIC_TREATMENT_LIST)

        # Attempt to augment disease list with real MedQuAD focus topics
        csv_to_use = dataset_csv or (DEFAULT_CSV_PATH if os.path.exists(DEFAULT_CSV_PATH) else SAMPLE_CSV_PATH)
        if os.path.exists(csv_to_use):
            try:
                df = pd.read_csv(csv_to_use)
                if "focus" in df.columns:
                    # Clean and normalize focus names
                    focus_names = df["focus"].dropna().astype(str).str.lower().str.strip()
                    # Add names and remove empty strings
                    for name in focus_names:
                        if name:
                            # Strip off common subheadings if any, or just clean
                            clean_name = re.sub(r'\s*\([^)]*\)', '', name).strip()
                            if clean_name:
                                self.diseases.add(clean_name)
                print(f"[Info] Dynamically loaded {len(self.diseases)} unique disease entities from dataset focus lists.")
            except Exception as e:
                print(f"[Warning] Failed to augment disease list from {csv_to_use}: {e}")

        # Pre-process lists to sort by length descending to match longest phrases first
        self.sorted_diseases = sorted(list(self.diseases), key=len, reverse=True)
        self.sorted_symptoms = sorted(list(self.symptoms), key=len, reverse=True)
        self.sorted_treatments = sorted(list(self.treatments), key=len, reverse=True)

    def extract_entities(self, text):
        """
        Scans input text for diseases, symptoms, and treatments.
        Returns a dictionary containing list of found entities in each category.
        """
        normalized_text = " " + re.sub(r'[^\w\s\-\']', ' ', text.lower()).strip() + " "
        
        found_diseases = []
        found_symptoms = []
        found_treatments = []

        # Find Diseases
        for disease in self.sorted_diseases:
            # Escape regex characters
            pattern = r'\b' + re.escape(disease) + r'\b'
            if re.search(pattern, normalized_text):
                found_diseases.append(disease)
                # Remove matched part to prevent duplicate matching in other sub-categories if needed,
                # but since we return distinct lists it's fine.

        # Find Symptoms
        for symptom in self.sorted_symptoms:
            pattern = r'\b' + re.escape(symptom) + r'\b'
            if re.search(pattern, normalized_text):
                found_symptoms.append(symptom)

        # Find Treatments
        for treatment in self.sorted_treatments:
            pattern = r'\b' + re.escape(treatment) + r'\b'
            if re.search(pattern, normalized_text):
                found_treatments.append(treatment)

        return {
            "diseases": sorted(list(set(found_diseases))),
            "symptoms": sorted(list(set(found_symptoms))),
            "treatments": sorted(list(set(found_treatments)))
        }


if __name__ == "__main__":
    # If run standalone, test entity recognition
    recognizer = MedicalEntityRecognizer()
    test_queries = [
        "What are the symptoms of type 2 diabetes?",
        "Is there an inhaler for treating my severe asthma wheezing?",
        "My lupus flare is causing intense joint pain and fever, what prednisone dosage is common?",
        "How is cancer chemotherapy different from radiation therapy?"
    ]
    
    print("\n--- Testing Medical Entity Recognizer ---")
    for q in test_queries:
        print(f"\nQuery: \"{q}\"")
        entities = recognizer.extract_entities(q)
        print("Tagged Entities:")
        print(f"  Diseases:   {entities['diseases']}")
        print(f"  Symptoms:   {entities['symptoms']}")
        print(f"  Treatments: {entities['treatments']}")
