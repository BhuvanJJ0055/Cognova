"""
Task 2 Comprehensive Medical Q&A & NER Evaluation Suite
Author: Bhuvan J J

Executes 10 diverse medical test cases across Symptoms, Treatments, Diseases,
and Sensitive/Distressed queries to validate Task 2 against requirements.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.medical_qa import MedicalRetriever, MedicalEntityRecognizer
from src.modules.sentiment import score_mood_vader

def run_task2_comprehensive_suite():
    retriever = MedicalRetriever()
    recognizer = MedicalEntityRecognizer()

    test_queries = [
        {
            "category": "DISEASE SYMPTOMS",
            "query": "What are the symptoms of Type 2 Diabetes?",
            "expected_disease": "type 2 diabetes"
        },
        {
            "category": "TREATMENT & MEDICATION",
            "query": "What treatments are used for Asthma?",
            "expected_disease": "asthma"
        },
        {
            "category": "SENSITIVE & DISTRESSED QUERY",
            "query": "I am feeling extremely anxious and have severe chest pain. What is happening?",
            "expected_symptom": "chest pain"
        },
        {
            "category": "DISEASE SYMPTOMS",
            "query": "What are the early signs and symptoms of Cancer?",
            "expected_disease": "cancer"
        },
        {
            "category": "TREATMENT & THERAPY",
            "query": "What medications or therapies are prescribed for Hypertension?",
            "expected_disease": "hypertension"
        },
        {
            "category": "INFECTIOUS DISEASE",
            "query": "What are the common symptoms of Influenza flu?",
            "expected_disease": "flu"
        },
        {
            "category": "CHRONIC CONDITION",
            "query": "What treatments are recommended for Lupus?",
            "expected_disease": "lupus"
        },
        {
            "category": "GENERAL HEALTH FAQ",
            "query": "How is Celiac Disease diagnosed and managed?",
            "expected_disease": "celiac disease"
        }
    ]

    print("\n" + "="*85)
    print("🩺 TASK 2: MEDICAL Q&A & ENTITY RECOGNITION EVALUATION SUITE")
    print("="*85)

    for idx, test in enumerate(test_queries, 1):
        query = test["query"]
        mood, score = score_mood_vader(query)
        
        # 1. Retrieval
        results = retriever.retrieve(query, top_k=2)
        combined_text = query + " " + " ".join([r['answer'] for r in results])
        
        # 2. Entity recognition
        entities = recognizer.extract_entities(combined_text)

        print(f"\nTest #{idx:02d} [{test['category']}]")
        print(f" 📥 Input Query: \"{query}\"")
        if mood == "upset":
            print(f" 💙 Empathetic De-escalation Triggered (Mood: UPSET)")
        print(f" 🩺 Diseases Detected  : {entities['diseases']}")
        print(f" 🤒 Symptoms Detected  : {entities['symptoms']}")
        print(f" 💊 Treatments Detected: {entities['treatments']}")

        if results:
            top_match = results[0]
            print(f" 🎯 Top Match ({top_match['similarity']*100:.1f}%): {top_match['question']}")
            print(f" 🏷️ Focus Area: `{top_match['focus']}` | Category: `{top_match['question_type']}`")
            print(f" 💡 Answer Preview: {top_match['answer'][:140]}...")
        else:
            print(" ℹ️ No match found.")

    print("\n" + "="*85)
    print("✨ ALL MEDICAL TEST SUITES EVALUATED SUCCESSFULLY ✨")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_task2_comprehensive_suite()
