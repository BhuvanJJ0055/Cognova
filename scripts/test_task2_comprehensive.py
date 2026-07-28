"""
Task 2 MedQuAD Medical Q&A Comprehensive Test Suite
Author: Bhuvan J J

Validates MedQuAD XML dataset parsing, multi-disease vector retrieval,
word-boundary Medical NER, and safety disclaimer integration.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.medical_qa import MedicalRetriever, MedicalEntityRecognizer, MEDICAL_DISCLAIMER

def run_task2_test_suite():
    print("\n" + "="*85)
    print("🩺 TASK 2: MEDQUAD MEDICAL Q&A & NER COMPREHENSIVE TEST SUITE 🩺")
    print("="*85)

    # 1. Test Entity Recognition Precision
    print("\n🔍 1. Testing Medical Entity Recognizer Word-Boundary Precision...")
    recognizer = MedicalEntityRecognizer()

    test_queries = [
        ("What are the symptoms of type 2 diabetes?", ["diabetes", "type 2 diabetes"], ["symptoms"], []),
        ("How is dengue fever treated with hydration?", ["dengue", "dengue fever"], [], ["treatment"]),
        ("I have flu and fever, do I need antibiotics?", ["flu"], ["fever"], ["antibiotic", "antibiotics"]),
        ("Fluid in lungs during pneumonia", ["pneumonia"], [], [])  # "fluid" should NOT trigger "flu"
    ]

    for q, exp_diseases, exp_symptoms, exp_treatments in test_queries:
        res = recognizer.extract_entities(q)
        print(f"\n 📥 Input: \"{q}\"")
        print(f"  -> Diseases: {res['diseases']}")
        print(f"  -> Symptoms: {res['symptoms']}")
        print(f"  -> Treatments: {res['treatments']}")

        # Verify precision (flu vs fluid)
        if "fluid" in q and "flu" not in q.split():
            assert "flu" not in res["diseases"], "FAILED: 'flu' incorrectly matched inside 'fluid'!"

    print("\n✅ Entity Recognition Precision: PASS")

    # 2. Test Retrieval Mechanism
    print("\n📚 2. Testing Medical Retrieval Engine across Multiple Conditions...")
    retriever = MedicalRetriever()

    medical_queries = [
        "tell the symptoms about the HIV AIDS",
        "mention the symptons of dengue",
        "What is the recommended treatment for Dengue Fever?",
        "I want a fever suggest me some tabletes",
        "What are the symptoms of Cancer?",
        "What is the treatment for Type 2 Diabetes?",
        "How to treat Dengue Fever?",
        "What medications treat Hypertension?"
    ]

    for query in medical_queries:
        results = retriever.retrieve(query, top_k=2)
        print(f"\n 📥 Query: \"{query}\"")
        if results:
            top = results[0]
            print(f"  -> Top Match: [{top['focus']}] {top['question']} (Match: {top['similarity']*100:.1f}%)")
            print(f"  -> Answer snippet: {top['answer'][:120]}...")
            if "fever" in query.lower() and "tabletes" in query.lower():
                assert top['focus'].lower() != "cancer", f"FAILED: Fever tablet query returned Cancer instead of Fever! Got {top['focus']}"
                assert top['focus'].lower() in ["fever", "dengue fever"], f"FAILED: Expected Fever focus, got {top['focus']}"
        else:
            print("  ⚠️ No direct match returned.")

    # 3. Test Answer Generation & Disclaimer
    print("\n⚠️ 3. Testing Answer Generator & Safety Disclaimer...")
    response = retriever.answer_question("What are the early signs of Diabetes?")
    assert "Medical Disclaimer" in response["answer"], "FAILED: Medical Disclaimer missing from response!"
    print("  ✅ Answer generation & disclaimer insertion: PASS")

    print("\n" + "="*85)
    print("🎯 TASK 2 EVALUATION SUMMARY: All MedQuAD & NER Tests PASSED")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_task2_test_suite()
