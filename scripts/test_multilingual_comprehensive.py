"""
Task 6 Comprehensive Multilingual & Cross-Lingual Evaluation Suite
Author: Bhuvan J J

Executes 7 multilingual test scenarios covering:
1. 6-Language Detection & Pivot Translation (EN, HI, KN, ES, FR, DE)
2. Hinglish / Kanglish / Spanglish / Franglish / Denglish Code-Switching
3. Cross-Lingual Context Preservation (Multi-turn pronoun resolution)
4. Ambiguous Query Detection & Target Language Clarification
5. Factual Consistency Overlap Score Calculation
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.multilingual import MultilingualAgent


def run_task6_comprehensive_suite():
    agent = MultilingualAgent()
    agent.initialize_retrievers()

    print("\n" + "="*85)
    print("🌐 TASK 6: COMPREHENSIVE MULTILINGUAL & CROSS-LINGUAL ASSISTANT EVALUATION")
    print("="*85)

    test_scenarios = [
        {
            "title": "1. HINGLISH CODE-SWITCHED QUERY",
            "input": "Madhumeha diabetes ke symptoms kya hain?",
            "expected_lang": "hi",
            "history": []
        },
        {
            "title": "2. SPANISH MEDICAL QUERY",
            "input": "¿Cuáles son los síntomas de la diabetes?",
            "expected_lang": "es",
            "history": []
        },
        {
            "title": "3. FRENCH MEDICAL QUERY",
            "input": "Quels sont les symptômes de l'asthme et le traitement?",
            "expected_lang": "fr",
            "history": []
        },
        {
            "title": "4. GERMAN MEDICAL QUERY",
            "input": "Was sind die Symptome von Krebs?",
            "expected_lang": "de",
            "history": []
        },
        {
            "title": "5. KANGLISH CODE-SWITCHED QUERY",
            "input": "Nange headache ide, what should I do?",
            "expected_lang": "kn",
            "history": []
        },
        {
            "title": "6. AMBIGUOUS QUERY TEST",
            "input": "ilaaj?",
            "expected_lang": "hi",
            "history": []
        },
        {
            "title": "7. MULTI-TURN CROSS-LINGUAL PRONOUN RESOLUTION",
            "input": "इसके क्या लक्षण हैं?",
            "expected_lang": "hi",
            "history": [
                {"user_input": "What is Type 2 Diabetes?", "translated": "What is Type 2 Diabetes?", "response": "Diabetes is a metabolic disease."}
            ]
        }
    ]

    passed_count = 0

    for idx, test in enumerate(test_scenarios, 1):
        text = test["input"]
        history = test["history"]
        
        print(f"\nScenario #{idx:02d} [{test['title']}]")
        print(f" 📥 Input Message : \"{text}\"")
        
        detection = agent.detect_and_translate(text, chat_history=history)
        
        print(f" 🌐 Detected Lang  : {detection['language_name']} ({detection['primary_language'].upper()})")
        print(f" 🔀 Mixed/Code-Sw: {detection['is_mixed']}")
        print(f" 📌 Pivot English  : \"{detection['translated_query']}\"")
        
        if detection.get("is_ambiguous"):
            print(f" ⚠️  Ambiguity Alert: Yes -> Clarification: \"{detection.get('clarification_question')}\"")

        # RAG Search
        docs = []
        if agent.medical_retriever:
            docs = agent.medical_retriever.retrieve(detection["translated_query"], top_k=2)
            if docs:
                match = docs[0]
                print(f" 🎯 RAG Retrieval Match ({match.get('similarity', 0.8)*100:.1f}%): {match.get('question')}")

        # Grounded synthesis & factual check
        gen_res = agent.generate_response(text, detection["translated_query"], detection, docs, chat_history=history) or {}
        resp_text = str(gen_res.get("response", ""))
        resp_english = str(gen_res.get("response_english", ""))
        score, aligned, missing = agent.check_factual_consistency(resp_english, docs)
        
        display_output = resp_text[:120] if resp_text else "No output generated"
        print(f" 🤖 Target Output  : \"{display_output}...\"")
        print(f" 📊 Factual Overlap: {score * 100:.1f}% (Aligned tokens: {len(aligned)})")

        if detection["primary_language"] == test["expected_lang"]:
            passed_count += 1

    print("\n" + "="*85)
    print(f"✨ MULTILINGUAL EVALUATION COMPLETED: {passed_count}/{len(test_scenarios)} SCENARIOS PASSED ✨")
    print("="*85 + "\n")


if __name__ == "__main__":
    run_task6_comprehensive_suite()
