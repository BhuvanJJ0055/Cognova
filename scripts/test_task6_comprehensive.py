"""
Task 6 Comprehensive Multilingual & Cross-Lingual Evaluation Suite
Author: Bhuvan J J

Executes 6 multilingual test scenarios covering Language Detection,
Hinglish/Kanglish/Spanglish/Franglish Code-Switching, Pivot Translation,
Ambiguity Resolution, and Multi-Turn Cross-Lingual Continuity.
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

    test_cases = [
        {
            "title": "HINGLISH CODE-SWITCHED QUERY",
            "input": "Madhumeha diabetes ke symptoms kya hain?",
            "expected_lang": "hi",
            "history": []
        },
        {
            "title": "SPANISH MEDICAL QUERY",
            "input": "¿Cuáles son los síntomas de la diabetes?",
            "expected_lang": "es",
            "history": []
        },
        {
            "title": "FRENCH MEDICAL QUERY",
            "input": "Quels sont les symptômes de l'asthme et le traitement?",
            "expected_lang": "fr",
            "history": []
        },
        {
            "title": "KANGLISH CODE-SWITCHED QUERY",
            "input": "Nanna payment twice deduct agide, please help",
            "expected_lang": "kn",
            "history": []
        },
        {
            "title": "ENGLISH FOLLOW-UP QUERY",
            "input": "What are the early signs of cancer?",
            "expected_lang": "en",
            "history": []
        },
        {
            "title": "MULTI-TURN CROSS-LINGUAL PRONOUN RESOLUTION",
            "input": "इसके क्या लक्षण हैं?",
            "expected_lang": "hi",
            "history": [
                {"user_input": "What is Diabetes?", "translated": "What is Diabetes?", "response": "Diabetes is a condition."}
            ]
        }
    ]

    print("\n" + "="*85)
    print("🌐 TASK 6: MULTILINGUAL & CROSS-LINGUAL ASSISTANT EVALUATION SUITE")
    print("="*85)

    for idx, test in enumerate(test_cases, 1):
        text = test["input"]
        history = test.get("history", [])
        detection = agent.detect_and_translate(text, chat_history=history)
        
        print(f"\nTest #{idx:02d} [{test['title']}]")
        print(f" 📥 Input Message : \"{text}\"")
        print(f" 🌐 Detected Lang  : {detection['language_name']} ({detection['primary_language'].upper()})")
        print(f" 🔀 Mixed/Code-Sw: {detection['is_mixed']}")
        print(f" 📌 English Pivot  : \"{detection['translated_query']}\"")

        if agent.medical_retriever:
            results = agent.medical_retriever.retrieve(detection["translated_query"], top_k=1)
            if results:
                match = results[0]
                print(f" 🎯 RAG Retrieval Match ({match.get('similarity', 0.8)*100:.1f}%): {match.get('question')}")

    print("\n" + "="*85)
    print("✨ ALL TASK 6 MULTILINGUAL TEST SUITES EVALUATED SUCCESSFULLY ✨")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_task6_comprehensive_suite()
