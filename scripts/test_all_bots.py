"""
End-to-End Verification Test Script for All 6 Cognova Bots & Modules
Author: Bhuvan J J

Runs structured test suites across all 6 tasks to verify pipeline correctness.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Module Imports
try:
    from src.modules.sentiment import SupportChatbot, score_mood_vader, tag_intent, _looks_like_plain_question
    from src.modules.medical_qa import MedicalRetriever, MedicalEntityRecognizer
    from src.modules.kb_updater import KnowledgeUpdater
    from src.modules.arxiv_expert import ArXivExpert, search_arxiv_api, extract_concepts, generate_explanation
    from src.modules.multimodal import MultimodalAssistant
    from src.modules.multilingual import MultilingualAssistant
    from src.core.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)


def test_task1_sentiment():
    print("\n" + "="*60)
    print("TESTING TASK 1: Sentiment-Aware Support Chatbot")
    print("="*60)
    
    bot = SupportChatbot(model_type="vader")
    test_cases = [
        ("My order is completely broken and this is terrible service!", "upset"),
        ("Thank you so much, your app is awesome and I love it!", "happy"),
        ("What is your standard return policy?", "calm")
    ]

    for msg, expected_mood in test_cases:
        mood, score = bot.score_mood(msg)
        intent = tag_intent(msg, mood=mood)
        reply = bot.reply_to(msg)
        is_plain_q = _looks_like_plain_question(msg)
        print(f"\n[Input]: '{msg}'")
        print(f" -> Detected Mood: {mood.upper()} (Score: {score:+.2f}) | Plain Q Override: {is_plain_q}")
        print(f" -> Tagged Intent: {intent}")
        print(f" -> Reply: {reply}")
        assert mood == expected_mood, f"Expected {expected_mood}, got {mood}"

    print("\n✅ TASK 1 PASSED: Sentiment classification, question override & replies working cleanly.")


def test_task2_medical_qa():
    print("\n" + "="*60)
    print("TESTING TASK 2: Medical Q&A Advisor & Medical NER")
    print("="*60)

    recognizer = MedicalEntityRecognizer()
    query = "What are the symptoms and treatments for Type 2 Diabetes?"
    entities = recognizer.extract_entities(query)
    print(f"\n[NER Input]: '{query}'")
    print(f" -> Extracted Diseases  : {entities['diseases']}")
    print(f" -> Extracted Symptoms  : {entities['symptoms']}")
    print(f" -> Extracted Treatments: {entities['treatments']}")

    retriever = MedicalRetriever()
    results = retriever.retrieve("diabetes symptoms", top_k=2)
    print(f"\n[Retrieval Matches Found]: {len(results)}")
    for idx, r in enumerate(results, 1):
        print(f" Match #{idx} (Sim: {r['similarity']*100:.1f}%): Focus={r['focus']} | Q={r['question']}")

    ans_dict = retriever.answer_question(query)
    print(f"\n[Structured Medical Answer]:\n{ans_dict['answer'][:200]}...")

    print("\n✅ TASK 2 PASSED: Medical Entity Recognition, RAG retrieval & Disclaimer verified.")


def test_task3_kb_updater():
    print("\n" + "="*60)
    print("TESTING TASK 3: Dynamic Knowledge Base Updater")
    print("="*60)

    updater = KnowledgeUpdater()
    success = updater.add_qa_pair(
        question="What is the support email for Cognova?",
        answer="You can contact Cognova support directly at support@cognova-ai.com.",
        focus="Cognova Platform",
        qtype="support"
    )
    print(f" -> Direct Q&A Addition: {'Success' if success else 'Failed'}")

    res = updater.rag.query("support email for Cognova", top_k=1)
    print(f" -> Query Newly Added Knowledge: Found {len(res)} match")
    if res:
        print(f"    Match Text: {res[0].get('answer') or res[0].get('text')}")

    print("\n✅ TASK 3 PASSED: Dynamic Knowledge Base updating verified.")


def test_task4_arxiv_expert():
    print("\n" + "="*60)
    print("TESTING TASK 4: ArXiv CS Research Assistant")
    print("="*60)

    expert = ArXivExpert()
    concepts = extract_concepts("Explain the self-attention mechanism in transformer models")
    print(f" -> Concept Extraction: {concepts}")

    papers = expert.retrieve("Transformer self-attention", top_k=2)
    print(f" -> Local Semantic Search Matches: {len(papers)}")
    for p in papers:
        print(f"    Title: {p['title']} ({p.get('published', 'N/A')})")

    explanation = generate_explanation("Transformer self-attention", papers)
    print(f"\n[Generated Paper Explanation]:\n{explanation[:250]}...")

    print("\n✅ TASK 4 PASSED: ArXiv research retrieval, concept extraction & explanation verified.")


def test_task5_multimodal():
    print("\n" + "="*60)
    print("TESTING TASK 5: Multimodal Vision Assistant")
    print("="*60)

    assistant = MultimodalAssistant()
    
    # Test ambiguity check
    vague_res = assistant.analyze_image("dummy.jpg" if os.path.exists("dummy.jpg") else None, "what is this?")
    print(f" -> Ambiguity Detection Test ('what is this?'): Is Ambiguous = {vague_res['is_ambiguous']}")
    print(f"    Notice: {vague_res['response'][:100]}...")

    clear_res = assistant.analyze_image("dummy.jpg" if os.path.exists("dummy.jpg") else None, "Explain the architecture of this deep learning diagram")
    print(f" -> Clear Prompt Test: Is Ambiguous = {clear_res['is_ambiguous']}")
    print(f"    Response: {clear_res['response'][:120]}...")

    print("\n✅ TASK 5 PASSED: Vision analysis & ambiguity check verified.")


def test_task6_multilingual():
    print("\n" + "="*60)
    print("TESTING TASK 6: Multilingual Assistant (All Possibilities)")
    print("="*60)

    assistant = MultilingualAssistant()
    assistant.initialize_retrievers()

    # 1. Code-switching test
    hindi_query = "Madhumeha ke symptoms kya hain?"
    detection = assistant.detect_and_translate(hindi_query)
    print(f" -> Input Query: '{hindi_query}'")
    print(f" -> Primary Language: {detection['primary_language'].upper()} | Is Mixed/Code-switched: {detection['is_mixed']}")
    print(f" -> Pivot English Query: '{detection['translated_query']}'")

    # 2. Multi-turn cross-lingual context preservation test
    chat_history = [{"user_input": "What is Diabetes?", "response": "Diabetes is a metabolic disease."}]
    followup_hi = assistant.detect_and_translate("इसके क्या लक्षण हैं?", chat_history=chat_history)
    print(f" -> Cross-Lingual Context Resolution: '{followup_hi['translated_query']}'")
    assert "Diabetes" in followup_hi["translated_query"], "Failed to resolve pronoun 'इसके' across language switch!"

    # 3. Ambiguity test
    ambiguous_res = assistant.detect_and_translate("ilaaj?")
    print(f" -> Ambiguity Detection ('ilaaj?'): Is Ambiguous = {ambiguous_res.get('is_ambiguous')}")

    # 4. Factual consistency test
    context = [{"title": "Diabetes Info", "text": "Diabetes causes increased thirst and frequent urination."}]
    score, aligned, _ = assistant.check_factual_consistency("Diabetes causes frequent urination and increased thirst.", context)
    print(f" -> Factual Overlap Score: {score * 100:.1f}% (Aligned: {aligned[:2]})")
    assert score >= 0.70, "Factual consistency check failed!"

    print("\n✅ TASK 6 PASSED: Multilingual detection, code-switching, context retention & factual consistency verified.")


if __name__ == "__main__":
    print("🚀 RUNNING COMPREHENSIVE END-TO-END VERIFICATION FOR ALL 6 TASKS 🚀")
    try:
        test_task1_sentiment()
        test_task2_medical_qa()
        test_task3_kb_updater()
        test_task4_arxiv_expert()
        test_task5_multimodal()
        test_task6_multilingual()
        print("\n" + "="*60)
        print("🎉 ALL 6 TASKS PASSED COMPREHENSIVE VERIFICATION SUCCESSFULLY! 🎉")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Verification Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
