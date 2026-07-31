"""
Exhaustive Master Multilingual Test Suite - All Possibilities Evaluator
Author: Bhuvan J J

Tests ALL possibilities across 6 categories:
1. Pure 6-Language Detection & Translation (EN, HI, KN, ES, FR, DE)
2. Code-Switching / Mixed-Language Parsing (Hinglish, Kanglish, Spanglish, Franglish, Denglish)
3. Multi-Turn Cross-Lingual Context Retention (6-turn language switch chain)
4. Ambiguity Detection & Target Language Clarification (6 languages)
5. Factual Consistency Overlap Scoring & Hallucination Audit
6. Cross-Lingual RAG Document Retrieval Integration
"""

import sys
import os
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.multilingual import MultilingualAgent


def run_all_possibilities_suite():
    print("\n" + "="*95)
    print("🚀 EXHAUSTIVE MULTILINGUAL ASSISTANT TEST SUITE - TESTING ALL POSSIBILITIES")
    print("="*95)

    agent = MultilingualAgent()
    agent.initialize_retrievers()

    report_lines = []
    report_lines.append("# Exhaustive Multilingual Test Report (All Possibilities)\n")
    report_lines.append(f"**Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("| Category | Test ID | Scenario | Input Text | Expected Lang | Detected Lang | Code-Switched | Pivot Query | Status |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|")

    total_tests = 0
    passed_tests = 0

    # -------------------------------------------------------------
    # CATEGORY A: Pure 6-Language Detection & Pivot Translation
    # -------------------------------------------------------------
    category_a = [
        ("A1", "English Query", "What are the symptoms and treatments for Type 2 Diabetes?", "en"),
        ("A2", "Hindi Query", "टाइप 2 मधुमेह के लक्षण और उपचार क्या हैं?", "hi"),
        ("A3", "Kannada Query", "ಟೈಪ್ 2 ಮಧುಮೇಹದ ಲಕ್ಷಣಗಳು ಮತ್ತು ಚಿಕಿತ್ಸೆಗಳು ಯಾವುವು?", "kn"),
        ("A4", "Spanish Query", "¿Cuáles son los síntomas y tratamientos de la diabetes tipo 2?", "es"),
        ("A5", "French Query", "Quels sont les symptômes et les traitements du diabète de type 2 ?", "fr"),
        ("A6", "German Query", "Was sind die Symptome und Behandlungen von Typ-2-Diabetes?", "de"),
    ]

    print("\n🔹 CATEGORY A: Pure 6-Language Detection & Pivot Translation")
    print("-" * 80)
    for tid, title, text, exp_lang in category_a:
        total_tests += 1
        det = agent.detect_and_translate(text)
        is_pass = det["primary_language"] == exp_lang
        if is_pass: passed_tests += 1
        status = "✅ PASS" if is_pass else "❌ FAIL"
        
        print(f"[{tid}] {title:18s} | In: \"{text[:35]}...\" | Lang: {det['primary_language'].upper()} | Pivot: \"{det['translated_query'][:35]}\" | {status}")
        report_lines.append(f"| Pure Language | {tid} | {title} | `{text}` | `{exp_lang}` | `{det['primary_language']}` | `{det['is_mixed']}` | `{det['translated_query']}` | {status} |")

    # -------------------------------------------------------------
    # CATEGORY B: Code-Switching / Mixed-Language Parsing
    # -------------------------------------------------------------
    category_b = [
        ("B1", "Hinglish Mixed", "Madhumeha sugar disease ke symptoms aur ilaj kya hai?", "hi"),
        ("B2", "Kanglish Mixed", "Nange severe talenovu and jvara agide, what should I do?", "kn"),
        ("B3", "Spanglish Mixed", "Tengo dolor de cabeza and fever, cuáles son los remedies?", "es"),
        ("B4", "Franglish Mixed", "J'ai un mal de tête and high fever, quel est le traitement?", "fr"),
        ("B5", "Denglish Mixed", "Ich habe Kopfschmerzen and fever, was ist die Medizin?", "de"),
    ]

    print("\n🔹 CATEGORY B: Code-Switching / Mixed-Language Parsing")
    print("-" * 80)
    for tid, title, text, exp_lang in category_b:
        total_tests += 1
        det = agent.detect_and_translate(text)
        is_pass = det["primary_language"] == exp_lang and det["is_mixed"]
        if is_pass: passed_tests += 1
        status = "✅ PASS" if is_pass else "❌ FAIL"
        
        print(f"[{tid}] {title:18s} | In: \"{text[:35]}...\" | Lang: {det['primary_language'].upper()} | Mixed: {det['is_mixed']} | Pivot: \"{det['translated_query'][:35]}\" | {status}")
        report_lines.append(f"| Code-Switching | {tid} | {title} | `{text}` | `{exp_lang}` | `{det['primary_language']}` | `{det['is_mixed']}` | `{det['translated_query']}` | {status} |")

    # -------------------------------------------------------------
    # CATEGORY C: Multi-Turn Cross-Lingual Context Preservation (6-Turn Chain)
    # -------------------------------------------------------------
    category_c_chain = [
        ("C1", "Turn 1 (EN Init)", "What is Asthma?", "en", "Asthma"),
        ("C2", "Turn 2 (HI Switch)", "इसके क्या लक्षण हैं?", "hi", "Asthma"),
        ("C3", "Turn 3 (KN Switch)", "ಇದರ ಚಿಕಿತ್ಸೆ ಏನು?", "kn", "Asthma"),
        ("C4", "Turn 4 (ES Switch)", "¿Cómo prevenir sus ataques?", "es", "Asthma"),
        ("C5", "Turn 5 (FR Switch)", "Existe-t-il un remède contre cela ?", "fr", "Asthma"),
        ("C6", "Turn 6 (DE Switch)", "Ist diese Krankheit gefährlich?", "de", "Asthma"),
    ]

    print("\n🔹 CATEGORY C: Multi-Turn Cross-Lingual Context Preservation (6-Turn Chain)")
    print("-" * 80)
    chain_history = []
    for tid, title, text, exp_lang, expected_entity in category_c_chain:
        total_tests += 1
        det = agent.detect_and_translate(text, chat_history=chain_history)
        resolved_query = det["translated_query"]
        
        has_entity = expected_entity.lower() in resolved_query.lower()
        is_pass = det["primary_language"] == exp_lang and (has_entity or tid == "C1")
        if is_pass: passed_tests += 1
        status = "✅ PASS" if is_pass else "❌ FAIL"

        print(f"[{tid}] {title:18s} | In: \"{text}\" -> Pivot: \"{resolved_query}\" | {status}")
        report_lines.append(f"| Multi-Turn Context | {tid} | {title} | `{text}` | `{exp_lang}` | `{det['primary_language']}` | `{det['is_mixed']}` | `{resolved_query}` | {status} |")

        # Append turn to chain history
        gen_res = agent.generate_response(text, resolved_query, det, [], chat_history=chain_history) or {}
        chain_history.append({
            "user_input": text,
            "prompt": text,
            "translated": resolved_query,
            "response": gen_res.get("response", ""),
            "answer": gen_res.get("response", "")
        })

    # -------------------------------------------------------------
    # CATEGORY D: Ambiguity Detection & Target Language Clarification
    # -------------------------------------------------------------
    category_d = [
        ("D1", "EN Ambiguous", "treatment?", "en"),
        ("D2", "HI Ambiguous", "इलाज?", "hi"),
        ("D3", "KN Ambiguous", "ಚಿಕಿತ್ಸೆ?", "kn"),
        ("D4", "ES Ambiguous", "¿tratamiento?", "es"),
        ("D5", "FR Ambiguous", "traitement ?", "fr"),
        ("D6", "DE Ambiguous", "Behandlung?", "de"),
    ]

    print("\n🔹 CATEGORY D: Ambiguity Detection & Target Language Clarification")
    print("-" * 80)
    for tid, title, text, exp_lang in category_d:
        total_tests += 1
        det = agent.detect_and_translate(text)
        is_pass = det.get("is_ambiguous", False) and bool(det.get("clarification_question"))
        if is_pass: passed_tests += 1
        status = "✅ PASS" if is_pass else "❌ FAIL"

        print(f"[{tid}] {title:18s} | In: \"{text:12s}\" | Ambiguous: {det.get('is_ambiguous')} | Clarification: \"{det.get('clarification_question')[:35]}...\" | {status}")
        report_lines.append(f"| Ambiguity Resolution | {tid} | {title} | `{text}` | `{exp_lang}` | `{det['primary_language']}` | `{det['is_mixed']}` | Clarification: {det.get('clarification_question')} | {status} |")

    # -------------------------------------------------------------
    # CATEGORY E: Factual Consistency & Groundedness Auditing
    # -------------------------------------------------------------
    print("\n🔹 CATEGORY E: Factual Consistency & Groundedness Auditing")
    print("-" * 80)
    
    # E1: High Overlap Test
    total_tests += 1
    sample_context = [{"title": "Asthma Info", "text": "Asthma causes shortness of breath, wheezing, and coughing."}]
    good_resp = "Asthma is characterized by coughing and shortness of breath."
    score_good, aligned_good, _ = agent.check_factual_consistency(good_resp, sample_context)
    pass_e1 = score_good >= 0.70
    if pass_e1: passed_tests += 1
    status_e1 = "✅ PASS" if pass_e1 else "❌ FAIL"
    print(f"[E1] Grounded Response Check | Score: {score_good*100:.1f}% | Aligned: {aligned_good[:3]} | {status_e1}")
    report_lines.append(f"| Groundedness Audit | E1 | Grounded Response Check | `{good_resp}` | `en` | `en` | `False` | Overlap Score: {score_good*100:.1f}% | {status_e1} |")

    # E2: Hallucination Detection Test
    total_tests += 1
    hallucinated_resp = "Asthma is treated with insulin injections and antibiotic penicillin."
    score_hall, _, missing_hall = agent.check_factual_consistency(hallucinated_resp, sample_context)
    pass_e2 = score_hall <= 0.40 and "penicillin" in missing_hall
    if pass_e2: passed_tests += 1
    status_e2 = "✅ PASS" if pass_e2 else "❌ FAIL"
    print(f"[E2] Hallucination Audit    | Score: {score_hall*100:.1f}% | Missing: {missing_hall[:2]} | {status_e2}")
    report_lines.append(f"| Groundedness Audit | E2 | Hallucination Audit | `{hallucinated_resp}` | `en` | `en` | `False` | Overlap Score: {score_hall*100:.1f}% | {status_e2} |")

    # -------------------------------------------------------------
    # CATEGORY F: RAG Index Retrieval Across Modules
    # -------------------------------------------------------------
    print("\n🔹 CATEGORY F: RAG Index Retrieval Across Modules")
    print("-" * 80)

    rag_queries = [
        ("F1", "Diabetes RAG Query", "Madhumeha ke symptoms kya hain?", "diabetes"),
        ("F2", "Asthma RAG Query", "Quels sont les symptômes de l'asthme?", "asthma")
    ]

    for tid, title, text, keyword in rag_queries:
        total_tests += 1
        det = agent.detect_and_translate(text)
        docs = []
        if agent.medical_retriever:
            docs = agent.medical_retriever.retrieve(det["translated_query"], top_k=1)
        
        has_match = len(docs) > 0
        if has_match: passed_tests += 1
        status = "✅ PASS" if has_match else "❌ FAIL"

        match_q = docs[0].get("question") if docs else "No match"
        print(f"[{tid}] {title:18s} | In: \"{text[:35]}...\" | Match: \"{match_q[:35]}...\" | {status}")
        report_lines.append(f"| RAG Integration | {tid} | {title} | `{text}` | `en` | `{det['primary_language']}` | `{det['is_mixed']}` | Match: {match_q} | {status} |")

    # -------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print(f"🏆 ALL POSSIBILITIES TEST RESULTS: {passed_tests}/{total_tests} TESTS PASSED ({passed_tests/total_tests*100:.1f}%) 🏆")
    print("="*95 + "\n")

    # Save detailed markdown report
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "multilingual_exhaustive_test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        f.write(f"\n\n## Summary\n- **Total Test Cases**: {total_tests}\n- **Passed Test Cases**: {passed_tests}\n- **Success Rate**: {passed_tests/total_tests*100:.1f}%\n")

    print(f"📄 Exhaustive Test Report saved to: {report_path}")


if __name__ == "__main__":
    run_all_possibilities_suite()
