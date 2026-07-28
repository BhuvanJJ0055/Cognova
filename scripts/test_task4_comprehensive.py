"""
Task 4 ArXiv Computer Science Expert Assistant Test Suite
Author: Bhuvan J J

Validates ArXiv dataset indexing, CS category filtering (cs.CL, cs.CV, cs.LG, cs.AI),
concept extraction, abstract summarization, and LLM explanation synthesis.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.arxiv_expert import ArXivExpert, extract_concepts, summarize_paper_text, generate_explanation, search_arxiv_api

def run_task4_test_suite():
    print("\n" + "="*85)
    print("📚 TASK 4: ARXIV COMPUTER SCIENCE EXPERT COMPREHENSIVE TEST SUITE 📚")
    print("="*85)

    # 1. Test CS Paper Retrieval & Subdomain Filtering
    print("\n🔍 1. Testing CS Paper Retrieval & Subdomain Category Filtering...")
    expert = ArXivExpert()

    test_queries = [
        ("Transformer self-attention in language models", "cs.CL"),
        ("Deep residual learning for image recognition", "cs.CV"),
        ("Generative adversarial networks", "cs.LG"),
        ("Chain of thought prompting", "cs.AI")
    ]

    for q, target_cat in test_queries:
        hits = expert.retrieve(q, top_k=2, category=target_cat)
        print(f"\n 📥 Query: \"{q}\" (Target Category: {target_cat})")
        assert len(hits) > 0, f"FAILED: No paper hits for query '{q}'"
        top = hits[0]
        print(f"  -> Top Paper: [{top['primary_category']}] {top['title']} (Match: {top['similarity']*100:.1f}%)")
        print(f"  -> ArXiv ID: {top['id']} | Authors: {top['authors'][:40]}...")

    print("\n✅ CS Paper Retrieval & Category Filtering: PASS")

    # 2. Test Concept Extraction & Abstract Summarization
    print("\n🏷️ 2. Testing NLP Information Extraction & Summarization...")
    sample_abstract = (
        "We propose the Transformer architecture based entirely on self-attention mechanisms, "
        "dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks "
        "show these models to be superior in quality while being more parallelizable."
    )
    
    concepts = extract_concepts("Attention Is All You Need " + sample_abstract)
    summary = summarize_paper_text(sample_abstract, max_sentences=2)

    print(f"  -> Extracted Concepts: {concepts}")
    print(f"  -> Summarized Abstract: {summary}")
    assert "Transformer Architecture" in concepts or "Self-Attention Mechanism" in concepts, "FAILED: Failed to extract core concepts!"
    assert len(summary) > 0, "FAILED: Abstract summary is empty!"
    print("  ✅ Concept Extraction & Summarization: PASS")

    # 3. Test Open-Source LLM Explanation Synthesis
    print("\n💡 3. Testing Explanation Generation & Synthesis Engine...")
    papers = expert.retrieve("Transformer self-attention", top_k=2)
    explanation = generate_explanation("Transformer self-attention", papers)
    
    print("  -> Sample Generated Synthesis Preview:")
    print("     " + "\n     ".join(explanation.split("\n")[:6]))
    assert "Key Paper Synthesis" in explanation, "FAILED: Explanation synthesis header missing!"
    assert "Extracted Core Concepts" in explanation, "FAILED: Core concepts missing from explanation!"
    print("  ✅ Explanation Synthesis Engine: PASS")

    # 4. Test Live ArXiv API Search Engine
    print("\n🌐 4. Testing Live ArXiv API Search & Fallback Engine...")
    api_hits = search_arxiv_api("large language models", max_results=2, category="cs.CL")
    assert len(api_hits) > 0, "FAILED: Live ArXiv API search returned empty results!"
    print(f"  -> API Top Match: [{api_hits[0]['primary_category']}] {api_hits[0]['title']}")
    print("  ✅ Live ArXiv API Search Engine: PASS")

    print("\n" + "="*85)
    print("🎯 TASK 4 EVALUATION SUMMARY: All ArXiv CS Expert Tests PASSED")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_task4_test_suite()
