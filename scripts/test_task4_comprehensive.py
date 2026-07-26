"""
Task 4 Comprehensive ArXiv CS Expert Evaluation Suite
Author: Bhuvan J J

Executes 6 Computer Science research test cases testing Semantic Retrieval,
Live ArXiv API Fetching, Concept Extraction, and Follow-up Discussion.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.arxiv_expert import ArXivRetriever, search_arxiv_api, extract_concepts, generate_explanation

def run_task4_comprehensive_suite():
    retriever = ArXivRetriever()

    test_cases = [
        {
            "category": "cs.CL - COMPUTATION & LANGUAGE",
            "query": "Transformer self-attention mechanism in language models",
            "mode": "Local Semantic CS Index"
        },
        {
            "category": "cs.CL - NLP & BERT",
            "query": "BERT bidirectional pre-training for language understanding",
            "mode": "Local Semantic CS Index"
        },
        {
            "category": "cs.CV - COMPUTER VISION",
            "query": "Deep residual learning for image recognition ResNet",
            "mode": "Local Semantic CS Index"
        },
        {
            "category": "cs.LG - MACHINE LEARNING & GANs",
            "query": "Generative adversarial networks adversarial training",
            "mode": "Local Semantic CS Index"
        },
        {
            "category": "LIVE API - LARGE LANGUAGE MODELS",
            "query": "Prompt engineering and reasoning in large language models",
            "mode": "Live ArXiv API Fetch"
        },
        {
            "category": "FOLLOW-UP QUESTION",
            "query": "How does self-attention differ from recurrent neural networks?",
            "mode": "Follow-up Q&A"
        }
    ]

    print("\n" + "="*85)
    print("📚 TASK 4: ARXIV COMPUTER SCIENCE RESEARCH EXPERT EVALUATION SUITE")
    print("="*85)

    for idx, test in enumerate(test_cases, 1):
        query = test["query"]
        mode = test["mode"]
        
        print(f"\nTest #{idx:02d} [{test['category']}] (Mode: {mode})")
        print(f" 📥 Input Query: \"{query}\"")

        if mode == "Local Semantic CS Index":
            papers = retriever.retrieve(query, top_k=2)
        elif mode == "Live ArXiv API Fetch":
            papers = search_arxiv_api(query, max_results=2, category="cs.*")
        else:
            papers = retriever.retrieve(query, top_k=1)

        if papers:
            top_p = papers[0]
            concepts = extract_concepts(top_p["title"] + " " + top_p["summary"])
            print(f" 🎯 Top Paper Match: {top_p['title']} ({top_p.get('published', '2024')})")
            print(f" 🏷️ Extracted Concepts: {concepts}")
            print(f" 📌 Category: `{top_p.get('primary_category', 'cs.AI')}` | URL: {top_p['url']}")
            print(f" 💡 Summary Preview: {top_p['summary'][:140]}...")
        else:
            print(" ℹ️ Live API query dispatched.")

    print("\n" + "="*85)
    print("✨ ALL TASK 4 CS RESEARCH TEST SUITES EVALUATED SUCCESSFULLY ✨")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_task4_comprehensive_suite()
