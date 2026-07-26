"""
Task 3 - test_arxiv_pipeline.py
Author: Bhuvan J J

Automated tests to verify the scientific paper expert chatbot pipelines:
1. Local database loading
2. TF-IDF retrieval accuracy
3. Regex-based NER concept tagging
4. Extractive text summarization
"""

import os
import pandas as pd
from arxiv_loader import get_local_papers, initialize_database
from build_arxiv_index import ArXivRetriever
from nlp_utils import extract_concepts, summarize_text

def test_arxiv_pipeline():
    print("[Test] Initializing test database...")
    initialize_database()
    
    # 1. Load Local Papers test
    print("[Test] Loading local papers...")
    df = get_local_papers()
    assert not df.empty, "Database loaded as empty."
    assert "title" in df.columns, "Missing title column in database."
    assert "summary" in df.columns, "Missing summary column in database."
    print(f"  Passed: Database contains {len(df)} papers.")
    
    # 2. NER Concept Extraction test
    print("[Test] Testing Named Entity Recognition (NER)...")
    sample_text = (
        "We pre-train a Transformer model with self-attention using gradient descent "
        "and show how to mitigate overfitting using regularisation."
    )
    concepts = extract_concepts(sample_text)
    print(f"  Extracted concepts: {concepts}")
    
    assert "transformer" in concepts, "Failed to extract 'transformer'."
    assert "self-attention" in concepts, "Failed to extract 'self-attention'."
    assert "gradient descent" in concepts, "Failed to extract 'gradient descent'."
    assert "overfitting" in concepts, "Failed to extract 'overfitting'."
    print("  Passed: NER tagged key technical terms correctly.")

    # 3. TF-IDF Semantic Retrieval test
    print("[Test] Testing TF-IDF index retrieval...")
    retriever = ArXivRetriever()
    
    # Search for BERT specifically
    hits = retriever.retrieve("BERT pre-training bidirectional representations", threshold=0.05, top_k=2)
    assert len(hits) > 0, "Retriever returned no matches."
    assert "BERT" in hits[0]["title"], f"Expected BERT paper as top hit, got: {hits[0]['title']}"
    assert hits[0]["similarity_score"] > 0.05, f"Similarity score too low: {hits[0]['similarity_score']}"
    print(f"  Passed: Query matched paper '{hits[0]['title']}' with score {hits[0]['similarity_score']:.4f}")

    # 4. Extractive Summarization test
    print("[Test] Testing Extractive Summarizer...")
    text_to_summarize = (
        "Deeper neural networks are more difficult to train. "
        "We present a residual learning framework to ease the training of networks that are substantially deeper. "
        "We explicitly reformulate the layers as learning residual functions. "
        "We provide comprehensive empirical evidence showing that these residual networks are easier to optimize."
    )
    summary = summarize_text(text_to_summarize, num_sentences=2)
    print(f"  Generated Summary: {summary}")
    assert len(summary) > 0, "Extracted summary is empty."
    # Ensure it's shorter than the original text
    assert len(summary.split(". ")) < len(text_to_summarize.split(". ")), "Summary did not compress text."
    print("  Passed: Summarization compressed text correctly.")
    
    print("\n[Success] All ArXiv Expert Chatbot unit tests completed successfully!")

if __name__ == "__main__":
    test_arxiv_pipeline()
