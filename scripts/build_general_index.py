"""
CLI Script to build and serialize a general knowledge base index.
Author: Bhuvan J J

Usage:
    python scripts/build_general_index.py
"""

import os
import sys
import argparse
import pandas as pd

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.core.rag_pipeline import RAGPipeline

DOCS_DIR = os.path.join(BASE_DIR, "data", "general_docs")
INDEX_PATH = os.path.join(BASE_DIR, "data", "general_kb_index", "general_index.joblib")


def build_general_index():
    print(f"[Info] Scanning documents in {DOCS_DIR}...")
    documents = []

    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "filename": filename,
                        "text": content,
                        "title": os.path.splitext(filename)[0].replace("_", " ").title()
                    })

    if not documents:
        # Default sample documents
        documents = [
            {
                "filename": "sample_faq.txt",
                "title": "Cognova System FAQ",
                "text": "Cognova is a unified multi-agent GenAI platform providing Sentiment-Aware Customer Support, Medical Q&A, ArXiv Paper Summarization, Multimodal Vision, and Multilingual translation."
            }
        ]

    pipeline = RAGPipeline(index_path=INDEX_PATH)
    count = pipeline.build_from_texts(documents)
    print(f"[Success] Built and saved general index with {count} documents to {INDEX_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build General Knowledge Base Index")
    args = parser.parse_args()
    build_general_index()
