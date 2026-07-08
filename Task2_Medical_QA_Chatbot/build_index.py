"""
Task 2 - build_index.py
Author: Bhuvan J J

This script implements the retrieval mechanism for our Medical Q&A Chatbot.
It uses TF-IDF vectorization + Cosine Similarity over the question dataset.
This approach is chosen because:
- It is fast, deterministic, and has zero external dependencies (no API keys, no heavy neural models to download).
- It is highly explainable, returning direct word overlaps and exact match indicators.
- It is memory-efficient and can run comfortably on simple local instances.

The script provides a serialized 'MedicalRetriever' class which saves:
1. The fitted TfidfVectorizer.
2. The document-term TF-IDF matrix representing all stored questions.
3. Parallel lists of metadata (raw questions, answers, disease focus, and question types).
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "medquad_qa.csv")
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "sample_medquad_qa.csv")
INDEX_SAVE_PATH = os.path.join(DATA_DIR, "retriever_index.joblib")


class MedicalRetriever:
    """Manages text representation, index loading/saving, and question retrieval."""

    def __init__(self, index_path=INDEX_SAVE_PATH, fallback_csv_path=SAMPLE_CSV_PATH):
        self.index_path = index_path
        self.vectorizer = None
        self.tfidf_matrix = None
        self.metadata = []  # List of dicts containing question, answer, focus, type

        # Attempt to load serialized index. If missing, compile from csv.
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            print(f"[Warning] Index file {self.index_path} not found. Attempting to build from CSV...")
            csv_candidates = [
                os.path.join(DATA_DIR, "medical_qa.csv"),
                DEFAULT_CSV_PATH,
                fallback_csv_path
            ]
            csv_to_use = None
            for path in csv_candidates:
                if os.path.exists(path):
                    csv_to_use = path
                    break
            
            if csv_to_use:
                self.build_and_save_index(csv_to_use)
            else:
                raise FileNotFoundError(
                    f"No index file found at {self.index_path} and no CSV found to rebuild."
                )

    def build_and_save_index(self, csv_path):
        """Fits TF-IDF vectorizer on questions, builds index, and serializes."""
        print(f"[Info] Building retrieval index from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Clean data
        df = df.dropna(subset=["question", "answer"])
        df["question"] = df["question"].astype(str)
        df["answer"] = df["answer"].astype(str)
        
        # Populate missing columns dynamically if they do not exist
        if "focus" not in df.columns:
            df["focus"] = "General"
        if "question_type" not in df.columns:
            df["question_type"] = "general"
            
        df["focus"] = df["focus"].fillna("General").astype(str)
        df["question_type"] = df["question_type"].fillna("general").astype(str)
        
        # Initialize TF-IDF Vectorizer
        # We use unigrams and bigrams, lowercase, and english stop words.
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        
        # Fit vectorizer and transform questions
        self.tfidf_matrix = self.vectorizer.fit_transform(df["question"])
        
        # Gather metadata for alignment
        self.metadata = df[["question", "answer", "focus", "question_type"]].to_dict(orient="records")
        
        # Serialize the built index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "metadata": self.metadata
        }, self.index_path)
        
        print(f"[Success] Retrieval index containing {len(self.metadata)} documents saved to {self.index_path}")

    def load_index(self):
        """Loads fitted vectorizer and TF-IDF matrix from joblib file."""
        try:
            payload = joblib.load(self.index_path)
            self.vectorizer = payload["vectorizer"]
            self.tfidf_matrix = payload["tfidf_matrix"]
            self.metadata = payload["metadata"]
            print(f"[Success] Loaded retrieval index containing {len(self.metadata)} QA records from {self.index_path}")
        except Exception as e:
            print(f"[Error] Failed to load index from {self.index_path}: {e}")
            raise e

    def retrieve(self, query, threshold=0.15, top_k=3):
        """
        Retrieves top_k relevant Q&A pairs matching the query.
        Returns a list of dictionaries with matching QA pairs and their cosine similarities.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarities between query and all indexed questions
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Rank similarity indices in descending order
        ranked_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            
            # Stop if score falls below similarity cutoff threshold
            if score < threshold:
                break
                
            match = self.metadata[idx].copy()
            match["similarity_score"] = score
            results.append(match)
            
            if len(results) >= top_k:
                break
                
        return results


if __name__ == "__main__":
    # If run standalone, build the index using whatever CSV is available
    csv_path = DEFAULT_CSV_PATH if os.path.exists(DEFAULT_CSV_PATH) else SAMPLE_CSV_PATH
    retriever = MedicalRetriever(fallback_csv_path=csv_path)
    
    # Test query
    test_q = "What are the treatments for Lupus?"
    print(f"\n[Testing Retriever] Query: '{test_q}'")
    hits = retriever.retrieve(test_q, threshold=0.1, top_k=2)
    for i, hit in enumerate(hits):
        print(f"\nMatch #{i+1} (Similarity: {hit['similarity_score']:.4f})")
        print(f"Focus: {hit['focus']} | Type: {hit['question_type']}")
        print(f"Q: {hit['question']}")
        print(f"A: {hit['answer'][:150]}...")
