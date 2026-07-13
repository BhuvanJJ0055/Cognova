"""
Task 3 - build_arxiv_index.py
Author: Bhuvan J J

Implements the retrieval engine for ArXiv scientific papers.
Computes TF-IDF representations over paper titles + abstracts, saves the fitted model,
and performs cosine similarity queries to return relevant papers.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "arxiv_cs_papers.csv")
INDEX_PATH = os.path.join(DATA_DIR, "arxiv_retriever_index.joblib")

class ArXivRetriever:
    """Manages paper vectorization, index storage, and semantic retrieval."""

    def __init__(self, index_path=INDEX_PATH, fallback_csv_path=CSV_PATH):
        self.index_path = index_path
        self.fallback_csv_path = fallback_csv_path
        self.vectorizer = None
        self.tfidf_matrix = None
        self.metadata = []

        if os.path.exists(self.index_path):
            self.load_index()
        else:
            print(f"[Warning] Index file {self.index_path} not found. Building from CSV...")
            if os.path.exists(self.fallback_csv_path):
                self.build_and_save_index(self.fallback_csv_path)
            else:
                # Initialize base database and try again
                from arxiv_loader import initialize_database
                initialize_database()
                if os.path.exists(self.fallback_csv_path):
                    self.build_and_save_index(self.fallback_csv_path)
                else:
                    raise FileNotFoundError(f"Cannot find or build papers database at {self.fallback_csv_path}")

    def build_and_save_index(self, csv_path):
        """Fits TF-IDF vectorizer on paper title + summary, builds index, and serializes."""
        print(f"[Info] Building ArXiv index from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Fill NaN values
        df = df.fillna("")
        df["title"] = df["title"].astype(str)
        df["summary"] = df["summary"].astype(str)
        df["authors"] = df["authors"].astype(str)
        df["primary_category"] = df["primary_category"].astype(str)
        df["published"] = df["published"].astype(str)
        df["url"] = df["url"].astype(str)
        
        # Combine title and summary for rich semantic indexing
        # Double weight on title to emphasize title match over abstract text
        corpus_texts = df["title"] + " " + df["title"] + " " + df["summary"]
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        
        # Fit vectorizer and transform text
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        
        # Save metadata
        self.metadata = df[["id", "title", "authors", "summary", "published", "primary_category", "url"]].to_dict(orient="records")
        
        # Serialize the built index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "metadata": self.metadata
        }, self.index_path)
        
        print(f"[Success] ArXiv retriever index containing {len(self.metadata)} papers saved to {self.index_path}")

    def load_index(self):
        """Loads fitted vectorizer and TF-IDF matrix from joblib file."""
        try:
            payload = joblib.load(self.index_path)
            self.vectorizer = payload["vectorizer"]
            self.tfidf_matrix = payload["tfidf_matrix"]
            self.metadata = payload["metadata"]
            print(f"[Success] Loaded ArXiv retrieval index containing {len(self.metadata)} papers.")
        except Exception as e:
            print(f"[Error] Failed to load ArXiv index from {self.index_path}: {e}")
            raise e

    def retrieve(self, query, threshold=0.08, top_k=3):
        """
        Retrieves top_k relevant papers matching the query.
        Returns a list of dictionaries with matching paper details and similarity scores.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Rank similarity indices descending
        ranked_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            
            # Stop if score falls below similarity cutoff
            if score < threshold:
                break
                
            match = self.metadata[idx].copy()
            match["similarity_score"] = score
            results.append(match)
            
            if len(results) >= top_k:
                break
                
        return results

if __name__ == "__main__":
    retriever = ArXivRetriever()
    test_q = "self-attention mechanism in language processing"
    print(f"\n[Testing Retriever] Query: '{test_q}'")
    hits = retriever.retrieve(test_q, threshold=0.05, top_k=2)
    for i, hit in enumerate(hits):
        print(f"\nMatch #{i+1} (Score: {hit['similarity_score']:.4f})")
        print(f"Title: {hit['title']} ({hit['primary_category']})")
        print(f"Authors: {hit['authors']}")
        print(f"Abstract: {hit['summary'][:150]}...")
