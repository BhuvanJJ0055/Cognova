"""
Core RAG Pipeline - Base Indexing, Retrieval, and Response Generation
Author: Bhuvan J J

Core RAG architecture powering the Cognova platform.
Provides text chunking, TF-IDF vector embedding, index serialization,
similarity relevance thresholding, fallback keyword matching, and single top-match response synthesis.
"""

import os
import re
from typing import Optional, List, Any
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGPipeline:
    """Unified document vector store, retriever, and RAG answer engine."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None
    ):
        self.index_path = index_path
        self.text_fields = text_fields or ["text"]
        self.metadata_fields = metadata_fields or ["text"]
        self.vectorizer = None
        self.tfidf_matrix = None
        self.metadata: List[Any] = []

        if self.index_path and os.path.exists(self.index_path):
            self.load_index()

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        text_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None
    ):
        """Fits TF-IDF vectorizer on specified dataframe columns and stores metadata records."""
        if text_fields:
            self.text_fields = text_fields
        if metadata_fields:
            self.metadata_fields = metadata_fields

        df = df.fillna("")
        combined_texts = []
        for _, row in df.iterrows():
            parts = [str(row[field]) for field in self.text_fields if field in row and str(row[field]).strip()]
            combined_texts.append(" ".join(parts))

        if not combined_texts:
            return 0

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_texts)
        self.metadata = df.to_dict(orient="records")

        if self.index_path:
            self.save_index()

        return len(self.metadata)

    def add_texts(self, texts: List[str], metadata: Optional[List[dict]] = None):
        """Appends new text documents dynamically and refits the vector store."""
        if not texts:
            return 0

        new_records = []
        for idx, t in enumerate(texts):
            rec = {"text": t}
            if metadata and idx < len(metadata):
                rec.update(metadata[idx])
            new_records.append(rec)

        all_records = self.metadata + new_records
        df = pd.DataFrame(all_records)
        return self.build_from_dataframe(df, text_fields=self.text_fields, metadata_fields=self.metadata_fields)

    def query(self, query_text: str, top_k: int = 3, threshold: float = 0.00):
        """Calculates similarity scores and returns matching document records."""
        if self.vectorizer is None or self.tfidf_matrix is None or not self.metadata:
            return []

        try:
            query_vec = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = similarities.argsort()[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > threshold:
                    item = dict(self.metadata[idx])
                    item["score"] = score
                    results.append(item)
        except Exception:
            results = []

        # Fallback keyword overlap matcher if TF-IDF score returned empty results
        if not results and self.metadata:
            query_words = set(re.findall(r'\b\w+\b', query_text.lower()))
            stop_words = {"what", "are", "the", "of", "and", "is", "for", "in", "to", "how"}
            meaningful_words = query_words - stop_words

            scored_meta = []
            for item in self.metadata:
                text_content = " ".join([str(v) for v in item.values()]).lower()
                item_words = set(re.findall(r'\b\w+\b', text_content))
                overlap = len(meaningful_words.intersection(item_words))
                if overlap > 0:
                    scored_meta.append((overlap, item))

            scored_meta.sort(key=lambda x: x[0], reverse=True)
            for ov, item in scored_meta[:top_k]:
                m_copy = dict(item)
                m_copy["score"] = min(0.60 + ov * 0.1, 1.0)
                results.append(m_copy)

        return results

    def answer(self, query: str, threshold: float = 0.00):
        """
        Retrieves top match, evaluates relevance, and synthesizes structured response.
        Returns: (response_text, top_match_dict, is_relevant_bool, score_float)
        """
        results = self.query(query, top_k=1, threshold=threshold)
        if not results:
            return self.build_fallback_response(query, mood="calm"), None, False, 0.0

        top_match = results[0]
        score = top_match.get("score", 0.0)

        answer_body = top_match.get("answer") or top_match.get("text") or top_match.get("summary") or ""
        return answer_body, top_match, True, score

    def build_fallback_response(self, query: str, mood: str = "calm") -> str:
        """Generates a sentiment-tailored fallback message when no relevant chunk matches."""
        if mood == "upset":
            return (
                "I am really sorry for the trouble you are experiencing. "
                "I couldn't find an exact answer in our knowledge base for your query. "
                "Please connect with our support team directly for immediate assistance."
            )
        elif mood == "happy":
            return (
                "Thanks for reaching out! I couldn't locate specific details for that query in our knowledge base, "
                "but I'd be happy to assist if you have any other questions."
            )
        else:
            return "I could not locate specific guidelines for that query in the index."

    def save_index(self):
        """Serializes vectorizer, TF-IDF matrix, and metadata to disk."""
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            joblib.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "metadata": self.metadata,
                "text_fields": self.text_fields,
                "metadata_fields": self.metadata_fields
            }, self.index_path)

    def load_index(self, path: Optional[str] = None):
        """Loads serialized vectorizer, TF-IDF matrix, and metadata from disk."""
        target_path = path or self.index_path
        if target_path and os.path.exists(target_path):
            data = joblib.load(target_path)
            self.vectorizer = data.get("vectorizer")
            self.tfidf_matrix = data.get("tfidf_matrix")
            self.metadata = data.get("metadata", [])
            self.text_fields = data.get("text_fields", self.text_fields)
            self.metadata_fields = data.get("metadata_fields", self.metadata_fields)
            return len(self.metadata)
        return 0
