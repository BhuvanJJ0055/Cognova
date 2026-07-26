"""
Task 3 - Dynamic Knowledge Base Updater Module
Author: Bhuvan J J

Monitors data/incoming_docs/ and sources in data/kb_sources.json,
hashes file content via MD5 to deduplicate, runs background periodic updates,
and dynamically expands the vector database with new information in real-time.
"""

import os
import json
import hashlib
import time
import threading
import pandas as pd
from typing import Optional, List

try:
    from src.core.rag_pipeline import RAGPipeline
except ImportError:
    from core.rag_pipeline import RAGPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming_docs")
CONFIG_PATH = os.path.join(DATA_DIR, "kb_sources.json")
HASH_STORE_PATH = os.path.join(DATA_DIR, "processed_hashes.json")
GENERAL_INDEX_PATH = os.path.join(DATA_DIR, "general_kb_index", "general_index.joblib")


def compute_md5(file_path: str) -> str:
    """Computes MD5 hash of file content for deduplication."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


class KnowledgeUpdater:
    """Dynamic file watcher, periodic background scheduler, and RAG index updater."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.incoming_dir = os.path.join(data_dir, "incoming_docs")
        self.config_path = os.path.join(data_dir, "kb_sources.json")
        self.hash_store_path = os.path.join(data_dir, "processed_hashes.json")
        self.processed_hashes = self._load_hashes()
        self.rag = RAGPipeline(index_path=GENERAL_INDEX_PATH)
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def _load_hashes(self) -> set:
        """Loads processed file MD5 hashes from disk."""
        if os.path.exists(self.hash_store_path):
            try:
                with open(self.hash_store_path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _save_hashes(self):
        """Saves processed file MD5 hashes to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.hash_store_path, "w", encoding="utf-8") as f:
            json.dump(list(self.processed_hashes), f, indent=2)

    def scan_and_update(self) -> int:
        """Scans incoming directory, computes hashes, dedupes, and updates general RAG index."""
        if not os.path.exists(self.incoming_dir):
            os.makedirs(self.incoming_dir, exist_ok=True)
            return 0

        new_texts = []
        new_metadata = []

        for filename in os.listdir(self.incoming_dir):
            if filename.startswith(".") or filename.lower().endswith(".txt.readme"):
                continue

            fp = os.path.join(self.incoming_dir, filename)
            if os.path.isfile(fp) and filename.endswith((".txt", ".csv", ".json")):
                file_hash = compute_md5(fp)
                if file_hash in self.processed_hashes:
                    continue

                self.processed_hashes.add(file_hash)

                if filename.endswith(".txt"):
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                        if text:
                            new_texts.append(text)
                            new_metadata.append({"source_file": filename, "hash": file_hash})

                elif filename.endswith(".csv"):
                    try:
                        df = pd.read_csv(fp)
                        for _, row in df.iterrows():
                            content = " ".join([str(v) for v in row.values if str(v).strip()])
                            if content:
                                new_texts.append(content)
                                new_metadata.append({"source_file": filename, "hash": file_hash})
                    except Exception:
                        pass

                elif filename.endswith(".json"):
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    text_val = json.dumps(item)
                                    new_texts.append(text_val)
                                    new_metadata.append({"source_file": filename, "hash": file_hash})
                    except Exception:
                        pass

        if new_texts:
            self.rag.add_texts(new_texts, metadata=new_metadata)
            self._save_hashes()
            print(f"[KnowledgeUpdater] Added {len(new_texts)} new items to general index.")
            return len(new_texts)

        return 0

    def add_qa_pair(self, question: str, answer: str, focus: str = "General", qtype: str = "general") -> bool:
        """Adds custom Q&A pair directly to the RAG pipeline."""
        content = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
        meta = {"question": question.strip(), "answer": answer.strip(), "focus": focus.strip(), "question_type": qtype.strip()}
        self.rag.add_texts([content], metadata=[meta])
        return True

    def start_periodic_updater(self, interval_seconds: int = 30):
        """Starts a background daemon thread that periodically updates the vector store."""
        if self._is_running:
            return

        self._is_running = True

        def _loop():
            while self._is_running:
                try:
                    self.scan_and_update()
                except Exception as e:
                    print(f"[KnowledgeUpdater Error]: {e}")
                time.sleep(interval_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        print(f"[KnowledgeUpdater] Periodic background updater started (Interval: {interval_seconds}s).")

    def stop_periodic_updater(self):
        """Stops the background updater thread."""
        self._is_running = False


if __name__ == "__main__":
    updater = KnowledgeUpdater()
    added = updater.scan_and_update()
    print(f"Incremental update finished. Items added: {added}")
