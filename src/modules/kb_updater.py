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

    def __init__(self, data_dir: str = DATA_DIR, target_index_path: Optional[str] = None):
        self.data_dir = data_dir
        self.incoming_dir = os.path.join(data_dir, "incoming_docs")
        self.config_path = os.path.join(data_dir, "kb_sources.json")
        self.hash_store_path = os.path.join(data_dir, "processed_hashes.json")
        self.processed_hashes = self._load_hashes()
        
        # Priority target index
        default_medical_index = os.path.join(BASE_DIR, "Medical_QA_Chatbot", "data", "retriever_index.joblib")
        if target_index_path and os.path.exists(target_index_path):
            self.target_index_path = target_index_path
        elif os.path.exists(default_medical_index):
            self.target_index_path = default_medical_index
        else:
            self.target_index_path = GENERAL_INDEX_PATH

        self.rag = RAGPipeline(index_path=self.target_index_path)
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self.last_scan_time = None
        self.total_records_added = 0

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

    def scan_and_update(self, target_rag: Optional[RAGPipeline] = None) -> int:
        """Scans incoming directory, computes hashes, dedupes, and dynamically updates RAG index."""
        if not os.path.exists(self.incoming_dir):
            os.makedirs(self.incoming_dir, exist_ok=True)
            return 0

        active_rag = target_rag or self.rag
        self.last_scan_time = time.strftime("%Y-%m-%d %H:%M:%S")

        new_texts = []
        new_metadata = []

        for filename in os.listdir(self.incoming_dir):
            if filename.startswith(".") or filename.lower().endswith(".txt.readme"):
                continue

            fp = os.path.join(self.incoming_dir, filename)
            if os.path.isfile(fp) and filename.lower().endswith((".txt", ".csv", ".json", ".md")):
                file_hash = compute_md5(fp)
                if file_hash in self.processed_hashes:
                    continue

                self.processed_hashes.add(file_hash)

                if filename.lower().endswith((".txt", ".md")):
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                        if text:
                            new_texts.append(text)
                            new_metadata.append({
                                "question": f"Document Snippet ({filename})",
                                "answer": text,
                                "focus": "General Document",
                                "question_type": "information",
                                "source_file": filename,
                                "hash": file_hash
                            })

                elif filename.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(fp)
                        for _, row in df.iterrows():
                            content = " ".join([str(v) for v in row.values if str(v).strip()])
                            if content:
                                q_val = str(row.get("question", f"CSV Entry ({filename})"))
                                a_val = str(row.get("answer", content))
                                focus_val = str(row.get("focus", "General"))
                                qtype_val = str(row.get("question_type", "general"))
                                new_texts.append(content)
                                new_metadata.append({
                                    "question": q_val,
                                    "answer": a_val,
                                    "focus": focus_val,
                                    "question_type": qtype_val,
                                    "source_file": filename,
                                    "hash": file_hash
                                })
                    except Exception:
                        pass

                elif filename.lower().endswith(".json"):
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        q_val = item.get("question", "JSON Query")
                                        a_val = item.get("answer", json.dumps(item))
                                        focus_val = item.get("focus", "General")
                                        qtype_val = item.get("question_type", "general")
                                    else:
                                        q_val = "JSON Entry"
                                        a_val = str(item)
                                        focus_val = "General"
                                        qtype_val = "general"

                                    new_texts.append(a_val)
                                    new_metadata.append({
                                        "question": q_val,
                                        "answer": a_val,
                                        "focus": focus_val,
                                        "question_type": qtype_val,
                                        "source_file": filename,
                                        "hash": file_hash
                                    })
                    except Exception:
                        pass

        if new_texts:
            active_rag.add_texts(new_texts, metadata=new_metadata)
            if active_rag.index_path:
                active_rag.save_index()
            self._save_hashes()
            added_count = len(new_texts)
            self.total_records_added += added_count
            print(f"[KnowledgeUpdater] Dynamically added {added_count} new items to vector store.")
            return added_count

        return 0

    def add_qa_pair(
        self,
        question: str,
        answer: str,
        focus: str = "General",
        qtype: str = "general",
        target_rag: Optional[RAGPipeline] = None
    ) -> bool:
        """Adds custom Q&A pair directly to the active RAG pipeline and persists index."""
        content = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
        meta = {
            "question": question.strip(),
            "answer": answer.strip(),
            "focus": focus.strip(),
            "question_type": qtype.strip()
        }
        active_rag = target_rag or self.rag
        active_rag.add_texts([content], metadata=[meta])
        if active_rag.index_path:
            active_rag.save_index()
        self.total_records_added += 1
        return True

    def start_periodic_updater(self, interval_seconds: int = 30, target_rag: Optional[RAGPipeline] = None):
        """Starts a background daemon thread that periodically updates the vector store."""
        if self._is_running:
            return

        self._is_running = True

        def _loop():
            while self._is_running:
                try:
                    self.scan_and_update(target_rag=target_rag)
                except Exception as e:
                    print(f"[KnowledgeUpdater Error]: {e}")
                time.sleep(interval_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        print(f"[KnowledgeUpdater] Periodic background updater started (Interval: {interval_seconds}s).")

    def stop_periodic_updater(self):
        """Stops the background updater thread."""
        self._is_running = False

    def is_running(self) -> bool:
        """Returns background scheduler status."""
        return self._is_running and self._thread is not None and self._thread.is_alive()


if __name__ == "__main__":
    updater = KnowledgeUpdater()
    added = updater.scan_and_update()
    print(f"Incremental update finished. Items added: {added}")
