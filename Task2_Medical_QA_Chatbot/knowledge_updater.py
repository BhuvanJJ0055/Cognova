"""
Task 3 - knowledge_updater.py
Author: Bhuvan J J / Antigravity AI

This script manages the dynamic knowledge base expansion system for our Medical Q&A Chatbot.
It handles:
1. Directory structures for incoming and processed data updates.
2. Configuration loading and saving (sources, sync intervals, scheduler toggle).
3. File parsing for CSV, JSON, and MedQuAD-compatible XML formats.
4. Remote file downloading and ingestion.
5. Ingestion of manual inputs directly from the Streamlit UI.
6. Deduplication and merging of new Q&As with the existing index, triggering index rebuilds.
7. Background thread scheduling for periodic polling.
"""

import os
import re
import json
import time
import shutil
import datetime
import threading
import requests
import pandas as pd
import xml.etree.ElementTree as ET

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PENDING_DIR = os.path.join(DATA_DIR, "pending_updates")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_updates")
CONFIG_PATH = os.path.join(DATA_DIR, "sources_config.json")
CUSTOM_CSV_PATH = os.path.join(DATA_DIR, "custom_updates.csv")
LOG_PATH = os.path.join(DATA_DIR, "update_history.json")
INDEX_PATH = os.path.join(DATA_DIR, "retriever_index.joblib")


class KnowledgeUpdater:
    """Manages custom Q&A sources, ingestion flow, deduplication, and index rebuilding."""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.pending_dir = os.path.join(data_dir, "pending_updates")
        self.processed_dir = os.path.join(data_dir, "processed_updates")
        self.config_path = os.path.join(data_dir, "sources_config.json")
        self.custom_csv_path = os.path.join(data_dir, "custom_updates.csv")
        self.log_path = os.path.join(data_dir, "update_history.json")
        self.index_path = os.path.join(data_dir, "retriever_index.joblib")

        self.initialize_directories()

    def initialize_directories(self):
        """Creates the data directories if they do not exist, and writes default config."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.pending_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # Backup pristine sample CSV if it exists and hasn't been backed up yet
        base_sample_path = os.path.join(self.data_dir, "sample_medquad_qa.csv")
        pristine_path = os.path.join(self.data_dir, "pristine_sample_medquad_qa.csv")
        if os.path.exists(base_sample_path) and not os.path.exists(pristine_path):
            try:
                shutil.copy(base_sample_path, pristine_path)
            except Exception as e:
                print(f"[Warning] Failed to backup pristine sample: {e}")

        # Create default config if missing
        if not os.path.exists(self.config_path):
            default_config = {
                "sync_interval_seconds": 60,
                "scheduler_active": True,
                "sources": [
                    {
                        "type": "folder",
                        "path": self.pending_dir
                    }
                ]
            }
            self.save_config(default_config)

    def load_config(self):
        """Loads configuration from JSON file."""
        if not os.path.exists(self.config_path):
            self.initialize_directories()
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load config from {self.config_path}: {e}")
            return {"sync_interval_seconds": 60, "scheduler_active": False, "sources": []}

    def save_config(self, config):
        """Saves configuration to JSON file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print(f"[Error] Failed to save config to {self.config_path}: {e}")
            return False

    def add_source(self, source_type, path_or_url):
        """Appends a new source to the configuration."""
        config = self.load_config()
        # Clean paths or URLs
        path_or_url = path_or_url.strip()
        
        # Check for duplicate
        for s in config.get("sources", []):
            if s.get("type") == source_type and (s.get("path") == path_or_url or s.get("url") == path_or_url):
                return False  # Already exists

        new_source = {"type": source_type}
        if source_type == "folder":
            new_source["path"] = path_or_url
        elif source_type == "url":
            new_source["url"] = path_or_url
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        config.setdefault("sources", []).append(new_source)
        return self.save_config(config)

    def remove_source(self, index):
        """Removes a source from the config by index."""
        config = self.load_config()
        sources = config.get("sources", [])
        if 0 <= index < len(sources):
            sources.pop(index)
            config["sources"] = sources
            return self.save_config(config)
        return False

    def load_custom_updates(self):
        """Loads existing custom updates from CSV."""
        if os.path.exists(self.custom_csv_path):
            try:
                df = pd.read_csv(self.custom_csv_path)
                # Ensure correct columns exist
                for col in ["focus", "question_type", "question", "answer"]:
                    if col not in df.columns:
                        df[col] = ""
                return df
            except Exception as e:
                print(f"[Error] Failed to load custom updates CSV: {e}")
        
        # Return empty DataFrame
        return pd.DataFrame(columns=["focus", "question_type", "question", "answer"])

    def save_custom_updates(self, df):
        """Saves custom updates to CSV."""
        try:
            df.to_csv(self.custom_csv_path, index=False, encoding="utf-8")
            return True
        except Exception as e:
            print(f"[Error] Failed to save custom updates to CSV: {e}")
            return False

    def get_active_csv_path(self):
        """Finds which database CSV is currently active on disk."""
        candidates = [
            os.path.join(self.data_dir, "medical_qa.csv"),
            os.path.join(self.data_dir, "medquad_qa.csv"),
            os.path.join(self.data_dir, "sample_medquad_qa.csv")
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return os.path.join(self.data_dir, "sample_medquad_qa.csv")

    def ingest_manual(self, focus, question_type, question, answer):
        """Adds a single manual Q&A entry, deduplicating and rebuilding the index."""
        focus = focus.strip()
        question_type = question_type.strip()
        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            return False, "Question and Answer cannot be empty."

        df_custom = self.load_custom_updates()
        
        # Check duplicate
        dup_mask = df_custom["question"].str.strip().str.lower() == question.lower()
        if dup_mask.any():
            # Update existing rather than duplicate
            df_custom.loc[dup_mask, "focus"] = focus
            df_custom.loc[dup_mask, "question_type"] = question_type
            df_custom.loc[dup_mask, "answer"] = answer
            action = "Updated existing custom entry"
        else:
            new_row = pd.DataFrame([{
                "focus": focus or "General",
                "question_type": question_type or "general",
                "question": question,
                "answer": answer
            }])
            df_custom = pd.concat([df_custom, new_row], ignore_index=True)
            action = "Added new custom entry"

        self.save_custom_updates(df_custom)
        
        # Sync immediately
        added_count = self.merge_and_rebuild()
        
        self.log_sync("manual_ingest", 1 if "new" in action else 0, f"{action}: '{question[:30]}...'")
        return True, f"Success: {action}."

    def parse_csv_file(self, file_path):
        """Parses Q&A rows from a CSV file. Expects columns: question, answer."""
        records = []
        try:
            df = pd.read_csv(file_path)
            # Normalize column names
            df.columns = [col.strip().lower() for col in df.columns]
            if "question" not in df.columns or "answer" not in df.columns:
                print(f"[Warning] CSV {file_path} missing required 'question' or 'answer' columns.")
                return []

            for _, row in df.iterrows():
                q = str(row["question"]).strip()
                a = str(row["answer"]).strip()
                if q and a:
                    records.append({
                        "focus": str(row.get("focus", "General")).strip(),
                        "question_type": str(row.get("question_type", "general")).strip(),
                        "question": q,
                        "answer": a
                    })
        except Exception as e:
            print(f"[Error] Parsing CSV file {file_path}: {e}")
        return records

    def parse_json_file(self, file_path):
        """Parses Q&A records from a JSON file. Supports a list of objects or a single object."""
        records = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert to list if single dict
            if isinstance(data, dict):
                data = [data]
                
            if isinstance(data, list):
                for item in data:
                    q = item.get("question", "").strip()
                    a = item.get("answer", "").strip()
                    if q and a:
                        records.append({
                            "focus": item.get("focus", "General").strip(),
                            "question_type": item.get("question_type", "general").strip(),
                            "question": q,
                            "answer": a
                        })
        except Exception as e:
            print(f"[Error] Parsing JSON file {file_path}: {e}")
        return records

    def parse_xml_file(self, file_path):
        """Parses a MedQuAD-conforming XML file."""
        records = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            focus_elem = root.find("Focus")
            focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else "General"
            
            for qa_pair in root.findall(".//QAPair"):
                q_elem = qa_pair.find("Question")
                a_elem = qa_pair.find("Answer")
                
                if q_elem is not None and a_elem is not None:
                    q = q_elem.text.strip() if q_elem.text else ""
                    a = a_elem.text.strip() if a_elem.text else ""
                    qtype = q_elem.get("qtype", "general").strip()
                    
                    if q and a:
                        records.append({
                            "focus": focus,
                            "question_type": qtype,
                            "question": q,
                            "answer": a
                        })
        except Exception as e:
            print(f"[Error] Parsing XML file {file_path}: {e}")
        return records

    def scan_local_folder(self, folder_path):
        """Scans folder for update files, ingests, and moves processed files."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            return []

        records = []
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            ext = os.path.splitext(file)[1].lower()[1:]
            file_records = []
            
            if ext == "csv":
                file_records = self.parse_csv_file(file_path)
            elif ext == "json":
                file_records = self.parse_json_file(file_path)
            elif ext == "xml":
                file_records = self.parse_xml_file(file_path)
            else:
                continue  # Unsupported extension
                
            if file_records:
                records.extend(file_records)
                print(f"[Ingestion] Successfully parsed {len(file_records)} Q&A pairs from {file}.")
            
            # Archive file to processed_updates folder to prevent duplicate scanning
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{timestamp}_{file}"
            os.makedirs(self.processed_dir, exist_ok=True)
            try:
                shutil.move(file_path, os.path.join(self.processed_dir, archive_name))
            except Exception as e:
                print(f"[Warning] Failed to archive file {file}: {e}")
                # Fallback: delete file if we can't move
                try:
                    os.remove(file_path)
                except:
                    pass
        return records

    def fetch_remote_url(self, url):
        """Downloads a remote file and attempts to parse it as CSV or JSON."""
        records = []
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # Guess type from headers or URL extension
            content_type = response.headers.get("Content-Type", "")
            is_json = "json" in content_type or url.lower().endswith(".json")
            is_csv = "csv" in content_type or url.lower().endswith(".csv")
            
            if is_json:
                data = response.json()
                if isinstance(data, dict):
                    data = [data]
                if isinstance(data, list):
                    for item in data:
                        q = item.get("question", "").strip()
                        a = item.get("answer", "").strip()
                        if q and a:
                            records.append({
                                "focus": item.get("focus", "General").strip(),
                                "question_type": item.get("question_type", "general").strip(),
                                "question": q,
                                "answer": a
                            })
            else:
                # Fallback / try CSV parsing
                import io
                df = pd.read_csv(io.StringIO(response.text))
                df.columns = [col.strip().lower() for col in df.columns]
                if "question" in df.columns and "answer" in df.columns:
                    for _, row in df.iterrows():
                        q = str(row["question"]).strip()
                        a = str(row["answer"]).strip()
                        if q and a:
                            records.append({
                                "focus": str(row.get("focus", "General")).strip(),
                                "question_type": str(row.get("question_type", "general")).strip(),
                                "question": q,
                                "answer": a
                            })
        except Exception as e:
            print(f"[Error] Failed to fetch remote url {url}: {e}")
        return records

    def sync_all_sources(self):
        """Polls all configured sources, parses new rows, deduplicates, and re-indexes."""
        config = self.load_config()
        sources = config.get("sources", [])
        new_records = []

        for source in sources:
            stype = source.get("type")
            if stype == "folder":
                folder_path = source.get("path", self.pending_dir)
                new_records.extend(self.scan_local_folder(folder_path))
            elif stype == "url":
                url = source.get("url")
                if url:
                    new_records.extend(self.fetch_remote_url(url))

        if not new_records:
            self.log_sync("scheduler", 0, "Polled sources: No new records found.")
            return 0

        # Load custom database
        df_custom = self.load_custom_updates()
        
        added_count = 0
        for rec in new_records:
            question = rec["question"]
            # Deduplicate against already added custom updates
            dup_mask = df_custom["question"].str.strip().str.lower() == question.strip().lower()
            if dup_mask.any():
                # Update answer
                df_custom.loc[dup_mask, "focus"] = rec["focus"]
                df_custom.loc[dup_mask, "question_type"] = rec["question_type"]
                df_custom.loc[dup_mask, "answer"] = rec["answer"]
            else:
                # Append new
                new_row = pd.DataFrame([rec])
                df_custom = pd.concat([df_custom, new_row], ignore_index=True)
                added_count += 1

        self.save_custom_updates(df_custom)

        # Merge and rebuild index
        self.merge_and_rebuild()

        self.log_sync("scheduler", added_count, f"Synced configured sources: Added {added_count} new entries.")
        return added_count

    def merge_and_rebuild(self):
        """Merges custom updates with base CSV and rebuilds the retriever TF-IDF matrix."""
        active_csv = self.get_active_csv_path()
        if not os.path.exists(active_csv):
            # Fallback to create sample CSV if it got deleted
            from build_index import SAMPLE_CSV_PATH
            active_csv = SAMPLE_CSV_PATH
            if not os.path.exists(active_csv):
                print(f"[Error] Active CSV path {active_csv} does not exist.")
                return 0

        # Load base CSV
        df_base = pd.read_csv(active_csv)
        
        # Load custom updates
        df_custom = self.load_custom_updates()
        
        if len(df_custom) == 0:
            return 0

        # Merge databases
        # Concatenate and drop duplicates by 'question', keeping the custom one (which allows overriding base facts if needed)
        df_merged = pd.concat([df_base, df_custom])
        df_merged = df_merged.drop_duplicates(subset=["question"], keep="last")
        
        # Save merged dataframe back to active path
        df_merged.to_csv(active_csv, index=False, encoding="utf-8")
        print(f"[Sync] Saved merged database ({len(df_merged)} rows) to {active_csv}")

        # Build retrieval index
        from build_index import MedicalRetriever
        retriever = MedicalRetriever(index_path=self.index_path, fallback_csv_path=active_csv)
        retriever.build_and_save_index(active_csv)
        
        return len(df_custom)

    def log_sync(self, trigger_type, added_count, details):
        """Appends a log record to the sync history JSON file."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "trigger": trigger_type,
            "added_count": added_count,
            "details": details
        }
        
        logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read sync log history: {e}")
                
        logs.append(log_entry)
        # Cap at last 100 entries to prevent files growing too large
        logs = logs[-100:]
        
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to write sync logs: {e}")

    def wipe_custom_knowledge(self):
        """Wipes the custom updates CSV and restores the base CSV to its original state (excluding custom)."""
        active_csv = self.get_active_csv_path()
        if not os.path.exists(active_csv):
            return False
            
        # Empty custom updates
        df_empty = pd.DataFrame(columns=["focus", "question_type", "question", "answer"])
        self.save_custom_updates(df_empty)
        
        # Parse XML or download base again to recreate clean CSV
        # Or, just drop any row from active CSV that does not exist in the master dataset.
        # But wait! A simpler, robust way is: if they are using sample_medquad_qa.csv,
        # we can rebuild it from XML or just overwrite it if we have a backup.
        # Actually, let's look at build_index.py. If we rebuild index with an empty custom CSV,
        # we still have custom rows in active_csv. How do we cleanly restore it?
        # Let's clean the active CSV by removing questions that aren't in the original.
        # How do we know what is original?
        # If medquad_qa.csv is the active CSV: we can parse XML files recursively to rebuild it!
        # If sample_medquad_qa.csv is active: we can just restore the first 40 sample pairs!
        # Let's implement a clean restore:
        try:
            print("[Restore] Restoring active database to pristine state...")
            if "sample_medquad_qa" in active_csv:
                # We can construct a list of standard sample rows or read them if we had a backup.
                # Or even simpler: we can rebuild it by filtering out any custom added rows.
                # But wait, how do we know which rows are custom?
                # We can just keep a copy of the pristine sample CSV if it's there.
                # Actually, when app initializes, we can save a copy `data/pristine_sample_medquad_qa.csv` or
                # simply restore by reading the XMLs if full data, or if sample, we can save a copy.
                # Wait! A pristine backup is the easiest:
                pristine_sample_path = os.path.join(self.data_dir, "pristine_sample_medquad_qa.csv")
                if os.path.exists(pristine_sample_path):
                    shutil.copy(pristine_sample_path, active_csv)
            else:
                # For full dataset, we can run the XML parser to rebuild the CSV cleanly
                from data_loader import build_dataframe_from_xml
                build_dataframe_from_xml() # Re-generates medquad_qa.csv cleanly
                
            # Rebuild index
            from build_index import MedicalRetriever
            retriever = MedicalRetriever(index_path=self.index_path, fallback_csv_path=active_csv)
            retriever.build_and_save_index(active_csv)
            
            self.log_sync("wipe", 0, "Restored database to clean base state. Wiped all custom knowledge.")
            return True
        except Exception as e:
            print(f"[Error] Failed to wipe custom knowledge: {e}")
            return False


def run_scheduler_loop(updater):
    """Loop function executed by background thread."""
    print("[Scheduler] Background thread scheduler loop started.")
    while True:
        # Load configuration on every cycle to check for dynamically modified parameters
        config = updater.load_config()
        
        # If scheduler is enabled, trigger synchronization check
        if config.get("scheduler_active", True):
            try:
                added = updater.sync_all_sources()
                if added > 0:
                    print(f"[Scheduler] Background sync found and processed {added} new records.")
            except Exception as e:
                print(f"[Scheduler] Error running background sync: {e}")
                
        # Sleep in short 1-second ticks so that the thread can shut down or adapt to interval edits immediately
        interval = config.get("sync_interval_seconds", 60)
        for _ in range(int(interval)):
            time.sleep(1)


def start_background_updater(updater):
    """Spawns the background scheduler thread if not already running."""
    # Check if thread is already running in current process
    for thread in threading.enumerate():
        if thread.name == "KnowledgeUpdaterScheduler":
            return thread
            
    scheduler_thread = threading.Thread(
        target=run_scheduler_loop, 
        args=(updater,), 
        name="KnowledgeUpdaterScheduler", 
        daemon=True
    )
    scheduler_thread.start()
    return scheduler_thread
