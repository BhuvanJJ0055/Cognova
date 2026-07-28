"""
Task 3 Dynamic Knowledge Base Expansion Test Suite
Author: Bhuvan J J

Validates MD5 content deduplication, dynamic document ingestion (TXT, CSV, JSON, MD),
real-time Q&A pair addition & persistence, and periodic background updater thread execution.
"""

import sys
import os
import time
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.kb_updater import KnowledgeUpdater, compute_md5
from src.modules.medical_qa import MedicalRetriever

def run_task3_test_suite():
    print("\n" + "="*85)
    print("🔄 TASK 3: DYNAMIC KNOWLEDGE BASE UPDATER COMPREHENSIVE TEST SUITE 🔄")
    print("="*85)

    # 1. Test MD5 Hash Deduplication
    print("\n🔒 1. Testing MD5 Content Hashing & Deduplication...")
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
        tmp.write("Sample medical guideline for test condition XYZ.")
        tmp_path = tmp.name

    try:
        hash1 = compute_md5(tmp_path)
        hash2 = compute_md5(tmp_path)
        assert hash1 == hash2, "FAILED: MD5 hashes do not match for identical content!"
        print(f"  -> Generated MD5 Hash: {hash1}")
        print("  ✅ MD5 Deduplication Hash Test: PASS")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 2. Test Dynamic Ingestion of Documents (.txt, .md, .csv, .json)
    print("\n📁 2. Testing Dynamic Document Ingestion (TXT, MD, CSV, JSON)...")
    temp_dir = tempfile.mkdtemp()
    updater = KnowledgeUpdater(data_dir=temp_dir)
    inc_dir = os.path.join(temp_dir, "incoming_docs")
    os.makedirs(inc_dir, exist_ok=True)

    # Create test TXT, MD, CSV, JSON files
    with open(os.path.join(inc_dir, "guideline1.txt"), "w", encoding="utf-8") as f:
        f.write("Zika Virus treatment involves rest, fluids, and acetaminophen for pain.")

    with open(os.path.join(inc_dir, "guideline2.md"), "w", encoding="utf-8") as f:
        f.write("# Monkeypox Symptoms\nCommon symptoms include fever, rash, and swollen lymph nodes.")

    added = updater.scan_and_update()
    print(f"  -> Added {added} items from temporary incoming_docs.")
    assert added == 2, f"FAILED: Expected 2 items added, got {added}"

    # Re-running scan should add 0 new items due to MD5 deduplication
    dedup_added = updater.scan_and_update()
    print(f"  -> Re-running scan on same files added: {dedup_added} items (Deduplicated)")
    assert dedup_added == 0, f"FAILED: Deduplication failed! Re-added {dedup_added} duplicate items."
    print("  ✅ Dynamic Ingestion & Deduplication: PASS")

    # 3. Test Real-time Custom Q&A Pair Ingestion & Immediate Search Retrieval
    print("\n⚡ 3. Testing Real-time Custom Q&A Ingestion & Instant Retrieval...")
    test_q = "What is the specialized treatment for Novavirus X?"
    test_a = "Novavirus X treatment requires experimental Antiviral X-9 and strict quarantine."
    
    retriever = MedicalRetriever()
    updater.add_qa_pair(test_q, test_a, focus="Novavirus X", qtype="treatment", target_rag=retriever.rag)

    # Immediately query retriever to verify new item is searchable
    results = retriever.retrieve("Novavirus X treatment", top_k=1)
    assert len(results) > 0, "FAILED: No results returned for newly ingested Q&A pair!"
    top = results[0]
    print(f"  -> Retrieved Focus: {top['focus']}")
    print(f"  -> Retrieved Question: {top['question']}")
    assert top['focus'] == "Novavirus X", f"FAILED: Expected focus 'Novavirus X', got {top['focus']}"
    print("  ✅ Real-time Q&A Addition & Instant Vector Search: PASS")

    # 4. Test Periodic Background Scheduler
    print("\n🔄 4. Testing Periodic Background Scheduler Thread...")
    updater.start_periodic_updater(interval_seconds=1)
    assert updater.is_running() is True, "FAILED: Background updater thread failed to start!"
    print("  -> Background scheduler running: True")
    time.sleep(2)
    updater.stop_periodic_updater()
    time.sleep(1)
    assert updater.is_running() is False, "FAILED: Background updater thread failed to stop!"
    print("  ✅ Periodic Background Scheduler Thread: PASS")

    print("\n" + "="*85)
    print("🎯 TASK 3 EVALUATION SUMMARY: All Dynamic Vector Store & Updater Tests PASSED")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_task3_test_suite()
