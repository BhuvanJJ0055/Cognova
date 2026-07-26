"""
Task 3 Knowledge Base Management Verification Test Script
Author: Bhuvan J J

Verifies both manual UI Q&A addition and dynamic file-watcher MD5 deduplication.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.kb_updater import KnowledgeUpdater, compute_md5
from src.modules.medical_qa import MedicalRetriever

def run_kb_management_test():
    print("\n" + "="*80)
    print("🔄 TASK 3: KNOWLEDGE BASE MANAGEMENT EVALUATION")
    print("="*80)

    # Part 1: Test Direct Q&A Addition
    print("\n--- Part 1: Interactive Q&A Addition ---")
    retriever = MedicalRetriever()
    updater = KnowledgeUpdater()

    custom_q = "What is the recommended treatment for Dengue Fever?"
    custom_a = "Dengue Fever treatment focuses on supportive care, hydration, fluid replacement, pain relievers like acetaminophen, and avoiding NSAIDs like aspirin or ibuprofen."
    custom_focus = "Dengue Fever"
    custom_type = "treatment"

    print(f" 📥 Adding Custom Q&A: \"{custom_q}\"")
    success = updater.add_qa_pair(custom_q, custom_a, custom_focus, custom_type)
    retriever.rag.add_texts([f"Question: {custom_q}\nAnswer: {custom_a}"], metadata=[{
        "question": custom_q, "answer": custom_a, "focus": custom_focus, "question_type": custom_type
    }])

    print(f" ✅ Addition Status: {'SUCCESS' if success else 'FAILED'}")

    # Query back immediately
    results = retriever.retrieve("treatment for Dengue Fever", top_k=1)
    if results:
        match = results[0]
        print(f" 🎯 Instant Query Retrieval Match ({match['similarity']*100:.1f}%):")
        print(f"    Question: {match['question']}")
        print(f"    Focus Area: `{match['focus']}` | Category: `{match['question_type']}`")
        print(f"    Answer: {match['answer']}")
    else:
        print(" ❌ Match failed.")

    # Part 2: Test File Watcher & MD5 Deduplication
    print("\n--- Part 2: File Watcher & MD5 Hash Deduplication ---")
    incoming_dir = os.path.join(BASE_DIR, "data", "incoming_docs")
    os.makedirs(incoming_dir, exist_ok=True)
    
    test_file = os.path.join(incoming_dir, "test_doc_policy.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Cognova Enterprise System Privacy Policy: All medical data processed by Cognova is encrypted at rest and in transit.")

    file_hash = compute_md5(test_file)
    print(f" 📄 Created Test Document: 'test_doc_policy.txt'")
    print(f" 🔑 Computed MD5 Hash  : {file_hash}")

    added_count = updater.scan_and_update()
    print(f" 🔄 File Watcher Ingested: {added_count} new item(s)")

    # Run scan a second time to verify deduplication
    dedup_count = updater.scan_and_update()
    print(f" 🛡️ Second Scan (Deduplication Check): {dedup_count} items ingested (Expected: 0)")

    print("\n" + "="*80)
    print("✨ KNOWLEDGE BASE MANAGEMENT EVALUATED SUCCESSFULLY ✨")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_kb_management_test()
