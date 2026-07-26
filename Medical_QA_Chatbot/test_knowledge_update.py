"""
Task 3 - test_knowledge_update.py
Author: Bhuvan J J / Antigravity AI

Automated unit tests to verify the dynamic knowledge base expansion system.
It tests:
1. Initialization of directories and default configurations.
2. Ingestion of manual entries.
3. Deduplication logic (handling duplicate questions).
4. Local file scanning (parsing XML and CSV files from a pending directory).
5. Dynamic index rebuilding and query retrieval verification.
6. Clean database restoration.
"""

import os
import shutil
import json
import pandas as pd
from knowledge_updater import KnowledgeUpdater
from build_index import MedicalRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "data", "test_knowledge_kb")


def test_dynamic_updater_pipeline():
    print("[Test] Setting up isolated test directory...")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)

    # 1. Create a dummy base CSV to represent our active database
    base_csv_path = os.path.join(TEST_DIR, "sample_medquad_qa.csv")
    base_data = pd.DataFrame([
        {
            "focus": "Influenza",
            "question_type": "symptoms",
            "question": "What are the symptoms of Influenza?",
            "answer": "Symptoms include fever, chills, cough, sore throat, and muscle aches."
        },
        {
            "focus": "Diabetes",
            "question_type": "treatment",
            "question": "How is diabetes treated?",
            "answer": "Treatment involves lifestyle changes, blood sugar monitoring, and insulin."
        }
    ])
    base_data.to_csv(base_csv_path, index=False, encoding="utf-8")
    print(f"  Created dummy base CSV at {base_csv_path}")

    # 2. Instantiate KnowledgeUpdater in test directory
    updater = KnowledgeUpdater(data_dir=TEST_DIR)
    print("  Initialized KnowledgeUpdater.")
    
    # Assert folders were created
    assert os.path.exists(updater.pending_dir), "Pending updates folder not created."
    assert os.path.exists(updater.processed_dir), "Processed updates folder not created."
    assert os.path.exists(updater.config_path), "Configuration JSON file not created."

    # 3. Test Manual Ingestion
    print("\n[Test] Running Manual Ingestion...")
    success, msg = updater.ingest_manual(
        focus="Cognova Fever",
        question_type="information",
        question="What is Cognova Fever?",
        answer="Cognova Fever is a mock disease causing extreme coding speeds."
    )
    assert success, f"Manual ingestion failed: {msg}"
    
    df_custom = updater.load_custom_updates()
    assert len(df_custom) == 1, f"Expected 1 custom update, found {len(df_custom)}"
    assert df_custom.iloc[0]["focus"] == "Cognova Fever"
    assert os.path.exists(updater.index_path), "Retriever index was not compiled/saved."

    # Test retrieval of manually ingested data
    retriever = MedicalRetriever(index_path=updater.index_path, fallback_csv_path=base_csv_path)
    hits = retriever.retrieve("What is Cognova Fever?", threshold=0.15, top_k=1)
    assert len(hits) > 0, "Failed to retrieve manually ingested record."
    assert hits[0]["focus"] == "Cognova Fever"
    assert "coding speeds" in hits[0]["answer"], "Retrieved answer does not match input."
    print("  Manual Ingestion and retrieval assertions passed.")

    # 4. Test Deduplication
    print("\n[Test] Testing Deduplication Logic...")
    # Attempt to ingest the same question with a different answer (should update it instead of inserting a duplicate)
    success, msg = updater.ingest_manual(
        focus="Cognova Fever",
        question_type="information",
        question="What is Cognova Fever?",
        answer="Cognova Fever causes coding speeds and 100% test success."
    )
    assert success
    df_custom = updater.load_custom_updates()
    assert len(df_custom) == 1, f"Expected still 1 custom entry due to deduplication, found {len(df_custom)}"
    assert "100% test success" in df_custom.iloc[0]["answer"], "Answer was not updated in database."
    print("  Deduplication assertions passed.")

    # 5. Test File Scanning (CSV and XML)
    print("\n[Test] Testing Folder Ingestion (CSV and XML)...")
    
    # Write a mock CSV update file to pending updates
    mock_csv_file = os.path.join(updater.pending_dir, "batch_update.csv")
    csv_update_data = pd.DataFrame([
        {
            "focus": "Beta Cold",
            "question_type": "treatment",
            "question": "How to treat Beta Cold?",
            "answer": "Treat Beta Cold with rest and warm liquids."
        },
        {
            "focus": "GCP Syndrome",
            "question_type": "susceptibility",
            "question": "Who gets GCP Syndrome?",
            "answer": "Developers working on cloud deployments."
        }
    ])
    csv_update_data.to_csv(mock_csv_file, index=False)
    
    # Write a mock XML update file conforming to MedQuAD schema to pending updates
    mock_xml_file = os.path.join(updater.pending_dir, "xml_update.xml")
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Document>
    <Focus>Gamma Rash</Focus>
    <QAPairs>
        <QAPair>
            <Question qtype="symptoms">What are the symptoms of Gamma Rash?</Question>
            <Answer>Symptoms of Gamma Rash are red spots and mild itching.</Answer>
        </QAPair>
    </QAPairs>
</Document>
"""
    with open(mock_xml_file, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print("  Created mock CSV and XML update files in pending_updates.")

    # Trigger Sync
    added_count = updater.sync_all_sources()
    assert added_count == 3, f"Expected 3 records added from folder sync, got {added_count}"

    # Verify files were archived
    pending_files = os.listdir(updater.pending_dir)
    assert len(pending_files) == 0, f"Pending directory should be empty, files: {pending_files}"
    processed_files = os.listdir(updater.processed_dir)
    assert len(processed_files) == 2, f"Processed directory should contain 2 archived files, got {len(processed_files)}"

    # Verify custom updates count
    df_custom = updater.load_custom_updates()
    assert len(df_custom) == 4, f"Expected 4 custom updates total, got {len(df_custom)}"

    # Reload retriever and verify retrieval of synced CSV/XML questions
    retriever.load_index()
    
    # Check CSV retrieval
    hits_csv = retriever.retrieve("Who gets GCP Syndrome?", threshold=0.15, top_k=1)
    assert len(hits_csv) > 0, "Failed to retrieve synced CSV record."
    assert hits_csv[0]["focus"] == "GCP Syndrome"
    
    # Check XML retrieval
    hits_xml = retriever.retrieve("What are the symptoms of Gamma Rash?", threshold=0.15, top_k=1)
    assert len(hits_xml) > 0, "Failed to retrieve synced XML record."
    assert hits_xml[0]["focus"] == "Gamma Rash"
    assert "red spots" in hits_xml[0]["answer"]
    print("  Folder sync (CSV/XML) parsing and retrieval assertions passed.")

    # 6. Test Database Wipe / Clean Restore
    print("\n[Test] Testing Database Wipe...")
    success = updater.wipe_custom_knowledge()
    assert success, "Wipe execution failed."
    
    df_custom_wiped = updater.load_custom_updates()
    assert len(df_custom_wiped) == 0, "Custom database should be empty after wipe."
    
    retriever.load_index()
    hits_wiped = retriever.retrieve("What is Cognova Fever?", threshold=0.15, top_k=1)
    assert len(hits_wiped) == 0, "Querying custom question after wipe should return zero matches."
    print("  Wipe database restore assertions passed.")

    # Cleanup test folder
    shutil.rmtree(TEST_DIR)
    print("\n[Success] All dynamic knowledge base updater tests completed successfully!")


if __name__ == "__main__":
    test_dynamic_updater_pipeline()
