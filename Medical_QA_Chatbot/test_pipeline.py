"""
Task 2 - test_pipeline.py
Author: Bhuvan J J

Automated test script to verify retrieval accuracy and pipeline integrity.
It:
1. Instantiates the MedicalRetriever using the sample Q&A CSV.
2. Triggers indexing dynamically.
3. Tests retrieval queries ("What is Asthma?" and "How is diabetes treated?")
4. Asserts that the top retrieved focus and similarity scores are within acceptable bounds.
"""

import os
import sys

# Ensure local modules can be loaded
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from build_index import MedicalRetriever

def test_retrieval_flow():
    print("[Test] Initializing MedicalRetriever with sample CSV...")
    sample_csv = os.path.join(BASE_DIR, "data", "sample_medquad_qa.csv")
    temp_index = os.path.join(BASE_DIR, "data", "test_retriever_index.joblib")
    
    # Clean up any residual test index
    if os.path.exists(temp_index):
        os.remove(temp_index)
        
    try:
        # Create retriever using the test index path and sample CSV
        retriever = MedicalRetriever(index_path=temp_index, fallback_csv_path=sample_csv)
        
        # Test Query 1: Asthma symptoms
        query_1 = "What is Asthma?"
        print(f"\n[Test] Querying: '{query_1}'")
        hits_1 = retriever.retrieve(query_1, threshold=0.15, top_k=1)
        
        assert len(hits_1) > 0, "Failed to retrieve any match for Asthma symptoms."
        best_match_1 = hits_1[0]
        print(f"  Best Match Topic: {best_match_1['focus']}")
        print(f"  Similarity Score: {best_match_1['similarity_score']:.4f}")
        assert best_match_1["focus"] == "Asthma", "Retrieved incorrect topic focus for Asthma query."
        assert "coughing" in best_match_1["answer"].lower(), "Retrieved answer lacks expected symptom terms."
        
        # Test Query 2: Diabetes treatment
        query_2 = "How is diabetes treated?"
        print(f"\n[Test] Querying: '{query_2}'")
        hits_2 = retriever.retrieve(query_2, threshold=0.15, top_k=1)
        
        assert len(hits_2) > 0, "Failed to retrieve any match for Diabetes treatment."
        best_match_2 = hits_2[0]
        print(f"  Best Match Topic: {best_match_2['focus']}")
        print(f"  Similarity Score: {best_match_2['similarity_score']:.4f}")
        assert best_match_2["focus"] == "Diabetes", "Retrieved incorrect topic focus for Diabetes query."
        assert "insulin" in best_match_2["answer"].lower(), "Retrieved answer lacks expected treatment terms."
        
        print("\n[Success] All pipeline retrieval assertions passed successfully!")
        
    finally:
        # Clean up temporary test index
        if os.path.exists(temp_index):
            os.remove(temp_index)

if __name__ == "__main__":
    test_retrieval_flow()
