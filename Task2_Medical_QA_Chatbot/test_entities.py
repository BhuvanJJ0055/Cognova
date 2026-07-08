"""
Task 2 - test_entities.py
Author: Bhuvan J J

Automated test script to verify Medical Entity Recognizer tagging.
It:
1. Instantiates the MedicalEntityRecognizer.
2. Evaluates query statements representing various patient contexts.
3. Asserts that the identified diseases, symptoms, and treatments match target values.
"""

import os
import sys

# Ensure local modules can be loaded
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from entity_recognition import MedicalEntityRecognizer

def test_entity_recognition():
    print("[Test] Initializing MedicalEntityRecognizer...")
    recognizer = MedicalEntityRecognizer()
    
    # Test case 1: Multiple entities
    query_1 = "My lupus flare is causing intense joint pain and fever, what prednisone dosage is common?"
    print(f"\n[Test] Parsing: '{query_1}'")
    e1 = recognizer.extract_entities(query_1)
    print(f"  Diseases:   {e1['diseases']}")
    print(f"  Symptoms:   {e1['symptoms']}")
    print(f"  Treatments: {e1['treatments']}")
    
    assert "lupus" in e1["diseases"], "Failed to tag disease 'lupus'"
    assert "joint pain" in e1["symptoms"], "Failed to tag symptom 'joint pain'"
    assert "fever" in e1["symptoms"], "Failed to tag symptom 'fever'"
    assert "prednisone" in e1["treatments"], "Failed to tag treatment 'prednisone'"
    
    # Test case 2: Negation / boundary verification
    # Make sure subwords like "pain" inside "paint" are NOT matched, 
    # but actual words are.
    query_2 = "I am painting my walls, but I have a throbbing headache."
    print(f"\n[Test] Parsing: '{query_2}'")
    e2 = recognizer.extract_entities(query_2)
    print(f"  Symptoms:   {e2['symptoms']}")
    
    assert "headache" in e2["symptoms"], "Failed to tag symptom 'headache'"
    assert "pain" not in e2["symptoms"], "Falsely matched 'pain' inside 'painting'"
    
    # Test case 3: Multi-word disease matching
    query_3 = "What are the first signs of celiac disease?"
    print(f"\n[Test] Parsing: '{query_3}'")
    e3 = recognizer.extract_entities(query_3)
    print(f"  Diseases:   {e3['diseases']}")
    
    assert "celiac disease" in e3["diseases"], "Failed to tag multi-word disease 'celiac disease'"
    
    print("\n[Success] All medical entity recognizer assertions passed successfully!")

if __name__ == "__main__":
    test_entity_recognition()
