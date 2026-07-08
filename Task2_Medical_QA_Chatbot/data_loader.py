"""
Task 2 - data_loader.py
Author: Bhuvan J J

This script is responsible for loading and parsing the MedQuAD (Medical Question Answering Dataset).
Since the full dataset is hosted on GitHub and contains over 11,000 XML files across multiple directories,
this script:
1. Automatically downloads the master zip of MedQuAD from GitHub on-demand if the local dataset folder is missing.
2. Extracts it recursively to a local 'data' directory.
3. Parses every single XML file using xml.etree.ElementTree to extract the focus disease/drug, the question, 
   the question type (e.g. symptoms, treatment), and the answer.
4. Flattens all entries and exports them to a unified CSV file (data/medquad_qa.csv) for vector retrieval indexing.
"""

import os
import glob
import zipfile
import requests
import xml.etree.ElementTree as ET
import pandas as pd

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
XML_EXTRACT_PATH = os.path.join(DATA_DIR, "MedQuAD-master")
FINAL_CSV_PATH = os.path.join(DATA_DIR, "medquad_qa.csv")
MEDQUAD_ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"


def download_and_extract_medquad():
    """Downloads the MedQuAD dataset from GitHub and extracts it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "medquad.zip")
    
    if os.path.exists(XML_EXTRACT_PATH):
        print(f"[Info] MedQuAD raw XML files already exist at {XML_EXTRACT_PATH}.")
        return

    print(f"[Info] Downloading MedQuAD dataset from {MEDQUAD_ZIP_URL}...")
    try:
        response = requests.get(MEDQUAD_ZIP_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("[Info] Download complete. Extracting files...")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        print(f"[Info] Extraction complete. Raw files at {XML_EXTRACT_PATH}.")
        
        # Clean up zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    except Exception as e:
        print(f"[Error] Failed to download/extract MedQuAD: {e}")
        print("[Info] Please clone or extract MedQuAD manually to " + XML_EXTRACT_PATH)


def parse_medquad_xml_file(file_path):
    """
    Parses a single MedQuAD XML file.
    XML schema contains a single root <Document> which has:
      - <Focus> (the disease or topic name)
      - <QAPairs> containing multiple <QAPair> elements
        - Each <QAPair> has:
          - <Question qtype="..."> (question type attribute and text)
          - <Answer> (answer text)
    """
    qa_pairs_list = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract Focus (Topic of the document)
        focus_elem = root.find("Focus")
        focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else "General"
        
        # Extract individual QA pairs
        for qa_pair in root.findall(".//QAPair"):
            question_elem = qa_pair.find("Question")
            answer_elem = qa_pair.find("Answer")
            
            if question_elem is not None and answer_elem is not None:
                question = question_elem.text.strip() if question_elem.text else ""
                answer = answer_elem.text.strip() if answer_elem.text else ""
                
                # Fetch question type attribute
                qtype = question_elem.get("qtype", "general").strip()
                
                # Skip instances where question or answer is missing (copyright restrictions sometimes apply)
                if question and answer:
                    qa_pairs_list.append({
                        "focus": focus,
                        "question_type": qtype,
                        "question": question,
                        "answer": answer
                    })
    except ET.ParseError as e:
        print(f"[Warning] Parse error in {file_path}: {e}")
    except Exception as e:
        print(f"[Warning] Error reading {file_path}: {e}")
        
    return qa_pairs_list


def build_dataframe_from_xml():
    """Scans raw XML folder recursively and creates a flat dataset CSV."""
    if not os.path.exists(XML_EXTRACT_PATH):
        print(f"[Warning] Raw XML directory {XML_EXTRACT_PATH} not found.")
        # Fallback: check if we can run download
        download_and_extract_medquad()
        if not os.path.exists(XML_EXTRACT_PATH):
            return None

    print("[Info] Recursively scanning for XML files...")
    # Find all xml files in any subfolder under MedQuAD-master
    search_pattern = os.path.join(XML_EXTRACT_PATH, "**", "*.xml")
    xml_files = glob.glob(search_pattern, recursive=True)
    
    print(f"[Info] Found {len(xml_files)} XML files. Starting parsing...")
    
    all_qa_pairs = []
    processed_count = 0
    
    for xml_file in xml_files:
        pairs = parse_medquad_xml_file(xml_file)
        all_qa_pairs.extend(pairs)
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"[Info] Processed {processed_count}/{len(xml_files)} XML files... ({len(all_qa_pairs)} QA pairs parsed)")
            
    print(f"[Info] Finished parsing. Total QA pairs extracted: {len(all_qa_pairs)}")
    
    if len(all_qa_pairs) == 0:
        print("[Error] No valid Q&A pairs extracted. (Verify if XML files contain answers - some sources omit them).")
        return None
        
    # Convert to pandas DataFrame
    df = pd.DataFrame(all_qa_pairs)
    
    # Save to CSV
    os.makedirs(os.path.dirname(FINAL_CSV_PATH), exist_ok=True)
    df.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8")
    print(f"[Success] Exported parsed dataset to {FINAL_CSV_PATH}")
    return df


if __name__ == "__main__":
    # If run standalone, execute the complete pipeline
    download_and_extract_medquad()
    build_dataframe_from_xml()
