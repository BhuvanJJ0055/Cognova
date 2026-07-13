"""
Task 3 - data_processor_arxiv.py
Author: Bhuvan J J

Automated script to download the official arXiv dataset from Kaggle,
stream-parse its 3GB+ JSON file, filter for Computer Science & AI papers,
and index them into our local TF-IDF retriever.
"""

import os
import json
import argparse
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "arxiv_cs_papers.csv")

def process_kaggle_dataset(limit=15000, progress_callback=None):
    """
    Downloads the Kaggle arXiv dataset using kagglehub and extracts 
    computer science research papers using a memory-efficient stream parser.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("The 'kagglehub' package is required. Run 'pip install kagglehub' to install it.")
        
    print("[Ingestion] Resolving arXiv dataset via kagglehub...")
    if progress_callback:
        progress_callback(0.05, "Resolving dataset download via kagglehub...")
        
    # Download dataset from Kaggle (resolves locally if already downloaded)
    dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
    json_path = os.path.join(dataset_path, "arxiv-metadata-oai-snapshot.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not locate metadata file 'arxiv-metadata-oai-snapshot.json' in downloaded path: {dataset_path}")
        
    print(f"[Ingestion] Found metadata file at: {json_path}")
    
    file_size = os.path.getsize(json_path)
    bytes_processed = 0
    cs_papers = []
    
    # CS target subcategories
    target_prefixes = ["cs.cl", "cs.lg", "cs.cv", "cs.ai", "cs.ne", "cs.ir", "cs.si"]
    
    print("[Ingestion] Beginning stream parsing of metadata (filtering for CS category)...")
    
    with open(json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            bytes_processed += len(line.encode("utf-8"))
            
            # Trigger callback every 25,000 scanned records
            if progress_callback and idx % 25000 == 0:
                progress = min(0.95, bytes_processed / file_size)
                # Keep scale between 0.1 and 0.95
                adjusted_progress = 0.1 + (progress * 0.85)
                progress_callback(adjusted_progress, f"Scanned {idx:,} records... Found {len(cs_papers):,} CS papers")
                
            try:
                record = json.loads(line)
                categories = str(record.get("categories", "")).lower().split()
                
                # Check if paper belongs to CS/AI categories
                is_cs = any(
                    cat in target_prefixes or any(cat.startswith(p + ".") for p in target_prefixes)
                    for cat in categories
                )
                
                if is_cs:
                    paper_id = record.get("id", "")
                    title = record.get("title", "").replace("\n", " ").strip()
                    authors = record.get("authors", "").replace("\n", " ").strip()
                    summary = record.get("abstract", "").replace("\n", " ").strip()
                    published = record.get("update_date", "")
                    
                    # Extract the first matching cs subcategory as primary category
                    cs_cats = [cat for cat in categories if cat.startswith("cs.")]
                    primary_category = cs_cats[0] if cs_cats else categories[0]
                    
                    paper_url = f"http://arxiv.org/abs/{paper_id}"
                    
                    cs_papers.append({
                        "id": paper_id,
                        "title": title,
                        "authors": authors,
                        "summary": summary,
                        "published": published,
                        "primary_category": primary_category,
                        "url": paper_url
                    })
                    
                    if len(cs_papers) >= limit:
                        print(f"[Ingestion] Reached paper limit of {limit:,}. Halting stream.")
                        break
            except Exception:
                continue
                
    if progress_callback:
        progress_callback(0.96, f"Extracted {len(cs_papers):,} CS papers. Overwriting database...")
        
    # Write to local CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    df_new = pd.DataFrame(cs_papers)
    df_new.to_csv(CSV_PATH, index=False, encoding="utf-8")
    
    print(f"[Ingestion] Saved {len(cs_papers)} papers to {CSV_PATH}")
    
    if progress_callback:
        progress_callback(0.98, "Re-compiling retrieval index...")
        
    # Rebuild index
    try:
        from build_arxiv_index import ArXivRetriever
    except ImportError:
        from Task3_ArXiv_CS_Chatbot.build_arxiv_index import ArXivRetriever
    retriever = ArXivRetriever(fallback_csv_path=CSV_PATH)
    retriever.build_and_save_index(CSV_PATH)
    
    if progress_callback:
        progress_callback(1.0, f"Successfully indexed {len(cs_papers):,} papers from Kaggle!")
        
    return len(cs_papers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle ArXiv Dataset Processor")
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum papers to ingest (default 500 for test run)"
    )
    args = parser.parse_args()
    
    print(f"Running Kaggle ArXiv Ingestion. Target limit: {args.limit}")
    try:
        count = process_kaggle_dataset(limit=args.limit)
        print(f"Success! Ingested {count} papers.")
    except Exception as e:
        print(f"Error executing processor: {e}")
