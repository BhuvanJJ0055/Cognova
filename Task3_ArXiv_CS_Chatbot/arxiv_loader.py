"""
Task 3 - arxiv_loader.py
Author: Bhuvan J J

Manages scientific papers. Loads a local CSV dataset, and allows searching 
and downloading metadata dynamically from the official arXiv API.
"""

import os
import requests
import pandas as pd
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "arxiv_cs_papers.csv")

# Semantic seminal AI/ML papers to initialize out-of-the-box functionality
SEMINAL_PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
        "summary": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.",
        "published": "2017-06-12",
        "primary_category": "cs.CL",
        "url": "http://arxiv.org/abs/1706.03762v7"
    },
    {
        "id": "1810.04805v2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "summary": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.",
        "published": "2018-10-11",
        "primary_category": "cs.CL",
        "url": "http://arxiv.org/abs/1810.04805v2"
    },
    {
        "id": "1512.03385v1",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
        "summary": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those previously used. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from greatly increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity.",
        "published": "2015-12-10",
        "primary_category": "cs.CV",
        "url": "http://arxiv.org/abs/1512.03385v1"
    },
    {
        "id": "1412.6980v9",
        "title": "Adam: A Method for Stochastic Optimization",
        "authors": "Diederik P. Kingma, Jimmy Ba",
        "summary": "We introduce Adam, a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. The method computes adaptive individual learning rates for different parameters from estimates of first and second moments of the gradients; the name Adam is derived from adaptive moment estimation. Our method is designed to combine the advantages of two recently popular methods: AdaGrad and RMSProp. We show Adam works well in practice and compares favorably to other stochastic optimizers.",
        "published": "2014-12-22",
        "primary_category": "cs.LG",
        "url": "http://arxiv.org/abs/1412.6980v9"
    },
    {
        "id": "1406.2661v1",
        "title": "Generative Adversarial Nets",
        "authors": "Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjervil Ozair, Aaron Courville, Yoshua Bengio",
        "summary": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere.",
        "published": "2014-06-10",
        "primary_category": "cs.LG",
        "url": "http://arxiv.org/abs/1406.2661v1"
    },
    {
        "id": "2005.14165v4",
        "title": "Language Models are Few-Shot Learners",
        "authors": "Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei",
        "summary": "We demonstrate that scaling up language models greatly improves few-shot performance, sometimes even matching prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.",
        "published": "2020-05-28",
        "primary_category": "cs.CL",
        "url": "http://arxiv.org/abs/2005.14165v4"
    }
]

def initialize_database():
    """Ensures the data directory and a default set of CS papers exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # If file doesn't exist or is completely empty (0 size), write seminal papers
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) <= 1:
        df = pd.DataFrame(SEMINAL_PAPERS)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        print(f"[Loader] Initialized database with {len(SEMINAL_PAPERS)} seminal papers.")
    else:
        # Check if the columns are correct
        try:
            df = pd.read_csv(CSV_PATH)
            required = ["id", "title", "authors", "summary", "published", "primary_category", "url"]
            if not all(col in df.columns for col in required):
                print("[Loader] CSV columns incorrect. Re-initializing...")
                df = pd.DataFrame(SEMINAL_PAPERS)
                df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        except Exception as e:
            print(f"[Loader] Error reading CSV, recreating default: {e}")
            df = pd.DataFrame(SEMINAL_PAPERS)
            df.to_csv(CSV_PATH, index=False, encoding="utf-8")

def get_local_papers() -> pd.DataFrame:
    """Loads and returns all local papers as a DataFrame."""
    initialize_database()
    try:
        df = pd.read_csv(CSV_PATH)
        # Handle nan values
        df = df.fillna("")
        return df
    except Exception as e:
        print(f"[Loader] Error loading local papers: {e}")
        return pd.DataFrame(columns=["id", "title", "authors", "summary", "published", "primary_category", "url"])

def save_local_papers(df: pd.DataFrame):
    """Saves the DataFrame back to the local database CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

def search_arxiv_api(query: str, max_results: int = 10) -> list:
    """Queries the live arXiv API and returns a list of dictionaries with paper metadata."""
    base_url = "http://export.arxiv.org/api/query?"
    
    # Formulate query: target Computer Science (cs) if not specified
    # Prefix with computer science categories unless already formatted
    clean_query = query.strip()
    if not any(prefix in clean_query for prefix in ["ti:", "au:", "abs:", "all:"]):
        # Default search across all fields, limiting to computer science subject class
        api_query = f"all:{clean_query} AND cat:cs.*"
    else:
        api_query = clean_query
        
    params = {
        "search_query": api_query,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    encoded = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    full_url = base_url + encoded
    
    results = []
    try:
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        namespace = "{http://www.w3.org/2005/Atom}"
        
        for entry in root.findall(f"{namespace}entry"):
            paper = {}
            authors = []
            
            # Check if it has an id - if not, skip
            id_node = entry.find(f"{namespace}id")
            if id_node is None or not id_node.text:
                continue
                
            paper["id"] = id_node.text.split("/abs/")[-1]
            paper["url"] = id_node.text.strip()
            
            title_node = entry.find(f"{namespace}title")
            paper["title"] = title_node.text.replace("\n", " ").strip() if title_node is not None and title_node.text else "Untitled"
            
            summary_node = entry.find(f"{namespace}summary")
            paper["summary"] = summary_node.text.replace("\n", " ").strip() if summary_node is not None and summary_node.text else ""
            
            pub_node = entry.find(f"{namespace}published")
            if pub_node is not None and pub_node.text:
                # Format to YYYY-MM-DD
                paper["published"] = pub_node.text.split("T")[0]
            else:
                paper["published"] = datetime.now().strftime("%Y-%m-%d")
                
            for author_node in entry.findall(f"{namespace}author"):
                name_node = author_node.find(f"{namespace}name")
                if name_node is not None and name_node.text:
                    authors.append(name_node.text.strip())
            paper["authors"] = ", ".join(authors)
            
            cat_node = entry.find(f"{namespace}primary_category")
            if cat_node is not None:
                paper["primary_category"] = cat_node.get("term", "cs.LG")
            else:
                paper["primary_category"] = "cs.LG"
                
            results.append(paper)
    except Exception as e:
        print(f"[Loader] Error querying arXiv API: {e}")
        
    return results

def import_papers_to_local(papers: list) -> int:
    """Merges a list of retrieved paper dicts into our local CSV, filtering duplicates."""
    if not papers:
        return 0
        
    df_local = get_local_papers()
    df_new = pd.DataFrame(papers)
    
    # Filter duplicate IDs
    if not df_local.empty:
        df_new = df_new[~df_new["id"].isin(df_local["id"])]
        
    if df_new.empty:
        return 0
        
    df_merged = pd.concat([df_local, df_new], ignore_index=True)
    save_local_papers(df_merged)
    return len(df_new)

if __name__ == "__main__":
    initialize_database()
    print("Local database contains:")
    print(get_local_papers()[["title", "primary_category"]])
