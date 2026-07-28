"""
Task 4 - ArXiv Research Expert Assistant Module
Author: Bhuvan J J

Streams ArXiv paper records, filters CS categories (cs.*), limits max_papers
for disk/memory efficiency, and provides semantic paper search, summarization,
concept extraction, and explanation generation.
"""

import os
import re
import json
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from typing import Optional, List

try:
    from src.core.rag_pipeline import RAGPipeline
except ImportError:
    from core.rag_pipeline import RAGPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "arxiv_index", "arxiv_retriever_index.joblib")


def summarize_paper_text(text: str, max_sentences: int = 3) -> str:
    """Extracts key sentences from abstract."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
    if not sentences:
        return text
    return " ".join(sentences[:max_sentences])


class ArXivExpert:
    """ArXiv CS research assistant powered by RAGPipeline."""

    def __init__(self, index_path=INDEX_PATH, max_papers=1500, category_prefix="cs.", fallback_csv_path=None):
        self.index_path = index_path
        self.max_papers = max_papers
        self.category_prefix = category_prefix
        self.fallback_csv_path = fallback_csv_path
        self.rag = RAGPipeline(
            index_path=index_path,
            text_fields=["title", "summary"],
            metadata_fields=["id", "title", "authors", "summary", "published", "primary_category", "url"]
        )
        self.load_or_build_index()

    def load_or_build_index(self, jsonl_file_path: Optional[str] = None):
        """Indexes papers filtering by category_prefix up to max_papers limit."""
        if os.path.exists(self.index_path) and len(self.rag.metadata) > 0:
            return len(self.rag.metadata)

        papers = []
        target_jsonl = jsonl_file_path or os.path.join(DATA_DIR, "arxiv", "arxiv-metadata-oai-snapshot.json")
        if target_jsonl and os.path.exists(target_jsonl):
            try:
                with open(target_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        cat = record.get("categories", "")
                        if self.category_prefix in cat:
                            papers.append({
                                "id": record.get("id", ""),
                                "title": record.get("title", "").replace("\n", " ").strip(),
                                "authors": record.get("authors", "").replace("\n", " ").strip(),
                                "summary": record.get("abstract", "").replace("\n", " ").strip(),
                                "published": record.get("update_date", "2024"),
                                "primary_category": cat.split()[0] if cat else "cs.AI",
                                "url": f"https://arxiv.org/abs/{record.get('id', '')}"
                            })
                            if len(papers) >= self.max_papers:
                                break
            except Exception:
                pass

        if not papers:
            # Baseline seminal computer science research dataset
            papers = [
                {
                    "id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
                    "summary": "We propose the Transformer architecture based entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments show superior quality and parallelizability for neural machine translation.",
                    "published": "2017-06-12",
                    "primary_category": "cs.CL",
                    "url": "https://arxiv.org/abs/1706.03762"
                },
                {
                    "id": "1810.04805",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                    "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
                    "summary": "BERT pre-trains deep bidirectional representations from unlabeled text by jointly conditioning on left and right context. Fine-tuning achieves state-of-the-art results on eleven natural language processing tasks.",
                    "published": "2018-10-11",
                    "primary_category": "cs.CL",
                    "url": "https://arxiv.org/abs/1810.04805"
                },
                {
                    "id": "2005.14165",
                    "title": "Language Models are Few-Shot Learners (GPT-3)",
                    "authors": "Tom B. Brown et al.",
                    "summary": "We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance, achieving strong performance on broad NLP tasks without fine-tuning.",
                    "published": "2020-05-28",
                    "primary_category": "cs.CL",
                    "url": "https://arxiv.org/abs/2005.14165"
                },
                {
                    "id": "2301.08745",
                    "title": "A Survey of Prompt Engineering Methods in Large Language Models",
                    "authors": "Jiawei Chen, Ruijie Wang et al.",
                    "summary": "Prompt engineering has emerged as a key paradigm for guiding Large Language Models (LLMs) to perform complex reasoning, zero-shot learning, and chain-of-thought generation without parameter updates.",
                    "published": "2023-01-20",
                    "primary_category": "cs.CL",
                    "url": "https://arxiv.org/abs/2301.08745"
                },
                {
                    "id": "1512.03385",
                    "title": "Deep Residual Learning for Image Recognition (ResNet)",
                    "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
                    "summary": "We present a residual learning framework to ease the training of networks that are substantially deeper than those previously used, winning 1st place in ILSVRC 2015.",
                    "published": "2015-12-10",
                    "primary_category": "cs.CV",
                    "url": "https://arxiv.org/abs/1512.03385"
                },
                {
                    "id": "1406.2661",
                    "title": "Generative Adversarial Nets (GANs)",
                    "authors": "Ian J. Goodfellow et al.",
                    "summary": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model and a discriminative model.",
                    "published": "2014-06-10",
                    "primary_category": "cs.LG",
                    "url": "https://arxiv.org/abs/1406.2661"
                }
            ]

        df = pd.DataFrame(papers)
        return self.rag.build_from_dataframe(
            df,
            text_fields=["title", "summary"],
            metadata_fields=["id", "title", "authors", "summary", "published", "primary_category", "url"]
        )

    def retrieve(self, query: str, top_k: int = 4, threshold: float = 0.00) -> list:
        """Retrieves matching computer science papers."""
        results = self.rag.query(query, top_k=top_k, threshold=0.00)
        formatted = []
        for res in results:
            formatted.append({
                "id": res.get("id", ""),
                "title": res.get("title", ""),
                "authors": res.get("authors", ""),
                "summary": res.get("summary", ""),
                "published": res.get("published", ""),
                "primary_category": res.get("primary_category", "cs.AI"),
                "url": res.get("url", ""),
                "similarity": res.get("score", 0.0)
            })
        return formatted


class ArXivRetriever(ArXivExpert):
    """Backwards compatible alias for ArXivExpert."""
    pass


def extract_concepts(text: str) -> List[str]:
    """Extracts key CS concepts from paper title and abstract."""
    lowered = text.lower()
    known_concepts = {
        "prompt": "Prompt Engineering",
        "language models": "Large Language Models (LLMs)",
        "transformer": "Transformer Architecture",
        "self-attention": "Self-Attention Mechanism",
        "attention": "Attention Mechanism",
        "bert": "BERT Pre-training",
        "gpt": "Generative Pre-trained Transformer",
        "cnn": "Convolutional Neural Networks",
        "resnet": "Residual Learning (ResNet)",
        "gan": "Generative Adversarial Networks",
        "deep learning": "Deep Learning",
        "machine learning": "Machine Learning",
        "nlp": "Natural Language Processing",
        "computer vision": "Computer Vision",
        "fine-tuning": "Supervised Fine-Tuning",
        "zero-shot": "Zero-Shot Learning",
        "few-shot": "Few-Shot Learning"
    }

    found = []
    for term, label in known_concepts.items():
        if term in lowered and label not in found:
            found.append(label)

    return found if found else ["Computer Science", "Deep Learning"]


def search_arxiv_api(query: str, max_results: int = 5, category: str = "cs.*", *args, **kwargs) -> list:
    """Queries live ArXiv API for papers matching query with fallback to local index."""
    target_cat = kwargs.get("category", category)
    if not target_cat or target_cat == "cs.*":
        cat_str = "cat:cs*"
    else:
        cat_str = f"cat:{target_cat}"

    # Build clean ArXiv query terms
    words = [w for w in re.findall(r'\b\w+\b', query.lower()) if len(w) > 2 and w not in {"the", "and", "for", "in", "with"}]
    if words:
        q_str = "+AND+".join([f"all:{w}" for w in words[:4]])
    else:
        q_str = f"all:{query.strip()}"

    search_query_url = f"http://export.arxiv.org/api/query?search_query={cat_str}+AND+{q_str}&start=0&max_results={max_results}"

    papers = []
    try:
        resp = requests.get(search_query_url, timeout=8)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

            for entry in root.findall('atom:entry', ns):
                id_elem = entry.find('atom:id', ns)
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                published_elem = entry.find('atom:published', ns)

                paper_id = id_elem.text.split('/')[-1] if id_elem is not None and id_elem.text else ""
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None and title_elem.text else "Untitled"
                summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None and summary_elem.text else ""
                published = published_elem.text[:10] if published_elem is not None and published_elem.text else "2024"

                authors = []
                for a in entry.findall('atom:author', ns):
                    name_elem = a.find('atom:name', ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)

                cat_elem = entry.find('arxiv:primary_category', ns)
                cat_term = cat_elem.attrib['term'] if (cat_elem is not None and 'term' in cat_elem.attrib) else "cs.CL"
                paper_url = id_elem.text if id_elem is not None and id_elem.text else f"https://arxiv.org/abs/{paper_id}"

                papers.append({
                    "id": paper_id,
                    "title": title,
                    "authors": ", ".join(authors[:5]) if authors else "Unknown",
                    "summary": summary,
                    "published": published,
                    "primary_category": cat_term,
                    "url": paper_url
                })
    except Exception:
        pass

    # Fallback to local semantic retriever if network or API returns no entries
    if not papers:
        retriever = ArXivExpert()
        papers = retriever.retrieve(query, top_k=max_results)

    return papers


def generate_explanation(query: str, retrieved_papers: list) -> str:
    """Generates structured concept synthesis and research paper summary."""
    if not retrieved_papers:
        return "No relevant CS research papers were found matching your query."

    paper = retrieved_papers[0]
    summary_text = summarize_paper_text(paper.get("summary", ""), max_sentences=3)
    concepts = extract_concepts(paper.get("title", "") + " " + paper.get("summary", ""))

    concept_tags = " • ".join([f"`{c}`" for c in concepts])

    return (
        f"### 📄 Key Paper Synthesis: **{paper.get('title', 'Paper')}**\n"
        f"**Authors**: {paper.get('authors', 'Unknown')}\n"
        f"**Category**: `{paper.get('primary_category', 'cs.AI')}` | **Published**: {paper.get('published', '2024')}\n"
        f"**ArXiv Link**: [{paper.get('url', '#')}]({paper.get('url', '#')})\n\n"
        f"🏷️ **Extracted Core Concepts**: {concept_tags}\n\n"
        f"#### 💡 **Technical Summary & Concept Explanation**:\n"
        f"{summary_text}\n\n"
        f"**Domain Insight**: This research paper contributes directly to key computer science advancements in "
        f"*{', '.join(concepts[:3])}*. It addresses performance bottlenecks by introducing structural algorithmic improvements."
    )
