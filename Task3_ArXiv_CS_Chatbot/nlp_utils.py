"""
Task 3 - nlp_utils.py
Author: Bhuvan J J

Implements advanced NLP techniques:
1. Extractive Summarization: Ranks sentences in text using word frequency counts.
2. Named Entity Recognition (NER): Extracts computer science and machine learning concepts.
"""

import re
import math
from collections import Counter

# Core AI/ML/CS concepts for NER
TECHNICAL_CONCEPTS = [
    # NLP / Transformers
    "self-attention", "attention mechanism", "transformer", "bert", "gpt", "lstm", "rnn", 
    "recurrent neural network", "encoder-decoder", "machine translation", "word embedding", 
    "language model", "natural language processing", "nlp", "tokenization",
    
    # Computer Vision
    "convolutional neural network", "cnn", "image recognition", "residual network", "resnet", 
    "object detection", "computer vision", "generative adversarial network", "gan", "autoencoder",
    
    # Optimization / Math
    "gradient descent", "backpropagation", "stochastic optimization", "adam optimizer", 
    "loss function", "activation function", "learning rate", "regularization", "overfitting", 
    "underfitting", "normalisation", "dropout",
    
    # Learning paradigms
    "reinforcement learning", "supervised learning", "unsupervised learning", "self-supervised learning", 
    "few-shot learning", "zero-shot learning", "pre-training", "fine-tuning", "neural network",
    "deep learning", "machine learning", "transfer learning"
]

# Sort concepts by length descending to match longest phrases first (e.g. "recurrent neural network" before "neural network")
TECHNICAL_CONCEPTS_SORTED = sorted(TECHNICAL_CONCEPTS, key=len, reverse=True)

def extract_concepts(text: str) -> list:
    """
    Extracts computer science and machine learning concepts from text.
    Uses regex word boundaries to prevent subword matching.
    """
    if not text:
        return []
        
    extracted = set()
    text_lower = text.lower()
    
    # Check each concept in sorted order
    for concept in TECHNICAL_CONCEPTS_SORTED:
        # Escape special characters (like '-' or '*')
        pattern = r"\b" + re.escape(concept) + r"\b"
        if re.search(pattern, text_lower):
            extracted.add(concept)
            # To avoid subphrase double-tagging (e.g. tagging "neural network" when "recurrent neural network" was matched),
            # we can temporarily mask the matched term in the search string
            # However, for simplicity and visibility, keeping standard tag matching is sufficient.
            
    return sorted(list(extracted), key=len, reverse=True)

def split_sentences(text: str) -> list:
    """Splits a body of text into sentences using basic regular expressions."""
    if not text:
        return []
    # Split on periods/exclamations/questions followed by space and capital letter, or end of line
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_end.split(text.replace("\n", " ").strip())
    return [s.strip() for s in sentences if s.strip()]

def summarize_text(text: str, num_sentences: int = 2) -> str:
    """
    Generates an extractive summary of the text.
    Computes word frequencies, scores sentences based on the sum of their word importances,
    and returns the top sentences sorted in their original order of appearance.
    """
    if not text:
        return ""
        
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return text
        
    # Preprocess text to count word frequencies
    # Lowercase and keep alphanumeric words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Standard English stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "at", "by", 
        "from", "for", "in", "out", "on", "off", "over", "under", "to", "with", "is", 
        "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", 
        "does", "did", "of", "that", "this", "these", "those", "it", "its", "we", "our", 
        "us", "you", "your", "they", "them", "their", "he", "him", "his", "she", "her",
        "i", "me", "my", "more", "most", "about", "show", "can", "will", "would", "should"
    }
    
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if not filtered_words:
        # Fallback if no content words
        return " ".join(sentences[:num_sentences])
        
    word_counts = Counter(filtered_words)
    max_freq = max(word_counts.values())
    
    # Normalize word frequencies
    word_weights = {word: count / max_freq for word, count in word_counts.items()}
    
    # Score sentences
    sentence_scores = {}
    for idx, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        score = 0
        word_count_in_sentence = 0
        
        for w in sentence_words:
            if w in word_weights:
                score += word_weights[w]
                word_count_in_sentence += 1
                
        # Length normalization to prevent favoring extremely long sentences
        if word_count_in_sentence > 0:
            sentence_scores[idx] = score / math.sqrt(word_count_in_sentence)
        else:
            sentence_scores[idx] = 0
            
    # Get top scoring sentences
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    # Sort indices back in original order of appearance
    top_indices.sort()
    
    summary_sentences = [sentences[idx] for idx in top_indices]
    return " ".join(summary_sentences)

if __name__ == "__main__":
    test_abstract = (
        "Deeper neural networks are more difficult to train. We present a residual learning framework "
        "to ease the training of networks that are substantially deeper than those previously used. "
        "We explicitly reformulate the layers as learning residual functions with reference to the "
        "layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical "
        "evidence showing that these residual networks are easier to optimize, and can gain accuracy "
        "from greatly increased depth. On the ImageNet dataset we evaluate residual nets with a depth "
        "of up to 152 layers---8x deeper than VGG nets but still having lower complexity."
    )
    
    print("NER Concepts Extracted:")
    print(extract_concepts(test_abstract))
    
    print("\nExtractive Summary:")
    print(summarize_text(test_abstract, num_sentences=2))
