"""
Task 3 - llm_explainer.py
Author: Bhuvan J J

Handles explanation generation using either:
1. Hugging Face Serverless Inference API (Llama-3/Mistral)
2. Google Gemini API (if API key provided)
3. A smart, deterministic local fallback template engine that answers questions 
   by synthesizing facts from retrieved paper abstracts and NER terms.

Maintains conversation history for follow-up question capabilities.
"""

import os
import requests
import json
from nlp_utils import extract_concepts, summarize_text

# Default open source model hosted on Hugging Face
DEFAULT_HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Dictionary of local static explanations for fallback mode if no APIs are available
CONCEPT_DICTIONARY = {
    "transformer": (
        "The Transformer is a neural network architecture introduced in 2017 in the paper 'Attention Is All You Need'. "
        "Unlike previous sequential architectures like RNNs or LSTMs, it processes the entire input sequence at once, "
        "leveraging 'self-attention' to compute representations in parallel. This makes it highly scalable and the "
        "foundation for modern LLMs like GPT, BERT, and Claude."
    ),
    "self-attention": (
        "Self-attention is a mechanism that allows a neural network to look at other words in an input sequence "
        "to get a better representation of the current word. For example, in the sentence 'The animal didn't cross the street "
        "because it was too tired', self-attention helps the model learn that 'it' refers to 'animal' rather than 'street'. "
        "It computes key, query, and value vectors for each word to weight their relevance relative to each other."
    ),
    "bert": (
        "BERT (Bidirectional Encoder Representations from Transformers) is an NLP model developed by Google in 2018. "
        "It is pre-trained bidirectionally, meaning it looks at both left and right context of a word simultaneously "
        "during training. BERT is excellent for understanding sentence context and is used for search ranking, classification, "
        "and named entity recognition."
    ),
    "gpt": (
        "GPT (Generative Pre-trained Transformer) is a family of autoregressive language models developed by OpenAI. "
        "Unlike BERT (which is encoder-only), GPT uses a decoder-only architecture designed for generative text tasks. "
        "It is trained to predict the next token in a sequence given all preceding tokens, enabling fluent text generation."
    ),
    "cnn": (
        "A Convolutional Neural Network (CNN) is a type of deep neural network mostly used for visual data (images and videos). "
        "It uses convolution operations with slide-over filters to extract spatial hierarchies of features, starting from "
        "low-level edges up to high-level complex objects. ResNet and VGG are famous examples."
    ),
    "lstm": (
        "LSTM (Long Short-Term Memory) is a specialized recurrent neural network (RNN) architecture designed to learn "
        "long-term dependencies. It uses input, output, and forget gates to regulate the flow of information, solving "
        "the vanishing gradient problem of simple RNNs. It was the standard for NLP before Transformers."
    ),
    "gradient descent": (
        "Gradient descent is an optimization algorithm used to minimize the loss function in neural network training. "
        "It calculates the gradient (derivative) of the loss function with respect to the network weights and updates "
        "the weights in the opposite direction of the gradient to step 'downhill' toward the global or local minimum."
    ),
    "overfitting": (
        "Overfitting occurs when a machine learning model learns the training data *too* well, capturing noise and random "
        "fluctuations rather than the underlying distribution. Consequently, the model performs exceptionally on training "
        "data but fails to generalize to unseen test data. It is mitigated using regularization, dropout, and early stopping."
    ),
    "reinforcement learning": (
        "Reinforcement Learning (RL) is a paradigm of machine learning where an agent learns to make decisions by "
        "interacting with an environment. The agent receives rewards for good actions and penalties for bad ones, and "
        "tries to learn a policy that maximizes the cumulative long-term reward. Q-learning and PPO are common RL algorithms."
    )
}

def generate_local_explanation(query: str, retrieved_papers: list, chat_history: list) -> str:
    """
    Generates a structured explanation locally using text synthesis and static rules.
    Used as a robust offline/fail-safe fallback.
    """
    extracted = extract_concepts(query)
    
    # 1. Resolve potential pronouns or follow-up intents (e.g. "what is that?", "explain the first paper")
    query_lower = query.lower()
    referenced_paper = None
    
    # Look for paper reference intents in history
    if retrieved_papers:
        if "first" in query_lower or "paper 1" in query_lower or "summarise" in query_lower:
            referenced_paper = retrieved_papers[0]
        elif "second" in query_lower or "paper 2" in query_lower and len(retrieved_papers) > 1:
            referenced_paper = retrieved_papers[1]
        else:
            # Default to the most relevant retrieved paper if any are returned
            referenced_paper = retrieved_papers[0]
            
    # Case A: Summarize a specific retrieved paper
    if ("summarise" in query_lower or "summary" in query_lower or "explain" in query_lower) and referenced_paper:
        summary_ext = summarize_text(referenced_paper["summary"], num_sentences=3)
        return (
            f"### 📝 Paper Summary: **{referenced_paper['title']}**\n\n"
            f"**Authors**: {referenced_paper['authors']} | **Category**: {referenced_paper['primary_category']}\n\n"
            f"Here is an extractive summary of the paper's key contents and methodology:\n"
            f"> {summary_ext}\n\n"
            f"You can read the full paper here: [arXiv Link]({referenced_paper['url']})"
        )

    # Case B: Concept Explanation Query
    if extracted:
        responses = []
        for concept in extracted:
            if concept in CONCEPT_DICTIONARY:
                responses.append(f"**{concept.title()}**:\n{CONCEPT_DICTIONARY[concept]}")
                
        if responses:
            base_expl = "\n\n".join(responses)
            paper_ctx = ""
            if retrieved_papers:
                paper_ctx = (
                    f"\n\n**Related Research**: For advanced details, you can look at the paper "
                    f"**\"{retrieved_papers[0]['title']}\"** by {retrieved_papers[0]['authors'].split(',')[0]} et al. "
                    f"which addresses relevant topics. Read abstract: *\"{summarize_text(retrieved_papers[0]['summary'], 1)}\"*"
                )
            return (
                f"### 💡 Concept Explanation\n\n"
                f"{base_expl}"
                f"{paper_ctx}"
            )
            
    # Case C: General Q&A / Follow-up fallback
    if retrieved_papers:
        paper = retrieved_papers[0]
        summary_ext = summarize_text(paper["summary"], num_sentences=2)
        return (
            f"Based on the scientific corpus, the paper **\"{paper['title']}\"** explains related concepts. "
            f"Here is a summary of the paper abstract:\n\n"
            f"\"{summary_ext}\"\n\n"
            f"Is there a specific machine learning model, term (like Transformers or Overfitting), or paper you want to discuss?"
        )
        
    return (
        "I could not locate specific scientific papers or match technical concepts in your query. "
        "Try asking about core deep learning architectures (e.g. Transformers, CNNs, LSTMs), optimizers (Adam), "
        "or specific ML terms like overfitting, reinforcement learning, or word embeddings."
    )

from typing import Any, Optional

def query_huggingface_api(messages: list, api_token: Optional[str] = None, model_id: str = DEFAULT_HF_MODEL) -> str:
    """Calls Hugging Face Serverless Inference API to generate a chat response."""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
        
    # Format prompt using chat templates structure
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}</s>\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}</s>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}</s>\n"
    prompt += "<|assistant|>\n"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "").strip()
                # Clean up any leftover prompt tags in output if present
                clean_text = text.replace("<|assistant|>", "").strip()
                return clean_text
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].strip()
        
        # Log error status
        print(f"[LLM] Hugging Face Inference API error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[LLM] Error contacting Hugging Face: {e}")
        
    return ""

def query_gemini_api(messages: list, api_key: str) -> str:
    """Calls Google Gemini API to generate a chat response."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    
    # Map messages to Gemini contents structure
    contents = []
    system_instruction = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_instruction = content
        else:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": content}]
            })
            
    payload: dict[str, Any] = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
    try:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                return text
        print(f"[LLM] Gemini API error {response.status_code}: {response.text}")
    except Exception as e:
        clean_error = str(e)
        if api_key:
            clean_error = clean_error.replace(api_key, "REDACTED_API_KEY")
        print(f"[LLM] Error contacting Gemini: {clean_error}")
        
    return ""

def generate_explanation(query: str, retrieved_papers: list, chat_history: list, hf_token: Optional[str] = None, gemini_key: Optional[str] = None) -> str:
    """
    Orchestrates explanation generation. Try LLM APIs if keys are provided,
    otherwise default to the smart local fallback explainer.
    """
    # Build System Instruction
    system_prompt = (
        "You are an expert scientific advisor and domain expert in Computer Science and Machine Learning. "
        "Your task is to answer user queries, explain complex technical concepts in simple terms, and summarize research papers. "
        "You are provided with a context of semantically retrieved papers. Restrict your technical scientific answers "
        "mostly to the context provided. If the context does not contain enough information, explain general concepts "
        "using standard computer science knowledge but prioritize the retrieved papers. Be educational, clear, and friendly.\n\n"
    )
    
    if retrieved_papers:
        system_prompt += "Retrieved scientific papers for context:\n"
        for idx, paper in enumerate(retrieved_papers):
            system_prompt += (
                f"Paper #{idx+1}:\n"
                f"- Title: {paper['title']}\n"
                f"- Authors: {paper['authors']}\n"
                f"- Primary Category: {paper['primary_category']}\n"
                f"- Abstract: {paper['summary']}\n"
                f"- Published: {paper['published']}\n"
                f"- URL: {paper['url']}\n\n"
            )
            
    # Format messages list from chat history
    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    # Add trailing conversation history (up to last 6 turns to keep context clean)
    for msg in chat_history[-6:]:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Append current query
    formatted_messages.append({"role": "user", "content": query})
    
    # Try Gemini API if key is available
    if gemini_key:
        response = query_gemini_api(formatted_messages, gemini_key)
        if response:
            return response
            
    # Try Hugging Face API if token/online option is enabled
    # We can try querying HF without a token (public access), but it is often throttled.
    # Therefore we only try if explicitly allowed or token is present
    if hf_token or os.environ.get("HF_API_TOKEN"):
        token = hf_token or os.environ.get("HF_API_TOKEN")
        response = query_huggingface_api(formatted_messages, token)
        if response:
            return response
            
    # Fallback to local template generator
    return generate_local_explanation(query, retrieved_papers, chat_history)
