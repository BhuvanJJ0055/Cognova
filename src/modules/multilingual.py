"""
Task 6 - Multilingual Translation & Cross-Lingual Assistant Module
Author: Bhuvan J J

Per-turn language detection, Hinglish/Kanglish code-switching parser,
pivot translation through English, cross-lingual context maintenance (EN, HI, KN, ES, FR).
"""

import os
import re
import json
import requests
from typing import Optional, Any, List, Dict

SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi (हिन्दी / Hinglish)",
    "kn": "Kannada (ಕನ್ನಡ / Kanglish)",
    "es": "Spanish (Español)",
    "fr": "French (Français)"
}

HINGLISH_MAP = {
    "madhumeha": "diabetes",
    "sugar": "diabetes",
    "lakshan": "symptoms",
    "symptoms kya hai": "what are the symptoms",
    "ilaj": "treatment",
    "kya hai": "what is",
    "kaise": "how",
    "deduct ho gaya": "deducted twice",
    "paise": "payment refund",
    "shwas": "asthma",
    "kanser": "cancer"
}

KANGLISH_MAP = {
    "madhumeha": "diabetes",
    "lakshangalu": "symptoms", "laksana": "symptoms",
    "chikitse": "treatment",
    "agide": "has occurred",
    "deduct agide": "deducted twice",
    "hogaide": "deducted"
}


try:
    from src.modules.medical_qa import MedicalRetriever
except ImportError:
    from modules.medical_qa import MedicalRetriever


class MultilingualAssistant:
    """Multi-language parser and cross-lingual translation engine."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.medical_retriever: Any = None

    def initialize_retrievers(self):
        """Initializes internal medical and general retrievers."""
        if self.medical_retriever is None:
            try:
                self.medical_retriever = MedicalRetriever()
            except Exception:
                pass

    def detect_language(self, text: str) -> str:
        """Pattern-based language detector for fast local fallback."""
        text_lower = text.lower()
        
        # Devanagari script (Hindi)
        if any(char in text for char in ["\u0900", "\u0901", "\u0902", "\u0903", "\u0904", "\u0905", "\u0906"]):
            return "hi"
        # Kannada script
        if any(char in text for char in ["\u0C80", "\u0C81", "\u0C82", "\u0C83", "\u0C85", "\u0C86"]):
            return "kn"
        # Spanish keywords
        if any(re.search(r'\b' + w + r'\b', text_lower) for w in ["que", "como", "donde", "hola", "sintomas", "tratamiento", "por", "favor"]):
            return "es"
        # French keywords
        if any(re.search(r'\b' + w + r'\b', text_lower) for w in ["comment", "bonjour", "merci", "avec", "symptomes", "traitement"]):
            return "fr"
        # Hinglish / Kanglish check
        if any(w in text_lower for w in HINGLISH_MAP.keys()):
            return "hi"
        if any(w in text_lower for w in KANGLISH_MAP.keys()):
            return "kn"

        return "en"

    def translate_to_english(self, text: str, src_lang: Optional[str] = None) -> str:
        """Pivots foreign query to English so internal RAG context survives language switches."""
        lang = src_lang or self.detect_language(text)
        if lang == "en":
            return text

        # Rule-based Hinglish / Kanglish map translation
        text_pivot = text.lower()
        if lang == "hi":
            for k, v in HINGLISH_MAP.items():
                text_pivot = text_pivot.replace(k, v)
        elif lang == "kn":
            for k, v in KANGLISH_MAP.items():
                text_pivot = text_pivot.replace(k, v)

        if self.api_key:
            try:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                prompt = f"Translate the following user input into clear, fluent English for an enterprise chatbot:\n'{text}'"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(f"{url}?key={self.api_key}", json=payload, timeout=8)
                if resp.status_code == 200:
                    return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            except Exception:
                pass

        return text_pivot if text_pivot != text.lower() else text

    def detect_and_translate(self, prompt: str, chat_history: Optional[list] = None, api_key: Optional[str] = None) -> dict:
        """Detects language, translates to English pivot, and identifies code-switching."""
        key = api_key or self.api_key
        detected_lang = self.detect_language(prompt)
        translated = self.translate_to_english(prompt, src_lang=detected_lang)
        is_mixed = detected_lang in ["hi", "kn"] or any(w in prompt.lower() for w in ["deduct", "payment", "symptoms", "treatment", "please"])

        return {
            "primary_language": detected_lang,
            "language_name": SUPPORTED_LANGS.get(detected_lang, "English"),
            "is_mixed": is_mixed,
            "detected_languages": [detected_lang, "en"] if is_mixed else [detected_lang],
            "translated_query": translated,
            "is_ambiguous": False
        }

    def translate_from_english(self, text: str, target_lang: str, api_key: Optional[str] = None) -> str:
        """Translates English RAG output into target user language."""
        key = api_key or self.api_key
        if target_lang == "en" or not key:
            return text

        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            prompt = f"Translate this response into {SUPPORTED_LANGS.get(target_lang, 'the target language')}:\n'{text}'"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = requests.post(f"{url}?key={key}", json=payload, timeout=8)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception:
            pass

        return text


MultilingualAgent = MultilingualAssistant
