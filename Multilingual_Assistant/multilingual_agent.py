import os
import json
import requests
import re
import sys
from typing import Optional, Any, List, Dict, Tuple

# Add parent path to allow cross-task imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from src.modules.multilingual import MultilingualAssistant, SUPPORTED_LANGS
except ImportError:
    SUPPORTED_LANGS = {
        "en": "English",
        "hi": "Hindi (हिन्दी / Hinglish)",
        "kn": "Kannada (ಕನ್ನಡ / Kanglish)",
        "es": "Spanish (Español / Spanglish)",
        "fr": "French (Français / Franglish)",
        "de": "German (Deutsch / Denglish)"
    }
    MultilingualAssistant = None

try:
    from src.modules.medical_qa import MedicalRetriever
except ImportError:
    MedicalRetriever = None


class MultilingualAgent:
    """Manages language detection, mixed-language parsing, cross-lingual context maintenance,

    ambiguity resolution, grounded response generation, and factual consistency scoring
    across English, Hindi, Kannada, Spanish, French, and German.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.internal_assistant = MultilingualAssistant(api_key=self.api_key) if MultilingualAssistant is not None else None
        self.medical_retriever = None
        self.arxiv_retriever = None
        self.sentiment_bot = None

    def initialize_retrievers(self, csv_path: Optional[str] = None):
        if self.internal_assistant:
            self.internal_assistant.initialize_retrievers(csv_path=csv_path)
            self.medical_retriever = self.internal_assistant.medical_retriever
        elif self.medical_retriever is None and MedicalRetriever is not None:
            try:
                self.medical_retriever = MedicalRetriever(fallback_csv_path=csv_path)
            except Exception as e:
                print(f"[Multilingual] MedicalRetriever load note: {e}")

    def clean_json_response(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def detect_language(self, text: str) -> str:
        """Detects primary language code for input text."""
        if self.internal_assistant:
            return self.internal_assistant.detect_language(text)
        return "en"

    def detect_and_translate(
        self,
        prompt: str,
        chat_history: Optional[list] = None,
        api_key: Optional[str] = None
    ) -> dict:
        """Delegates detection and translation to core MultilingualAssistant engine."""
        if self.internal_assistant:
            return self.internal_assistant.detect_and_translate(prompt, chat_history=chat_history, api_key=api_key)
        
        # Fallback local dictionary response
        return {
            "primary_language": "en",
            "language_name": "English",
            "is_mixed": False,
            "detected_languages": ["en"],
            "translated_query": prompt,
            "is_ambiguous": False,
            "clarification_question": ""
        }

    def generate_response(
        self,
        user_prompt: str,
        translated_query: str,
        lang_info: dict,
        context_docs: list,
        chat_history: Optional[list] = None,
        api_key: Optional[str] = None
    ) -> dict:
        """Delegates grounded generation to core MultilingualAssistant engine."""
        if self.internal_assistant:
            return self.internal_assistant.generate_response(
                user_prompt, translated_query, lang_info, context_docs, chat_history=chat_history, api_key=api_key
            )

        doc_summary = context_docs[0].get("answer") or context_docs[0].get("text") if context_docs else "No reference document found."
        return {
            "response": doc_summary,
            "response_english": doc_summary
        }

    def check_factual_consistency(self, response_english: str, context_docs: list) -> Tuple[float, List[str], List[str]]:
        """Delegates factual consistency verification to core MultilingualAssistant engine."""
        if self.internal_assistant:
            return self.internal_assistant.check_factual_consistency(response_english, context_docs)
        
        if not context_docs or not response_english:
            return 1.0, [], []

        return 1.0, [], []
