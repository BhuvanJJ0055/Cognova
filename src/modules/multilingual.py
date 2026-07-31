"""
Task 6 - Multilingual Translation & Cross-Lingual Assistant Module
Author: Bhuvan J J

Per-turn language detection, code-switching parser (Hinglish, Kanglish, Spanglish, Franglish, Denglish),
pivot translation through English, cross-lingual context maintenance (EN, HI, KN, ES, FR, DE),
ambiguity resolution, grounded response synthesis, and factual consistency evaluation.
"""

import os
import re
import json
import requests
from typing import Optional, Any, List, Dict, Tuple

SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi (हिन्दी / Hinglish)",
    "kn": "Kannada (ಕನ್ನಡ / Kanglish)",
    "es": "Spanish (Español / Spanglish)",
    "fr": "French (Français / Franglish)",
    "de": "German (Deutsch / Denglish)"
}

HINGLISH_MAP = {
    "madhumeha": "diabetes",
    "sugar": "diabetes",
    "lakshan": "symptoms",
    "lakshane": "symptoms",
    "symptoms kya hai": "what are the symptoms",
    "symptoms kya hain": "what are the symptoms",
    "ilaj": "treatment",
    "ilaaj": "treatment",
    "kya hai": "what is",
    "kaise": "how",
    "deduct ho gaya": "deducted twice",
    "paise": "payment refund",
    "shwas": "asthma",
    "kanser": "cancer",
    "bukhar": "fever",
    "khansi": "cough",
    "sir dard": "headache",
    "pet dard": "stomach ache",
    "bimar": "sick",
    "dawa": "medicine"
}

KANGLISH_MAP = {
    "madhumeha": "diabetes",
    "lakshangalu": "symptoms",
    "laksana": "symptoms",
    "chikitse": "treatment",
    "agide": "has occurred",
    "deduct agide": "deducted twice",
    "hogaide": "deducted",
    "talenovu": "headache",
    "jvara": "fever",
    "kasa": "cough",
    "aushadha": "medicine",
    "nange": "i have",
    "yenu": "what"
}

SPANGLISH_MAP = {
    "sintomas": "symptoms",
    "síntomas": "symptoms",
    "tratamiento": "treatment",
    "dolor de cabeza": "headache",
    "fiebre": "fever",
    "remendio": "remedy",
    "medicina": "medicine",
    "doctor": "doctor"
}

FRANGLISH_MAP = {
    "symptomes": "symptoms",
    "symptômes": "symptoms",
    "traitement": "treatment",
    "mal de tete": "headache",
    "mal de tête": "headache",
    "fievre": "fever",
    "fièvre": "fever",
    "medecin": "doctor"
}

DENGLISH_MAP = {
    "symptome": "symptoms",
    "symptomen": "symptoms",
    "behandlung": "treatment",
    "kopfschmerzen": "headache",
    "fieber": "fever",
    "medizin": "medicine",
    "arzt": "doctor"
}


try:
    from src.modules.medical_qa import MedicalRetriever
except ImportError:
    try:
        from modules.medical_qa import MedicalRetriever
    except ImportError:
        MedicalRetriever = None


class MultilingualAssistant:
    """Multi-language parser, context resolver, and cross-lingual translation engine."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.medical_retriever: Any = None
        self.arxiv_retriever: Any = None
        self.sentiment_bot: Any = None

    def initialize_retrievers(self, csv_path: Optional[str] = None):
        """Initializes internal medical retriever safely."""
        if self.medical_retriever is None and MedicalRetriever is not None:
            try:
                self.medical_retriever = MedicalRetriever(fallback_csv_path=csv_path)
            except Exception as e:
                print(f"[Multilingual] MedicalRetriever load note: {e}")

    def clean_json_response(self, text: str) -> str:
        """Cleans potential markdown wrapping (e.g. ```json) around JSON string."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def detect_language(self, text: str) -> str:
        """Pattern & script-based language detector for fast local processing."""
        text_lower = text.lower()
        
        # Devanagari script (Hindi)
        if any("\u0900" <= char <= "\u097F" for char in text):
            return "hi"
        # Kannada script
        if any("\u0C80" <= char <= "\u0CFF" for char in text):
            return "kn"
        # Spanish keywords / accents
        if any(char in text for char in ["¿", "¡", "ñ", "á", "é", "í", "ó", "ú"]) or \
           any(re.search(r'\b' + w + r'\b', text_lower) for w in ["que", "como", "donde", "hola", "sintomas", "tratamiento", "por", "favor", "enfermedad"]):
            return "es"
        # French keywords / accents
        if any(char in text for char in ["ç", "è", "ê", "à", "ù"]) or \
           any(re.search(r'\b' + w + r'\b', text_lower) for w in ["comment", "bonjour", "merci", "avec", "symptomes", "traitement", "pourquoi", "est"]):
            return "fr"
        # German keywords / accents
        if any(char in text for char in ["ä", "ö", "ü", "ß"]) or \
           any(re.search(r'\b' + w + r'\b', text_lower) for w in ["wie", "was", "wo", "hallo", "symptome", "behandlung", "danke", "bitte", "krankheit"]):
            return "de"
        # Hinglish / Kanglish check
        if any(w in text_lower for w in HINGLISH_MAP.keys()):
            return "hi"
        if any(w in text_lower for w in KANGLISH_MAP.keys()):
            return "kn"

        return "en"

    def translate_to_english_local(self, text: str, src_lang: str) -> Tuple[str, bool]:
        """Local open-source rule & dictionary pivot translator."""
        text_lower = text.lower()
        is_mixed = False
        translated = text_lower

        if src_lang == "hi":
            for k, v in HINGLISH_MAP.items():
                if k in translated:
                    translated = translated.replace(k, v)
                    is_mixed = True
        elif src_lang == "kn":
            for k, v in KANGLISH_MAP.items():
                if k in translated:
                    translated = translated.replace(k, v)
                    is_mixed = True
        elif src_lang == "es":
            for k, v in SPANGLISH_MAP.items():
                if k in translated:
                    translated = translated.replace(k, v)
                    is_mixed = True
        elif src_lang == "fr":
            for k, v in FRANGLISH_MAP.items():
                if k in translated:
                    translated = translated.replace(k, v)
                    is_mixed = True
        elif src_lang == "de":
            for k, v in DENGLISH_MAP.items():
                if k in translated:
                    translated = translated.replace(k, v)
                    is_mixed = True

        # General English code-switch markers check
        english_words = {"symptoms", "treatment", "headache", "fever", "cough", "doctor", "payment", "deduct", "help", "please", "what", "how"}
        input_words = set(re.findall(r'\b\w+\b', text_lower))
        if len(input_words.intersection(english_words)) > 0 and src_lang != "en":
            is_mixed = True

        # Capitalize first letter of translated query
        translated = translated.strip()
        if translated:
            translated = translated[0].upper() + translated[1:]
        else:
            translated = text

        return translated, is_mixed

    def resolve_context_references(self, query: str, chat_history: Optional[list]) -> str:
        """Resolves pronouns (its, this, यह, இதன், de esto) using previous chat turns."""
        if not chat_history:
            return query

        query_lower = query.lower()
        pronouns = ["its", "this", "it", "their", "यह", "इसका", "इसके", "ಇದರ", "ಇದಕ್ಕೆ", "de esto", "sus", "ses", "ce", "diese"]

        # Check if prompt contains a pronoun reference
        has_pronoun = any(re.search(r'\b' + re.escape(p) + r'\b', query_lower) for p in pronouns)
        if not has_pronoun:
            return query

        # Search past turns for disease/topic entities
        entities = []
        for turn in reversed(chat_history[-5:]):
            prev_text = (str(turn.get("prompt", "")) + " " + str(turn.get("user_input", "")) + " " + str(turn.get("translated", ""))).lower()
            topic_matches = re.findall(r'\b(diabetes|asthma|cancer|hypertension|covid|headache|fever|cough|arthritis|depression|malaria|tb)\b', prev_text)
            if topic_matches:
                entities.extend(topic_matches)
                break

        if entities:
            topic = entities[0].title()
            return f"{query} (referring to {topic})"

        return query

    def check_ambiguity_local(self, prompt: str, primary_lang: str) -> Tuple[bool, str]:
        """Local open-source ambiguity evaluator for short or vague inputs."""
        words = re.findall(r'\b\w+\b', prompt.strip().lower())
        stop_words = {"what", "how", "is", "kya", "yenu", "que", "comment", "wie", "hai", "ide", "aqui"}
        meaningful = [w for w in words if w not in stop_words]

        is_ambiguous = len(words) <= 2 and len(meaningful) <= 1
        
        clarification = ""
        if is_ambiguous:
            clarifications = {
                "en": "Could you please specify which medical condition or topic you are asking about?",
                "hi": "क्या आप कृपया स्पष्ट कर सकते हैं कि आप किस बीमारी या विषय के बारे में पूछ रहे हैं?",
                "kn": "ನೀವು ಯಾವ ಆರೋಗ್ಯ ಸಮಸ್ಯೆ ಅಥವಾ ವಿಷಯದ ಬಗ್ಗೆ ಕೇಳುತ್ತಿದ್ದೀರಿ ಎಂದು ಸ್ಪಷ್ಟಪಡಿಸುವಿರಾ?",
                "es": "¿Podría especificar sobre qué condición médica o tema está preguntando?",
                "fr": "Pourriez-vous s'il vous plaît préciser de quelle condition médicale ou de quel sujet il s'agit ?",
                "de": "Könnten Sie bitte angeben, nach welcher medizinischen Erkrankung oder welchem Thema Sie fragen?"
            }
            clarification = clarifications.get(primary_lang, clarifications["en"])

        return is_ambiguous, clarification

    def detect_and_translate(
        self,
        prompt: str,
        chat_history: Optional[list] = None,
        api_key: Optional[str] = None
    ) -> dict:
        """Detects language, translates to English pivot, handles code-switching,

        resolves context pronouns, and identifies ambiguous queries.
        """
        key = api_key or self.api_key

        # 1. Fallback / Local Analysis
        local_lang = self.detect_language(prompt)
        local_translated, local_is_mixed = self.translate_to_english_local(prompt, local_lang)
        local_resolved = self.resolve_context_references(local_translated, chat_history)
        local_ambiguous, local_clarification = self.check_ambiguity_local(prompt, local_lang)

        fallback_result = {
            "primary_language": local_lang,
            "language_name": SUPPORTED_LANGS.get(local_lang, "English"),
            "is_mixed": local_is_mixed,
            "detected_languages": [local_lang, "en"] if local_is_mixed else [local_lang],
            "translated_query": local_resolved,
            "is_ambiguous": local_ambiguous,
            "clarification_question": local_clarification
        }

        if not key:
            return fallback_result

        # 2. LLM Model Pipeline Call with automatic model fallback
        history_block = ""
        if chat_history:
            history_block = "Conversation Chat History so far:\n"
            for turn in chat_history[-5:]:
                u_text = turn.get("user_input") or turn.get("prompt", "")
                r_text = turn.get("response") or turn.get("answer", "")
                history_block += f"User: {u_text}\nAI: {r_text}\n"
            history_block += "\n"

        system_instruction = (
            "You are an expert Multilingual Parser. Analyze the user query within the context of chat history.\n"
            "Detect the primary language, code-switched/mixed language status, translate into clear English "
            "resolving pronouns (e.g. 'its symptoms' -> 'symptoms of diabetes'), and determine if the query is ambiguous.\n\n"
            "Supported languages:\n"
            "- en (English)\n- hi (Hindi/Hinglish)\n- kn (Kannada/Kanglish)\n"
            "- es (Spanish/Spanglish)\n- fr (French/Franglish)\n- de (German/Denglish)\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "primary_language": "hi" | "kn" | "es" | "fr" | "de" | "en",\n'
            '  "is_mixed": true | false,\n'
            '  "detected_languages": ["hi", "en"],\n'
            '  "translated_query": "English translation with context/pronouns fully resolved",\n'
            '  "is_ambiguous": true | false,\n'
            '  "clarification_question": "Polite clarification in primary_language if ambiguous, else empty string"\n'
            "}\n\n"
            f"{history_block}"
            f"User Input: {prompt}\n"
            "JSON Response:"
        )

        payload = {"contents": [{"parts": [{"text": system_instruction}]}]}
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}

        models_to_try = [
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-pro"
        ]

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=12)
                if res.status_code == 200:
                    data = res.json()
                    text = data.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    cleaned_text = self.clean_json_response(text)
                    parsed = json.loads(cleaned_text)
                    parsed["language_name"] = SUPPORTED_LANGS.get(parsed.get("primary_language", "en"), "English")
                    return parsed
            except Exception:
                continue

        return fallback_result

    def generate_response(
        self,
        user_prompt: str,
        translated_query: str,
        lang_info: dict,
        context_docs: list,
        chat_history: Optional[list] = None,
        api_key: Optional[str] = None
    ) -> dict:
        """Generates a grounded answer in the target user language,

        accompanied by an English version for factual consistency validation.
        """
        key = api_key or self.api_key
        target_lang = lang_info.get("primary_language", "en")
        is_mixed = lang_info.get("is_mixed", False)

        # Build local fallback response from documents
        doc_summary = ""
        if context_docs:
            top_doc = context_docs[0]
            doc_summary = top_doc.get("answer") or top_doc.get("text") or top_doc.get("content", "")
        else:
            doc_summary = "No exact reference document found in the knowledge base."

        # Template translation fallback
        fallback_target_response = doc_summary
        if target_lang != "en" and doc_summary and not doc_summary.startswith("No exact"):
            fallback_target_response = f"[{SUPPORTED_LANGS.get(target_lang, target_lang).upper()}] {doc_summary}"

        fallback_res = {
            "response": fallback_target_response,
            "response_english": doc_summary
        }

        if not key:
            return fallback_res

        context_block = ""
        if context_docs:
            context_block = "Factual Reference Documents retrieved from database:\n"
            for idx, doc in enumerate(context_docs):
                t = doc.get("title") or doc.get("question") or f"Doc #{idx+1}"
                c = doc.get("answer") or doc.get("text") or ""
                context_block += f"Document [{idx+1}] - {t}:\n{c}\n\n"
        else:
            context_block = "No reference documents found. Politely inform the user.\n\n"

        history_block = ""
        if chat_history:
            history_block = "Conversation History:\n"
            for turn in chat_history[-5:]:
                u_text = turn.get("user_input") or turn.get("prompt", "")
                r_text = turn.get("response") or turn.get("answer", "")
                history_block += f"User: {u_text}\nAI: {r_text}\n"
            history_block += "\n"

        instruction = (
            "You are a Multilingual Expert Assistant. Generate an answer strictly grounded in the provided factual reference documents.\n"
            f"1. Generate main response in user's target language: {target_lang}. "
            f"If is_mixed is True ({is_mixed}), adopt a natural code-switched style (e.g., Hinglish/Kanglish) while keeping medical details exact.\n"
            "2. Make sure all factual claims are strictly present in the reference documents.\n"
            "3. If context does not contain the answer, state that politely in the target language.\n"
            "4. Return ONLY a JSON object:\n"
            "{\n"
            '  "response": "Response in user target language or code-switched dialect",\n'
            '  "response_english": "Direct English translation of response for factual overlap auditing"\n'
            "}\n\n"
            f"{context_block}"
            f"{history_block}"
            f"User Original Prompt: {user_prompt}\n"
            f"User English Translation: {translated_query}\n"
            "JSON Response:"
        )

        payload = {"contents": [{"parts": [{"text": instruction}]}]}
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}

        models_to_try = [
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-pro"
        ]

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    text = data.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    cleaned_text = self.clean_json_response(text)
                    return json.loads(cleaned_text)
            except Exception:
                continue

        return fallback_res

    def check_factual_consistency(self, response_english: str, context_docs: list) -> Tuple[float, List[str], List[str]]:
        """Determines token overlap alignment between English response version and source documents."""
        if not context_docs or not response_english:
            return 1.0, [], []

        # Combine source text
        combined_source = ""
        for doc in context_docs:
            combined_source += " " + str(doc.get("text", "")) + " " + str(doc.get("answer", "")) + " " + str(doc.get("content", ""))
        combined_source = combined_source.lower()

        resp_cleaned = re.sub(r'[^\w\s]', '', response_english.lower())
        resp_words = set(resp_cleaned.split())

        stop_words = {
            'is', 'the', 'of', 'and', 'a', 'in', 'to', 'that', 'it', 'for', 'on', 'with', 'as', 'this',
            'are', 'was', 'by', 'an', 'be', 'at', 'from', 'or', 'your', 'have', 'has', 'not', 'will',
            'can', 'should', 'would', 'could', 'about', 'more', 'how', 'what', 'which', 'who', 'our',
            'we', 'they', 'he', 'she', 'you', 'if', 'then', 'else', 'but', 'there', 'their', 'them',
            'these', 'those', 'also', 'such', 'only', 'very', 'here', 'response', 'english', 'direct'
        }

        keywords = {w for w in resp_words if w not in stop_words and len(w) > 2}
        if not keywords:
            return 1.0, [], []

        aligned = []
        missing = []

        for word in keywords:
            if word in combined_source:
                aligned.append(word)
            else:
                missing.append(word)

        score = len(aligned) / len(keywords) if keywords else 1.0
        return round(score, 2), sorted(aligned), sorted(missing)


MultilingualAgent = MultilingualAssistant
