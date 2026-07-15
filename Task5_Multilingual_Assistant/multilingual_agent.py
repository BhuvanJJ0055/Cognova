import os
import json
import requests
import re
import sys

# Add parent path and task directories to allow cross-task imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "Task1_Sentiment_Chatbot"))
sys.path.append(os.path.join(parent_dir, "Task2_Medical_QA_Chatbot"))
sys.path.append(os.path.join(parent_dir, "Task3_ArXiv_CS_Chatbot"))
sys.path.append(os.path.join(parent_dir, "Task4_Multimodal_Assistant"))

from Task2_Medical_QA_Chatbot.build_index import MedicalRetriever
from Task3_ArXiv_CS_Chatbot.build_arxiv_index import ArXivRetriever
from Task1_Sentiment_Chatbot.chatbot_v2 import SupportChatbot, score_mood_vader

class MultilingualAgent:
    """Manages language detection, mixed-language parsing, cross-lingual context maintenance,

    and evidence-grounded response generation across Hindi, Kannada, and English.
    """

    def __init__(self):
        # We will instantiate retrievers lazily or pass them from app.py to conserve memory.
        self.medical_retriever = None
        self.arxiv_retriever = None
        self.sentiment_bot = None

    def initialize_retrievers(self, csv_path=None):
        if self.medical_retriever is None:
            try:
                self.medical_retriever = MedicalRetriever(fallback_csv_path=csv_path)
            except Exception as e:
                print(f"[Multilingual] Failed to load MedicalRetriever: {e}")
        if self.arxiv_retriever is None:
            try:
                self.arxiv_retriever = ArXivRetriever()
            except Exception as e:
                print(f"[Multilingual] Failed to load ArXivRetriever: {e}")
        if self.sentiment_bot is None:
            try:
                self.sentiment_bot = SupportChatbot(model_type="vader")
            except Exception as e:
                print(f"[Multilingual] Failed to load SupportChatbot: {e}")

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

    def detect_and_translate(self, prompt: str, chat_history: list, api_key: str) -> dict:
        """Calls Gemini API to analyze the prompt, detect language, and translate it to English.

        Takes chat history into account to resolve context/pronouns.
        """
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
        
        # Build chat history context block
        history_block = ""
        if chat_history:
            history_block = "Conversation Chat History so far:\n"
            for turn in chat_history[-5:]:  # Limit to last 5 turns for context efficiency
                history_block += f"User: {turn.get('prompt')}\n"
                history_block += f"AI: {turn.get('response')}\n"
            history_block += "\n"

        system_instruction = (
            "You are a Multilingual Language Parser. Your job is to analyze the user's input query "
            "within the context of the conversation history. Detect the primary language, determine "
            "if it is a mixed-language/code-switched input (like Hinglish or Kanglish), translate it into "
            "clear English resolving any pronouns or contextual references, and check if the query is too ambiguous.\n\n"
            "Supported languages:\n"
            "- en (English)\n"
            "- hi (Hindi/Hinglish)\n"
            "- kn (Kannada/Kanglish)\n\n"
            "Return ONLY a JSON object with this structure:\n"
            "{\n"
            "  \"primary_language\": \"hi\" | \"kn\" | \"en\",\n"
            "  \"is_mixed\": true | false,\n"
            "  \"detected_languages\": [\"en\", \"hi\"],\n"
            "  \"translated_query\": \"The query fully translated to clear English, resolving context references (e.g. 'its treatment' should resolve to 'the treatment of [disease name mentioned in history]')\",\n"
            "  \"is_ambiguous\": true | false,\n"
            "  \"clarification_question\": \"A polite question in the detected primary language asking for clarification, ONLY if is_ambiguous is true. Otherwise empty string.\"\n"
            "}\n\n"
            f"{history_block}"
            f"User Input: {prompt}\n"
            "JSON Response:"
        )

        payload = {
            "contents": [{
                "parts": [{"text": system_instruction}]
            }]
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                data = res.json()
                text = data.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                cleaned_text = self.clean_json_response(text)
                return json.loads(cleaned_text)
            else:
                # Fallback dictionary if API fails
                return {
                    "primary_language": "en",
                    "is_mixed": False,
                    "detected_languages": ["en"],
                    "translated_query": prompt,
                    "is_ambiguous": False,
                    "clarification_question": ""
                }
        except Exception as e:
            print(f"[Multilingual] Error detecting language: {e}")
            return {
                "primary_language": "en",
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
        chat_history: list,
        api_key: str
    ) -> dict:
        """Generates a grounded explanation answering the prompt in the user's target language,

        accompanied by an English version of the response for factual overlap validation.
        """
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
        
        # Format the context block from retrieved documents
        context_block = ""
        if context_docs:
            context_block = "Factual Reference Documents retrieved from database:\n"
            for idx, doc in enumerate(context_docs):
                context_block += f"Document [{idx+1}]: {doc.get('title')}\nContent: {doc.get('text')}\n\n"
        else:
            context_block = "No reference documents found. Use general knowledge but warn the user.\n\n"

        # Format conversation history
        history_block = ""
        if chat_history:
            history_block = "Conversation Chat History:\n"
            for turn in chat_history[-5:]:
                history_block += f"User: {turn.get('prompt')}\n"
                history_block += f"AI: {turn.get('response')}\n"
            history_block += "\n"

        target_lang = lang_info.get("primary_language", "en")
        is_mixed = lang_info.get("is_mixed", False)

        instruction = (
            "You are a Multilingual Expert Assistant. You answer questions strictly grounded in the provided factual reference documents.\n"
            "Your output must contain two parts: the response in the user's primary/target language, and a translation of that response in English.\n\n"
            "Instructions:\n"
            f"1. Generate your main response in the user's target language: {target_lang}. "
            f"If is_mixed is True ({is_mixed}), you can generate a natural, conversational response that fits the tone (e.g. Hinglish or Kanglish) while keeping the medical/scientific information clear.\n"
            "2. Make sure all medical/scientific claims are strictly grounded in the provided reference documents. Do not hallucinate or add facts not present in the context.\n"
            "3. If the reference documents do not contain the answer, politely state that you cannot find the answer in the database, in the user's language.\n"
            "4. Return your output ONLY as a JSON object with this structure:\n"
            "{\n"
            "  \"response\": \"Your grounded response in the target language (Hindi, Kannada, or code-switched Hinglish/Kanglish).\",\n"
            "  \"response_english\": \"Direct English translation of the response. This will be used to run factual validation check against source documents.\"\n"
            "}\n\n"
            f"{context_block}"
            f"{history_block}"
            f"User Original Prompt: {user_prompt}\n"
            f"User English Translation: {translated_query}\n"
            "JSON Response:"
        )

        payload = {
            "contents": [{
                "parts": [{"text": instruction}]
            }]
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                data = res.json()
                text = data.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                cleaned_text = self.clean_json_response(text)
                return json.loads(cleaned_text)
            else:
                return {
                    "response": f"Error: Failed to contact Gemini API (HTTP {res.status_code})",
                    "response_english": "Error communicating with API."
                }
        except Exception as e:
            print(f"[Multilingual] Error generating response: {e}")
            return {
                "response": f"Exception occurred: {e}",
                "response_english": "Exception during generation."
            }

    def check_factual_consistency(self, response_english: str, context_docs: list) -> tuple:
        """Determines overlap alignment between the English version of the response and retrieved source documents."""
        if not context_docs or not response_english:
            return 0.0, [], []

        # Combine all source context
        combined_source = " ".join([doc.get("text", "") for doc in context_docs]).lower()
        
        # Tokenize and clean response words
        resp_cleaned = re.sub(r'[^\w\s]', '', response_english.lower())
        resp_words = set(resp_cleaned.split())

        # Simple set of common stop words to filter out
        stop_words = {
            'is', 'the', 'of', 'and', 'a', 'in', 'to', 'that', 'it', 'for', 'on', 'with', 'as', 'this',
            'are', 'was', 'by', 'an', 'be', 'at', 'from', 'or', 'your', 'have', 'has', 'not', 'will',
            'can', 'should', 'would', 'could', 'about', 'more', 'how', 'what', 'which', 'who', 'our',
            'we', 'they', 'he', 'she', 'you', 'if', 'then', 'else', 'but', 'there', 'their', 'them',
            'these', 'those', 'also', 'such', 'only', 'very', 'here'
        }
        
        # Filter technical terms (nouns/adjectives)
        resp_keywords = {w for w in resp_words if w not in stop_words and len(w) > 2}

        if not resp_keywords:
            return 1.0, [], []

        aligned = []
        missing = []

        # Check alignment against source text
        for word in resp_keywords:
            if word in combined_source:
                aligned.append(word)
            else:
                missing.append(word)

        score = len(aligned) / len(resp_keywords)
        return score, aligned, missing
