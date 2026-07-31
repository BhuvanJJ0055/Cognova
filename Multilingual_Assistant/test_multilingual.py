import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Setup path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from Multilingual_Assistant.multilingual_agent import MultilingualAgent
except ImportError:
    from src.modules.multilingual import MultilingualAgent


class TestMultilingualAgent(unittest.TestCase):

    def setUp(self):
        self.agent = MultilingualAgent()

    def test_local_language_detection(self):
        # Hindi Devanagari
        self.assertEqual(self.agent.detect_language("मुझे छाती में दर्द है"), "hi")
        # Kannada script
        self.assertEqual(self.agent.detect_language("ನನಗೆ ತಲೆನೋವು ಇದೆ"), "kn")
        # Spanish keyword
        self.assertEqual(self.agent.detect_language("¿Cuáles son los síntomas de la diabetes?"), "es")
        # French keyword
        self.assertEqual(self.agent.detect_language("Quels sont les symptômes de l'asthme?"), "fr")
        # German keyword
        self.assertEqual(self.agent.detect_language("Was sind die Symptome von Krebs?"), "de")

    def test_code_switching_detection(self):
        # Hinglish
        res_hi = self.agent.detect_and_translate("Madhumeha diabetes ke symptoms kya hain?")
        self.assertEqual(res_hi["primary_language"], "hi")
        self.assertTrue(res_hi["is_mixed"])
        self.assertIn("diabetes", res_hi["translated_query"].lower())

        # Kanglish
        res_kn = self.agent.detect_and_translate("Nange headache ide, what should I do?")
        self.assertEqual(res_kn["primary_language"], "kn")
        self.assertTrue(res_kn["is_mixed"])

    def test_cross_lingual_context_pronoun_resolution(self):
        # Simulating past chat turn discussing Diabetes
        chat_history = [
            {"user_input": "What is Type 2 Diabetes?", "response": "Diabetes is a chronic condition."}
        ]
        
        # Current query in Hindi with pronoun "इसके" (its)
        res_hi = self.agent.detect_and_translate("इसके क्या लक्षण हैं?", chat_history=chat_history)
        self.assertIn("Diabetes", res_hi["translated_query"])

        # Current query in Kannada with pronoun "ಇದರ" (its)
        res_kn = self.agent.detect_and_translate("ಇದರ ಚಿಕಿತ್ಸೆ ಏನು?", chat_history=chat_history)
        self.assertIn("Diabetes", res_kn["translated_query"])

    def test_ambiguity_detection_and_clarification(self):
        # Very short vague input
        res = self.agent.detect_and_translate("ilaaj?")
        self.assertTrue(res["is_ambiguous"])
        self.assertNotEqual(res["clarification_question"], "")

        # Clear input
        res_clear = self.agent.detect_and_translate("What are the symptoms of asthma?")
        self.assertFalse(res_clear["is_ambiguous"])

    def test_factual_consistency_checker(self):
        # Good overlap
        context = [{"title": "Asthma", "text": "Asthma causes narrowing of the airways of the lungs."}]
        response_english = "Asthma leads to narrowing of the lungs airways."
        
        score, aligned, missing = self.agent.check_factual_consistency(response_english, context)
        self.assertGreater(score, 0.70)
        self.assertIn("narrowing", aligned)

        # Low overlap / hallucinated content
        response_hallucinated = "This is treated with penicillin and checking blood insulin levels."
        score_hall, aligned_hall, missing_hall = self.agent.check_factual_consistency(response_hallucinated, context)
        self.assertLess(score_hall, 0.40)
        self.assertIn("penicillin", missing_hall)

    @patch('requests.post')
    def test_api_integration_mock(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{\n  "primary_language": "hi",\n  "is_mixed": false,\n  "detected_languages": ["hi"],\n  "translated_query": "I have chest pain, what is this?",\n  "is_ambiguous": false,\n  "clarification_question": ""\n}'
                    }]
                }
            }]
        }
        mock_post.return_value = mock_response

        prompt = "मुझे छाती में दर्द है, यह क्या है?"
        result = self.agent.detect_and_translate(prompt, [], "mock_api_key")

        self.assertEqual(result["primary_language"], "hi")
        self.assertFalse(result["is_mixed"])
        self.assertEqual(result["translated_query"], "I have chest pain, what is this?")


if __name__ == "__main__":
    unittest.main()
