import os
import unittest
from unittest.mock import patch, MagicMock
import sys

# Setup path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "Task1_Sentiment_Chatbot"))
sys.path.append(os.path.join(parent_dir, "Task2_Medical_QA_Chatbot"))
sys.path.append(os.path.join(parent_dir, "Task3_ArXiv_CS_Chatbot"))
sys.path.append(os.path.join(parent_dir, "Task4_Multimodal_Assistant"))
sys.path.append(os.path.join(parent_dir, "Task5_Multilingual_Assistant"))

from Task5_Multilingual_Assistant.multilingual_agent import MultilingualAgent

class TestMultilingualAgent(unittest.TestCase):

    def setUp(self):
        self.agent = MultilingualAgent()

    @patch('requests.post')
    def test_detect_and_translate_hindi(self, mock_post):
        # Mock successful Gemini API response
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
        self.assertFalse(result["is_ambiguous"])

    @patch('requests.post')
    def test_detect_and_translate_kannada_mixed(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{\n  "primary_language": "kn",\n  "is_mixed": true,\n  "detected_languages": ["kn", "en"],\n  "translated_query": "I have headache, what should I do?",\n  "is_ambiguous": false,\n  "clarification_question": ""\n}'
                    }]
                }
            }]
        }
        mock_post.return_value = mock_response

        prompt = "nange headache ide, what should I do?"
        result = self.agent.detect_and_translate(prompt, [], "mock_api_key")

        self.assertEqual(result["primary_language"], "kn")
        self.assertTrue(result["is_mixed"])
        self.assertIn("kn", result["detected_languages"])

    @patch('requests.post')
    def test_generate_grounded_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{\n  "response": "ಆಸ್ತಮಾ ಶ್ವಾಸಕೋಶದ ಕಾಯಿಲೆಯಾಗಿದೆ.",\n  "response_english": "Asthma is a lung disease."\n}'
                    }]
                }
            }]
        }
        mock_post.return_value = mock_response

        context = [{"title": "Asthma info", "text": "Asthma is a lung disease that affects breathing.", "source": "NIH"}]
        lang_info = {"primary_language": "kn", "is_mixed": False}
        
        result = self.agent.generate_response("ಆಸ್ತಮಾ ಎಂದರೇನು?", "What is asthma?", lang_info, context, [], "mock_api_key")

        self.assertEqual(result["response"], "ಆಸ್ತಮಾ ಶ್ವಾಸಕೋಶದ ಕಾಯಿಲೆಯಾಗಿದೆ.")
        self.assertEqual(result["response_english"], "Asthma is a lung disease.")

    def test_factual_consistency_checker(self):
        # Case A: Good overlap
        context = [{"title": "Asthma", "text": "Asthma causes narrowing of the airways of the lungs."}]
        response_english = "Asthma leads to narrowing of the lungs airways."
        
        score, aligned, missing = self.agent.check_factual_consistency(response_english, context)
        self.assertGreater(score, 0.70)
        self.assertIn("narrowing", aligned)

        # Case B: Low overlap (hallucination)
        response_hallucinated = "This is treated with penicillin and checking blood insulin."
        score_hall, aligned_hall, missing_hall = self.agent.check_factual_consistency(response_hallucinated, context)
        self.assertLess(score_hall, 0.30)
        self.assertIn("penicillin", missing_hall)

if __name__ == "__main__":
    unittest.main()
