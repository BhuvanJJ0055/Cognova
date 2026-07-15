"""
Task 4 - test_multimodal.py
Author: Bhuvan J J

Unit tests for validating the Multi-Modal AI Assistant's:
1. Image property parsing
2. Heuristic-based visual categorization
3. Ambiguity detection
4. Domain routing and context retrieval
5. Factual consistency calculation
"""

import os
import unittest
import sys
from io import BytesIO
from PIL import Image

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Task4_Multimodal_Assistant.multimodal_agent import MultimodalAgent

class TestMultimodalAgent(unittest.TestCase):

    def setUp(self):
        # Initialize agent with sample/mock references to test logic
        self.agent = MultimodalAgent()

    def create_mock_image(self, width=100, height=100, color="red"):
        """Utility to generate a mock in-memory image for PIL tests."""
        file_obj = BytesIO()
        img = Image.new("RGB", (width, height), color=color)
        img.save(file_obj, format="PNG")
        file_obj.name = "mock_image.png"
        file_obj.seek(0)
        return file_obj

    def test_image_metadata_parsing(self):
        """Verify image property extraction using PIL."""
        mock_img = self.create_mock_image(200, 150)
        props = self.agent.parse_image_properties(mock_img)
        
        self.assertIn("format", props)
        self.assertEqual(props["format"], "PNG")
        self.assertEqual(props["width"], 200)
        self.assertEqual(props["height"], 150)
        self.assertEqual(props["aspect_ratio"], 1.33)

    def test_routing_heuristics(self):
        """Test heuristic visual categorization on various filename keywords."""
        # 1. Medical scan routing
        analysis_med = self.agent.run_local_visual_fallback(
            filename="my_chest_xray_pneumonia.png",
            image_properties={"width": 800, "height": 800, "mode": "RGB"},
            prompt="What is on this scan?"
        )
        self.assertEqual(analysis_med["routed_domain"], "medical")
        self.assertIn("pneumonia", analysis_med["description"].lower())

        # 2. Scientific chart routing
        analysis_sci = self.agent.run_local_visual_fallback(
            filename="pca_projection.jpg",
            image_properties={"width": 640, "height": 480, "mode": "RGB"},
            prompt="summarize this"
        )
        self.assertEqual(analysis_sci["routed_domain"], "scientific")
        self.assertIn("pca", analysis_sci["description"].lower())

        # 3. Customer support receipt routing
        analysis_sup = self.agent.run_local_visual_fallback(
            filename="receipt_screenshot.png",
            image_properties={"width": 300, "height": 600, "mode": "RGB"},
            prompt="Why was I charged this much?"
        )
        self.assertEqual(analysis_sup["routed_domain"], "support")
        self.assertIn("order #5432", analysis_sup["description"].lower())

    def test_ambiguity_handler(self):
        """Test if vague queries and generic filenames prompt for clarification."""
        # Generic case: should trigger ambiguity
        questions = self.agent.check_ambiguity(
            prompt="explain",
            filename="image.png",
            properties={"width": 100, "height": 100}
        )
        self.assertTrue(len(questions) > 0)
        self.assertIn("generic", questions[0].lower())

        # Specific query: should NOT trigger ambiguity
        questions_clear = self.agent.check_ambiguity(
            prompt="What are the symptoms of pneumonia described here?",
            filename="image.png",
            properties={"width": 100, "height": 100}
        )
        self.assertEqual(len(questions_clear), 0)

    def test_factual_consistency_checker(self):
        """Verify the alignment scoring system for generated text and source documents."""
        context = [
            {
                "title": "Asthma Treatment Guidelines",
                "text": "Inhaled corticosteroids are highly effective anti-inflammatory medications for chronic asthma.",
                "source": "NIH"
            }
        ]

        # Case A: Factual response (aligned)
        factual_resp = "Inhaled corticosteroids are effective anti-inflammatory therapy for chronic asthma."
        score, aligned, missing = self.agent.check_factual_consistency(factual_resp, context)
        self.assertGreater(score, 0.70)
        self.assertIn("corticosteroids", aligned)
        self.assertIn("asthma", aligned)

        # Case B: Hallucinated response (unaligned concepts)
        hallucinated_resp = "This patient needs to take penicillin antibiotics and check insulin blood sugar levels."
        score_hall, aligned_hall, missing_hall = self.agent.check_factual_consistency(hallucinated_resp, context)
        self.assertLess(score_hall, 0.30)
        self.assertIn("penicillin", missing_hall)
        self.assertIn("insulin", missing_hall)

if __name__ == "__main__":
    unittest.main()
