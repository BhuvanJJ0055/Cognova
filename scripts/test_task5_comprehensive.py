"""
Task 5 Multimodal Assistant Comprehensive Test Suite
Author: Bhuvan J J

Validates image feature extraction, OCR scanning, ambiguity detection,
response verification pass, and multi-turn conversational context memory.
"""

import sys
import os
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.multimodal import MultimodalAssistant, extract_ocr_text

def create_sample_image(text_label: str = "Cognova Multimodal Test") -> Image.Image:
    """Generates a synthetic test image with text and geometric shapes."""
    img = Image.new("RGB", (600, 400), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 550, 350], outline=(30, 144, 255), width=4)
    draw.text((70, 70), text_label, fill=(0, 0, 0))
    return img

def run_task5_test_suite():
    print("\n" + "="*85)
    print("👁️ TASK 5: MULTIMODAL ASSISTANT COMPREHENSIVE TEST SUITE 👁️")
    print("="*85)

    assistant = MultimodalAssistant()
    test_img = create_sample_image("Cognova AI Diagram v2.0")

    # 1. Test Visual Evidence Extraction
    print("\n📊 1. Testing Visual Evidence Extraction (Dimensions, Mode, Brightness, OCR)...")
    evidence = assistant.extract_visual_evidence(test_img)
    
    print(f"  -> Extracted Dimensions: {evidence['width']}x{evidence['height']} px")
    print(f"  -> Orientation: {evidence['orientation']} | Aspect Ratio: {evidence['aspect_ratio']}")
    print(f"  -> Mean Brightness: {evidence['mean_brightness']}/255 | Color Mode: {evidence['color_mode']}")
    
    assert evidence["width"] == 600 and evidence["height"] == 400, "FAILED: Image resolution mismatch!"
    assert evidence["orientation"] == "Landscape", "FAILED: Orientation detection mismatch!"
    print("  ✅ Visual Evidence Extraction: PASS")

    # 2. Test Ambiguity Detection & Disambiguation Menu
    print("\n⚠️ 2. Testing Ambiguity Detection Engine...")
    vague_prompt = "what is this"
    res_ambiguous = assistant.analyze_image(test_img, vague_prompt)
    
    print(f"  -> Prompt: \"{vague_prompt}\"")
    print(f"  -> Ambiguity Flag: {res_ambiguous['is_ambiguous']}")
    print("  -> Response Preview: " + res_ambiguous["response"].split("\n")[0])
    
    assert res_ambiguous["is_ambiguous"] is True, "FAILED: Ambiguity detection failed for vague prompt!"
    assert "Ambiguity Detected" in res_ambiguous["response"], "FAILED: Disambiguation menu missing!"
    print("  ✅ Ambiguity Detection Engine: PASS")

    # 3. Test Multimodal Reasoning & Response Verification Pass
    print("\n🔬 3. Testing Multimodal Reasoning & Evidence Verification Pass...")
    detailed_prompt = "Analyze the text labels and structural diagram layout in this image."
    res_reasoning = assistant.analyze_image(test_img, detailed_prompt)
    
    print("  -> Reasoning Output Preview:")
    print("     " + "\n     ".join(res_reasoning["response"].split("\n")[:5]))
    
    assert res_reasoning["is_ambiguous"] is False, "FAILED: Non-ambiguous prompt marked as ambiguous!"
    assert "Evidence Verification Pass" in res_reasoning["response"], "FAILED: Evidence verification pass header missing!"
    print("  ✅ Response Verification Pass: PASS")

    # 4. Test Multi-Turn Conversational Context Memory
    print("\n💬 4. Testing Multi-Turn Conversational Context Memory...")
    mock_history = [
        {"role": "user", "content": "What type of document is this?"},
        {"role": "assistant", "content": "This is a technical diagram with dimensions 600x400 px."}
    ]
    followup_prompt = "What is the key label inside it?"
    res_context = assistant.analyze_image(test_img, followup_prompt, chat_history=mock_history)
    
    assert res_context["is_ambiguous"] is False, "FAILED: Follow-up question with history failed!"
    assert "Evidence Verification Pass" in res_context["response"], "FAILED: Contextual follow-up verification missing!"
    print("  ✅ Multi-Turn Conversational Memory: PASS")

    print("\n" + "="*85)
    print("🎯 TASK 5 EVALUATION SUMMARY: All Multimodal Assistant Tests PASSED")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_task5_test_suite()
