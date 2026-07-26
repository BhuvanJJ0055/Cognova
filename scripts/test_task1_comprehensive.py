"""
Task 1 Universal Sentiment & Intent Comprehensive Test Suite
Author: Bhuvan J J

Validates the Universal 4-Layer Sentiment & Intent Engine across 15 arbitrary inputs.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.modules.sentiment import SupportChatbot, score_mood_vader, tag_intent, _looks_like_plain_question

def run_universal_test_suite():
    bot = SupportChatbot(model_type="vader")

    test_prompts = [
        # --- POSITIVE / HAPPY PROMPTS ---
        {
            "category": "Praise & Satisfaction",
            "prompt": "Thank you so much, your app is awesome and I love it!",
            "expected_sentiment": "happy",
            "expected_intent": "appreciation_feedback"
        },
        {
            "category": "Product Delight",
            "prompt": "Great job team! The interface is super smooth and impressive.",
            "expected_sentiment": "happy",
            "expected_intent": "appreciation_feedback"
        },
        {
            "category": "Happy Feature Question",
            "prompt": "I really love your app! Can you tell me how to enable dark mode?",
            "expected_sentiment": "happy",
            "expected_intent": "product_inquiry"
        },

        # --- FINANCIAL / BILLING PROMPTS (POLITE & DIRECT) ---
        {
            "category": "Polite Billing Complaint",
            "prompt": "Can you please help me? My payment was deducted twice.",
            "expected_sentiment": "upset",
            "expected_intent": "financial_refund"
        },
        {
            "category": "Direct Refund Demand",
            "prompt": "I want my money back! I was overcharged on my last invoice.",
            "expected_sentiment": "upset",
            "expected_intent": "financial_refund"
        },
        {
            "category": "Calm Payment Inquiry",
            "prompt": "Where can I download the tax invoice for my transaction?",
            "expected_sentiment": "calm",
            "expected_intent": "financial_refund"
        },

        # --- TECHNICAL & SERVICE COMPLAINTS ---
        {
            "category": "System Bug / Crash",
            "prompt": "The app keeps freezing whenever I try to upload a file.",
            "expected_sentiment": "upset",
            "expected_intent": "service_complaint"
        },
        {
            "category": "Product Defect",
            "prompt": "This update is terrible and completely broke my workflow.",
            "expected_sentiment": "upset",
            "expected_intent": "service_complaint"
        },

        # --- LOGISTICS & TRACKING ---
        {
            "category": "Shipping Inquiry",
            "prompt": "Where is my package? It was supposed to arrive yesterday.",
            "expected_sentiment": "calm",
            "expected_intent": "logistics_tracking"
        },
        {
            "category": "Order Courier Status",
            "prompt": "Can you track order ID #984721 for me?",
            "expected_sentiment": "calm",
            "expected_intent": "logistics_tracking"
        },

        # --- PRODUCT & FEATURE INQUIRIES ---
        {
            "category": "How-To Inquiry",
            "prompt": "How do I export my data to PDF or CSV format?",
            "expected_sentiment": "calm",
            "expected_intent": "product_inquiry"
        },

        # --- GENERAL QUESTIONS & GREETINGS ---
        {
            "category": "Plain FAQ Question",
            "prompt": "What is your standard return and warranty policy?",
            "expected_sentiment": "calm",
            "expected_intent": "general_query"
        },
        {
            "category": "Greeting",
            "prompt": "Hello! Good morning chatbot.",
            "expected_sentiment": "calm",
            "expected_intent": "greeting_salutation"
        }
    ]

    print("\n" + "="*85)
    print("🚀 TASK 1: UNIVERSAL SENTIMENT & INTENT COMPREHENSIVE EVALUATION 🚀")
    print("="*85)

    passed_count = 0
    for idx, test in enumerate(test_prompts, 1):
        prompt = test["prompt"]
        mood, score = bot.score_mood(prompt)
        intent = tag_intent(prompt, mood)
        reply = bot.reply_to(prompt)

        sentiment_pass = (mood == test["expected_sentiment"])
        intent_pass = (intent == test["expected_intent"])
        is_pass = sentiment_pass and intent_pass
        if is_pass:
            passed_count += 1

        status_str = "✅ PASS" if is_pass else f"⚠️ CHECK (Sentiment: {mood}, Intent: {intent})"

        print(f"\nTest #{idx:02d} [{test['category']}] -> {status_str}")
        print(f" 📥 Input: \"{prompt}\"")
        print(f" 📊 Sentiment: {mood.upper()} ({score:+.2f}) | Tagged Intent: {intent}")
        print(f" 💬 Reply: {reply}")

    print("\n" + "="*85)
    print(f"🎯 EVALUATION SUMMARY: Passed {passed_count}/{len(test_prompts)} Universal Scenarios")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_universal_test_suite()
