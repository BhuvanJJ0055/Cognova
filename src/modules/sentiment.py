"""
Task 1 - Enterprise Sentiment & Intent Analysis Engine (GenAI Internship Platform)
Author: Bhuvan J J

Universal, Multi-Layered PM-Grade Architecture for Any Arbitrary User Input:
1. Politeness Sanitization: Strips politeness markers ("please", "kindly", "could you")
   during sentiment calculation to prevent politeness from masking underlying complaints/distress.
2. Dual-Layer Sentiment Analysis (VADER + Contextual Problem/Emotion Lexicons):
   - Overrides VADER misclassifications when problem indicators exist.
   - Preserves plain-question neutral band for informational queries.
3. Generalized 7-Category Intent Engine:
   - financial_refund: All payment, billing, charge, debit, refund inquiries.
   - logistics_tracking: All shipping, delivery, tracking, order status queries.
   - service_complaint: All system bugs, crashes, defects, service failures.
   - product_inquiry: Product features, specifications, usage, how-to questions.
   - appreciation_feedback: All praise, compliments, gratitude, positive reviews.
   - greeting_salutation: All greetings, welcomes, farewells.
   - general_query: Universal fallback.
4. Emotion-Adaptive & Intent-Tailored Response Generator.
"""

import os
import re
import csv
import datetime
from typing import Tuple, Dict, Any, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "data", "chat_history.csv")

POSITIVE_CUTOFF = 0.25
NEGATIVE_CUTOFF = -0.15

# Comprehensive Sentiment & Intent Lexicons
POLITENESS_MARKERS = {
    "please", "kindly", "could you", "would you mind", "can you please",
    "if possible", "thanks in advance", "appreciate if"
}

HAPPY_LEXICON = {
    "awesome", "great", "excellent", "wonderful", "amazing", "fantastic",
    "brilliant", "love", "loved", "superb", "perfect", "delighted", "happy",
    "impressed", "kudos", "best", "thank you", "thanks", "thank"
}

PROBLEM_DISTRESS_LEXICON = {
    "broken", "failed", "failing", "crash", "crashed", "error", "bug",
    "deducted", "deduction", "charged", "charging", "charge", "debited",
    "debit", "overcharged", "twice", "double", "scam", "useless", "terrible",
    "horrible", "worst", "awful", "frustrated", "angry", "upset", "refund",
    "reimburse", "disappointed", "stolen", "lost", "slow", "freeze", "freezing",
    "issue", "problem", "not working", "cannot open", "cant open", "unable"
}


def sanitize_politeness(text: str) -> str:
    """Removes politeness markers to prevent masking of underlying distress or complaints."""
    lowered = text.lower()
    for marker in POLITENESS_MARKERS:
        lowered = lowered.replace(marker, "")
    return lowered.strip()


def _looks_like_plain_question(text: str) -> bool:
    """
    Forces neutral ('calm') classification for informational queries
    that do not express explicit emotional praise or distress.
    """
    lowered = text.lower().strip()
    words = set(re.findall(r'\b\w+\b', lowered))
    
    has_happy = bool(words.intersection(HAPPY_LEXICON))
    has_problem = bool(words.intersection(PROBLEM_DISTRESS_LEXICON))
    
    is_question_form = (
        lowered.endswith("?") or
        lowered.startswith(("what", "where", "how", "when", "why", "who", "which", "can i", "is there", "do you"))
    )

    return is_question_form and not has_happy and not has_problem


def is_weak_answer(answer: str, prompt_instruction: str) -> bool:
    """Detects weak or instruction-echoing LLM responses."""
    if not answer or len(answer.strip()) < 10:
        return True

    ans_words = set(re.findall(r'\b\w+\b', answer.lower()))
    instr_words = set(re.findall(r'\b\w+\b', prompt_instruction.lower()))

    if not ans_words:
        return True

    overlap = len(ans_words.intersection(instr_words)) / float(len(ans_words))
    return overlap > 0.75


def score_mood_vader(message: str, scorer: Optional[SentimentIntensityAnalyzer] = None) -> Tuple[str, float]:
    """
    Universal Dual-Layer Sentiment Scorer:
    - Strips politeness markers before evaluation.
    - Applies neutral override for plain informational questions.
    - Applies problem override if message contains distress/issue lexicons.
    """
    if _looks_like_plain_question(message):
        return "calm", 0.0

    sanitized_text = sanitize_politeness(message)

    if scorer is None:
        scorer = SentimentIntensityAnalyzer()

    # Raw VADER polarity on sanitized text
    result = scorer.polarity_scores(sanitized_text if sanitized_text else message)
    compound = result["compound"]

    words = set(re.findall(r'\b\w+\b', message.lower()))

    # Problem/Distress Rule Override: If problem keywords exist, sentiment cannot be happy
    if words.intersection(PROBLEM_DISTRESS_LEXICON) or any(p in message.lower() for p in ["deducted", "charged", "not working", "broken", "issue"]):
        if compound <= 0.30:
            return "upset", min(compound, -0.35)
        else:
            return "calm", 0.0

    # Happy Lexicon Override
    if words.intersection(HAPPY_LEXICON) and compound >= 0.15:
        return "happy", max(compound, 0.50)

    if compound <= NEGATIVE_CUTOFF:
        mood = "upset"
    elif compound >= POSITIVE_CUTOFF:
        mood = "happy"
    else:
        mood = "calm"

    return mood, compound


def tag_intent(message: str, mood: str = "calm", *args, **kwargs) -> str:
    """
    Universal 7-Category Intent Engine:
    Uses semantic priority pattern matching across financial, logistics, complaints,
    product inquiries, appreciation, greetings, and general queries.
    """
    if "mood" in kwargs:
        mood = kwargs["mood"]
    elif args and isinstance(args[0], str):
        mood = args[0]
        
    lowered = message.lower().strip()

    # Category 1: Financial & Refunds (Payment, billing, deductions, cards, invoices)
    financial_patterns = [
        "refund", "money back", "reimburse", "return my money", "chargeback",
        "billing", "deducted", "deduction", "charged", "charging", "debited",
        "debit", "overcharged", "twice", "double charge", "invoice", "receipt",
        "payment", "paid", "transaction", "fee", "cost"
    ]
    if any(p in lowered for p in financial_patterns):
        return "financial_refund"

    # Category 2: Logistics & Order Tracking (Shipping, delivery, tracking, status)
    logistics_patterns = [
        "track", "tracking", "shipment", "delivery", "dispatch", "dispatched",
        "package", "courier", "where is my order", "order status", "order id", "eta", "arrival"
    ]
    if any(p in lowered for p in logistics_patterns):
        return "logistics_tracking"

    # Category 3: Service & Technical Complaints (Bugs, defects, crashes, errors)
    complaint_patterns = [
        "broken", "not working", "defective", "terrible", "worst", "awful",
        "horrible", "waste of money", "disappointed", "crash", "crashed", "bug",
        "error", "failed", "failing", "freeze", "slow", "issue", "problem", "glitch"
    ]
    if any(p in lowered for p in complaint_patterns):
        return "service_complaint"

    # Category 4: Product & Feature Inquiries (How-to, specs, features, usage)
    product_patterns = [
        "how to", "how do i", "feature", "specification", "settings", "documentation",
        "guide", "instructions", "configure", "setup", "can i use", "where to find"
    ]
    if any(p in lowered for p in product_patterns):
        return "product_inquiry"

    # Category 5: Appreciation & Positive Feedback (Praise, compliments, reviews)
    appreciation_patterns = [
        "thank", "thanks", "great job", "awesome", "love it", "amazing",
        "wonderful", "excellent", "kudos", "appreciate", "great app", "perfect", "good work"
    ]
    if any(p in lowered for p in appreciation_patterns):
        return "appreciation_feedback"

    # Category 6: Greetings & Salutations (Hellos, goodbyes)
    greeting_patterns = [
        "hi", "hello", "hey", "good morning", "good evening", "greetings", "bye", "goodbye"
    ]
    if any(p in lowered for p in greeting_patterns):
        return "greeting_salutation"

    # Category 7: Sentiment Fallbacks
    if mood == "happy":
        return "appreciation_feedback"
    elif mood == "upset":
        return "service_complaint"

    return "general_query"


def append_log(message: str, mood: str, compound: float, intent: str, reply: str):
    """Logs interactions to CSV for quality assurance and continuous analytics."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if not file_exists:
            writer.writerow(["timestamp", "user_message", "detected_mood", "compound_score", "detected_intent", "bot_reply"])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            message, mood, round(compound, 3), intent, reply,
        ])


class SupportChatbot:
    """Sentiment-Aware Universal Support Engine."""

    def __init__(self, model_type="vader"):
        self.model_type = model_type.lower()
        self.scorer = SentimentIntensityAnalyzer()

    def score_mood(self, message: str) -> Tuple[str, float]:
        return score_mood_vader(message, self.scorer)

    def reply_to(self, message: str) -> str:
        mood, compound = self.score_mood(message)
        intent = tag_intent(message, mood=mood)

        # Universal Sentiment & Intent Adaptive Response Matrix
        response_matrix: Dict[Tuple[str, str], str] = {
            ("happy", "appreciation_feedback"): (
                "Thank you so much for your glowing feedback! 🌟 We're thrilled to hear about your great experience with our platform. "
                "Is there anything else we can assist you with today?"
            ),
            ("happy", "product_inquiry"): (
                "We're glad you're enjoying our platform! 🚀 Regarding your feature question: please let us know what specific configuration or usage detail you'd like to explore."
            ),
            ("upset", "financial_refund"): (
                "I am deeply sorry for any billing or payment concern! 💳 "
                "Your financial inquiry has been flagged as high-priority. Please provide your transaction ID or registered account email so our billing team can process your resolution immediately."
            ),
            ("upset", "service_complaint"): (
                "I sincerely apologize for the inconvenience and frustration caused! 😔 "
                "Your satisfaction is our top priority. Please share your account or error details so our senior technical team can resolve this right away."
            ),
            ("upset", "logistics_tracking"): (
                "I apologize for the delay or concern with your shipment! 📦 "
                "Please share your tracking or order ID so we can escalate your package delivery immediately."
            ),
            ("calm", "financial_refund"): (
                "I understand you have a question regarding billing or payment. 💳 "
                "Please share your order ID or reference number, and I will look up your payment details."
            ),
            ("calm", "logistics_tracking"): (
                "Certainly! I can check your order tracking right away. 📦 Please share your tracking number or order ID."
            ),
            ("calm", "product_inquiry"): (
                "I'm happy to help with product details and usage guidance! 📘 What specific feature or setting would you like to know more about?"
            ),
            ("calm", "greeting_salutation"): (
                "Hello! Welcome to Cognova Assistant Platform. 👋 How can I help you today?"
            ),
            ("calm", "general_query"): (
                "Thank you for reaching out to customer support. ℹ️ How can I assist you with your request today?"
            )
        }

        reply = response_matrix.get((mood, intent))
        if not reply:
            if mood == "happy":
                reply = "We truly appreciate your positive feedback! Let us know if you need any further assistance."
            elif mood == "upset":
                reply = "We apologize for the inconvenience caused. We are committed to making this right—how can we best assist you?"
            else:
                reply = "Thank you for contacting customer support. How can I assist you with your request today?"

        append_log(message, mood, compound, intent, reply)
        return reply
