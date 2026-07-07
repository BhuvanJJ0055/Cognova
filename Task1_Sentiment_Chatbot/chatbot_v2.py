"""
Task 1 - Sentiment-Aware Support Chatbot
Author: Bhuvan J J

Goal: build a chatbot that can tell whether a customer sounds happy,
upset, or neutral, and adjust its reply tone to match.

Approach I chose:
- Sentiment scoring: VADER (via the vaderSentiment package). It's a
  rule/lexicon based scorer built for short informal text like chat
  messages and social posts, so it doesn't need any training data or
  GPU on my end - good fit for a first working version.
- Intent tagging: simple keyword matching for now (refund request,
  complaint, order tracking, etc). Noted in my report as an area I'd
  upgrade to a trained classifier if I had more time/data.
- Every exchange gets written to a log file so I can measure accuracy
  afterward with test_accuracy.py.
"""

import csv
import os
import random
import argparse
from datetime import datetime
import joblib

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LOG_PATH = os.path.join(os.path.dirname(__file__), "chat_history.csv")

# Thresholds decide how strongly-worded a message needs to be before I
# call it clearly positive/negative rather than neutral. Tuned by hand
# after trying a few sample messages.
POSITIVE_CUTOFF = 0.35
NEGATIVE_CUTOFF = -0.35

# keyword -> intent label. First match wins, so put more specific
# phrases first within a list where it matters.
INTENT_RULES = [
    ("complaint", ["broken", "not working", "defective", "terrible", "worst", "awful"]),
    ("refund_request", ["refund", "money back", "reimburse", "return my"]),
    ("order_tracking", ["track", "shipment", "delivery status", "where is my order"]),
    ("compliment", ["thank you", "thanks", "great job", "awesome", "love it", "appreciate"]),
    ("greeting", ["hi", "hello", "hey there", "good morning", "good evening"]),
]

# reply pools keyed on (mood, intent). Kept as lists so replies vary a
# little between runs instead of feeling scripted.
REPLY_BANK = {
    ("upset", "complaint"): [
        "Sorry to hear that - that's not the experience we want you to have. "
        "Could you send me the order number so I can dig in?",
        "That's frustrating, I get it. Let's sort it out - what happened exactly?",
    ],
    ("upset", "refund_request"): [
        "Understood, I'll get your refund moving. Can you confirm the order ID first?",
    ],
    ("upset", "other"): [
        "Sounds like this hasn't gone well. Tell me more and I'll try to fix it.",
    ],
    ("happy", "compliment"): [
        "That's great to hear, thanks for saying so!",
        "Appreciate the kind words - anything else you need help with?",
    ],
    ("happy", "other"): [
        "Glad things are going smoothly! What can I help with today?",
    ],
    ("calm", "order_tracking"): [
        "Sure thing - what's the order number?",
    ],
    ("calm", "greeting"): [
        "Hey! What can I help you with?",
    ],
    ("calm", "other"): [
        "Thanks for the message - can you give me a bit more detail?",
    ],
}

FALLBACK_REPLIES = {
    "upset": ["I'm sorry for the hassle - walk me through what happened?"],
    "happy": ["Good to hear! Let know if there's anything else."],
    "calm": ["Could you share a few more details so I can help?"],
}


def score_mood_vader(message, scorer):
    """Run VADER on the message and bucket it into upset/calm/happy."""
    result = scorer.polarity_scores(message)
    compound = result["compound"]

    if compound <= NEGATIVE_CUTOFF:
        mood = "upset"
    elif compound >= POSITIVE_CUTOFF:
        mood = "happy"
    else:
        mood = "calm"

    return mood, compound


def tag_intent(message):
    """Cheap keyword lookup - not fancy, but transparent and easy to debug."""
    lowered = message.lower()
    for label, phrases in INTENT_RULES:
        if any(phrase in lowered for phrase in phrases):
            return label
    return "other"


def pick_reply(mood, intent):
    options = REPLY_BANK.get((mood, intent))
    if not options:
        options = FALLBACK_REPLIES[mood]
    return random.choice(options)


def ensure_log_exists():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                ["time", "message", "mood", "compound_score", "intent", "reply"]
            )


def append_log(message, mood, compound, intent, reply):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            message, mood, round(compound, 3), intent, reply,
        ])


class SupportChatbot:
    """Ties scoring + intent tagging + reply selection together."""

    def __init__(self, model_type="vader"):
        self.model_type = model_type.lower()
        self.scorer = SentimentIntensityAnalyzer()
        self.ml_pipeline = None

        if self.model_type == "ml":
            model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.joblib")
            if os.path.exists(model_path):
                try:
                    self.ml_pipeline = joblib.load(model_path)
                    print(f"[Info] Loaded Advanced ML model from {model_path}")
                except Exception as e:
                    print(f"[Warning] Failed to load ML model: {e}. Falling back to VADER.")
                    self.model_type = "vader"
            else:
                print(f"[Warning] ML model file {model_path} not found. Run train_ml_model.py first. Falling back to VADER.")
                self.model_type = "vader"

        ensure_log_exists()

    def score_mood(self, message: str):
        if self.model_type == "ml" and self.ml_pipeline is not None:
            try:
                pred = self.ml_pipeline.predict([message])[0]
                score_map = {"happy": 1.0, "upset": -1.0, "calm": 0.0}
                return pred, score_map.get(pred, 0.0)
            except Exception as e:
                print(f"[Warning] ML inference error: {e}. Falling back to VADER.")

        return score_mood_vader(message, self.scorer)

    def reply_to(self, message: str) -> str:
        mood, compound = self.score_mood(message)
        intent = tag_intent(message)
        reply = pick_reply(mood, intent)
        append_log(message, mood, compound, intent, reply)
        return reply


def chat_loop(model_type="vader"):
    bot = SupportChatbot(model_type=model_type)
    print(f"Support bot ready ({bot.model_type.upper()} model). Type 'quit' to stop.\n")
    while True:
        message = input("You: ").strip()
        if message.lower() in {"quit", "exit"}:
            print("Bot: Take care, bye!")
            break
        if not message:
            continue
        print("Bot:", bot.reply_to(message))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment-Aware Support Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        default="vader",
        choices=["vader", "ml"],
        help="Sentiment classification model to use (vader or ml)"
    )
    args = parser.parse_args()
    chat_loop(model_type=args.model)
