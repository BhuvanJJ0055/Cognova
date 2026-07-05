"""
Task 1 - accuracy check for the mood-detection part of chatbot_v2.py

Runs a small hand-labeled set of messages through score_mood() and
reports accuracy plus a confusion matrix, so I have numbers to back up
the "Evaluation Criteria" section of the task (accuracy of sentiment
detection).
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from chatbot_v2 import score_mood

LABELED_EXAMPLES = [
    ("I absolutely love this, best purchase ever!", "happy"),
    ("This is the worst service I have ever experienced.", "upset"),
    ("Can you tell me your working hours?", "calm"),
    ("My package arrived damaged and I'm furious.", "upset"),
    ("Thanks a lot, you were super helpful!", "happy"),
    ("I would like to check my order status.", "calm"),
    ("This app keeps crashing, it's so frustrating.", "upset"),
    ("Great support, really appreciate it!", "happy"),
    ("What is your return policy?", "calm"),
    ("Terrible experience, I want a refund immediately.", "upset"),
]


def run_eval():
    scorer = SentimentIntensityAnalyzer()
    hits = 0
    confusion = {}

    for message, expected in LABELED_EXAMPLES:
        predicted, _ = score_mood(message, scorer)
        confusion.setdefault(expected, {}).setdefault(predicted, 0)
        confusion[expected][predicted] += 1
        mark = "OK " if predicted == expected else "MISS"
        if predicted == expected:
            hits += 1
        print(f"[{mark}] expected={expected:6s} got={predicted:6s} | {message}")

    acc = hits / len(LABELED_EXAMPLES)
    print(f"\nAccuracy: {acc:.0%} ({hits}/{len(LABELED_EXAMPLES)})")

    moods = ["happy", "upset", "calm"]
    print("\nConfusion matrix (row = expected, col = predicted):")
    print(" " * 8 + "".join(f"{m:>8s}" for m in moods))
    for expected in moods:
        row = confusion.get(expected, {})
        print(f"{expected:8s}" + "".join(f"{row.get(m, 0):>8d}" for m in moods))


if __name__ == "__main__":
    run_eval()
