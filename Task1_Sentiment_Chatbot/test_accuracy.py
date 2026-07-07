"""
Task 1 - Sentiment-Aware Support Chatbot (Model Evaluation & Comparison)
Author: Bhuvan J J

This script compares the sentiment classification accuracy of three approaches:
1. Keyword-Based Baseline (v1 draft approach)
2. VADER Lexicon-Based Baseline (original v2 approach)
3. Logistic Regression Classifier (Advanced ML Model)

It runs evaluation on the held-out test split of our custom customer support dataset,
and displays comparative metrics (Accuracy, Precision, Recall, F1-score) and
confusion matrices.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from chatbot_v2 import score_mood_vader

# Setup paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "sentiment_dataset.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "sentiment_model.joblib")

# Keywords for the baseline model (replicated from drafts/chatbot.py)
POSITIVE_KEYWORDS = {"good", "great", "happy", "love", "wonderful", "excellent", "awesome", "joy", "nice", "glad"}
NEGATIVE_KEYWORDS = {"bad", "sad", "angry", "hate", "terrible", "awful", "frustrated", "annoyed", "wrong", "annoy", "furious", "defective", "broken"}


def predict_keyword(text):
    """Predicts sentiment based on simple keyword count."""
    words = text.lower().split()
    pos_count = sum(1 for w in words if w in POSITIVE_KEYWORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_KEYWORDS)
    
    if pos_count > neg_count:
        return "happy"
    elif neg_count > pos_count:
        return "upset"
    else:
        return "calm"


def predict_vader(text, scorer):
    """Predicts sentiment using VADER compound score."""
    mood, _ = score_mood_vader(text, scorer)
    return mood


def print_confusion_matrix(cm, labels):
    """Helper to pretty print a confusion matrix."""
    print(f"{'':<12} | " + " | ".join(f"Pred {lbl[:3].upper()}" for lbl in labels))
    print("-" * (15 + 13 * len(labels)))
    for i, actual in enumerate(labels):
        row_str = f"Actual {actual[:3].upper():<4} | " + " | ".join(f"{cm[i, j]:^8d}" for j in range(len(labels)))
        print(row_str)


def run_evaluation():
    print("=" * 80)
    print("Evaluating Sentiment Detection Models")
    print("=" * 80)

    # 1. Load the dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please generate it first.")
    
    df = pd.read_csv(DATA_PATH)
    
    # 2. Re-create the exact same test split (80-20, random_state=42)
    _, X_test, _, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment']
    )
    
    print(f"Loaded test dataset ({len(X_test)} samples)")
    print("-" * 80)

    # 3. Load ML model pipeline
    if not os.path.exists(MODEL_PATH):
        print(f"[Warning] ML model file {MODEL_PATH} not found. Please train it first by running train_ml_model.py.")
        ml_model = None
    else:
        ml_model = joblib.load(MODEL_PATH)
        print(f"[Info] Loaded Advanced ML model successfully.")

    vader_scorer = SentimentIntensityAnalyzer()
    labels = ["happy", "upset", "calm"]

    # 4. Generate Predictions
    preds_keyword = []
    preds_vader = []
    preds_ml = []

    for text in X_test:
        preds_keyword.append(predict_keyword(text))
        preds_vader.append(predict_vader(text, vader_scorer))
        if ml_model is not None:
            preds_ml.append(ml_model.predict([text])[0])

    # 5. Report results
    print("\n" + "#" * 40)
    print(" 1. KEYWORD-BASED BASELINE MODEL")
    print("#" * 40)
    print(f"Accuracy: {accuracy_score(y_test, preds_keyword):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds_keyword, target_names=labels, zero_division=0))
    print("Confusion Matrix:")
    cm_kw = confusion_matrix(y_test, preds_keyword, labels=labels)
    print_confusion_matrix(cm_kw, labels)

    print("\n" + "#" * 40)
    print(" 2. VADER LEXICON-BASED BASELINE MODEL")
    print("#" * 40)
    print(f"Accuracy: {accuracy_score(y_test, preds_vader):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds_vader, target_names=labels, zero_division=0))
    print("Confusion Matrix:")
    cm_vd = confusion_matrix(y_test, preds_vader, labels=labels)
    print_confusion_matrix(cm_vd, labels)

    if ml_model is not None:
        print("\n" + "#" * 40)
        print(" 3. ADVANCED MACHINE LEARNING MODEL (TF-IDF + Logistic Regression)")
        print("#" * 40)
        print(f"Accuracy: {accuracy_score(y_test, preds_ml):.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds_ml, target_names=labels, zero_division=0))
        print("Confusion Matrix:")
        cm_ml = confusion_matrix(y_test, preds_ml, labels=labels)
        print_confusion_matrix(cm_ml, labels)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
