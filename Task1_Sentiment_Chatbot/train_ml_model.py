"""
Task 1 - Sentiment-Aware Support Chatbot (ML Model Trainer)
Author: Bhuvan J J

This script trains a Machine Learning model (Logistic Regression with TF-IDF features)
on the custom customer support sentiment dataset. The trained pipeline is saved
to disk so that the chatbot can load it for real-time sentiment analysis.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Setup paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "sentiment_dataset.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "sentiment_model.joblib")


def train_model():
    print("=" * 60)
    print("Training Sentiment Analysis Classifier (Advanced ML Model)")
    print("=" * 60)

    # 1. Load the dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run dataset generation first.")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from {DATA_PATH} ({len(df)} samples)")
    print("Class distribution:")
    print(df['sentiment'].value_counts())
    print("-" * 60)

    # 2. Train-Test Split (80% Train, 20% Test)
    # Using stratify ensures class balance in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment']
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print("-" * 60)

    # 3. Create Pipeline: Feature Extraction (TF-IDF) + Classifier (Logistic Regression)
    # We use a simple ngram range (1, 2) to capture word combinations (e.g. "not happy")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=10.0,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 4. Train the model
    print("Training model pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on Test Set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nEvaluation Results on Test Split:")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 60)

    # 6. Save the trained pipeline
    print(f"Saving trained model pipeline to: {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
