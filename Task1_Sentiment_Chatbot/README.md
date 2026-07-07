# Sentiment-Aware Support Chatbot (Task 1)

This project integrates sentiment analysis into a customer support chatbot to detect customer emotions (`happy`, `upset`, or `calm`) and tailor the response tone accordingly, combined with keyword-based intent routing.

---

## 1. Problem Statement
Support chatbots frequently sound mechanical and detached, ignoring the user's emotional state. This task implements a sentiment-aware chatbot that:
1. Recognizes whether a user is happy, upset, or calm.
2. Identifies customer intent (refund request, complaint, order tracking, compliment, greeting).
3. Tailors the reply tone to match the mood (e.g. empathetic de-escalation for frustrated customers).
4. Keeps a conversation history log for auditing and metrics evaluation.

---

## 2. Dataset
We compiled a balanced dataset of 90 customer support utterances in `data/sentiment_dataset.csv`.
- **Happy**: Positive compliments and satisfied comments (e.g., "Thanks a lot, you were super helpful!").
- **Upset**: Angry complaints, refund requests, and frustration (e.g., "My package arrived damaged and I'm furious.").
- **Calm**: Neutral questions and status checks (e.g., "I would like to check my order status.").

---

## 3. Methodology & Model Selection
We compare three sentiment analysis models:
1. **Keyword Baseline (v1)**: Matches user input against static lists of positive/negative keywords.
2. **VADER Baseline (v2)**: Uses a lexicon-based intensity score specifically tuned for informal/chat messages.
3. **Logistic Regression (Advanced ML Model)**: Vectorizes text using a TF-IDF vectorizer (capturing unigrams and bigrams to handle negations) and classifies sentiment using a Logistic Regression model.

---

## 4. Performance & Evaluation Results
Evaluating all three models on the test split (20% of the dataset) yields the following results:

| Model | Accuracy | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **Keyword Baseline** | ~55.56% | Fast, transparent, no training needed. | Fails on negations ("not happy") and context. |
| **VADER Baseline** | ~72.22% | Understands emoji, punctuation, and simple negations. | Misclassifies support-specific neutral terms as emotional. |
| **Logistic Regression ML** | **94.44%** | Captures bigrams (negations) and adapts directly to customer support vocabulary. | Requires a training phase and labeled training dataset. |

### Visualizations
The experiment generates the following visual outputs (which can be viewed in the Jupyter Notebook):
- `class_distribution.png`: Visualizes label counts to check dataset balance.
- `model_comparison.png`: Bar chart comparing the accuracies of the three models.
- `confusion_matrices.png`: Side-by-side confusion matrices showing misclassification details.

---

## 5. How to Run and Reproduce

### Install Dependencies
Activate your virtual environment and install the requirements:
```bash
pip install -r requirements.txt
```

### Train the ML Model
Before running the chatbot in ML mode, train the Logistic Regression model:
```bash
python train_ml_model.py
```
This will save the trained pipeline as `sentiment_model.joblib`.

### Run Model Evaluation
Compare all three models' accuracies and output text confusion matrices:
```bash
python test_accuracy.py
```

### Run the Chatbot
You can run the interactive CLI chatbot in two modes:
1. **VADER Mode (default)**:
   ```bash
   python chatbot_v2.py --model vader
   ```
2. **ML Mode (Trained Logistic Regression)**:
   ```bash
   python chatbot_v2.py --model ml
   ```

*Note: Conversations are logged to `chat_history.csv` for audit trails.*

---

## 6. Project Layout
- `data/sentiment_dataset.csv` — Labeled training/evaluation dataset.
- `chatbot_v2.py` — Main interactive chatbot CLI (accepts VADER or ML mode).
- `train_ml_model.py` — Training script that vectorizes text and fits the ML model.
- `test_accuracy.py` — Evaluation script that compares Keyword, VADER, and ML accuracies.
- `sentiment_analysis_experiment.ipynb` — Jupyter Notebook detailing the analysis, EDA, training, and plots.
- `internship_report.md` — Official internship report covering learning objectives, challenges, and impact.
- `chat_history.csv` — Generated chat log storing user inputs, moods, intent, and bot responses.
