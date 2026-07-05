# Sentiment-Aware Support Chatbot (Task 1)

## What this is
A rule-based chatbot that detects customer sentiment (mood: upset / calm /
happy) using VADER and tailors its response tone accordingly, with basic
intent tagging (refund request, complaint, order tracking, compliment,
greeting).

Use `chatbot_v2.py` + `test_accuracy.py` as the version to submit - it's
the final, cleaned-up rewrite. (`chatbot.py` / `evaluate.py` were an
earlier draft, kept for reference.)

## Why these design choices
- **VADER over a deep-learning model**: VADER is lexicon-based, needs no
  training data or GPU, runs instantly, and is specifically tuned for short,
  informal text (chat messages, social media) — a good fit here and easy to
  explain in a report. A natural "next step" to mention in your report is
  swapping it for a fine-tuned transformer (e.g. DistilBERT) if higher
  nuance is needed.
- **Keyword-based intent detection**: transparent and easy to justify/debug
  for a first version. Mention in your report that a trained intent
  classifier (e.g. scikit-learn on labeled utterances) would be the next
  iteration.
- **CSV logging**: every turn is logged with its detected sentiment score
  and intent, which is what makes the `evaluate.py` metrics and your
  report's "Evaluation Criteria" section possible to back up with evidence.

## How to run
```bash
pip install vaderSentiment
python chatbot_v2.py       # interactive CLI chat
python test_accuracy.py    # accuracy + confusion matrix on a labeled test set
```

## Files
- `chatbot_v2.py` — core chatbot (score_mood, tag_intent, pick_reply, SupportChatbot) — **submit this one**
- `test_accuracy.py` — accuracy evaluation against a labeled test set — **submit this one**
- `chat_history.csv` — auto-generated log of every conversation turn
- `chatbot.py`, `evaluate.py`, `conversation_log.csv` — earlier draft, kept for your own reference/report notes on how the design evolved

## Extending toward Task 3 (dynamic knowledge base)
Add a `KnowledgeBase` class that:
1. Stores documents/FAQs as embeddings in a vector store (e.g. `chromadb` or `faiss`, both pip-installable).
2. On a schedule (cron / simple loop with `time.sleep`), re-embeds and upserts new source documents.
3. In `ResponseGenerator`, before falling back to a template, do a similarity search against the vector store and use the top match to compose the answer.

## Extending toward Task 6 (multilingual)
1. Add a language-detection step (`langdetect` or `fasttext`, both pip-installable) before sentiment analysis.
2. Translate non-English input to English internally (e.g. `deep-translator`, pip-installable, no API key needed for Google backend) so your existing VADER + intent logic keeps working.
3. Translate the generated response back to the user's detected language before replying.
4. Keep a `session_language` variable per conversation so it persists across turns even if the user briefly switches language mid-conversation.

## Note on originality
This implementation is written from scratch for this task. If you reference
any external tutorial or repo while building your own version, rewrite the
logic in your own words/code rather than copying it directly, and cite the
source (e.g. "sentiment scoring approach adapted from VADER's published
methodology, Hutto & Gilbert, 2014") in your internship report's
"Background" section — that keeps you clearly on the right side of the
plagiarism rule in the assignment email.
