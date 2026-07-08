# Cognova

An AI assistant that actually pays attention to *how* you're asking, not just *what* you're asking.

Most chatbots pick one trick and stick with it — either they answer questions from a knowledge base, or they detect sentiment, or they handle multiple languages. Cognova tries to do all of it at once, in the same conversation, without losing track of context when things change mid-chat.

This started as a set of internship tasks (sentiment detection, medical Q&A, dynamic knowledge updates, domain expertise, multi-modal reasoning, multilingual support) that I decided to build as one connected system instead of six separate scripts.

## What it does

- **Reads the room.** Detects whether you sound frustrated, happy, or neutral, and adjusts its response tone accordingly.
- **Knows its stuff.** Answers domain-specific questions using retrieval — right now that's medical Q&A (via MedQuAD) and research paper explanations (via arXiv), but the retrieval layer is built to plug into any dataset.
- **Keeps learning.** New information gets pulled in and embedded on a schedule, so the knowledge base doesn't go stale the day after you deploy it.
- **Understands images too.** You can show it something, not just tell it something, and it'll reason across both.
- **Switches languages without losing the plot.** Start a conversation in English, drop into Hindi halfway through, and it keeps the context intact instead of resetting.

## How it's put together

```
Input (text / image / language detection)
        ↓
Understanding (sentiment + entity/intent recognition)
        ↓
Knowledge layer (vector DB + scheduled re-ingestion)
        ↓
Generation (open-source LLM, language-aware output)
```

Nothing exotic here — it's mostly about getting these pieces to share one consistent state instead of running in isolation.

## Stack

- **UI:** Streamlit
- **Vector store:** FAISS / ChromaDB
- **LLM:** open-source (Llama / Mistral, swappable)
- **Sentiment:** transformer-based classifier
- **Language detection:** langdetect / fastText
- **Orchestration:** LangChain

None of this is locked in — swap any piece out as needed.

## Project layout

```
cognova/
├── data/              datasets used for retrieval (MedQuAD, arXiv subset, etc.)
├── ingestion/          scripts that keep the vector store updated
├── modules/
│   ├── sentiment.py
│   ├── language.py
│   ├── retrieval.py
│   └── multimodal.py
├── app.py              Streamlit entry point
├── requirements.txt
└── README.md
```

## Running it locally

```bash
git clone https://github.com/BhuvanJJ0055/Cognova.git
cd Cognova
pip install -r requirements.txt
streamlit run app.py
```

## Where this could actually be useful

- Support chatbots that don't sound robotic when someone's annoyed
- A first-pass medical symptom checker (not a replacement for a doctor, obviously)
- Explaining research papers in plain language, with follow-up questions
- Support desks handling users who switch languages mid-sentence
- Troubleshooting bots where a screenshot says more than a paragraph of text

## Still to do

- [x] Get the core sentiment-aware chat loop solid in one language/domain first
- [x] Wire up RAG for medical + research domains (MedQuAD indexing and retrieval completed)
- [ ] Automate the knowledge base refresh
- [ ] Add image understanding
- [ ] Handle language switching mid-conversation
- [ ] Deploy a public demo

## License

MIT — use it, break it, learn from it.

## Author

Bhuvan J J
