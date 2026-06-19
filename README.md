# Semantic Book Recommender

A book recommendation app that lets you describe the kind of book you're looking for in plain language (e.g. *"A story about forgiveness"*) and get back a gallery of matching titles, filterable by category and emotional tone.

It combines semantic search over book descriptions with category and emotion classification, served through a Gradio dashboard.

## How it works

1. **`data-exploration.ipynb`** — loads the raw book metadata (via `kagglehub`), cleans missing values, and filters out books with missing or too-short descriptions.
2. **`zero-shot.ipynb`** — maps each book to a simple category (e.g. Fiction / Nonfiction) using a zero-shot classification model for books without enough labeled data.
3. **`sentiment.ipynb`** — runs each book's description through a fine-tuned emotion classifier (`emotion-english-distilroberta-base`) to score it on joy, anger, sadness, fear, and surprise.
4. **`gradio-dashboard.py`** — the app itself. Book descriptions are embedded with OpenAI embeddings and indexed in a Chroma vector store. A user query is matched semantically against descriptions, then results are filtered by category and sorted by emotional tone.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Running the app

```bash
python gradio-dashboard.py
```

This launches a local Gradio dashboard where you can enter a book description and filter by category and tone.

## Data

- `clean_data.csv` — cleaned raw book metadata
- `books_with_categories.csv` — books with simple category labels added
- `books_with_emotions.csv` — books with emotion scores added (used by the dashboard)
- `tagged_descriptions.txt` — book descriptions tagged with ISBN, used to build the vector store
