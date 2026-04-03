---
name: nlp-fundamentals
description: "Build NLP pipelines with spaCy, Hugging Face Transformers, and NLTK including tokenization, named entity recognition, sentiment analysis, text classification, and sequence labeling. Use when processing text, training NLP models, or implementing text analysis pipelines."
---

# NLP Fundamentals

Build text processing and natural language understanding pipelines.

## Expert Agent

For designing ML pipelines with NLP components, delegate to the expert agent:

- **`ml-expert`**: Classical and applied ML specialist for feature engineering, model selection, and evaluation.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: Pipeline design, feature extraction, model evaluation, hyperparameter tuning.

## Text Preprocessing Pipeline

```python
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text: str) -> list[str]:
    """Clean and tokenize text with spaCy."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    doc = nlp(text.lower())
    return [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token) > 2
    ]
```

## Tokenization Strategies

| Strategy | Library | Use Case |
|----------|---------|----------|
| Word-level | spaCy, NLTK | Traditional NLP, simple tasks |
| BPE | Hugging Face (GPT-2) | Subword for generative models |
| WordPiece | Hugging Face (BERT) | Subword for encoders |
| SentencePiece | Hugging Face (T5) | Language-agnostic subword |
| Unigram | Hugging Face (XLNet) | Probabilistic subword |

## Named Entity Recognition

```python
import spacy

nlp = spacy.load("en_core_web_trf")  # Transformer-based

def extract_entities(text: str) -> list[dict]:
    """Extract named entities with labels and positions."""
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]

# Custom NER training
from spacy.training import Example

def train_custom_ner(nlp, train_data: list[tuple], n_iter: int = 30):
    """Fine-tune NER on domain-specific data."""
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[2])
    optimizer = nlp.resume_training()
    for i in range(n_iter):
        losses = {}
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.3, losses=losses)
```

## Hugging Face Text Classification

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Quick inference
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("This movie was fantastic!")

# Fine-tuning
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

dataset = load_dataset("imdb")

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True)
```

## Sentiment Analysis

```python
from transformers import pipeline

# Multi-class sentiment
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_sentiment(texts: list[str]) -> list[dict]:
    """Batch sentiment analysis with confidence scores."""
    results = sentiment(texts, batch_size=32)
    return [
        {"text": t, "label": r["label"], "score": round(r["score"], 4)}
        for t, r in zip(texts, results)
    ]
```

## Embeddings and Similarity

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    embeddings = model.encode([text_a, text_b])
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(cos_sim)

def semantic_search(query: str, corpus: list[str], top_k: int = 5) -> list[tuple]:
    """Return top-k most similar documents to query."""
    query_emb = model.encode([query])
    corpus_emb = model.encode(corpus)
    scores = np.dot(corpus_emb, query_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i], float(scores[i])) for i in top_indices]
```

## POS Tagging and Dependency Parsing

```python
def analyze_syntax(text: str) -> dict:
    """Extract POS tags and dependency relations."""
    doc = nlp(text)
    return {
        "tokens": [
            {
                "text": t.text,
                "pos": t.pos_,
                "dep": t.dep_,
                "head": t.head.text,
            }
            for t in doc
        ],
        "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
    }
```

## NLP Pipeline Checklist

- [ ] Define task type: classification, NER, QA, generation, similarity
- [ ] Choose model size based on latency/accuracy tradeoff
- [ ] Preprocess: lowercasing, URL removal, special tokens
- [ ] Split data: train/val/test with stratification for imbalanced classes
- [ ] Fine-tune with early stopping on validation metric
- [ ] Evaluate: precision, recall, F1 (macro for multi-class)
- [ ] Test on out-of-domain examples for robustness
- [ ] Quantize model (INT8) for production deployment
