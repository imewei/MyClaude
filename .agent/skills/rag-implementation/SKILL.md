---
name: rag-implementation
version: "1.0.7"
description: Build production RAG systems with vector databases (Pinecone, Weaviate, Chroma), embeddings, chunking strategies, hybrid search (dense + BM25), reranking, and grounded prompts. Use when implementing document Q&A, knowledge base chatbots, or reducing LLM hallucinations.
---

# RAG Implementation

Retrieval-Augmented Generation for accurate, grounded LLM responses.

## Core Components

### Vector Databases

| Database | Best For | Features |
|----------|----------|----------|
| Pinecone | Production, managed | Scalable, fast queries |
| Weaviate | Hybrid search | Open-source, GraphQL |
| Chroma | Prototyping | Lightweight, easy setup |
| Qdrant | Filtered search | Fast, on-premise |
| FAISS | Local deployment | Meta's library, efficient |

### Embedding Models

| Model | Dims | Best For |
|-------|------|----------|
| text-embedding-ada-002 | 1536 | General purpose |
| all-MiniLM-L6-v2 | 384 | Fast, lightweight |
| bge-large-en-v1.5 | 1024 | SOTA performance |
| e5-large-v2 | 1024 | Multilingual |

## Quick Start

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load and chunk documents
loader = DirectoryLoader('./docs', glob="**/*.txt")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(loader.load())

# 2. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 3. Create retrieval chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True)

# 4. Query
result = qa({"query": "What are the main features?"})
```

## Chunking Strategies

| Strategy | Use Case | Splitter |
|----------|----------|----------|
| Recursive | General text | RecursiveCharacterTextSplitter |
| Token-based | Token limits | TokenTextSplitter |
| Semantic | Meaning-based | SemanticChunker |
| Markdown headers | Documentation | MarkdownHeaderTextSplitter |

```python
# Semantic chunking
from langchain.text_splitters import SemanticChunker

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile")
```

## Advanced Retrieval

### Hybrid Search

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5
dense = vectorstore.as_retriever(search_kwargs={"k": 5})

hybrid = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.3, 0.7])  # Dense-weighted
```

### Multi-Query

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=OpenAI())
```

### Parent Document

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000))
```

## Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get initial results
candidates = vectorstore.similarity_search(query, k=20)

# Rerank
pairs = [[query, doc.page_content] for doc in candidates]
scores = reranker.predict(pairs)
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```

### MMR (Maximal Marginal Relevance)

```python
results = vectorstore.max_marginal_relevance_search(
    query, k=5, fetch_k=20, lambda_mult=0.5)  # Balance diversity/relevance
```

## Grounding Prompts

### With Citations

```python
prompt = """Answer using only the context below. Cite sources using [1], [2].

Context:
{context}

Question: {question}

Answer (with citations):"""
```

### Confidence Scoring

```python
prompt = """Answer the question. If uncertain, say "I don't know."
Provide confidence (0-100%).

Context: {context}
Question: {question}

Answer:
Confidence:"""
```

## Metadata Filtering

```python
# Add metadata during indexing
for chunk in chunks:
    chunk.metadata = {"category": "technical", "source": chunk.metadata["source"]}

# Filter during retrieval
results = vectorstore.similarity_search(
    query, filter={"category": "technical"}, k=5)
```

## Evaluation

```python
def evaluate_rag(qa_chain, test_cases):
    metrics = {'accuracy': [], 'retrieval_quality': [], 'groundedness': []}

    for test in test_cases:
        result = qa_chain({"query": test['question']})
        metrics['accuracy'].append(score_answer(result['result'], test['expected']))
        metrics['retrieval_quality'].append(
            evaluate_retrieval(result['source_documents'], test['relevant_docs']))
        metrics['groundedness'].append(
            check_grounded(result['result'], result['source_documents']))

    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Chunk size | 500-1000 tokens, balance context/specificity |
| Overlap | 10-20% to preserve boundary context |
| Metadata | Source, page, timestamp for filtering |
| Hybrid search | Combine dense + sparse for best results |
| Reranking | Cross-encoder for top-k improvement |
| Citations | Always return source documents |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Poor retrieval | Check embeddings, chunk size, query |
| Irrelevant results | Add metadata filtering, hybrid search |
| Hallucinations | Improve grounding prompt, add verification |
| Slow queries | Optimize vector store, reduce k |
| Missing info | Verify indexing coverage |

## Checklist

- [ ] Chunking strategy matches content type
- [ ] Embeddings model selected for use case
- [ ] Metadata attached for filtering
- [ ] Retrieval strategy chosen (dense/hybrid/multi-query)
- [ ] Reranking implemented for quality
- [ ] Grounding prompt enforces factuality
- [ ] Evaluation metrics tracked
