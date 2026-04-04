---
name: memory-system-patterns
description: Design and implement AI memory systems including vector databases, semantic/episodic memory, retrieval-augmented generation, and knowledge persistence. Use when building AI agents that need long-term memory, context management, or knowledge retrieval.
---

# Memory System Patterns

## Expert Agent

For memory architecture design and context management, delegate to:

- **`context-specialist`**: Manages context windows, memory hierarchies, and knowledge retrieval strategies.
  - *Location*: `plugins/agent-core/agents/context-specialist.md`

Comprehensive guide for designing memory systems that give AI agents persistent, retrievable knowledge.

---

## 1. Memory Hierarchy

### Three-Tier Architecture

| Tier | Analogy | Storage | Latency | Capacity |
|------|---------|---------|---------|----------|
| **Working Memory** | CPU registers | Context window | ~0ms | 4K-200K tokens |
| **Short-Term Memory** | RAM | Session cache / Redis | ~5ms | Conversation history |
| **Long-Term Memory** | Disk | Vector DB / SQL | ~50ms | Unbounded |

### When to Use Each Tier

- **Working Memory**: Current task context, active instructions, recent tool outputs.
- **Short-Term Memory**: Conversation turns within a session, recently retrieved documents.
- **Long-Term Memory**: User preferences, past decisions, domain knowledge.

---

## 2. Vector Database Selection

### Comparison Matrix

| Database | Hosting | Max Vectors | Filtering | Best For |
|----------|---------|-------------|-----------|----------|
| **Pinecone** | Managed | Billions | Metadata | Production SaaS |
| **Chroma** | Self-hosted | Millions | Metadata | Prototyping, local dev |
| **Weaviate** | Both | Billions | GraphQL | Hybrid search |
| **Qdrant** | Both | Billions | Payload | High-performance filtering |
| **pgvector** | Self-hosted | Millions | SQL | Existing Postgres stack |

### Embedding Strategy

```python
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    content: str
    embedding: list[float]
    metadata: dict
    timestamp: str
    source: str
    access_count: int = 0

def chunk_document(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

---

## 3. Retrieval Patterns

### Semantic Search

Query the vector store with an embedding of the current question. Return top-K nearest neighbors.

```python
def retrieve_memories(
    query: str,
    vector_store: VectorStore,
    top_k: int = 5,
    score_threshold: float = 0.75,
) -> list[MemoryEntry]:
    """Retrieve relevant memories above a similarity threshold."""
    query_embedding = embed(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    return [r for r in results if r.score >= score_threshold]
```

### Hybrid Search

Combine semantic similarity with keyword matching (BM25) for higher recall.

| Component | Strength | Weakness |
|-----------|----------|----------|
| Semantic (dense) | Meaning, paraphrase | Exact terms, names |
| Keyword (sparse) | Exact match, rare terms | Synonyms, context |
| **Hybrid** | Best of both | Slightly higher latency |

### Recency-Weighted Retrieval

Apply a time-decay factor to prioritize recent memories while keeping old relevant ones accessible.

```python
import math

def recency_score(age_hours: float, half_life: float = 168.0) -> float:
    """Exponential decay with configurable half-life (default 1 week)."""
    return math.pow(0.5, age_hours / half_life)

def combined_score(similarity: float, age_hours: float) -> float:
    """Blend semantic similarity with recency."""
    return 0.7 * similarity + 0.3 * recency_score(age_hours)
```

---

## 4. Memory Consolidation

### Summarization Pipeline

Periodically compress old memories to prevent unbounded growth.

| Stage | Trigger | Action |
|-------|---------|--------|
| **Raw** | Every interaction | Store verbatim |
| **Summarized** | After 24 hours | Compress to key facts |
| **Archived** | After 30 days | Extract only durable knowledge |
| **Pruned** | Quarterly | Remove contradicted or stale entries |

### Conflict Resolution

When new information contradicts stored memory:

1. Flag the conflict with both old and new entries.
2. Keep the newer entry as primary, mark the older as superseded.
3. Log the conflict for human review if confidence is below threshold.

---

## 5. Context Window Management

### Budget Allocation

| Section | Budget % | Purpose |
|---------|----------|---------|
| System prompt | 10-15% | Instructions, persona, constraints |
| Retrieved context | 30-40% | Relevant memories and documents |
| Conversation history | 20-30% | Recent turns |
| Working space | 20-30% | Room for reasoning and output |

### Overflow Strategies

- **Sliding Window**: Drop oldest conversation turns first.
- **Summarize-and-Replace**: Compress old turns into a summary block.
- **Priority Queue**: Rank context items by relevance, drop lowest first.

---

## 6. Memory System Checklist

- [ ] Memory tiers defined (working, short-term, long-term)
- [ ] Vector database selected and provisioned
- [ ] Embedding model chosen with appropriate dimensions
- [ ] Chunking strategy defined (size, overlap, boundaries)
- [ ] Retrieval threshold tuned on representative queries
- [ ] Consolidation schedule configured
- [ ] Context budget allocated across sections
- [ ] Data retention and privacy policy applied

---

## Related Skills

- `llm-application-patterns` -- RAG implementation patterns that build on memory retrieval
- `multi-agent-coordination` -- Multi-agent systems that share memory across agents
