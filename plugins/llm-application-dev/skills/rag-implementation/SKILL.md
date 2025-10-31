---
name: rag-implementation
description: Build production-ready Retrieval-Augmented Generation (RAG) systems for LLM applications using vector databases, embeddings, and semantic search to ground AI responses in factual knowledge. Use when writing or editing Python files that implement document Q&A systems, knowledge base chatbots, semantic search APIs, or RAG pipelines. Apply this skill when setting up vector databases (Pinecone, Weaviate, Chroma, Qdrant, Milvus, FAISS), implementing document loaders and text splitters, creating embeddings with OpenAI, Cohere, or sentence-transformers models, building retrieval chains with LangChain or LlamaIndex, implementing hybrid search combining vector similarity and keyword matching (BM25), adding reranking with cross-encoders or Cohere Rerank API, designing chunking strategies (recursive, semantic, token-based, markdown-aware), optimizing retrieval with metadata filtering and maximal marginal relevance, reducing hallucinations through grounded prompts and source citations, or evaluating RAG systems for accuracy, retrieval quality, and groundedness.
---

# RAG Implementation

Master Retrieval-Augmented Generation (RAG) to build LLM applications that provide accurate, grounded responses using external knowledge sources.

## When to Use This Skill

- Writing or editing Python files (`.py`) that implement RAG pipelines or document Q&A systems
- Building Q&A systems over proprietary documents, knowledge bases, or documentation
- Creating chatbots with current, factual information grounded in retrieved context
- Implementing semantic search APIs with natural language queries
- Reducing hallucinations with grounded responses using retrieval-augmented prompts
- Enabling LLMs to access domain-specific knowledge from vector databases
- Building documentation assistants that cite sources for their answers
- Creating research tools with source citation and evidence-based responses
- Setting up vector databases (Pinecone, Weaviate, Chroma, Qdrant, Milvus, FAISS)
- Implementing document loaders for PDFs, Word docs, web pages, or text files
- Creating text splitters and chunking strategies (recursive, semantic, token-based, markdown headers)
- Generating and storing embeddings using OpenAI, Cohere, or sentence-transformers models
- Building retrieval chains with LangChain (RetrievalQA, ConversationalRetrievalChain) or LlamaIndex
- Implementing hybrid search combining dense retrieval (embeddings) and sparse retrieval (BM25, TF-IDF)
- Adding reranking stages with cross-encoders (ms-marco models) or Cohere Rerank API
- Optimizing retrieval with metadata filtering, maximal marginal relevance (MMR), or parent document patterns
- Designing prompts that enforce grounding and prevent hallucinations ("answer only from context")
- Evaluating RAG systems for retrieval quality (precision@k, recall@k, NDCG) and answer accuracy
- Implementing contextual compression to extract only relevant portions of retrieved documents

## Core Components

### 1. Vector Databases
**Purpose**: Store and retrieve document embeddings efficiently

**Options:**
- **Pinecone**: Managed, scalable, fast queries
- **Weaviate**: Open-source, hybrid search
- **Milvus**: High performance, on-premise
- **Chroma**: Lightweight, easy to use
- **Qdrant**: Fast, filtered search
- **FAISS**: Meta's library, local deployment

### 2. Embeddings
**Purpose**: Convert text to numerical vectors for similarity search

**Models:**
- **text-embedding-ada-002** (OpenAI): General purpose, 1536 dims
- **all-MiniLM-L6-v2** (Sentence Transformers): Fast, lightweight
- **e5-large-v2**: High quality, multilingual
- **Instructor**: Task-specific instructions
- **bge-large-en-v1.5**: SOTA performance

### 3. Retrieval Strategies
**Approaches:**
- **Dense Retrieval**: Semantic similarity via embeddings
- **Sparse Retrieval**: Keyword matching (BM25, TF-IDF)
- **Hybrid Search**: Combine dense + sparse
- **Multi-Query**: Generate multiple query variations
- **HyDE**: Generate hypothetical documents

### 4. Reranking
**Purpose**: Improve retrieval quality by reordering results

**Methods:**
- **Cross-Encoders**: BERT-based reranking
- **Cohere Rerank**: API-based reranking
- **Maximal Marginal Relevance (MMR)**: Diversity + relevance
- **LLM-based**: Use LLM to score relevance

## Quick Start

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load documents
loader = DirectoryLoader('./docs', glob="**/*.txt")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# 5. Query
result = qa_chain({"query": "What are the main features?"})
print(result['result'])
print(result['source_documents'])
```

## Advanced RAG Patterns

### Pattern 1: Hybrid Search
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Sparse retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Dense retriever (embeddings)
embedding_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, embedding_retriever],
    weights=[0.3, 0.7]
)
```

### Pattern 2: Multi-Query Retrieval
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Generate multiple query perspectives
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=OpenAI()
)

# Single query → multiple variations → combined results
results = retriever.get_relevant_documents("What is the main topic?")
```

### Pattern 3: Contextual Compression
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Returns only relevant parts of documents
compressed_docs = compression_retriever.get_relevant_documents("query")
```

### Pattern 4: Parent Document Retriever
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store for parent documents
store = InMemoryStore()

# Small chunks for retrieval, large chunks for context
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
```

## Document Chunking Strategies

### Recursive Character Text Splitter
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Try these in order
)
```

### Token-Based Splitting
```python
from langchain.text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
```

### Semantic Chunking
```python
from langchain.text_splitters import SemanticChunker

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)
```

### Markdown Header Splitter
```python
from langchain.text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
```

## Vector Store Configurations

### Pinecone
```python
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

index = pinecone.Index("your-index-name")

vectorstore = Pinecone(index, embeddings.embed_query, "text")
```

### Weaviate
```python
import weaviate
from langchain.vectorstores import Weaviate

client = weaviate.Client("http://localhost:8080")

vectorstore = Weaviate(client, "Document", "content", embeddings)
```

### Chroma (Local)
```python
from langchain.vectorstores import Chroma

vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

## Retrieval Optimization

### 1. Metadata Filtering
```python
# Add metadata during indexing
chunks_with_metadata = []
for i, chunk in enumerate(chunks):
    chunk.metadata = {
        "source": chunk.metadata.get("source"),
        "page": i,
        "category": determine_category(chunk.page_content)
    }
    chunks_with_metadata.append(chunk)

# Filter during retrieval
results = vectorstore.similarity_search(
    "query",
    filter={"category": "technical"},
    k=5
)
```

### 2. Maximal Marginal Relevance
```python
# Balance relevance with diversity
results = vectorstore.max_marginal_relevance_search(
    "query",
    k=5,
    fetch_k=20,  # Fetch 20, return top 5 diverse
    lambda_mult=0.5  # 0=max diversity, 1=max relevance
)
```

### 3. Reranking with Cross-Encoder
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get initial results
candidates = vectorstore.similarity_search("query", k=20)

# Rerank
pairs = [[query, doc.page_content] for doc in candidates]
scores = reranker.predict(pairs)

# Sort by score and take top k
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```

## Prompt Engineering for RAG

### Contextual Prompt
```python
prompt_template = """Use the following context to answer the question. If you cannot answer based on the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
```

### With Citations
```python
prompt_template = """Answer the question based on the context below. Include citations using [1], [2], etc.

Context:
{context}

Question: {question}

Answer (with citations):"""
```

### With Confidence
```python
prompt_template = """Answer the question using the context. Provide a confidence score (0-100%) for your answer.

Context:
{context}

Question: {question}

Answer:
Confidence:"""
```

## Evaluation Metrics

```python
def evaluate_rag_system(qa_chain, test_cases):
    metrics = {
        'accuracy': [],
        'retrieval_quality': [],
        'groundedness': []
    }

    for test in test_cases:
        result = qa_chain({"query": test['question']})

        # Check if answer matches expected
        accuracy = calculate_accuracy(result['result'], test['expected'])
        metrics['accuracy'].append(accuracy)

        # Check if relevant docs were retrieved
        retrieval_quality = evaluate_retrieved_docs(
            result['source_documents'],
            test['relevant_docs']
        )
        metrics['retrieval_quality'].append(retrieval_quality)

        # Check if answer is grounded in context
        groundedness = check_groundedness(
            result['result'],
            result['source_documents']
        )
        metrics['groundedness'].append(groundedness)

    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

## Resources

- **references/vector-databases.md**: Detailed comparison of vector DBs
- **references/embeddings.md**: Embedding model selection guide
- **references/retrieval-strategies.md**: Advanced retrieval techniques
- **references/reranking.md**: Reranking methods and when to use them
- **references/context-window.md**: Managing context limits
- **assets/vector-store-config.yaml**: Configuration templates
- **assets/retriever-pipeline.py**: Complete RAG pipeline
- **assets/embedding-models.md**: Model comparison and benchmarks

## Best Practices

1. **Chunk Size**: Balance between context and specificity (500-1000 tokens)
2. **Overlap**: Use 10-20% overlap to preserve context at boundaries
3. **Metadata**: Include source, page, timestamp for filtering and debugging
4. **Hybrid Search**: Combine semantic and keyword search for best results
5. **Reranking**: Improve top results with cross-encoder
6. **Citations**: Always return source documents for transparency
7. **Evaluation**: Continuously test retrieval quality and answer accuracy
8. **Monitoring**: Track retrieval metrics in production

## Common Issues

- **Poor Retrieval**: Check embedding quality, chunk size, query formulation
- **Irrelevant Results**: Add metadata filtering, use hybrid search, rerank
- **Missing Information**: Ensure documents are properly indexed
- **Slow Queries**: Optimize vector store, use caching, reduce k
- **Hallucinations**: Improve grounding prompt, add verification step
