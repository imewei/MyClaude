---
name: langchain-architecture
description: Design LLM applications with LangChain agents, chains, memory, and tools. Use when building autonomous agents, RAG systems, multi-step workflows, conversational AI with memory, or custom tool integrations.
---

# LangChain Architecture

For LLM application development with LangChain, delegate to the `ai-engineer` agent.

## Expert Agent

- **`ai-engineer`**: LLM applications, RAG systems, and agentic AI
  - *Location*: `plugins/science-suite/agents/ai-engineer.md`

## Quick Reference

| Component | Purpose |
|-----------|---------|
| Chains | Sequential LLM operations |
| Agents | Autonomous tool-using LLM |
| Memory | Conversation persistence |
| Retrievers | RAG document retrieval |

See `llm-application-dev` and `rag-implementation` skills for detailed patterns.

## Chain Patterns

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LCEL chain composition
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following in {style} style."),
    ("human", "{text}")
])
chain = prompt | ChatOpenAI(model="gpt-4") | StrOutputParser()
result = chain.invoke({"style": "academic", "text": doc})
```

## Agent Architecture

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for relevant documents."""
    return retriever.invoke(query)

agent = create_tool_calling_agent(llm, [search_documents], prompt)
executor = AgentExecutor(agent=agent, tools=[search_documents], max_iterations=5)
```

## Memory Patterns

| Memory Type | Use Case | Persistence |
|-------------|----------|-------------|
| ConversationBufferMemory | Short dialogues | In-memory |
| ConversationSummaryMemory | Long dialogues | In-memory |
| VectorStoreRetrieverMemory | Semantic recall | Vector DB |
| SQLChatMessageHistory | Multi-session | Database |

## RAG Pipeline

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

# Ingest
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

# Retrieve and generate
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## Production Patterns

- **Fallbacks**: Chain `.with_fallbacks([backup_llm])` for resilience against provider outages.
- **Rate limiting**: Wrap LLM calls with `langchain_core.rate_limiter.InMemoryRateLimiter`.
- **Caching**: Enable `langchain.globals.set_llm_cache(SQLiteCache())` to reduce duplicate calls.
- **Streaming**: Use `chain.astream()` for token-by-token delivery to frontends.
- **Tracing**: Integrate LangSmith for production observability via `LANGCHAIN_TRACING_V2=true`.

## Checklist

- [ ] Chains composed with LCEL pipe syntax (not legacy `LLMChain`)
- [ ] Tools have docstrings (required for agent tool selection)
- [ ] Memory backend selected for expected conversation length
- [ ] Retriever `k` and `score_threshold` tuned for precision/recall
- [ ] Fallback chains configured for production reliability
- [ ] LangSmith tracing enabled for debugging and evaluation
