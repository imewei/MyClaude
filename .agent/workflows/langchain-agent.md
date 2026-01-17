---
description: Create production-ready LangChain agents with LangGraph, RAG, and observability
triggers:
- /langchain-agent
- create production ready langchain agents
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<agent_description>`
The agent should parse these arguments from the user's request.

# LangChain/LangGraph Agent Development

Build production-ready AI agent for: $ARGUMENTS

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 5-10 min | Basic ReAct agent, simple tools, in-memory buffer |
| Standard (default) | 15-25 min | + RAG pipeline, memory with summarization, LangSmith |
| Comprehensive | 30-45 min | + Multi-agent orchestration, advanced RAG, full observability |

## External Documentation

| Topic | Reference | Lines |
|-------|-----------|-------|
| Advanced Patterns | [langchain-advanced-patterns.md](../../plugins/llm-application-dev/docs/langchain-advanced-patterns.md) | ~300 |
| LLM Integration | [llm-integration-patterns.md](../../plugins/llm-application-dev/docs/llm-integration-patterns.md) | ~400 |
| Testing & Deployment | [ai-testing-deployment.md](../../plugins/llm-application-dev/docs/ai-testing-deployment.md) | ~500 |

## Core Requirements

- LangChain 0.1+ and LangGraph APIs
- Async patterns throughout
- Comprehensive error handling
- LangSmith observability
- Security best practices
- Cost optimization

## Model & Embeddings

| Component | Recommended |
|-----------|-------------|
| Primary LLM | Claude Sonnet 4.5 (`claude-sonnet-4-5`) |
| Embeddings | Voyage AI (`voyage-3-large`) - Anthropic recommended |
| Code | `voyage-code-3` |
| Finance | `voyage-finance-2` |
| Legal | `voyage-law-2` |

## Agent Types

| Type | Use Case | Pattern |
|------|----------|---------|
| ReAct | General multi-step reasoning | `create_react_agent(llm, tools)` |
| Plan-and-Execute | Complex upfront planning | Separate planning/execution nodes |
| Multi-Agent | Specialized with routing | Supervisor + `Command[Literal[...]]` |

## Memory Systems

| Type | Use Case |
|------|----------|
| ConversationTokenBufferMemory | Token-based windowing |
| ConversationSummaryMemory | Compress long histories |
| ConversationEntityMemory | Track entities |
| VectorStoreRetrieverMemory | Semantic search |
| Hybrid | Combine multiple types |

## RAG Pipeline

### Components

| Component | Implementation |
|-----------|----------------|
| Embeddings | VoyageAIEmbeddings |
| Vector store | Pinecone, Chroma, Weaviate |
| Retriever | Hybrid search with reranking |

### Advanced Patterns
- **HyDE**: Hypothetical documents for better retrieval
- **RAG Fusion**: Multiple query perspectives
- **Reranking**: Cohere Rerank for relevance

## Tools Integration

| Element | Requirement |
|---------|-------------|
| Schema | Pydantic BaseModel with Field descriptions |
| Async | `async def` with `coroutine=` parameter |
| Error handling | Try/except with informative returns |

## Production Deployment

### FastAPI Server
- Streaming with `StreamingResponse`
- Health checks for LLM, tools, memory
- Structured logging with `structlog`

### Optimization
| Strategy | Purpose |
|----------|---------|
| Redis caching | Response caching with TTL |
| Connection pooling | Reuse vector DB connections |
| Load balancing | Multiple agent workers |
| Timeouts | All async operations |
| Retry logic | Exponential backoff |

### Monitoring
- LangSmith traces
- Prometheus metrics
- Health endpoints

## LangGraph State Pattern

```python
builder = StateGraph(MessagesState)
builder.add_node("node", func)
builder.add_conditional_edges("node", router, {"a": "next", "b": END})
agent = builder.compile(checkpointer=checkpointer)
```

## Implementation Checklist

- [ ] Initialize LLM with Claude Sonnet 4.5
- [ ] Setup Voyage AI embeddings
- [ ] Create tools with async + error handling
- [ ] Implement memory system
- [ ] Build state graph with LangGraph
- [ ] Add LangSmith tracing
- [ ] Implement streaming responses
- [ ] Setup health checks
- [ ] Add caching layer (Redis)
- [ ] Configure retry logic
- [ ] Write evaluation tests
- [ ] Document API endpoints

## Best Practices

1. **Always async**: `ainvoke`, `astream`
2. **Handle errors**: Try/except with fallbacks
3. **Monitor everything**: Trace, log, metric
4. **Optimize costs**: Cache, token limits
5. **Secure secrets**: Environment variables only
6. **Test thoroughly**: Unit, integration, evaluation
7. **Version control state**: Use checkpointers

## Related Commands

- `/ai-assistant` - Build custom AI assistants
- `/prompt-optimize` - Optimize agent prompts
