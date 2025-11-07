# LangChain Advanced Patterns

Advanced RAG, memory strategies, multi-agent orchestration, and LangSmith tracing for production LangChain applications.

## Extended RAG Patterns

### HyDE (Hypothetical Document Embeddings)

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_voyageai import VoyageAIEmbeddings

base_embeddings = VoyageAIEmbeddings(model="voyage-3-large")
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatAnthropic(model="claude-sonnet-4-5"),
    base_embeddings=base_embeddings,
    prompt_key="web_search"
)

# Generate hypothetical documents for better retrieval
docs = vectorstore.similarity_search(query, embeddings=hyde_embeddings)
```

### RAG Fusion

```python
def rag_fusion(query: str, vectorstore):
    """Multiple query perspectives for comprehensive results"""
    # Generate multiple queries
    query_variants = llm.generate([
        f"Rephrase this question: {query}",
        f"What are related questions to: {query}",
        f"Alternative formulation of: {query}"
    ])
    
    # Retrieve with all variants
    all_docs = []
    for variant in query_variants:
        docs = vectorstore.similarity_search(variant, k=5)
        all_docs.extend(docs)
    
    # Rerank and deduplicate
    reranked = reranker.rerank(query, all_docs)
    return reranked[:10]
```

### Self-Query Retrieval

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(name="source", description="The source of the document", type="string"),
    AttributeInfo(name="date", description="The publication date", type="date"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatAnthropic(model="claude-sonnet-4-5"),
    vectorstore=vectorstore,
    document_contents="Technical documentation",
    metadata_field_info=metadata_field_info
)

# Automatically extracts filters from query
docs = retriever.get_relevant_documents(
    "Show me recent Python documentation from 2024"
)
```

## Advanced Memory Strategies

### Hierarchical Memory

```python
class HierarchicalMemory:
    """Multi-level memory system"""
    
    def __init__(self):
        self.working_memory = ConversationBufferMemory()  # Last 5 turns
        self.summary_memory = ConversationSummaryMemory()  # Session summary
        self.vector_memory = VectorStoreRetrieverMemory()  # Semantic search
        
    async def get_context(self, query: str):
        """Retrieve from all memory levels"""
        context = {
            'recent': await self.working_memory.load_memory_variables({}),
            'summary': await self.summary_memory.load_memory_variables({}),
            'relevant': await self.vector_memory.load_memory_variables({'query': query})
        }
        return self.merge_context(context)
```

### Entity Memory

```python
from langchain.memory import ConversationEntityMemory

entity_memory = ConversationEntityMemory(
    llm=ChatAnthropic(model="claude-sonnet-4-5"),
    entity_extraction_prompt=entity_extraction_template,
    entity_summarization_prompt=entity_summary_template
)

# Tracks entities across conversation
# Example: "John ordered pizza" -> Stores: {John: {ordered: pizza}}
```

## Multi-Agent Orchestration

### Supervisor Pattern

```python
from langgraph.graph import StateGraph, END
from typing import Literal, Annotated
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    messages: Annotated[list, "conversation history"]
    next_agent: str

def supervisor_node(state: AgentState):
    """Supervisor decides which agent to route to"""
    last_message = state["messages"][-1].content
    
    # LLM decides routing
    response = llm_with_functions.invoke([
        SystemMessage(content="Route to appropriate specialist agent"),
        HumanMessage(content=last_message)
    ])
    
    return {"next_agent": response.tool_calls[0]["args"]["next"]}

# Build graph
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_agent)
builder.add_node("coder", code_agent)
builder.add_node("analyst", analysis_agent)

builder.add_conditional_edges(
    "supervisor",
    lambda x: x["next_agent"],
    {
        "researcher": "researcher",
        "coder": "coder",
        "analyst": "analyst",
        "END": END
    }
)
```

### Parallel Agent Execution

```python
async def parallel_agents(query: str):
    """Run multiple agents concurrently"""
    agents = {
        'web_search': web_search_agent,
        'doc_retrieval': doc_retrieval_agent,
        'code_search': code_search_agent
    }
    
    tasks = [
        agent.ainvoke({"query": query})
        for agent in agents.values()
    ]
    
    results = await asyncio.gather(*tasks)
    return synthesize_results(results)
```

## Production Optimization

### Streaming with LangGraph

```python
async def stream_agent_response(query: str):
    """Stream agent execution"""
    agent = create_react_agent(llm, tools)
    
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=query)]},
        version="v1"
    ):
        if event["event"] == "on_chat_model_stream":
            yield event["data"]["chunk"].content
```

### Caching Strategy

```python
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache

# Cache LLM responses
set_llm_cache(RedisCache(redis_url="redis://localhost:6379"))

# Cache embeddings
from langchain.storage import RedisStore

embedding_cache = RedisStore(redis_url="redis://localhost:6379")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=VoyageAIEmbeddings(),
    document_embedding_cache=embedding_cache
)
```

## LangSmith Tracing

### Comprehensive Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "production-app"

# Automatic tracing of all LangChain calls
agent = create_react_agent(llm, tools)
result = await agent.ainvoke({"messages": [...]})

# View in LangSmith dashboard:
# - Full execution trace
# - Token usage per step
# - Latency breakdown
# - Error tracking
```

### Custom Run Names and Metadata

```python
from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer(
    project_name="production-app",
    tags=["version:1.0", "environment:prod"]
)

result = await agent.ainvoke(
    {"messages": [HumanMessage(content=query)]},
    config={
        "callbacks": [tracer],
        "run_name": f"user_query_{user_id}",
        "metadata": {"user_id": user_id, "session_id": session_id}
    }
)
```

---

**See Also**:
- [LLM Integration Patterns](./llm-integration-patterns.md) - Provider integration
- [AI Testing & Deployment](./ai-testing-deployment.md) - Testing strategies
- Command: `/langchain-agent`
