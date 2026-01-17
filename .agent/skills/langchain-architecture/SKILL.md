---
name: langchain-architecture
version: "1.0.7"
description: Design LLM applications with LangChain agents, chains, memory, and tools. Use when building autonomous agents, RAG systems, multi-step workflows, conversational AI with memory, or custom tool integrations.
---

# LangChain Architecture

Build LLM applications with agents, chains, memory, and tool integration.

## Core Components

| Component | Purpose | Types |
|-----------|---------|-------|
| Agents | Autonomous decision-making | ReAct, OpenAI Functions, Conversational |
| Chains | Sequential LLM calls | LLMChain, Sequential, Router, MapReduce |
| Memory | Context persistence | Buffer, Summary, Window, Entity, Vector |
| Tools | External capabilities | Search, databases, APIs, custom @tool |
| Callbacks | Monitoring/logging | Token tracking, latency, errors |

## Quick Start

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools, llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory, verbose=True
)
result = agent.run("What's the weather in SF? Then calculate 25 * 4")
```

## RAG Pattern

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load and split
loader = TextLoader('docs.txt')
texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\
    .split_documents(loader.load())

# Vector store + retrieval
vectorstore = Chroma.from_documents(texts, OpenAIEmbeddings())
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
result = qa_chain({"query": "What is the main topic?"})
```

## Custom Tools

```python
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """Search internal database for information."""
    return f"Results for: {query}"

@tool
def send_email(recipient: str, content: str) -> str:
    """Send an email to specified recipient."""
    return f"Email sent to {recipient}"

agent = initialize_agent(
    [search_database, send_email], llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
```

## Sequential Chain

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

extract_chain = LLMChain(
    llm=llm, output_key="entities",
    prompt=PromptTemplate(input_variables=["text"],
        template="Extract key entities from: {text}")
)
analyze_chain = LLMChain(
    llm=llm, output_key="analysis",
    prompt=PromptTemplate(input_variables=["entities"],
        template="Analyze: {entities}")
)

overall = SequentialChain(
    chains=[extract_chain, analyze_chain],
    input_variables=["text"],
    output_variables=["entities", "analysis"]
)
```

## Memory Selection

| Type | Use Case | Example |
|------|----------|---------|
| Buffer | Short conversations (<10 msg) | `ConversationBufferMemory()` |
| Summary | Long conversations | `ConversationSummaryMemory(llm=llm)` |
| Window | Sliding window (last N) | `ConversationBufferWindowMemory(k=5)` |
| Entity | Track entities | `ConversationEntityMemory(llm=llm)` |
| Vector | Semantic retrieval | `VectorStoreRetrieverMemory(retriever)` |

## Callback Handler

```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started: {prompts}")
    def on_llm_end(self, response, **kwargs):
        print(f"LLM ended: {response}")
    def on_agent_action(self, action, **kwargs):
        print(f"Agent action: {action}")

agent.run("query", callbacks=[CustomHandler()])
```

## Performance

```python
# Caching
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

# Streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Error handling | Wrap agent.run in try/except |
| Token tracking | Use callbacks to monitor usage |
| Timeout limits | Set max execution time |
| Input validation | Validate before agent execution |
| Version prompts | Track prompt templates in Git |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Memory overflow | Use Summary/Window memory for long chats |
| Poor tool selection | Write clear tool descriptions |
| Context overflow | Manage history length |
| No error handling | Implement fallback strategies |
| Slow retrieval | Optimize vector store queries |

## Checklist

- [ ] Error handling implemented
- [ ] Request/response logging
- [ ] Token usage monitoring
- [ ] Timeout limits set
- [ ] Rate limiting configured
- [ ] Input validation added
- [ ] Edge cases tested
- [ ] Observability (callbacks) setup
- [ ] Fallback strategies defined
- [ ] Prompts version controlled
