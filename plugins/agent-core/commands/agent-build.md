---
version: "2.1.0"
description: Unified AI agent creation, optimization, and prompt engineering
argument-hint: <action> <target> [options]
category: agent-core
execution_time:
  quick: "5-15 minutes"
  standard: "15-45 minutes"
  deep: "1-2 hours"
color: magenta
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep, Bash(uv:*)]
external_docs:
  - agent-optimization-guide.md
  - langchain-advanced-patterns.md
  - prompt-patterns.md
  - llm-integration-patterns.md
tags: [agents, langchain, langgraph, prompts, optimization, multi-agent]
---

# AI Agent Development

$ARGUMENTS

## Actions

| Action | Description |
|--------|-------------|
| `create` | Create production-ready LangChain/LangGraph agents |
| `improve` | Optimize existing agent performance |
| `multi` | Coordinate multiple agents for optimization |
| `prompt` | Optimize prompts for better LLM performance |

**Examples:**
```bash
/agent-build create "customer support chatbot with RAG"
/agent-build improve my-agent --mode=check
/agent-build multi src/simulation/ --mode=scan
/agent-build prompt "You are a helpful assistant..."
```

## Options

**All actions:**
- `--mode <depth>`: quick, standard (default), deep/comprehensive

**Create:**
- `--framework <type>`: langchain, langgraph, anthropic-sdk

**Improve:**
- `--phase <n>`: 1 (analysis), 2 (engineering), 3 (testing), 4 (deploy)
- `--focus <area>`: tool-selection, reasoning, memory, safety

**Multi:**
- `--agents <list>`: Specify agents to use
- `--parallel`: Run agents in parallel

**Prompt:**
- `--model <target>`: gpt-4, claude, gemini

---

# Action: Create Agent

Build production-ready AI agents with LangChain/LangGraph.

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 5-10 min | Basic ReAct agent, simple tools |
| Standard | 15-25 min | + RAG pipeline, memory, LangSmith |
| Deep | 30-45 min | + Multi-agent, advanced RAG, full observability |

## Core Components

### Model Selection

| Component | Recommended |
|-----------|-------------|
| Primary LLM | Claude Sonnet 4.5 (`claude-sonnet-4-5`) |
| Embeddings | Voyage AI (`voyage-3-large`) |
| Code | `voyage-code-3` |
| Finance | `voyage-finance-2` |

### Agent Types

| Type | Use Case | Pattern |
|------|----------|---------|
| ReAct | General multi-step reasoning | `create_react_agent(llm, tools)` |
| Plan-and-Execute | Complex upfront planning | Separate planning/execution nodes |
| Multi-Agent | Specialized with routing | Supervisor + `Command[Literal[...]]` |

### Memory Systems

| Type | Use Case |
|------|----------|
| ConversationTokenBufferMemory | Token-based windowing |
| ConversationSummaryMemory | Compress long histories |
| ConversationEntityMemory | Track entities |
| VectorStoreRetrieverMemory | Semantic search |

### RAG Pipeline

| Component | Implementation |
|-----------|----------------|
| Embeddings | VoyageAIEmbeddings |
| Vector store | Pinecone, Chroma, Weaviate |
| Retriever | Hybrid search with reranking |

**Advanced patterns:** HyDE, RAG Fusion, Cohere Reranking

### Production Checklist

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

---

# Action: Improve Agent

Systematic agent improvement through performance analysis and prompt engineering.

## Mode Selection

| Command | Duration | Output |
|---------|----------|--------|
| `--mode=check` | 2-5 min | Health report with top 3 opportunities |
| `--phase=N` | 10-30 min | Targeted improvements for specific phase |
| `--mode=optimize` | 1-2 hours | Complete 4-phase improvement |

## Phases

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 1 | Performance analysis | Baseline metrics, failure modes |
| 2 | Prompt engineering | Chain-of-thought, few-shot, constitutional AI |
| 3 | Testing & validation | Test suite, A/B testing, evaluation |
| 4 | Deployment & monitoring | Versioning, staged rollout, monitoring |

## Health Report Format

```
Agent Health Report: <name>
Overall Score: X/100
├─ Success Rate: X% (target: >85%)
├─ Avg Corrections: X/task (target: <1.5)
├─ Tool Efficiency: X% (target: >80%)
└─ User Satisfaction: X/10

Top 3 Issues:
1. [Issue] → Fix: [Specific recommendation]
2. [Issue] → Fix: [Specific recommendation]
3. [Issue] → Fix: [Specific recommendation]
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Task success rate | +15% |
| User corrections | -25% |
| Safety violations | No increase |
| Response time | Within 10% of baseline |
| Cost per task | No increase >5% |

---

# Action: Multi-Agent Optimization

Coordinate specialized agents for comprehensive code optimization.

## Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| Scan | 2-5 min | Priority list of quick wins |
| Analyze | 10-30 min | Comprehensive report with patches |
| Apply | varies | Applied patches with validation |

## Pattern Detection

| Pattern | Impact |
|---------|--------|
| `for.*in range` in Python | Vectorization opportunity |
| `.apply(` in pandas | 10-100x speedup possible |
| Missing `@jit` on pure functions | 5-50x speedup |
| Missing `@lru_cache` | Repeated call optimization |

## Optimization Patterns

| Pattern | Speedup |
|---------|---------|
| Vectorization | 10-100x |
| JIT Compilation | 5-50x |
| Caching | 2-10x |
| Parallelization | Nx (N=cores) |
| GPU Acceleration | 10-1000x |

## Validation Gates

| Gate | Requirement |
|------|-------------|
| Tests | All pass (no regressions) |
| Performance | Improved or unchanged |
| Numerical accuracy | Within tolerance |
| Memory | Not increased >20% |

---

# Action: Prompt Optimization

Transform basic instructions into production-ready prompts.

**Expected improvements:** +40% accuracy, -30% hallucinations, -50-80% costs

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 5-10 min | Analysis + one technique, 3 tests |
| Standard | 15-25 min | Full optimization, 10 tests |
| Deep | 30-45 min | + meta-prompt, A/B strategy, 20+ tests |

## Optimization Techniques

### Chain-of-Thought Patterns

| Pattern | When to Use |
|---------|-------------|
| Zero-Shot CoT | Add "Let's think step-by-step" |
| Few-Shot CoT | Provide examples with reasoning |
| Tree-of-Thoughts | Explore multiple solution paths |

### Few-Shot Learning

| Type | Purpose |
|------|---------|
| Simple case | Demonstrates basic pattern |
| Edge case | Shows complexity handling |
| Counter-example | What NOT to do |

### Constitutional AI

1. Generate initial response
2. Review against principles (ACCURACY, SAFETY, QUALITY)
3. Produce refined response

**Benefits:** -40% harmful outputs, +25% factual accuracy

## Model-Specific Formats

### GPT-4 Style
```
##CONTEXT##
##OBJECTIVE##
##INSTRUCTIONS## (numbered)
##OUTPUT FORMAT## (JSON/structured)
```

### Claude Style
```xml
<context>background</context>
<task>objective</task>
<thinking>step-by-step</thinking>
<output_format>structure</output_format>
```

## Test Protocol

| Category | Count | Purpose |
|----------|-------|---------|
| Typical | 10 | Standard inputs |
| Edge | 5 | Boundary conditions |
| Adversarial | 3 | Stress testing |
| Out-of-scope | 2 | Rejection behavior |

---

## Best Practices

1. **Start simple** - Add complexity as needed
2. **Version control** - Commit prompts and agents
3. **Test thoroughly** - Unit, integration, evaluation
4. **Measure impact** - Success rate, quality, cost
5. **Monitor continuously** - Track post-deployment
6. **Safety first** - Never skip testing phase
7. **Iterate quickly** - Use targeted phase execution

## External Documentation

| Document | Purpose |
|----------|---------|
| agent-optimization-guide.md | Complete improvement methodology |
| langchain-advanced-patterns.md | LangGraph, RAG, multi-agent |
| prompt-patterns.md | Technique library |
| llm-integration-patterns.md | Model integration |
