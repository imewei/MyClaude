---
version: "1.0.6"
category: "llm-application-dev"
command: "/ai-assistant"
description: Build production-ready AI assistants with NLU, conversation management, and intelligent response generation
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<assistant_description>"
color: blue
execution_modes:
  quick: "5-10 minutes"
  standard: "15-25 minutes"
  comprehensive: "30-45 minutes"
agents:
  primary:
    - ai-engineer
  conditional:
    - agent: prompt-engineer
      trigger: pattern "prompt|llm.*prompt|response.*generation"
  orchestrated: false
---

# AI Assistant Development

Build production-ready AI assistants with NLU, conversation management, and intelligent responses.

## Requirements

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 5-10 min | Basic architecture, template responses, in-memory context |
| Standard (default) | 15-25 min | Full NLP pipeline, LLM integration, Docker deployment |
| Comprehensive | 30-45 min | Multi-provider LLM, hierarchical memory, K8s, full monitoring |

---

## Phase 1: Architecture Design

### Core Components

| Component | Purpose |
|-----------|---------|
| **NLU** | Intent classification, entity extraction, sentiment analysis |
| **Dialog Manager** | Conversation state, flow control, action selection |
| **Response Generator** | Template or LLM-powered responses, personalization |
| **Context Manager** | Short-term (conversation), long-term (user profile) memory |

### Pipeline Flow
```
User Message → NLU → Dialog Manager → Response Generator → Context Update → Response
```

---

## Phase 2: NLP Pipeline

### NLU Tasks

| Task | Output |
|------|--------|
| Intent Detection | Intent with confidence score |
| Entity Extraction | Named entities with types |
| Sentiment Analysis | Emotion, urgency, satisfaction |

### Implementation Strategy
- Parallel processing of NLU tasks
- Fallback handling for low-confidence intents
- Custom domain entity recognition

---

## Phase 3: Conversation Flows

### Flow Types

| Flow | Purpose |
|------|---------|
| Greeting | Welcome, understand needs |
| Task Completion | Slot filling, confirmation, execution |
| Error Handling | Clarification, fallback |
| Farewell | Conversation closure |

### Flow Engine Pattern
- State machine with nodes (nlu_processing, validation, slot_filling, action)
- Conditional transitions based on NLU results
- Graceful degradation on errors

---

## Phase 4: LLM Integration

### Provider Setup

| Provider | Model | Use Case |
|----------|-------|----------|
| Anthropic | Claude Sonnet 4.5 | Primary |
| OpenAI | GPT-4 | Fallback |
| Local | Ollama/vLLM | Privacy/cost |

### Integration Pattern
- Fallback chain: Primary → Secondary → Local → Static response
- Temperature tuning per task type
- Token limit management

---

## Phase 5: Context Management

### Memory Hierarchy

| Level | Content | Duration |
|-------|---------|----------|
| Working | Last N messages | Request |
| Short-term | Session context | Session |
| Long-term | User profile, preferences | Persistent |

### Context Operations
- Reference resolution (pronouns, temporal)
- Topic shift detection
- Entity state tracking
- Context pruning for token limits

---

## Phase 6: Testing & Deployment

### Testing Strategy

| Test Type | Scope |
|-----------|-------|
| Unit | Individual components |
| Integration | Component interactions |
| Conversation | Multi-turn flows |
| Performance | Latency, throughput |

### Deployment Options

| Platform | Use Case |
|----------|----------|
| Docker | Standard deployment |
| Kubernetes | Scalable production |
| Serverless | Event-driven, cost-optimized |

---

## Phase 7: Monitoring

### Key Metrics

| Category | Metrics |
|----------|---------|
| Real-time | Active sessions, response time, success rate |
| Conversation | Avg length, completion rate, satisfaction |
| System | Inference time, cache hit rate, errors |

### Alerts
- High fallback rate (>20%)
- Slow response (p95 >2s)
- Error rate spike

---

## Common Patterns

| Pattern | Apply | Avoid |
|---------|-------|-------|
| Intent | Multi-intent, hierarchical taxonomy | Over-fitting with too many intents |
| Response | Hybrid (templates + LLM), personalization | Over-reliance on LLM for simple cases |
| Context | Pruning, entity tracking | Unbounded memory |

---

## Success Criteria

- ✅ Architecture with all core components
- ✅ NLP pipeline (intent, entities, sentiment)
- ✅ Conversation flows for key use cases
- ✅ LLM integration with fallbacks
- ✅ Context management implemented
- ✅ Testing framework created
- ✅ Deployment configured
- ✅ Monitoring set up

---

## External Documentation

- `ai-assistant-architecture.md` - Complete patterns (~600 lines)
- `llm-integration-patterns.md` - Provider integration (~400 lines)
- `ai-testing-deployment.md` - Testing and deployment (~500 lines)
