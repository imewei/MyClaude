---
description: Build production-ready AI assistants with NLU, conversation management,
  and intelligent response generation
triggers:
- /ai-assistant
- build production ready ai assistants
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<assistant_description>`
The agent should parse these arguments from the user's request.

# AI Assistant Development

Build AI assistant: $ARGUMENTS

## Mode
- Quick (5-10m): Basic architecture, template responses
- Standard (15-25m): Full NLP, LLM, Docker
- Comprehensive (30-45m): Multi-LLM, hierarchical memory, K8s, monitoring

## Architecture
**NLU** → **Dialog Manager** → **Response Generator** → **Context Manager**
- Intent classification, entity extraction, sentiment
- Conversation state, flow control, action selection
- Template or LLM responses, personalization
- Short/long-term memory

## Phases

1. **NLU Pipeline**: Intent, entities, sentiment with parallel processing, fallback handling
2. **Conversation Flows**: State machine (greeting, task_completion, error, farewell) with slot filling
3. **LLM Integration**: Claude Sonnet 4.5 (primary) → GPT-4 (fallback) → Ollama (local) → static
4. **Context**: Working (request) → Short-term (session) → Long-term (persistent) with pruning
5. **Testing**: Unit, integration, conversation, performance
6. **Deployment**: Docker/K8s/Serverless
7. **Monitoring**: Response time, success rate, fallback rate, inference time

## Success Criteria
Architecture, NLP pipeline, conversation flows, LLM with fallbacks, context management, testing, deployment, monitoring

Refs: ai-assistant-architecture.md, llm-integration-patterns.md, ai-testing-deployment.md
