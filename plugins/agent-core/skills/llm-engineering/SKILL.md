---
name: llm-engineering
description: Meta-orchestrator for general LLM application engineering, prompt systems, RAG architecture, MCP integration, and safety guardrails. Use when building non-domain-specific LLM features, designing RAG pipelines, integrating MCP tools, or implementing safety constraints. For scientific LLM pipelines, JAX/Julia codegen prompts, experiment automation, or LLM evaluation in scientific workflows, use science-suite llm-and-ai or sci-workflow-engineer.
---

# LLM Engineering

Orchestrator for LLM application engineering. Routes problems to the appropriate specialized skill based on whether the task involves application architecture, prompt design, external tool integration, or safety enforcement.

## Expert Agent

For complex LLM engineering problems requiring deep context-state design, delegate to the expert agent:

- **`context-specialist`**: Specialist for context-window strategy, persistent memory, retrieval state, and cross-agent context handoff in LLM systems.
  - *Location*: `plugins/agent-core/agents/context-specialist.md`
  - *Capabilities*: Token-budget planning, memory architecture, knowledge retrieval, and context handoff protocols.

## Core Skills

### [thinkfirst — Interview-First Prompt Crafting](../thinkfirst/SKILL.md)
Conversational interview workflow that transforms brain dumps, rough ideas, or unstructured notes into structured prompts. Use as the upstream step whenever the user arrives with a vague need — clarifies intent through the Seven Dimensions before any draft is written. Pairs with prompt-engineering-patterns for the production-grade refinement step.

### [LLM Application Patterns](../llm-application-patterns/SKILL.md)
Architecture patterns, RAG pipelines, CoT integration, few-shot design, and evaluation harnesses. Use when designing the overall structure of an LLM-powered feature or application.

### [Prompt Engineering Patterns](../prompt-engineering-patterns/SKILL.md)
Production prompt design, reusable templates, and systematic prompt optimization. Use when crafting, refining, or standardizing prompts for reliability at scale.

> For **closed-loop** prompt optimization (DSPy, TextGrad, evolutionary search, constitutional self-critique) see the sibling skill [`self-improving-agents`](../self-improving-agents/SKILL.md) in the reasoning-and-memory hub. This skill focuses on hand-authored production prompts; `self-improving-agents` is the programmatic / learned counterpart.

### [MCP Integration](../mcp-integration/SKILL.md)
MCP server configuration, tool registration, and multi-tool coordination. Use when connecting an LLM agent to external tools or services via the Model Context Protocol.

### [Safety Guardrails](../safety-guardrails/SKILL.md)
Content filtering, output validation, jailbreak mitigation, and responsible AI constraints. Use when enforcing behavioral boundaries or compliance requirements on LLM outputs.

## Routing Decision Tree

```
What is the primary engineering concern?
|
+-- Starting from a vague idea, brain dump, or unstructured requirements?
|   --> thinkfirst (interview-first clarification, then draft)
|
+-- Designing the overall LLM application or RAG architecture?
|   --> llm-application-patterns (architecture, RAG, evaluation)
|
+-- Writing or optimizing prompts for production use (already have requirements)?
|   --> prompt-engineering-patterns (templates, optimization, versioning)
|
+-- Connecting the agent to external tools via MCP?
|   --> mcp-integration (server config, tool coordination)
|
+-- Enforcing content or behavioral safety constraints?
    --> safety-guardrails (filtering, validation, responsible AI)
```

## Checklist

- [ ] For vague or unstructured user requests, start with thinkfirst before reaching for templates
- [ ] Identify the primary concern using the routing decision tree before selecting a sub-skill
- [ ] Confirm RAG retrieval pipeline includes chunk size validation and relevance thresholds
- [ ] Verify all production prompts are versioned and tested against a regression suite
- [ ] Ensure MCP tool schemas are typed and include error response contracts
- [ ] Validate safety guardrails are applied at both input and output boundaries
- [ ] Document model assumptions (context window, token limits) in the application design
