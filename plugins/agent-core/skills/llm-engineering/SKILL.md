---
name: llm-engineering
description: Meta-orchestrator for LLM application engineering. Routes to specialized skills for prompt design, RAG systems, MCP integration, and safety guardrails. Use when building LLM-powered features, designing RAG pipelines, integrating MCP tools, or implementing safety constraints.
---

# LLM Engineering

Orchestrator for LLM application engineering. Routes problems to the appropriate specialized skill based on whether the task involves application architecture, prompt design, external tool integration, or safety enforcement.

## Expert Agent

For complex LLM engineering problems requiring deep context and safety expertise, delegate to the expert agent:

- **`context-specialist`**: Specialist for LLM application patterns, prompt optimization, MCP integration, and safety guardrail design.
  - *Location*: `plugins/agent-core/agents/context-specialist.md`
  - *Capabilities*: RAG pipeline design, prompt template engineering, MCP server configuration, and content safety systems.

## Core Skills

### [LLM Application Patterns](../llm-application-patterns/SKILL.md)
Architecture patterns, RAG pipelines, CoT integration, few-shot design, and evaluation harnesses. Use when designing the overall structure of an LLM-powered feature or application.

### [Prompt Engineering Patterns](../prompt-engineering-patterns/SKILL.md)
Production prompt design, reusable templates, and systematic prompt optimization. Use when crafting, refining, or standardizing prompts for reliability at scale.

### [MCP Integration](../mcp-integration/SKILL.md)
MCP server configuration, tool registration, and multi-tool coordination. Use when connecting an LLM agent to external tools or services via the Model Context Protocol.

### [Safety Guardrails](../safety-guardrails/SKILL.md)
Content filtering, output validation, jailbreak mitigation, and responsible AI constraints. Use when enforcing behavioral boundaries or compliance requirements on LLM outputs.

## Routing Decision Tree

```
What is the primary engineering concern?
|
+-- Designing the overall LLM application or RAG architecture?
|   --> llm-application-patterns (architecture, RAG, evaluation)
|
+-- Writing or optimizing prompts for production use?
|   --> prompt-engineering-patterns (templates, optimization)
|
+-- Connecting the agent to external tools via MCP?
|   --> mcp-integration (server config, tool coordination)
|
+-- Enforcing content or behavioral safety constraints?
    --> safety-guardrails (filtering, validation, responsible AI)
```

## Checklist

- [ ] Identify the primary concern using the routing decision tree before selecting a sub-skill
- [ ] Confirm RAG retrieval pipeline includes chunk size validation and relevance thresholds
- [ ] Verify all production prompts are versioned and tested against a regression suite
- [ ] Ensure MCP tool schemas are typed and include error response contracts
- [ ] Validate safety guardrails are applied at both input and output boundaries
- [ ] Document model assumptions (context window, token limits) in the application design
