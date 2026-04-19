# Agent Core

Meta-agent suite for multi-agent orchestration, advanced reasoning, and context engineering. Coordinates all other suites but is never delegated to from below. Optimized for Claude Opus.

## Overview

Agent Core is the coordination layer of MyClaude. Its 3 agents (orchestrator, reasoning-engine, context-specialist) manage multi-agent workflows, decompose complex problems, and maintain session context. It provides 4 hub skills routing to 13 sub-skills covering orchestration patterns, reasoning frameworks, LLM engineering, and intent clarification. Use agent-core when you need to coordinate multiple specialists or tackle problems requiring structured reasoning.

## Quick Start / Usage Examples

```bash
# Assemble a team for a complex task
/team-assemble backend-dev

# Deep structured reasoning on a hard problem
/ultra-think "Analyze the trade-offs of event sourcing vs CRUD for this domain"

# Ask the orchestrator to coordinate specialists
@orchestrator "Coordinate a code review across architecture, security, and testing"
```

## Features

- **Multi-Agent Orchestration**: Expert coordination of specialized agent teams and complex distributed workflows.
- **Advanced Reasoning**: Systematic problem-solving using structured frameworks (First Principles, Systems Thinking, OODA Loop).
- **Context Engineering**: Intelligent management of long-running session context, vector databases, and knowledge graphs.
- **LLM Application Patterns**: Production-ready RAG implementation, prompt optimization, and evaluation frameworks.
- **MCP Integration**: First-class support for Model Context Protocol servers (Serena, GitHub, Context7).

## Agents

| Agent | Model | Specialization |
|-------|-------|----------------|
| `orchestrator` | opus | Multi-agent coordination, team assembly, task delegation |
| `reasoning-engine` | opus | Advanced reasoning, prompt design, cognitive tasks |
| `context-specialist` | opus | Context engineering, memory systems, knowledge graphs |

## Commands

| Command | Description |
|---------|-------------|
| `/agent-build` | Unified AI agent creation, optimization, and prompt engineering |
| `/ai-assistant` | Build production-ready AI assistants with NLU and response generation |
| `/docs-lookup` | Query library documentation via Context7 MCP |
| `/reflection` | Meta-cognitive reflection framework execution |
| `/ultra-think` | Advanced structured reasoning with branching exploration |
| `/team-assemble` | Codebase-aware recommender with 10 focused teams (20 variants) + long-running workflow protocol (v3.4.1) |

## Skills (4 hubs → 13 sub-skills)

Hub skills route to specialized sub-skills via decision trees:

| Hub | Sub-skills | Focus |
|-----|------------|-------|
| `agent-systems` | 4 | Orchestration, coordination, performance optimization, evaluation |
| `reasoning-and-memory` | 5 | Reasoning frameworks, reflection, memory patterns, knowledge graphs, self-improving agents |
| `llm-engineering` | 4 | LLM app patterns, prompt engineering, MCP integration, safety guardrails |
| `thinkfirst` | 0 (standalone) | Interview-first intent clarification via Seven Dimensions framework |

Sub-skills include: reasoning-frameworks, reflection-framework, agent-performance-optimization, agent-evaluation, multi-agent-coordination, llm-application-patterns, mcp-integration, prompt-engineering-patterns, memory-system-patterns, safety-guardrails, tool-use-patterns, knowledge-graph-patterns, `self-improving-agents` *(new in v3.1.4)*

## Hooks (12 events)

SessionStart, SessionEnd, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStart, SubagentStop, PermissionDenied, TaskCreated, TaskCompleted, StopFailure

(PreSubagentUse, ExecutionError, PermissionPrompt, ContextOverflow, and CostThreshold handlers were removed in v3.4.0 — not supported by the CC v2.1.113 CLI event schema.)

## Integration / Workflow

Agent-core is the **coordination layer** — other suites delegate *up* to it when they need multi-agent orchestration or structured reasoning, and agent-core delegates *down* to them when a domain specialist is needed. See `docs/integration-map.rst` for the full delegation graph.

## Installation

```bash
/plugin marketplace add imewei/MyClaude
/plugin install agent-core@marketplace
```

After installation, restart Claude Code for changes to take effect.

## License

MIT License
