# Agent Core

Meta-agent suite for multi-agent orchestration, advanced reasoning, and context engineering. Coordinates all other suites but is never delegated to from below. Optimized for Claude Opus 4.6.

## Overview

Agent Core is the coordination layer of MyClaude. Its 3 agents (orchestrator, reasoning-engine, context-specialist) manage multi-agent workflows, decompose complex problems, and maintain session context. It provides 3 hub skills routing to 14 sub-skills covering orchestration patterns, reasoning frameworks, and LLM engineering. Use agent-core when you need to coordinate multiple specialists or tackle problems requiring structured reasoning.

## Quick Start

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
| `context-specialist` | sonnet | Context engineering, memory systems, knowledge graphs |

## Commands

| Command | Description |
|---------|-------------|
| `/agent-build` | Unified AI agent creation, optimization, and prompt engineering |
| `/ai-assistant` | Build production-ready AI assistants with NLU and response generation |
| `/docs-lookup` | Query library documentation via Context7 MCP |
| `/reflection` | Meta-cognitive reflection framework execution |
| `/ultra-think` | Advanced structured reasoning with branching exploration |
| `/team-assemble` | Codebase-aware recommender with 10 focused teams (20 variants) + long-running workflow protocol (v3.3.0) |

## Skills (3 hubs → 14 sub-skills)

Hub skills route to specialized sub-skills via decision trees:

| Hub | Sub-skills | Focus |
|-----|------------|-------|
| `agent-systems` | 4 | Orchestration, coordination, performance optimization, evaluation |
| `reasoning-and-memory` | 5 | Reasoning frameworks, reflection, memory patterns, knowledge graphs, self-improving agents |
| `llm-engineering` | 5 | Intent clarification (thinkfirst), LLM app patterns, prompt engineering, MCP integration, safety guardrails |

Sub-skills include: reasoning-frameworks, reflection-framework, agent-performance-optimization, agent-evaluation, multi-agent-coordination, llm-application-patterns, mcp-integration, prompt-engineering-patterns, memory-system-patterns, safety-guardrails, tool-use-patterns, knowledge-graph-patterns, `thinkfirst` *(new in v3.1.3)*, `self-improving-agents` *(new in v3.1.4)*

## Hooks (15 events)

SessionStart, SessionEnd, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStart, SubagentStop, PermissionDenied, TaskCreated, TaskCompleted, StopFailure, PreSubagentUse, ExecutionError, PermissionPrompt

## Installation

```bash
/plugin marketplace add imewei/MyClaude
/plugin install agent-core@marketplace
```

After installation, restart Claude Code for changes to take effect.

## License

MIT License
