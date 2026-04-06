# Agent Core

Meta-agent suite for multi-agent orchestration, advanced reasoning, and context engineering. Coordinates all other suites but is never delegated to from below. Optimized for Claude Opus 4.6.

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
| `/team-assemble` | Generate agent team configs from 21 pre-built templates |

## Skills (3 hubs → 12 sub-skills)

Hub skills route to specialized sub-skills via decision trees:

| Hub | Sub-skills | Focus |
|-----|------------|-------|
| `agent-systems` | 4 | Orchestration, coordination, performance optimization, evaluation |
| `reasoning-and-memory` | 4 | Reasoning frameworks, reflection, memory patterns, knowledge graphs |
| `llm-engineering` | 4 | LLM app patterns, prompt engineering, MCP integration, safety guardrails |

Sub-skills include: reasoning-frameworks, reflection-framework, agent-performance-optimization, agent-evaluation, multi-agent-coordination, llm-application-patterns, mcp-integration, prompt-engineering-patterns, memory-system-patterns, safety-guardrails, tool-use-patterns, knowledge-graph-patterns

## Hooks (8 events)

SessionStart, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStop, PermissionDenied, TaskCompleted

## Installation

```bash
/plugin marketplace add imewei/MyClaude
/plugin install agent-core@marketplace
```

After installation, restart Claude Code for changes to take effect.

## License

MIT License
