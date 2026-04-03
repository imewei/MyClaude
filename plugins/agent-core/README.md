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
| `/team-assemble` | Generate agent team configs from 38 pre-built templates |

## Skills (13)

- **Reasoning Frameworks**: Unified advanced reasoning and structured thinking
- **Reflection Framework**: Meta-cognitive and comprehensive reflection
- **Agent Orchestration**: Workflow coordination, DAG-based tasks, and team management
- **Agent Performance Optimization**: Monitoring, metrics, caching, and load balancing
- **Agent Evaluation**: Benchmark design, metrics collection, A/B testing, quality scoring
- **LLM Application Patterns**: Prompt engineering (CoT, few-shot), RAG, and evaluation
- **MCP Integration**: Guide for serena, github, sequential-thinking, and context7
- **Multi-Agent Coordination**: Inter-agent communication and task allocation
- **Prompt Engineering Patterns**: Chain-of-thought, few-shot learning, prompt versioning
- **Memory System Patterns**: Vector databases, semantic memory, retrieval strategies
- **Safety Guardrails**: Content filtering, constitutional AI, jailbreak detection
- **Tool Use Patterns**: Function calling, tool selection, chaining, error recovery
- **Knowledge Graph Patterns**: Entity modeling, graph databases, semantic search

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
