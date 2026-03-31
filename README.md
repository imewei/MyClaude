# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-3-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-22-green.svg)](docs/reference/agents.md)
[![Commands](https://img.shields.io/badge/Commands-33-orange.svg)](docs/reference/commands.md)
[![Skills](https://img.shields.io/badge/Skills-124-purple.svg)](docs/reference/cheatsheet.md)
[![Version](https://img.shields.io/badge/Version-2.2.1-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **3 focused suites**, **22 expert agents**, **33 slash commands**, and **124 skills** optimized for AI-powered development, scientific computing, and research workflows. Built for Claude Opus 4.6 with tiered model assignments (Opus/Sonnet/Haiku), 10 lifecycle hooks, and full v2.1.88 spec compliance.

## The 3-Suite Architecture

The MyClaude ecosystem is organized into three suites by delegation topology:

1.  **[Agent Core](plugins/agent-core/)**: Meta-agents for orchestration, reasoning, and context engineering. Coordinates everything but is never delegated to from below.
2.  **[Dev Suite](plugins/dev-suite/)**: The complete software development lifecycle — architecture, implementation, CI/CD, testing, debugging, and deployment. 9 agents that freely delegate to each other without cross-suite overhead.
3.  **[Science Suite](plugins/science-suite/)**: Domain-specific scientific computing — JAX, Julia, ML, physics, and research. Agents primarily collaborate within the suite.

## Specialist Agents

The system features 22 specialized agents across all suites, including:

| Agent | Suite | Specialization |
|-------|-------|----------------|
| `@orchestrator` | Agent Core | Multi-agent coordination and task delegation |
| `@software-architect` | Dev | Backend systems, microservices, and API design |
| `@debugger-pro` | Dev | Root cause analysis, log correlation, memory profiling |
| `@devops-architect` | Dev | Cloud infrastructure, Kubernetes, and IaC |
| `@quality-specialist` | Dev | Code review, security auditing, and test automation |
| `@jax-pro` | Science | Core JAX, Bayesian inference, and physics apps |
| `@python-pro` | Science | Modern Python systems engineering and performance |
| `@reasoning-engine` | Agent Core | Advanced reasoning and structured thinking |
| `@app-developer` | Dev | Web, iOS, and Android development (React, Flutter) |

## Installation

### Step 1: Add the Marketplace

In Claude Code, add this marketplace:

```bash
/plugin marketplace add imewei/MyClaude
```

### Step 2: Install Suites

```bash
# Install all suites
/plugin install agent-core@marketplace
/plugin install dev-suite@marketplace
/plugin install science-suite@marketplace
```

**Note:** After installation, restart Claude Code for changes to take effect.

## Quick Start

Once installed, suites provide agents, commands, and skills that are automatically available:

**Using Specialized Agents**
```
Ask Claude: "@python-pro help me optimize this async function"
Ask Claude: "@orchestrator coordinate a team for this new feature"
Ask Claude: "@jax-pro implement this differentiable physics model"
```

**Running Commands**
```bash
/agent-core:ultra-think "Analyze the architecture of this system"
/dev-suite:double-check my-feature
/dev-suite:fix-commit-errors
```

## Documentation

- **[Full Documentation](https://myclaude.readthedocs.io/en/latest/)**
- **[Plugin Cheatsheet](docs/reference/cheatsheet.md)**
- **[Complete Agents List](docs/reference/agents.md)**
- **[Complete Commands List](docs/reference/commands.md)**
- **[Agent Teams Guide](docs/agent-teams-guide.md)** — 38 ready-to-use team configurations

## License

MIT License (see [LICENSE](LICENSE))

---

**Built by Wei Chen** | [Documentation](https://myclaude.readthedocs.io/en/latest/) | [GitHub](https://github.com/imewei/MyClaude)
