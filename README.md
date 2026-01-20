# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-5-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-22-green.svg)](docs/reference/agents.md)
[![Commands](https://img.shields.io/badge/Commands-32-orange.svg)](docs/reference/commands.md)
[![Version](https://img.shields.io/badge/Version-2.1.0-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **5 consolidated suites**, **22 expert agents**, and **32 slash commands** optimized for AI-powered development, scientific computing, and research workflows.

## 🚀 The 5-Suite System

The MyClaude ecosystem has been consolidated into five powerful suites:

1.  **[Agent Core Suite](plugins/agent-core/)**: Multi-agent coordination, advanced reasoning, and context engineering.
2.  **[Software Engineering Suite](plugins/engineering-suite/)**: Full-stack development, systems programming, and legacy modernization.
3.  **[Infrastructure & Ops Suite](plugins/infrastructure-suite/)**: CI/CD automation, observability, and Git workflows.
4.  **[Quality & Maintenance Suite](plugins/quality-suite/)**: Code quality, test automation, and intelligent debugging.
5.  **[Scientific Computing Suite](plugins/science-suite/)**: HPC, JAX/Julia mastery, and specialized physics simulations.

## 🤖 Specialist Agents

The system features 22 specialized agents across all suites, including:

| Agent | Suite | Specialization |
|-------|-------|----------------|
| `@orchestrator` | Agent Core | Multi-agent coordination and task delegation |
| `@software-architect` | Engineering | Backend systems, microservices, and API design |
| `@jax-pro` | Science | Core JAX, Bayesian inference, and physics apps |
| `@quality-specialist` | Quality | Code review, security auditing, and test automation |
| `@devops-architect` | Infrastructure | Cloud infrastructure, Kubernetes, and IaC |
| `@python-pro` | Science | Modern Python systems engineering and performance |
| `@reasoning-engine` | Agent Core | Advanced reasoning and structured thinking |
| `@sre-expert` | Infrastructure | Reliability, observability, and incident response |
| `@app-developer` | Engineering | Web, iOS, and Android development (React, Flutter) |

## 📦 Installation

### Step 1: Add the Marketplace

In Claude Code, add this marketplace:

```bash
/plugin marketplace add imewei/MyClaude
```

### Step 2: Install Suites

```bash
# Install specific suites
/plugin install agent-core@marketplace
/plugin install engineering-suite@marketplace
/plugin install infrastructure-suite@marketplace
/plugin install quality-suite@marketplace
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
/quality-suite:double-check my-feature
/infrastructure-suite:fix-commit-errors
```

## Documentation

- **[Full Documentation](https://myclaude.readthedocs.io/en/latest/)**
- **[Plugin Cheatsheet](docs/reference/cheatsheet.md)**
- **[Complete Agents List](docs/reference/agents.md)**
- **[Complete Commands List](docs/reference/commands.md)**

## License

MIT License (see [LICENSE](LICENSE))

---

**Built by Wei Chen** | [Documentation](https://myclaude.readthedocs.io/en/latest/) | [GitHub](https://github.com/imewei/MyClaude)
