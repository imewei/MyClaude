# Plugin Consolidation Strategy

This document outlines the strategy for consolidating 31 fragmented plugins into 5 cohesive "Mega-Plugins" to reduce redundancy, simplify maintenance, and improve the developer experience.

## 1. Proposed Consolidation Architecture

The following "Mega-Plugins" will serve as the core suites for the ecosystem:

### 1.1 Infrastructure & Ops (`infrastructure-suite`)
Combines operational and automation workflows.
- **Source Plugins**: `cicd-automation`, `observability-monitoring`, `git-pr-workflows`.
- **Focus**: Deployment pipelines, monitoring, log analysis, and Git lifecycle management.

### 1.2 Software Engineering (`engineering-suite`)
Combines application development across languages and platforms.
- **Source Plugins**: `backend-development`, `frontend-mobile-development`, `multi-platform-apps`, `python-development`, `javascript-typescript`, `systems-programming`, `cli-tool-design`, `framework-migration`.
- **Focus**: Full-stack development, language-specific best practices, and platform-specific implementations.

### 1.3 Scientific Computing (`science-suite`)
Combines high-performance and scientific research tools.
- **Source Plugins**: `hpc-computing`, `jax-implementation`, `molecular-simulation`, `statistical-physics`, `deep-learning`, `machine-learning`, `data-visualization`, `research-methodology`, `julia-development`.
- **Focus**: Numerical methods, specialized physics/chemistry simulations, and data science workflows.

### 1.4 Quality & Maintenance (`quality-suite`)
Combines testing, review, and documentation tools.
- **Source Plugins**: `quality-engineering`, `unit-testing`, `codebase-cleanup`, `code-documentation`, `code-migration`, `comprehensive-review`, `debugging-toolkit`.
- **Focus**: Code quality, test automation, legacy modernization, and debugging.

### 1.5 Agent Core (`agent-core`)
Combines internal agent orchestration and reasoning capabilities.
- **Source Plugins**: `agent-orchestration`, `ai-reasoning`, `full-stack-orchestration`, `llm-application-dev`.
- **Focus**: Multi-agent coordination, deep reasoning, and specialized LLM application development.

## 2. Migration Path

### 2.1 Directory Structure
Each Mega-Plugin will follow the standard structure but organized by domain:
```
plugins/<mega-plugin>/
├── agents/            # Deduplicated agents
├── commands/          # Unified slash commands
├── skills/            # Consolidated skill sets
├── docs/              # Merged documentation
├── plugin.json        # Unified manifest
└── README.md          # Comprehensive overview
```

### 2.2 Manifest Merging Strategy
- **Name/ID**: New suite-level ID (e.g., `engineering-suite`).
- **Keywords**: Union of all source keywords, deduplicated.
- **Categories**: Primary category for the suite.
- **Dependencies**: Consolidated and version-aligned.

### 2.3 Agent Deduplication & Merging
To resolve the redundancies identified in the analysis:

| Target Agent | Merged From | Strategy |
| :--- | :--- | :--- |
| `software-architect` | `backend-architect`, `systems-architect`, `ai-systems-architect` | Create a generalist architect with domain-specific sub-instructions. |
| `quality-specialist` | `code-reviewer`, `test-automator`, `security-auditor` | Consolidate into a single quality agent with multi-modal capabilities. |
| `devops-engineer` | `deployment-engineer`, `performance-engineer` | Merge CI/CD and observability focus into a single operational agent. |
| `app-developer` | `mobile-developer`, `frontend-developer` | Single agent capable of multi-platform UI/UX. |
| `research-expert` | `scientific-data-expert`, `physics-expert`, `jax-pro` | Unified scientific agent with specialized skills. |

### 2.4 Implementation Phases
1.  **Phase 1: Skeleton Creation**: Create the new suite directories and basic `plugin.json`.
2.  **Phase 2: Skill Consolidation**: Move and deduplicate skills (e.g., merging duplicate GPU skills).
3.  **Phase 3: Agent Unification**: Draft the new unified agent prompts based on source agents.
4.  **Phase 4: Command Porting**: Migrate slash commands to the new suites.
5.  **Phase 5: Deprecation**: Mark old plugins as deprecated and provide migration paths for users.

## 3. Benefits of Consolidation
- **Reduced Binary Size**: Fewer redundant agents and prompt files.
- **Clearer Discovery**: Users find tools by domain rather than specific implementation details.
- **Easier Maintenance**: Single point of update for shared logic (e.g., updating a code reviewer).
- **Improved Performance**: Smaller index and faster discovery for Claude Code.
