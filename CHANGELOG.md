# Changelog

## v2.2.1 (2026-02-14)

**Version Consistency**

* Bumped all 208 versioned files from 2.2.0 to 2.2.1.
* Synchronized marketplace.json, plugin.json, agents, commands, skills, and docs.
* Added `.serena/` to `.gitignore`.
* Added `team-assemble` command to marketplace agent-core entry.

## v2.2.0 (2026-02-14)

**Agent Teams & Core**

* **New Feature**: Added Agent Teams support with `team-assemble` command and guide.
* Added project-level configuration support (`.serena/project.yml`).
* Implemented new hooks system (`pre_task`, `session_start`).

**Suite Updates**

* **Science Suite**: Updated agents and skills to v2.2.0; enhanced metadata validation.
* **Quality Suite**: Comprehensive skill updates and new validation tools.
* **Engineering & Infrastructure**: Full suite updates with new skills and agents.

**Marketplace & Configuration**

* Bumped all marketplace.json versions from 2.1.0 to 2.2.0.
* Added `team-assemble` command to agent-core marketplace entry.
* Added `.serena/` to `.gitignore`.

**Maintenance**

* Updated dependencies and lockfile.
* Added context budget checker tool.

## v2.1.0 (2026-01-20)

**Suite Consolidation**

* Consolidated 31 legacy plugins into 5 powerful suites:

  - **agent-core**: Multi-agent coordination, reasoning, and LLM applications
  - **engineering-suite**: Full-stack development and platform-specific implementations
  - **infrastructure-suite**: CI/CD automation, observability, and Git workflows
  - **quality-suite**: Code quality, testing, debugging, and documentation
  - **science-suite**: Scientific computing, HPC, JAX/Julia, and data science

**Flattened Skills Architecture**

* Restructured all skills to a **flat directory structure** for reliable auto-discovery.
* Previously nested skills (e.g., `jax-mastery/jax-bayesian-pro`) are now peers
  at the same directory level (e.g., `skills/jax-bayesian-pro/`).
* Meta-skills (`jax-mastery`, `julia-mastery`) remain as aggregators linking
  to related skills.
* Science suite now contains 80 flattened skills for comprehensive coverage.

**Agent Improvements**

* Standardized agent metadata with consistent colors, versions, and examples.
* Enhanced agent system prompts with explicit activation rules.
* Aligned all agents and skills to version 2.1.0.

**Command Updates**

* Renamed `feature-dev` to `eng-feature-dev` to prevent conflicts with
  the core `feature-dev` plugin.
* Added version metadata to all commands.

**Documentation**

* Updated documentation system for the new 5-suite architecture.
* Auto-generated suite RST files from plugin.json manifests.

## v2.0.0 (2025-12-15)

* Initial release of the consolidated architecture.
