Changelog
=========

v2.2.0 (2026-02-14)
-------------------

**Claude Opus 4.6 Compatibility**

* Upgraded all 5 plugin suites to v2.2.0 for Claude Opus 4.6 compatibility.
* Updated all 22 agents and 131 skills to version 2.2.0.

**Agent Teams System**

* New ``/team-assemble`` command with 33 pre-built team configurations.
* Teams span Development & Operations (1-10), Scientific Computing (11-16),
  Cross-Suite Specialized (17-25), and Official Plugin Integration (26-33).
* Each team provides a ready-to-paste prompt with role assignments, file
  ownership, and workflow ordering.
* Integrated 20 official plugin agents (pr-review-toolkit, feature-dev,
  coderabbit, plugin-dev, hookify, huggingface-skills, agent-sdk-dev, superpowers).
* Quality Gate Enhancers section for adding review agents to any team.
* Comprehensive reference guide at ``docs/agent-teams-guide.md``.

**Agent Enhancements**

* Added adaptive thinking references to reasoning-engine agent.
* Integrated Agent Teams coordination into orchestrator agent.
* Added ``memory`` frontmatter to 11 key agents for persistent context.

**Hooks Infrastructure**

* Added hooks support to agent-core suite (``SessionStart``, ``PreToolUse``).
* New ``hooks/hooks.json`` configuration in agent-core plugin manifest.

**Tooling**

* Added context budget checker tool (``tools/validation/context_budget_checker.py``).
* Refined descriptions across all 80 science-suite skills.

v2.1.0 (2026-01-20)
-------------------

**Suite Consolidation**

* Consolidated 31 legacy plugins into 5 powerful suites:

  - **agent-core**: Multi-agent coordination, reasoning, and LLM applications
  - **engineering-suite**: Full-stack development and platform-specific implementations
  - **infrastructure-suite**: CI/CD automation, observability, and Git workflows
  - **quality-suite**: Code quality, testing, debugging, and documentation
  - **science-suite**: Scientific computing, HPC, JAX/Julia, and data science

**Flattened Skills Architecture**

* Restructured all skills to a **flat directory structure** for reliable auto-discovery.
* Previously nested skills (e.g., ``jax-mastery/jax-bayesian-pro``) are now peers
  at the same directory level (e.g., ``skills/jax-bayesian-pro/``).
* Meta-skills (``jax-mastery``, ``julia-mastery``) remain as aggregators linking
  to related skills.
* Science suite now contains 80 flattened skills for comprehensive coverage.

**Agent Improvements**

* Standardized agent metadata with consistent colors, versions, and examples.
* Enhanced agent system prompts with explicit activation rules.
* Aligned all agents and skills to version 2.1.0.

**Command Updates**

* Renamed ``feature-dev`` to ``eng-feature-dev`` to prevent conflicts with
  the core ``feature-dev`` plugin.
* Added version metadata to all commands.

**Documentation**

* Updated documentation system for the new 5-suite architecture.
* Auto-generated suite RST files from plugin.json manifests.

v2.0.0 (2025-12-15)
-------------------
* Initial release of the consolidated architecture.
