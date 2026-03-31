Changelog
=========

v2.3.0 (2026-03-31)
-------------------

**v2.1.88 Spec Compliance Upgrade**

* Migrated all 5 plugin manifests to ``.claude-plugin/plugin.json`` per official spec.
* Consolidated 131 skills to 124 with zero function loss (7 semantic merges).
* Hardened all 22 agents with ``effort``, ``memory``, and ``tools`` fields.
* Removed non-spec ``version``/``color`` fields from all agent and command frontmatter.
* Added explicit ``name`` field to 27 commands that were missing it.

**Model Tier Optimization**

* Assigned Opus to 6 deep-reasoning agents (orchestrator, reasoning-engine,
  software-architect, debugger-pro, research-expert, statistical-physicist).
* Assigned Haiku to documentation-expert for speed-optimized docs generation.
* Fixed neural-network-master from ``inherit`` to explicit ``sonnet``.

**Hook Expansion**

* Expanded hook events from 3 to 8: added PostToolUse, PostCompact,
  SubagentStop, PermissionDenied, TaskCompleted.
* PostToolUse fires on Write/Edit for Python lint suggestions.

**New Infrastructure**

* Added ``output-styles/`` directory to agent-core (terse and verbose modes).
* Added ``settings.json`` with default agent configuration to all 5 suites.
* Updated metadata validator schema from v2.1.42 to v2.1.88.

**Skill Consolidations (7 merges)**

* advanced-reasoning + structured-reasoning → reasoning-frameworks
* meta-cognitive-reflection + comprehensive-reflection-framework → reflection-framework
* ai-assisted-debugging + debugging-strategies → debugging-toolkit
* comprehensive-validation-framework merged into comprehensive-validation
* machine-learning-essentials absorbed into machine-learning
* parallel-computing-strategy absorbed into parallel-computing
* python-testing-patterns + javascript-testing-patterns → testing-patterns

v2.2.1 (2026-02-14)
-------------------

**Claude Opus 4.6 Compatibility**

* Upgraded all 5 plugin suites to v2.2.1 for Claude Opus 4.6 compatibility.
* Updated all 22 agents and 131 skills to version 2.2.1.

**Agent Teams System**

* New ``/team-assemble`` command with 38 pre-built team configurations.
* Teams span Development & Operations (1-10), Scientific Computing (11-16),
  Cross-Suite Specialized (17-25), Official Plugin Integration (26-33), and Debugging (34-38).
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
