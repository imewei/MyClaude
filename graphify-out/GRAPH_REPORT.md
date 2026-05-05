# Graph Report - docs  (2026-05-04)

## Corpus Check
- Corpus is ~22,944 words - fits in a single context window. You may not need a graph.

## Summary
- 188 nodes · 385 edges · 10 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Science Suite Agents|Science Suite Agents]]
- [[_COMMUNITY_Dev Suite Commands & Workflows|Dev Suite Commands & Workflows]]
- [[_COMMUNITY_Dev Suite Agents & Python Stack|Dev Suite Agents & Python Stack]]
- [[_COMMUNITY_Agent Core & Quick Reference|Agent Core & Quick Reference]]
- [[_COMMUNITY_Architecture Governance & Changelog|Architecture Governance & Changelog]]
- [[_COMMUNITY_Hub-Skill Architecture & Glossary|Hub-Skill Architecture & Glossary]]
- [[_COMMUNITY_Sphinx Plugin Directives|Sphinx Plugin Directives]]
- [[_COMMUNITY_Scientific Review & Mode Flag|Scientific Review & Mode Flag]]
- [[_COMMUNITY_Simulation HPC Sub-Skills|Simulation HPC Sub-Skills]]

## God Nodes (most connected - your core abstractions)
1. `Science Suite` - 30 edges
2. `Agent Reference` - 25 edges
3. `Dev Suite` - 23 edges
4. `Scientific Workflows Guide` - 19 edges
5. `Agent Teams Guide` - 17 edges
6. `Command Reference` - 17 edges
7. `DevOps Workflows Guide` - 15 edges
8. `Agent Core Suite` - 13 edges
9. `Research Suite` - 13 edges
10. `Model Tier: sonnet` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Research-Spark 8-Stage Pipeline` --semantically_similar_to--> `Long-Running Workflow Protocol (PROGRESS.md + incremental git commits)`  [INFERRED] [semantically similar]
  docs/suites/research-suite.rst → docs/agent-teams-guide.md
- `Hub: llm-and-ai (science-suite)` --semantically_similar_to--> `Hub: llm-engineering (agent-core)`  [INFERRED] [semantically similar]
  docs/suites/science-suite.rst → docs/suites/agent-core.rst
- `Hub: research-and-domains (science-suite)` --semantically_similar_to--> `Hub: research-practice (research-suite)`  [INFERRED] [semantically similar]
  docs/suites/science-suite.rst → docs/suites/research-suite.rst
- `Hub-Skill Two-Tier Architecture` --semantically_similar_to--> `scientific-review --mode flag design pattern`  [INFERRED] [semantically similar]
  docs/guides/devops-workflows.rst → docs/superpowers/specs/2026-04-30-scientific-review-mode-flag-design.md
- `Dev Category Page` --references--> `Dev Suite`  [EXTRACTED]
  docs/categories/dev.rst → docs/suites/dev-suite.rst

## Hyperedges (group relationships)
- **Research-Spark Orchestrator Cross-Suite Fan-Out: orchestrator delegates Stage 6 JAX/Julia/MD to science-suite agents** — agent_research_spark_orchestrator, agent_jax_pro, agent_julia_pro, agent_simulation_expert [EXTRACTED 1.00]
- **Hub-Skill Routing Pattern: plugin.json → hub SKILL.md → routing tree → sub-skill** — concept_hub_skill, concept_routing_tree, concept_sub_skill [EXTRACTED 1.00]
- **sci-compute team covers all science agents across 7 variants (bayesian, julia-sciml, julia-ml, dynamics, md-sim, desktop, reproduce)** — team_sci_compute, suite_science_suite, concept_variant_system [EXTRACTED 1.00]
- **CI/CD Pipeline Setup: automation-engineer + ci-cd-pipelines hub + workflow-automate command** — agent_automation_engineer, hub_ci_cd_pipelines, cmd_workflow_automate, cicd_pipeline_workflow [EXTRACTED 1.00]
- **Bayesian Inference Pipeline: jax-pro + bayesian-inference hub + jax-computing hub** — agent_jax_pro, hub_bayesian_inference, hub_jax_computing, bayesian_pipeline_workflow [EXTRACTED 1.00]
- **scientific-review --mode flag: SKILL.md + review_structure.md + integrity_checks.md conditionally loaded** — scientific_review_skill_md, scientific_review_review_structure_md, scientific_review_integrity_checks_md, scientific_review_mode_flag [EXTRACTED 1.00]

## Communities (10 total, 0 thin omitted)

### Community 0 - "Science Suite Agents"
Cohesion: 0.09
Nodes (44): Agent: ai-engineer (science-suite), Agent: jax-pro (science-suite), Agent: julia-ml-hpc (science-suite), Agent: julia-pro (science-suite), Agent: ml-expert (science-suite), Agent: neural-network-master (science-suite, opus), Agent: nonlinear-dynamics-expert (science-suite, opus), Agent: prompt-engineer (science-suite) (+36 more)

### Community 1 - "Dev Suite Commands & Workflows"
Cohesion: 0.07
Nodes (38): CI/CD Pipeline Setup Workflow, Command: /commit, Command: /docs, Command: /double-check, Command: /eng-feature-dev, Command: /fix-commit-errors, Command: /merge-all, Command: /modernize (+30 more)

### Community 2 - "Dev Suite Agents & Python Stack"
Cohesion: 0.13
Nodes (32): Agent: app-developer (dev-suite), Agent: automation-engineer (dev-suite), Agent: debugger-pro (dev-suite), Agent: devops-architect (dev-suite), Agent: documentation-expert (dev-suite, haiku), Agent: python-pro (science-suite), Agent: quality-specialist (dev-suite), Agent: software-architect (dev-suite) (+24 more)

### Community 3 - "Agent Core & Quick Reference"
Cohesion: 0.12
Nodes (27): Agent: context-specialist (agent-core), Agent Core Suite, Agent: orchestrator (agent-core), Agent: reasoning-engine (agent-core), Quick Reference Cheatsheet, Command: /ultra-think, Core Suite Category, Cross-Suite Agent Teams Pattern (+19 more)

### Community 4 - "Architecture Governance & Changelog"
Cohesion: 0.13
Nodes (18): Three Adversarial Patterns (Reviewer 2, Stepwise Derivation, Instrument Margin), 2% Context Budget Rule per Skill, Research-Spark 8-Stage Pipeline, v3.4.0 Research-Suite Extraction from Science-Suite: research-expert + 5 methodology skills split out so science-suite stays purely computational, Skill Size Governance (>3000 bytes = review required, >80% = at-risk, >90% = refactor), Style Enforcement via style_lint.py (no em dashes, no banned vocabulary, quantified language), Version Single Source of Truth in plugin.json, Changelog (+10 more)

### Community 5 - "Hub-Skill Architecture & Glossary"
Cohesion: 0.31
Nodes (9): Command: /team-assemble, Agent Team Configuration System, Hub Skill Architecture, Hub-Skill Routing: plugin.json declares hubs only; sub-skills discovered via routing trees using ../ relative links — eliminates flat-list ambiguity, Routing Decision Tree, Sub-Skill, Team Variant System (--var MODE=x), Glossary (+1 more)

### Community 6 - "Sphinx Plugin Directives"
Cohesion: 0.36
Nodes (5): AgentDirective, BasePluginDirective, CommandDirective, SkillDirective, SphinxDirective

### Community 7 - "Scientific Review & Mode Flag"
Cohesion: 0.57
Nodes (7): scientific-review references/integrity_checks.md, scientific-review --mode flag design pattern, Scientific Review Mode Flag Design Spec, Scientific Review Mode Flag Implementation Plan, scientific-review references/review_structure.md, scientific-review SKILL.md (file), Skill: scientific-review

### Community 8 - "Simulation HPC Sub-Skills"
Cohesion: 0.5
Nodes (4): Hub Skill: simulation-and-hpc, Sub-Skill: advanced-simulations, Sub-Skill: md-simulation-setup, Sub-Skill: trajectory-analysis

## Knowledge Gaps
- **51 isolated node(s):** `Quality Gate Enhancers (append official plugin agents to any team)`, `Three Adversarial Patterns (Reviewer 2, Stepwise Derivation, Instrument Margin)`, `Hub: agent-systems (agent-core)`, `Hub: thinkfirst (agent-core, sub-skill of llm-engineering)`, `Hub: frontend-and-mobile (dev-suite)` (+46 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Science Suite` connect `Science Suite Agents` to `Dev Suite Agents & Python Stack`, `Agent Core & Quick Reference`, `Architecture Governance & Changelog`?**
  _High betweenness centrality (0.204) - this node is a cross-community bridge._
- **Why does `Dev Suite` connect `Dev Suite Agents & Python Stack` to `Dev Suite Commands & Workflows`, `Agent Core & Quick Reference`, `Architecture Governance & Changelog`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `Scientific Workflows Guide` connect `Science Suite Agents` to `Simulation HPC Sub-Skills`, `Scientific Review & Mode Flag`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **What connects `Quality Gate Enhancers (append official plugin agents to any team)`, `Three Adversarial Patterns (Reviewer 2, Stepwise Derivation, Instrument Margin)`, `Hub: agent-systems (agent-core)` to the rest of the system?**
  _51 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Science Suite Agents` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._
- **Should `Dev Suite Commands & Workflows` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._
- **Should `Dev Suite Agents & Python Stack` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._