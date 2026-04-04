Agent Core Suite
================

Core orchestration, advanced reasoning, and context engineering. Optimized for Claude Opus 4.6 with adaptive thinking and Agent Teams support.

**Version:** 3.1.0 | **3 Agents** | **2 Registered Commands** | **3 Hubs → 12 Sub-skills** | **8 Hook Events**

Agents
------

.. agent:: orchestrator
   :description: Multi-agent orchestrator specializing in workflow coordination, agent team assembly, and task allocation.
   :model: opus
   :version: 3.1.0

.. agent:: reasoning-engine
   :description: Expert in advanced reasoning, prompt design, and cognitive tasks. Masters Chain-of-Thought and structured frameworks.
   :model: opus
   :version: 3.1.0

.. agent:: context-specialist
   :description: Elite AI context engineering specialist mastering dynamic context management, vector databases, and memory systems.
   :model: sonnet
   :version: 3.1.0

Registered Commands
-------------------

.. command:: /ultra-think
   :description: Comprehensive analysis with full reasoning framework execution.

.. command:: /team-assemble
   :description: Generate ready-to-use agent team configurations from pre-built templates.

Skill-Invoked Commands
----------------------

These commands are triggered by skills, not directly by users:

.. command:: agent-build
   :description: Unified AI agent creation, optimization, and prompt engineering.

.. command:: ai-assistant
   :description: Build production-ready AI assistants with NLU and intelligent response generation.

.. command:: docs-lookup
   :description: Query library documentation using Context7 MCP for up-to-date API references.

.. command:: reflection
   :description: AI reasoning analysis, session retrospectives, and research optimization.

Hub Skills
----------

Skills use a hub architecture: 3 hub skills route to 12 specialized sub-skills.

Hub: agent-systems
^^^^^^^^^^^^^^^^^^

Multi-agent coordination, performance optimization, evaluation, and tool use patterns.

- ``agent-evaluation`` — Evaluate AI agent performance through systematic testing and benchmarking
- ``agent-performance-optimization`` — Monitor, cache, and load-balance agent systems for production
- ``multi-agent-coordination`` — Workflow orchestration, task allocation, and inter-agent communication
- ``tool-use-patterns`` — Tool selection, chaining, error handling, and result synthesis

Hub: reasoning-and-memory
^^^^^^^^^^^^^^^^^^^^^^^^^

Reasoning frameworks, reflection, knowledge graphs, and memory systems.

- ``reasoning-frameworks`` — First Principles, RCA, Decision Analysis, Systems Thinking, OODA Loop
- ``reflection-framework`` — Meta-cognitive analysis, bias detection, and session reflection
- ``knowledge-graph-patterns`` — Knowledge graphs for structured retrieval and semantic reasoning
- ``memory-system-patterns`` — Persistent memory systems with vector stores and context management

Hub: llm-engineering
^^^^^^^^^^^^^^^^^^^^

Prompt engineering, LLM application patterns, MCP integration, and safety.

- ``llm-application-patterns`` — Prompt engineering principles (CoT, few-shot), RAG design, evaluation
- ``mcp-integration`` — MCP server configuration, tool naming conventions, and cross-tool workflows
- ``prompt-engineering-patterns`` — Advanced prompting with chain-of-thought and production templates
- ``safety-guardrails`` — Content filtering, output validation, and responsible AI practices

Hooks
-----

8 hook events with Python script implementations:

- ``SessionStart`` — Session initialization
- ``PreToolUse`` — Before tool execution
- ``PostToolUse`` — After tool execution
- ``PreCompact`` — Before context compaction
- ``PostCompact`` — After context compaction
- ``SubagentStop`` — When a subagent completes
- ``PermissionDenied`` — When a tool call is denied
- ``TaskCompleted`` — When a task finishes
