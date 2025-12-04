Plugin Marketplace Documentation
=================================

Welcome to the comprehensive documentation for the Claude Code Plugin Marketplace. This marketplace provides 31 specialized plugins for scientific computing, software development, DevOps, AI/ML, and research workflows.

Overview
--------

The Plugin Marketplace offers:

- **31 Specialized Plugins** across 6 major categories
- **74 Expert Agents** for AI-powered development assistance
- **48 Slash Commands** for automated workflows
- **119 Skills** for context-aware intelligence
- **16 Tools** for validation and profiling
- **Integrated Ecosystem** with extensive cross-plugin collaboration

Categories
----------

Plugins are organized into the following categories:

Scientific Computing (8 plugins)
  Julia development, JAX implementation, HPC computing, molecular simulation, statistical physics, deep learning, data visualization, and research methodology

Development (10 plugins)
  Python, JavaScript/TypeScript, backend, frontend/mobile, systems programming, multi-platform apps, LLM applications, CLI tools, full-stack orchestration, and agent orchestration

AI & Machine Learning (2 plugins)
  Machine learning pipelines and AI reasoning frameworks

DevOps & Infrastructure (3 plugins)
  CI/CD automation, Git/PR workflows, and observability monitoring

Quality & Testing (4 plugins)
  Unit testing, comprehensive review, codebase cleanup, and quality engineering

Tools & Migration (4 plugins)
  Code documentation, code migration, framework migration, and debugging toolkit

Statistics
----------

**Version 1.0.4** (All plugins updated December 3, 2025)

**Total Resources:**

- 31 plugins
- 74 agents
- 48 commands
- 119 skills
- 16 tools

**By Category:**

- Scientific Computing: 8 plugins, 18 agents, 4 commands, 54 skills
- Development: 10 plugins, 24 agents, 14 commands, 30 skills
- AI & Machine Learning: 2 plugins, 6 agents, 3 commands, 10 skills
- DevOps & Infrastructure: 3 plugins, 10 agents, 8 commands, 12 skills
- Quality & Testing: 4 plugins, 7 agents, 10 commands, 3 skills
- Tools & Migration: 4 plugins, 9 agents, 9 commands, 7 skills

Features
--------

**Specialized Expertise**
  Each plugin provides focused expertise in specific domains, with specialized agents, commands, and skills.

**v1.0.4 Agent Enhancements**
  All 74 agents follow the nlsq-pro template with Pre-Response Validation Framework, When to Invoke sections, and Constitutional AI principles.

**Cross-Plugin Integration**
  Plugins are designed to work together, enabling complex multi-plugin workflows. See the :doc:`integration-map` for detailed integration patterns.

**Comprehensive Documentation**
  Every plugin includes detailed documentation covering description, usage, examples, and integration points.

**Quality Assurance**
  All plugins follow consistent metadata standards and include testing, CI/CD, and contribution guidelines.

Quick Links
-----------

Popular Plugins
~~~~~~~~~~~~~~~

- :doc:`/plugins/julia-development` - Comprehensive Julia development with SciML and Bayesian inference
- :doc:`/plugins/python-development` - Python programming with async patterns and packaging
- :doc:`/plugins/backend-development` - Backend API design and microservices
- :doc:`/plugins/jax-implementation` - JAX for numerical computing and optimization
- :doc:`/plugins/cicd-automation` - CI/CD pipelines and deployment automation

Getting Started
~~~~~~~~~~~~~~~

- :doc:`guides/scientific-workflows` - Scientific computing workflows
- :doc:`guides/development-workflows` - Development workflows
- :doc:`guides/devops-workflows` - DevOps workflows
- :doc:`guides/integration-patterns` - Integration patterns and best practices

Reference
~~~~~~~~~

- :doc:`integration-map` - Plugin integration matrix
- :doc:`tools-reference` - 16 utility scripts and tools
- :doc:`glossary` - Technical terminology reference
- :doc:`changelog` - Version history and updates

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Categories

   categories/scientific-computing
   categories/development
   categories/ai-ml
   categories/devops
   categories/tools
   categories/orchestration
   categories/quality
   categories/developer-tools
   categories/dev-tools
   categories/uncategorized

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   integration-map
   tools-reference
   glossary
   changelog

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Claude Code installed
- Git installed
- Python 3.12+ (for tools)

Quick Setup
~~~~~~~~~~~

**Option 1: Add Marketplace and Browse**

.. code-block:: bash

   /plugin marketplace add imewei/MyClaude
   # Select "Browse and install plugins" -> "scientific-computing-workflows" -> Select plugin

**Option 2: Install Specific Plugins**

.. code-block:: bash

   /plugin install python-development@scientific-computing-workflows
   /plugin install backend-development@scientific-computing-workflows
   /plugin install julia-development@scientific-computing-workflows

**Option 3: Install All 31 Plugins**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/imewei/MyClaude.git
   cd MyClaude

   # Enable all plugins
   make plugin-enable-all

   # Or use the Python script directly
   python3 tools/enable-all-plugins.py

**Note:** After installation, restart Claude Code for changes to take effect.

Using Plugins
~~~~~~~~~~~~~

Once installed, plugins provide agents, commands, and skills that are automatically available:

**Using Specialized Agents**

.. code-block:: text

   Ask Claude: "@python-pro help me optimize this async function"
   Ask Claude: "@julia-pro implement this differential equation using SciML"
   Ask Claude: "@jax-pro optimize this neural network training loop"

**Running Commands**

.. code-block:: bash

   /ai-reasoning:ultra-think "Analyze the architecture of this system"
   /quality-engineering:double-check --mode=standard
   /unit-testing:run-all-tests --fix --coverage

Contributing
------------

We welcome contributions! See the contribution guidelines for:

- Adding new plugins
- Improving existing plugins
- Reporting issues
- Documentation improvements

License
-------

This marketplace is licensed under the MIT License. Individual plugins may have their own licenses - see each plugin's documentation for details.

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`search`
