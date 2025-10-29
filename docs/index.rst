Plugin Marketplace Documentation
=================================

Welcome to the comprehensive documentation for the Claude Code Plugin Marketplace. This marketplace provides 31 specialized plugins for scientific computing, software development, DevOps, AI/ML, and research workflows.

Overview
--------

The Plugin Marketplace offers:

- **31 Specialized Plugins** across 9 categories
- **Comprehensive Coverage** for scientific computing, development, DevOps, and more
- **Independent Modification** without affecting source plugins
- **Git-Based Version Control** for all customizations
- **Integrated Ecosystem** with extensive cross-plugin collaboration

Categories
----------

Plugins are organized into the following categories:

Scientific Computing (2 plugins)
  High-performance computing, numerical analysis, Julia development, SciML, and data visualization

Development (7 plugins)
  Backend, frontend, systems programming, LLM applications, and code migration

AI & Machine Learning (2 plugins)
  Deep learning, machine learning pipelines, and MLOps

DevOps (2 plugins)
  CI/CD automation, Git workflows, and observability

Tools (14 plugins)
  General-purpose utilities for testing, documentation, code quality, and more

Orchestration (1 plugin)
  Full-stack workflow coordination and multi-layer application management

Quality Engineering (1 plugin)
  Comprehensive code review and security analysis

Developer Tools (1 plugin)
  Command-line tool design and automation

Development Tools (1 plugin)
  Interactive debugging and developer experience

Statistics
----------

**Total Resources:**

- 31 plugins
- 69 agents
- 40 commands
- 109 skills

**By Category:**

- Scientific Computing: 2 plugins, 5 agents, 4 commands, 24 skills
- Development: 7 plugins, 18 agents, 16 commands, 20 skills
- AI & ML: 2 plugins, 5 agents, 0 commands, 13 skills
- DevOps: 2 plugins, 5 agents, 5 commands, 6 skills
- Tools: 14 plugins, 27 agents, 15 commands, 41 skills
- Orchestration: 1 plugin, 4 agents, 1 command, 0 skills
- Quality: 1 plugin, 3 agents, 2 commands, 1 skill
- Developer Tools: 1 plugin, 1 agent, 0 commands, 2 skills
- Dev Tools: 1 plugin, 1 agent, 0 commands, 2 skills

Features
--------

**Specialized Expertise**
  Each plugin provides focused expertise in specific domains, with specialized agents, commands, and skills.

**Cross-Plugin Integration**
  Plugins are designed to work together, enabling complex multi-plugin workflows. See the :doc:`integration-map` for detailed integration patterns.

**Comprehensive Documentation**
  Every plugin includes detailed documentation covering description, usage, examples, and integration points.

**Quality Assurance**
  All plugins follow consistent metadata standards and include testing, CI/CD, and contribution guidelines.

**Versioned Documentation**
  Documentation supports versioning to reference specific plugin versions and track evolution over time.

Quick Links
-----------

Popular Plugins
~~~~~~~~~~~~~~~

- :doc:`/plugins/julia-development` - Comprehensive Julia development with SciML and Bayesian inference
- :doc:`/plugins/python-development` - Python programming with async patterns and packaging
- :doc:`/plugins/backend-development` - Backend API design and microservices
- :doc:`/plugins/deep-learning` - Neural architectures and training workflows
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
   guides/scientific-workflows
   guides/development-workflows
   guides/devops-workflows
   guides/infrastructure-workflows
   guides/integration-patterns

.. toctree::
   :maxdepth: 1
   :caption: Reference

   integration-map
   glossary
   changelog

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Claude Code installed
- Git installed
- ``jq`` installed (for metadata generation)

.. code-block:: bash

   # macOS
   brew install jq

   # Linux
   sudo apt-get install jq

Quick Setup
~~~~~~~~~~~

.. code-block:: bash

   # Clone the marketplace
   git clone https://github.com/your-org/claude-code-marketplace.git
   cd claude-code-marketplace

   # Link marketplace to Claude Code
   ln -s "$(pwd)" "$HOME/.claude/marketplace"

   # Verify installation
   claude list-plugins

Using Plugins
~~~~~~~~~~~~~

Activate plugins using Claude Code's plugin system. Each plugin provides specialized agents, commands, and skills that integrate seamlessly with your workflows.

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
