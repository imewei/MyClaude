Plugin Marketplace Documentation
=================================

Welcome to the comprehensive documentation for the Claude Code Plugin Marketplace. This marketplace provides 5 consolidated suites for scientific computing, software engineering, infrastructure, quality, and agent orchestration.

Overview
--------

The Plugin Marketplace offers:

- **5 Specialized Suites** consolidated from 31 legacy plugins
- **22 Core Agents** for AI-powered development assistance
- **Extensive Slash Commands** for automated workflows
- **Deep Skill Integration** for context-aware intelligence
- **Integrated Ecosystem** with extensive cross-suite collaboration

Suites
------

The marketplace is organized into the following suites:

Agent Core
  Consolidated suite for multi-agent coordination, deep reasoning, and specialized LLM application development.

Software Engineering
  Consolidated suite for full-stack engineering, language-specific development, and platform-specific implementations.

Infrastructure & Ops
  Consolidated suite for CI/CD automation, observability monitoring, and Git PR workflows.

Quality & Maintenance
  Consolidated suite for code quality, test automation, legacy modernization, and debugging.

Scientific Computing
  Consolidated suite for high-performance computing, specialized physics/chemistry simulations, and data science workflows.

Features
--------

**v2.0 Architecture**
  Consolidated 31 plugins into 5 high-performance suites for better maintainability and faster loading.

**New Furo Theme**
  Modern documentation theme with light/dark mode support, keyboard navigation, and enhanced mobile experience.

**Automated Documentation**
  Suite documentation is now automatically generated from plugin metadata, ensuring consistency across the ecosystem.

**Cross-Suite Integration**
  Suites are designed to work together, enabling complex multi-agent workflows.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Suites

   suites/agent-core
   suites/engineering-suite
   suites/infrastructure-suite
   suites/quality-suite
   suites/science-suite

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   integration-map
   tools-reference
   api/index
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
   # Select "Browse and install suites" -> Select suite

**Option 2: Install Specific Suites**

.. code-block:: bash

   /plugin install engineering-suite@marketplace
   /plugin install science-suite@marketplace

**Option 3: Install All Suites**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/imewei/MyClaude.git
   cd MyClaude

   # Enable all suites
   make plugin-enable-all

**Note:** After installation, restart Claude Code for changes to take effect.

Using Suites
~~~~~~~~~~~~

Once installed, suites provide agents, commands, and skills that are automatically available:

**Using Specialized Agents**

.. code-block:: text

   Ask Claude: "@software-architect help me design this microservice"
   Ask Claude: "@jax-pro optimize this neural network training loop"

**Running Commands**

.. code-block:: bash

   /agent-build:create "customer support chatbot"
   /quality-suite:double-check --mode=standard

Contributing
------------

We welcome contributions! See the contribution guidelines for:

- Adding new suites
- Improving existing suites
- Reporting issues
- Documentation improvements

License
-------

This marketplace is licensed under the MIT License. Individual suites may have their own licenses - see each suite's documentation for details.

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`search`
