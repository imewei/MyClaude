Quick-Start Guides
==================

This section provides quick-start guides for common multi-plugin workflows. Each guide demonstrates how to combine multiple plugins to accomplish specific tasks.

Available Guides
----------------

.. toctree::
   :maxdepth: 1

   scientific-workflows
   development-workflows
   devops-workflows
   infrastructure-workflows
   integration-patterns

Guide Overview
--------------

Scientific Workflows
~~~~~~~~~~~~~~~~~~~~

Learn how to combine scientific computing plugins for high-performance simulations, data analysis, and visualization.

**Plugins:** julia-development, jax-implementation, hpc-computing, molecular-simulation, statistical-physics, deep-learning, data-visualization, research-methodology

:doc:`scientific-workflows`

Development Workflows
~~~~~~~~~~~~~~~~~~~~~

Discover development workflows that integrate frontend, backend, testing, and deployment plugins.

**Plugins:** python-development, backend-development, frontend-mobile-development, javascript-typescript, systems-programming, multi-platform-apps, llm-application-dev

:doc:`development-workflows`

DevOps Workflows
~~~~~~~~~~~~~~~~

Master DevOps automation by combining CI/CD, monitoring, and infrastructure plugins.

**Plugins:** cicd-automation, git-pr-workflows, observability-monitoring

:doc:`devops-workflows`

Infrastructure Workflows
~~~~~~~~~~~~~~~~~~~~~~~~

Build robust cloud infrastructure with integrated monitoring and management.

:doc:`infrastructure-workflows`

Integration Patterns
~~~~~~~~~~~~~~~~~~~~

Best practices for combining plugins and creating custom workflows.

:doc:`integration-patterns`

Reference Documents
-------------------

LaTeX Reference Documents
~~~~~~~~~~~~~~~~~~~~~~~~~

Printable reference documents are available in LaTeX format for offline use:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Document
     - Description
     - Format
   * - ``AGENTS_LIST.tex``
     - Complete 74-agent reference with professional styling
     - LaTeX (~27KB)
   * - ``COMMANDS_LIST.tex``
     - Complete 48-command reference with professional styling
     - LaTeX (~22KB)
   * - ``plugin-cheatsheet.tex``
     - Quick reference cheatsheet (landscape, 3-column)
     - LaTeX (~5KB)
   * - ``latex_compile.sh``
     - Compilation script for all LaTeX documents
     - Bash
   * - ``LATEX_README.md``
     - LaTeX compilation guide and troubleshooting
     - Markdown

**To compile LaTeX documents:**

.. code-block:: bash

   cd docs/guides/

   # Option 1: Use the compilation script (recommended)
   chmod +x latex_compile.sh
   ./latex_compile.sh

   # Option 2: Compile manually
   pdflatex AGENTS_LIST.tex && pdflatex AGENTS_LIST.tex
   pdflatex COMMANDS_LIST.tex && pdflatex COMMANDS_LIST.tex
   pdflatex plugin-cheatsheet.tex && pdflatex plugin-cheatsheet.tex

Markdown References
~~~~~~~~~~~~~~~~~~~

- `AGENTS_LIST.md <https://github.com/imewei/MyClaude/blob/main/AGENTS_LIST.md>`_ - Complete agent catalog with descriptions
- `COMMANDS_LIST.md <https://github.com/imewei/MyClaude/blob/main/COMMANDS_LIST.md>`_ - Complete command catalog with usage examples
- `PLUGIN_CHEATSHEET.md <https://github.com/imewei/MyClaude/blob/main/PLUGIN_CHEATSHEET.md>`_ - Quick reference guide

Quick Reference
---------------

Execution Modes
~~~~~~~~~~~~~~~

Most commands support three execution modes:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Mode
     - Time
     - Description
   * - ``quick``
     - 30min-2h
     - Fast analysis, syntax checking, basic scaffolding
   * - ``standard``
     - 2-6h
     - Full implementation with testing and documentation
   * - ``enterprise``
     - 1-2d
     - Complete solution with advanced features and compliance

Agent Format
~~~~~~~~~~~~

Agents use the format ``plugin:agent`` (single colon):

.. code-block:: text

   julia-development:julia-pro
   jax-implementation:jax-pro
   python-development:python-pro
   backend-development:backend-architect

Command Format
~~~~~~~~~~~~~~

Commands use the format ``/plugin:command`` (slash prefix):

.. code-block:: bash

   /julia-development:sciml-setup "problem description"
   /quality-engineering:double-check --mode=quick
   /cicd-automation:fix-commit-errors --auto-fix

Statistics
----------

**Version 1.0.4** (December 2025)

- **31 Plugins** across 6 categories
- **74 Agents** enhanced with nlsq-pro template
- **48 Commands** with execution mode support
- **114 Skills** for context-aware intelligence
- **16 Tools** for validation and profiling

See Also
--------

- :doc:`../index` - Documentation home
- :doc:`../tools-reference` - 16 utility scripts
- :doc:`../integration-map` - Plugin integration matrix
- :doc:`../glossary` - Technical terminology
