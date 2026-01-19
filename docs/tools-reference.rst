Tools Reference
===============

This section documents the 16 utility scripts and tools included in the MyClaude plugin marketplace for validation, profiling, and analysis.

Overview
--------

The marketplace includes Python utilities organized into four categories:

- **Plugin Management** - Enable and configure plugins
- **Validation Tools** - Validate metadata, documentation, and references
- **Performance Profilers** - Measure load time, activation, and memory usage
- **Analysis Tools** - Analyze dependencies, terminology, and workflows

All tools are located in the ``tools/`` directory and require Python 3.12+.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Purpose
     - Target Metric
   * - ``activation_profiler.py``
     - Measure agent activation time
     - <50ms
   * - ``load_profiler.py``
     - Measure plugin load time
     - <100ms
   * - ``memory_analyzer.py``
     - Profile memory consumption
     - <5MB per plugin
   * - ``metadata_validator.py``
     - Validate plugin.json schema
     - 100% compliance
   * - ``skill_validator.py``
     - Test skill pattern matching
     - <5% over-trigger rate
   * - ``plugin_review_script.py``
     - Comprehensive plugin validation
     - All checks pass

Plugin Management
-----------------

enable_all_plugins.py
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/enable_all_plugins.py``

Enable all 31 plugins from the scientific-computing-workflows marketplace in Claude Code.

**Usage:**

.. code-block:: bash

   python3 tools/enable_all_plugins.py

**Features:**

- Reads plugins from marketplace.json
- Updates Claude Code settings.json
- Tracks newly enabled vs already enabled plugins
- Provides summary of enabled plugins

Validation Tools
----------------

metadata_validator.py
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/metadata_validator.py``

Validates plugin.json files against the marketplace schema.

**Usage:**

.. code-block:: bash

   python3 tools/metadata_validator.py <plugin-name>
   python3 tools/metadata_validator.py --all

**Validations:**

- Required fields (name, version, description, category)
- Version format (semver)
- Agent/command/skill array structure
- Status field values (active, deprecated, experimental)
- Keyword format and uniqueness

skill_validator.py
~~~~~~~~~~~~~~~~~~

**Location:** ``tools/skill_validator.py``

Tests skill pattern matching and validates skill recommendations to check for over-triggering issues.

**Usage:**

.. code-block:: bash

   python3 tools/skill_validator.py
   python3 tools/skill_validator.py --plugins-dir /path/to/plugins
   python3 tools/skill_validator.py --corpus-dir /path/to/test-corpus
   python3 tools/skill_validator.py --plugin julia-development

**Metrics Generated:**

- Overall Accuracy
- Precision and Recall
- Over-Trigger Rate (target: <5%)
- Under-Trigger Rate (target: <5%)

**Classes:**

- ``Skill``: Represents a plugin skill with keywords and patterns
- ``SkillContext``: Context for skill application (file info, imports, functions)
- ``SkillApplicationResult``: Result of skill application test
- ``SkillValidationMetrics``: Aggregated validation metrics

activation_tester.py
~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/activation_tester.py``

Tests plugin activation accuracy against test corpus samples, measuring false positive and false negative rates.

**Usage:**

.. code-block:: bash

   python3 tools/activation_tester.py
   python3 tools/activation_tester.py --plugin julia-development
   python3 tools/activation_tester.py --corpus-dir /path/to/test-corpus

**Metrics:**

- True positive rate
- True negative rate
- False positive rate (target: <5%)
- False negative rate (target: <5%)

doc_checker.py
~~~~~~~~~~~~~~

**Location:** ``tools/doc_checker.py``

Validates plugin documentation for completeness and formatting.

**Usage:**

.. code-block:: bash

   python3 tools/doc_checker.py <plugin-name>
   python3 tools/doc_checker.py --all

**Validations:**

- README.md required sections
- Markdown formatting
- Code block syntax
- Cross-reference accuracy
- Link validation

xref_validator.py
~~~~~~~~~~~~~~~~~

**Location:** ``tools/xref_validator.py``

Validates cross-plugin references to identify broken links.

**Usage:**

.. code-block:: bash

   python3 tools/xref_validator.py
   python3 tools/xref_validator.py --plugins-dir /path/to/plugins

**Features:**

- Checks all cross-plugin references
- Validates agent/command/skill mentions
- Identifies broken references
- Generates validation reports

plugin_review_script.py
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/plugin_review_script.py``

Comprehensive plugin validation combining multiple checks.

**Usage:**

.. code-block:: bash

   python3 tools/plugin_review_script.py <plugin-name>
   python3 tools/plugin_review_script.py --all

**Checks Performed:**

- Metadata validation
- Documentation completeness
- Performance profiling
- Cross-reference validation
- Skill pattern testing

Performance Profilers
---------------------

activation_profiler.py
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/activation_profiler.py``

Measures agent activation performance to identify bottlenecks in triggering logic.

**Usage:**

.. code-block:: bash

   python3 tools/activation_profiler.py <plugin-name>
   python3 tools/activation_profiler.py <plugin-name> /path/to/plugins
   python3 tools/activation_profiler.py --all

**Target:** <50ms activation time

**Metrics Tracked:**

- Metadata extraction time
- Keyword matching time
- Agent selection time
- Agent description parsing time
- Context relevance scoring time

load_profiler.py
~~~~~~~~~~~~~~~~

**Location:** ``tools/load_profiler.py``

Measures plugin loading performance to identify bottlenecks and optimize initialization.

**Usage:**

.. code-block:: bash

   python3 tools/load_profiler.py <plugin-name>
   python3 tools/load_profiler.py <plugin-name> /path/to/plugins
   python3 tools/load_profiler.py --all

**Target:** <100ms load time per plugin

**Metrics Tracked:**

- plugin.json parsing time
- Agents directory scan time
- Commands directory scan time
- Skills directory scan time
- README.md loading time

memory_analyzer.py
~~~~~~~~~~~~~~~~~~

**Location:** ``tools/memory_analyzer.py``

Measures plugin memory consumption to identify memory leaks and inefficiencies.

**Usage:**

.. code-block:: bash

   python3 tools/memory_analyzer.py <plugin-name>
   python3 tools/memory_analyzer.py <plugin-name> /path/to/plugins
   python3 tools/memory_analyzer.py --all

**Features:**

- Measures baseline memory consumption
- Tracks memory during typical operations
- Identifies memory leaks
- Profiles data structure efficiency

performance_reporter.py
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/performance_reporter.py``

Aggregates performance metrics across all profiling tools and generates comprehensive reports.

**Usage:**

.. code-block:: bash

   python3 tools/performance_reporter.py <plugin-name>
   python3 tools/performance_reporter.py --all
   python3 tools/performance_reporter.py --compare before.json after.json
   python3 tools/performance_reporter.py --export csv output.csv
   python3 tools/performance_reporter.py --export json output.json

**Features:**

- Aggregates metrics across all plugins
- Generates before/after comparison reports
- Exports results to CSV/JSON
- Visualizes performance trends

Analysis Tools
--------------

command_analyzer.py
~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/command_analyzer.py``

Analyzes command suggestion relevance, timing, and priority ranking accuracy.

**Usage:**

.. code-block:: bash

   python3 tools/command_analyzer.py
   python3 tools/command_analyzer.py --plugin julia-development
   python3 tools/command_analyzer.py --corpus-dir /path/to/test-corpus

**Features:**

- Command relevance scoring
- Suggestion timing validation
- Priority ranking accuracy
- Context-aware analysis

dependency_mapper.py
~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/dependency_mapper.py``

Analyzes cross-plugin relationships and generates dependency maps.

**Usage:**

.. code-block:: bash

   python3 tools/dependency_mapper.py
   python3 tools/dependency_mapper.py --output graph.json

**Features:**

- Parses all plugin.json files
- Extracts agent, command, and skill references
- Builds dependency graph
- Identifies integration patterns
- Generates visual and textual dependency maps

terminology_analyzer.py
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/terminology_analyzer.py``

Analyzes terminology usage and consistency across all plugins.

**Usage:**

.. code-block:: bash

   python3 tools/terminology_analyzer.py
   python3 tools/terminology_analyzer.py --plugins-dir /path/to/plugins

**Features:**

- Extracts technical terms from all plugins
- Identifies terminology variations
- Maps synonyms and inconsistencies
- Suggests standardization

workflow_generator.py
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/workflow_generator.py``

Creates integration workflow documentation for multi-plugin use cases.

**Usage:**

.. code-block:: bash

   python3 tools/workflow_generator.py
   python3 tools/workflow_generator.py --output workflows.md

**Features:**

- Identifies common plugin combinations
- Generates workflow documentation templates
- Creates integration examples
- Documents multi-plugin use cases

triggering_reporter.py
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/triggering_reporter.py``

Aggregates triggering metrics across all validation tools.

**Usage:**

.. code-block:: bash

   python3 tools/triggering_reporter.py
   python3 tools/triggering_reporter.py --export report.json

**Features:**

- Aggregates activation accuracy metrics
- Generates comprehensive triggering reports
- Identifies plugins with high over-trigger rates
- Tracks improvements over time

test_corpus_generator.py
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``tools/test_corpus_generator.py``

Generates diverse sample projects for testing plugin activation accuracy.

**Usage:**

.. code-block:: bash

   python3 tools/test_corpus_generator.py
   python3 tools/test_corpus_generator.py --output-dir custom-test-corpus
   python3 tools/test_corpus_generator.py --categories scientific-computing development

**Features:**

- Generates samples for each plugin category
- Includes edge cases and multi-language projects
- Creates negative test samples
- Provides expected plugin mappings

Sphinx Custom Directives
------------------------

plugin_directives.py
~~~~~~~~~~~~~~~~~~~~

**Location:** ``docs/_ext/plugin_directives.py``

Custom Sphinx directives for plugin documentation.

**Directives:**

.. code-block:: rst

   .. agent:: agent-name
      :status: active

      Description of the agent.

   .. command:: /command-name
      :status: active
      :priority: high

      Description of the command.

   .. skill:: skill-name
      :status: active

      Description of the skill.

**Usage in RST Files:**

.. code-block:: rst

   Agents
   ------

   .. agent:: python-pro

      Master Python 3.12+ with modern features

      **Status:** active

Tool Categories Summary
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Category
     - Tools
     - Purpose
   * - Plugin Management
     - 1
     - Enable and configure plugins in Claude Code
   * - Validation
     - 6
     - Validate metadata, docs, skills, activation, cross-references
   * - Performance
     - 4
     - Profile load time, activation, memory, generate reports
   * - Analysis
     - 5
     - Analyze dependencies, terminology, commands, workflows, test corpus

See Also
--------

- :doc:`index` - Documentation home
- :doc:`glossary` - Technical terminology
- :doc:`guides/index` - Quick-start guides
