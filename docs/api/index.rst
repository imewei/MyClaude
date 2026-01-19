API Reference
=============

This section provides the API documentation for the Python tools included in the MyClaude plugin marketplace.

.. note::

   All tools require Python 3.12+ and use only the standard library.

Overview
--------

The tools are organized into the following modules:

- **Common Module** - Shared data models and utilities used across all tools
- **Validation Tools** - Plugin metadata, documentation, and reference validators
- **Performance Tools** - Load time, activation, and memory profilers
- **Analysis Tools** - Dependency, terminology, and workflow analyzers

Common Module
-------------

The ``tools/common/`` module contains shared components:

.. toctree::
   :maxdepth: 2

   common

Validation Tools
----------------

Tools for validating plugin structure, metadata, and documentation:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - Purpose
   * - ``metadata_validator.py``
     - Validates plugin.json against schema
   * - ``doc_checker.py``
     - Validates documentation completeness
   * - ``skill_validator.py``
     - Tests skill pattern matching accuracy
   * - ``activation_tester.py``
     - Tests plugin activation accuracy
   * - ``xref_validator.py``
     - Validates cross-plugin references
   * - ``plugin_review_script.py``
     - Comprehensive plugin review

Performance Tools
-----------------

Tools for profiling plugin performance:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - Purpose
   * - ``load_profiler.py``
     - Measures plugin load time (<100ms target)
   * - ``activation_profiler.py``
     - Measures agent activation time (<50ms target)
   * - ``memory_analyzer.py``
     - Profiles memory consumption (<5MB target)
   * - ``performance_reporter.py``
     - Aggregates performance metrics

Analysis Tools
--------------

Tools for analyzing plugin relationships and patterns:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - Purpose
   * - ``dependency_mapper.py``
     - Maps cross-plugin dependencies
   * - ``terminology_analyzer.py``
     - Analyzes terminology consistency
   * - ``workflow_generator.py``
     - Generates integration workflows
   * - ``command_analyzer.py``
     - Analyzes command relevance
   * - ``triggering_reporter.py``
     - Comprehensive triggering reports
   * - ``test_corpus_generator.py``
     - Generates test samples

Data Models
-----------

All tools use shared data models defined in ``tools/common/models.py``:

ValidationIssue
~~~~~~~~~~~~~~~

Represents a validation issue (error or warning).

.. code-block:: python

   @dataclass
   class ValidationIssue:
       field: str                      # Field name with issue
       severity: str                   # 'critical', 'error', 'warning', 'info'
       message: str                    # Description of the issue
       suggestion: Optional[str]       # Suggested fix
       file_path: Optional[str]        # File location
       line_number: int                # Line number (0 if unknown)

ValidationResult
~~~~~~~~~~~~~~~~

Standardized validation result container.

.. code-block:: python

   @dataclass
   class ValidationResult:
       plugin_name: str
       plugin_path: Optional[Path]
       is_valid: bool
       issues: list[ValidationIssue]

PluginMetadata
~~~~~~~~~~~~~~

Plugin metadata extracted from plugin.json.

.. code-block:: python

   @dataclass
   class PluginMetadata:
       name: str
       version: str
       description: str
       category: str
       path: Optional[Path]
       agents: list[dict]
       commands: list[dict]
       skills: list[dict]
       keywords: list[str]

ProfileMetric
~~~~~~~~~~~~~

Timing metric for profiling operations.

.. code-block:: python

   @dataclass
   class ProfileMetric:
       name: str           # Operation name
       duration_ms: float  # Duration in milliseconds
       status: str         # 'pass', 'warn', 'fail', 'error'
       details: str        # Additional details

Usage Examples
--------------

Loading Plugins
~~~~~~~~~~~~~~~

.. code-block:: python

   from tools.common.loader import PluginLoader
   from pathlib import Path

   # Initialize loader
   loader = PluginLoader(Path("plugins"))

   # Load single plugin
   metadata = loader.load_plugin("julia-development")
   print(f"Loaded: {metadata.name} v{metadata.version}")
   print(f"Agents: {metadata.agent_count}")

   # Load all plugins
   all_plugins = loader.load_all_plugins()
   print(f"Loaded {len(all_plugins)} plugins")

   # Get plugins by category
   by_category = loader.get_plugins_by_category()
   for category, plugins in by_category.items():
       print(f"{category}: {len(plugins)} plugins")

Running Validators
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Validate single plugin
   python tools/metadata_validator.py plugins/julia-development

   # Validate all plugins
   python tools/metadata_validator.py --all

   # Run comprehensive review
   python tools/plugin_review_script.py julia-development

Running Profilers
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Profile load time
   python tools/load_profiler.py julia-development

   # Profile activation time
   python tools/activation_profiler.py --all

   # Generate performance report
   python tools/performance_reporter.py --all --export json results.json

See Also
--------

- :doc:`/tools-reference` - Complete tools reference
- :doc:`/index` - Documentation home
