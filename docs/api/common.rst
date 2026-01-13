Common Module
=============

The ``tools/common/`` module provides shared data models and utilities used across all plugin validation and analysis tools.

.. contents:: Module Contents
   :local:
   :depth: 2

Module Overview
---------------

The common module contains two primary components:

- ``models.py`` - Shared dataclasses for validation results, plugin metadata, and metrics
- ``loader.py`` - Unified plugin.json loader with caching

models.py
---------

Shared data models for plugin validation tools. Consolidates duplicate dataclass definitions from multiple tools.

Classes
~~~~~~~

ValidationIssue
^^^^^^^^^^^^^^^

Represents a validation issue (error or warning).

**Attributes:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Attribute
     - Type
     - Description
   * - ``field``
     - ``str``
     - Field name where the issue was found
   * - ``severity``
     - ``str``
     - Severity level: 'critical', 'error', 'warning', 'info'
   * - ``message``
     - ``str``
     - Description of the issue
   * - ``suggestion``
     - ``Optional[str]``
     - Suggested fix for the issue
   * - ``file_path``
     - ``Optional[str]``
     - Path to the file with the issue
   * - ``line_number``
     - ``int``
     - Line number (0 if unknown)

**Properties:**

- ``is_error`` - Returns ``True`` if this is an error-level issue
- ``emoji`` - Returns emoji indicator for severity level

**Example:**

.. code-block:: python

   from tools.common.models import ValidationIssue

   issue = ValidationIssue(
       field="version",
       severity="error",
       message="Invalid version format",
       suggestion="Use semver format: X.Y.Z"
   )
   print(f"{issue.emoji} {issue.message}")  # 🟠 Invalid version format

ValidationResult
^^^^^^^^^^^^^^^^

Standardized validation result for all validators.

**Attributes:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Attribute
     - Type
     - Description
   * - ``plugin_name``
     - ``str``
     - Name of the validated plugin
   * - ``plugin_path``
     - ``Optional[Path]``
     - Path to the plugin directory
   * - ``is_valid``
     - ``bool``
     - Whether validation passed
   * - ``issues``
     - ``list[ValidationIssue]``
     - List of validation issues found

**Methods:**

- ``add_error(field, message, suggestion, file_path, line_number)`` - Add a validation error
- ``add_warning(field, message, suggestion, file_path, line_number)`` - Add a validation warning
- ``add_info(field, message)`` - Add an informational message
- ``get_issue_count_by_severity()`` - Count issues by severity level

**Properties:**

- ``errors`` - List of error-level issues
- ``warnings`` - List of warning-level issues
- ``error_count`` - Number of errors
- ``warning_count`` - Number of warnings

**Example:**

.. code-block:: python

   from tools.common.models import ValidationResult

   result = ValidationResult(plugin_name="my-plugin")
   result.add_error("version", "Missing version field")
   result.add_warning("description", "Description too short")

   print(f"Valid: {result.is_valid}")  # False
   print(f"Errors: {result.error_count}")  # 1
   print(f"Warnings: {result.warning_count}")  # 1

PluginMetadata
^^^^^^^^^^^^^^

Plugin metadata extracted from plugin.json.

**Attributes:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Attribute
     - Type
     - Description
   * - ``name``
     - ``str``
     - Plugin name
   * - ``version``
     - ``str``
     - Plugin version
   * - ``description``
     - ``str``
     - Plugin description
   * - ``category``
     - ``str``
     - Plugin category
   * - ``path``
     - ``Optional[Path]``
     - Path to plugin directory
   * - ``agents``
     - ``list[dict]``
     - List of agent definitions
   * - ``commands``
     - ``list[dict]``
     - List of command definitions
   * - ``skills``
     - ``list[dict]``
     - List of skill definitions
   * - ``keywords``
     - ``list[str]``
     - List of keywords

**Properties:**

- ``agent_names`` - List of agent names
- ``command_names`` - List of command names
- ``skill_names`` - List of skill names
- ``agent_count`` - Number of agents
- ``command_count`` - Number of commands
- ``skill_count`` - Number of skills

ProfileMetric
^^^^^^^^^^^^^

Timing metric for profiling operations.

**Attributes:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Attribute
     - Type
     - Description
   * - ``name``
     - ``str``
     - Operation name
   * - ``duration_ms``
     - ``float``
     - Duration in milliseconds
   * - ``status``
     - ``str``
     - Status: 'pass', 'warn', 'fail', 'error'
   * - ``details``
     - ``str``
     - Additional details

**Properties:**

- ``status_emoji`` - Visual status indicator

**Class Methods:**

- ``from_duration(name, duration_ms, pass_threshold, warn_threshold, details)`` - Create metric with automatic status

loader.py
---------

Unified plugin.json loader with caching.

PluginLoader Class
~~~~~~~~~~~~~~~~~~

Loads and caches plugin metadata from plugin.json files.

**Constructor:**

.. code-block:: python

   loader = PluginLoader(plugins_dir: Path)

**Parameters:**

- ``plugins_dir`` - Path to the plugins directory

**Methods:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``load_plugin(name)``
     - Load a single plugin's metadata
   * - ``load_all_plugins()``
     - Load all plugins from directory
   * - ``get_plugin(name)``
     - Get from cache or load if not cached
   * - ``get_all_cached()``
     - Get all currently cached plugins
   * - ``get_errors(name)``
     - Get load errors for a plugin
   * - ``clear_cache()``
     - Clear the plugin cache
   * - ``get_plugin_names()``
     - List of all loaded plugin names
   * - ``get_plugins_by_category()``
     - Group plugins by category
   * - ``find_agent(name)``
     - Find which plugin contains an agent
   * - ``find_command(name)``
     - Find which plugin contains a command
   * - ``find_skill(name)``
     - Find which plugin contains a skill
   * - ``get_total_counts()``
     - Get total counts across all plugins

**Example:**

.. code-block:: python

   from tools.common.loader import PluginLoader
   from pathlib import Path

   # Initialize loader
   loader = PluginLoader(Path("plugins"))

   # Load all plugins
   plugins = loader.load_all_plugins()
   print(f"Loaded {len(plugins)} plugins")

   # Get counts
   counts = loader.get_total_counts()
   print(f"Agents: {counts['agents']}")
   print(f"Commands: {counts['commands']}")
   print(f"Skills: {counts['skills']}")

   # Find where an agent is defined
   result = loader.find_agent("python-pro")
   if result:
       plugin_name, agent_def = result
       print(f"Found in: {plugin_name}")

   # Group by category
   by_category = loader.get_plugins_by_category()
   for category, plugins in by_category.items():
       print(f"{category}: {len(plugins)} plugins")

See Also
--------

- :doc:`index` - API Reference overview
- :doc:`/tools-reference` - Complete tools documentation
