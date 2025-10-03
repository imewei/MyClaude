# Plugin Development Guide

Complete guide to developing plugins for the Claude Code command executor framework.

## Table of Contents

1. [Overview](#overview)
2. [Plugin Types](#plugin-types)
3. [Getting Started](#getting-started)
4. [Plugin Structure](#plugin-structure)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Publishing](#publishing)

## Overview

The plugin system enables extensibility of the Claude Code command executor framework through custom:
- Commands (slash commands)
- Agents (analysis agents)
- Validators (validation logic)
- Cache providers (storage backends)
- Reporters (output formats)
- Integrations (external services)

### Key Features

- **Auto-discovery**: Plugins are automatically discovered from configured directories
- **Hot-reloading**: Plugins can be reloaded without restarting the system
- **Dependency management**: Automatic dependency resolution
- **Security**: Sandboxing and permission system
- **Hook system**: Extend framework behavior at various points

## Plugin Types

### 1. Command Plugin

Add new slash commands to the framework.

**Use Cases:**
- Custom analysis commands
- Deployment automation
- Project scaffolding
- Custom workflows

**Base Class:** `CommandPlugin`

**Example:**
```python
from core.plugin_base import CommandPlugin, PluginContext, PluginResult
from api.command_api import CommandAPI

class MyCommandPlugin(CommandPlugin):
    def load(self):
        return True

    def execute(self, context: PluginContext) -> PluginResult:
        return CommandAPI.success_result(
            self.metadata.name,
            {"message": "Hello from plugin!"}
        )

    def get_command_info(self):
        return {
            "name": "my-command",
            "description": "My custom command",
            "usage": "/my-command"
        }
```

### 2. Agent Plugin

Add custom agents to the multi-agent system.

**Use Cases:**
- Domain-specific experts
- Custom analysis capabilities
- Specialized review agents

**Base Class:** `AgentPlugin`

**Example:**
```python
from core.plugin_base import AgentPlugin
from api.agent_api import AgentAPI

class MyAgentPlugin(AgentPlugin):
    def load(self):
        return True

    def get_agent_profile(self):
        return AgentAPI.create_agent_profile(
            capabilities=['custom_analysis'],
            specializations=['domain expertise'],
            languages=['python'],
            frameworks=['custom'],
            priority=8
        )

    def analyze(self, context):
        # Perform analysis
        return {
            'findings': [],
            'recommendations': []
        }
```

### 3. Integration Plugin

Integrate with external services.

**Use Cases:**
- Slack/Discord notifications
- JIRA/GitHub integration
- Metrics export (Prometheus, etc.)
- CI/CD integration

**Base Class:** `IntegrationPlugin`

### 4. Validator Plugin

Add custom validation logic.

**Base Class:** `ValidatorPlugin`

### 5. Cache Provider Plugin

Implement custom cache backends.

**Base Class:** `CacheProviderPlugin`

### 6. Reporter Plugin

Add custom report formats.

**Base Class:** `ReporterPlugin`

## Getting Started

### Step 1: Create Plugin Directory

```bash
mkdir -p ~/.claude/commands/plugins/my-plugin
cd ~/.claude/commands/plugins/my-plugin
```

### Step 2: Create Plugin Manifest

Create `plugin.json`:

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "type": "command",
  "description": "My custom plugin",
  "author": "Your Name",
  "framework_version": ">=2.0.0",
  "python_version": ">=3.8",
  "dependencies": [],
  "capabilities": [],
  "default_config": {},
  "permissions": ["read"],
  "sandbox": true,
  "license": "MIT",
  "tags": ["custom"]
}
```

### Step 3: Implement Plugin

Create `plugin.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.plugin_base import CommandPlugin, PluginContext, PluginResult
from api.command_api import CommandAPI

class MyPlugin(CommandPlugin):
    def load(self):
        self.logger.info(f"Loading {self.metadata.name}")
        return True

    def execute(self, context: PluginContext) -> PluginResult:
        # Your plugin logic here
        return CommandAPI.success_result(
            self.metadata.name,
            {"result": "success"}
        )

    def get_command_info(self):
        return {
            "name": "my-plugin",
            "description": "My custom plugin",
            "usage": "/my-plugin"
        }
```

### Step 4: Test Plugin

```bash
python plugin.py
```

## Plugin Structure

### Directory Layout

```
my-plugin/
├── plugin.json          # Plugin manifest (required)
├── plugin.py            # Main plugin file (required)
├── README.md           # Documentation (recommended)
├── requirements.txt    # Python dependencies (optional)
├── tests/              # Tests (recommended)
│   └── test_plugin.py
└── examples/           # Usage examples (optional)
    └── example.py
```

### Plugin Manifest Fields

**Required Fields:**
- `name`: Plugin identifier (kebab-case)
- `version`: Semantic version (e.g., "1.0.0")
- `type`: Plugin type (command, agent, integration, etc.)
- `description`: Brief description
- `author`: Author name

**Optional Fields:**
- `framework_version`: Required framework version
- `python_version`: Required Python version
- `dependencies`: Python package dependencies
- `capabilities`: Plugin capabilities
- `supported_commands`: Commands this plugin enhances
- `config_schema`: Configuration schema
- `default_config`: Default configuration
- `permissions`: Required permissions
- `sandbox`: Enable sandboxing (default: true)
- `homepage`: Plugin homepage URL
- `repository`: Source repository URL
- `license`: License identifier
- `tags`: Search tags

## Development Workflow

### 1. Initialize Plugin

```bash
# Use template (coming soon)
cp -r plugins/templates/command_plugin_template my-plugin
cd my-plugin
```

### 2. Implement Core Logic

Focus on the `execute()` method:

```python
def execute(self, context: PluginContext) -> PluginResult:
    # 1. Get configuration
    config_value = self.get_config('key', 'default')

    # 2. Access context
    work_dir = context.work_dir
    command_name = context.command_name

    # 3. Perform plugin logic
    result_data = self.do_work(work_dir)

    # 4. Return result
    return CommandAPI.success_result(
        self.metadata.name,
        data=result_data
    )
```

### 3. Add Configuration

```python
def load(self):
    # Validate configuration
    required_key = self.get_config('required_key')
    if not required_key:
        self.logger.error("required_key not configured")
        return False

    return True
```

### 4. Register Hooks

```python
from core.plugin_base import HookType

def load(self):
    # Register hooks
    self.register_hook(HookType.PRE_EXECUTION, self.pre_exec_hook)
    self.register_hook(HookType.POST_EXECUTION, self.post_exec_hook)
    return True

def pre_exec_hook(self, context, data):
    # Called before command execution
    self.logger.info("Pre-execution hook")
    return data

def post_exec_hook(self, context, data):
    # Called after command execution
    self.logger.info("Post-execution hook")
    return data
```

### 5. Handle State

```python
def execute(self, context):
    # Get state
    counter = self.get_state('counter', 0)

    # Update state
    self.set_state('counter', counter + 1)

    return CommandAPI.success_result(
        self.metadata.name,
        {"counter": counter + 1}
    )
```

## Testing

### Unit Tests

Create `tests/test_plugin.py`:

```python
import unittest
from pathlib import Path
from core.plugin_base import PluginMetadata, PluginContext, PluginType
from plugin import MyPlugin

class TestMyPlugin(unittest.TestCase):
    def setUp(self):
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            plugin_type=PluginType.COMMAND,
            description="Test",
            author="Test"
        )
        self.plugin = MyPlugin(metadata)
        self.plugin.load()

    def test_execute(self):
        context = PluginContext(
            plugin_name="test",
            command_name="test",
            work_dir=Path.cwd(),
            config={},
            framework_version="2.0.0"
        )

        result = self.plugin.execute(context)

        self.assertTrue(result.success)
        self.assertEqual(result.plugin_name, "test-plugin")

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

Test with the plugin manager:

```python
from core.plugin_manager import PluginManager

manager = PluginManager()
manager.initialize()

plugin = manager.get_plugin('my-plugin')
assert plugin is not None

# Test execution
context = PluginContext(...)
result = plugin.execute(context)
assert result.success
```

### Manual Testing

```bash
# Test plugin directly
python plugin.py

# Test with framework
python -m core.plugin_manager
```

## Publishing

### Option 1: Directory Plugin

Simply place your plugin directory in:
- `~/.claude/commands/plugins/`
- `~/.claude/plugins/`
- `./plugins/` (project-local)

### Option 2: pip-Installable Plugin

Create `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name='claude-code-my-plugin',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        # dependencies
    ],
    entry_points={
        'claude_code_plugins': [
            'my-plugin = my_plugin.plugin:MyPlugin',
        ],
    },
)
```

Install:
```bash
pip install -e .
```

### Option 3: Share as Package

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

Then users can install:
```bash
pip install claude-code-my-plugin
```

## Best Practices

1. **Follow naming conventions**
   - Use kebab-case for plugin names
   - Prefix pip packages with `claude-code-`

2. **Validate configuration**
   - Check required config in `load()`
   - Provide sensible defaults
   - Document configuration options

3. **Error handling**
   - Always catch exceptions in `execute()`
   - Return meaningful error messages
   - Log errors appropriately

4. **Resource cleanup**
   - Implement `cleanup()` for resource cleanup
   - Close files, connections, etc.

5. **Documentation**
   - Include README.md
   - Document configuration options
   - Provide usage examples

6. **Testing**
   - Write unit tests
   - Test edge cases
   - Test with framework integration

7. **Security**
   - Validate all inputs
   - Use sandboxing when possible
   - Request only necessary permissions

8. **Performance**
   - Cache expensive operations
   - Use async when appropriate
   - Minimize dependencies

## Next Steps

- Review [Plugin API Reference](PLUGIN_API_REFERENCE.md)
- Check [Example Plugins](PLUGIN_EXAMPLES.md)
- Learn [Best Practices](PLUGIN_BEST_PRACTICES.md)
- Understand [Security](PLUGIN_SECURITY.md)

## Support

- GitHub Issues: [Report bugs or request features]
- Documentation: [Full framework documentation]
- Examples: See `plugins/examples/` directory