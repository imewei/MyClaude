# Plugin System - Complete Index

## Quick Navigation

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Plugin Types](#plugin-types)
- [Example Plugins](#example-plugins)
- [API Reference](#api-reference)
- [Development Guide](#development-guide)
- [Testing](#testing)

## Architecture Overview

The plugin system consists of 8 main components:

1. **Core System** - Base classes and plugin management
2. **API Layer** - Development APIs and utilities
3. **Example Plugins** - Working plugin implementations
4. **Templates** - Plugin scaffolding
5. **Utilities** - Helper functions
6. **Configuration** - Schemas and configs
7. **Documentation** - Guides and references
8. **Testing** - Test framework

## Core Components

### Plugin Base Classes (`core/plugin_base.py`)
- **BasePlugin**: Abstract base for all plugins
- **CommandPlugin**: Base for command plugins
- **AgentPlugin**: Base for agent plugins
- **ValidatorPlugin**: Base for validator plugins
- **CacheProviderPlugin**: Base for cache providers
- **ReporterPlugin**: Base for reporters
- **IntegrationPlugin**: Base for integrations

### Plugin Manager (`core/plugin_manager.py`)
- **PluginManager**: Central management system
- **PluginDiscovery**: Auto-discovery system
- **PluginLoader**: Dynamic loading

### Hook System (`core/plugin_hooks.py`)
- **HookRegistry**: Central hook registration
- **HookManager**: Hook coordination
- **11 Hook Types**: Pre/post execution, validation, agents, cache, errors

### Validator (`core/plugin_validator.py`)
- Security validation
- Manifest validation
- Dependency scanning
- Code analysis

## Plugin Types

### 1. Command Plugin
**Purpose**: Add custom slash commands

**Base Class**: `CommandPlugin`

**Key Methods**:
- `load()`: Initialize plugin
- `execute(context)`: Execute command logic
- `get_command_info()`: Command metadata

**Example**: `examples/example_hello_world/`

### 2. Agent Plugin
**Purpose**: Add custom analysis agents

**Base Class**: `AgentPlugin`

**Key Methods**:
- `load()`: Initialize agent
- `get_agent_profile()`: Agent capabilities
- `analyze(context)`: Perform analysis
- `execute(context)`: Execute agent

**Example**: `examples/example_domain_expert/`

### 3. Integration Plugin
**Purpose**: Connect to external services

**Base Class**: `IntegrationPlugin`

**Key Methods**:
- `load()`: Initialize connection
- `connect()`: Establish connection
- `send(data)`: Send data
- `disconnect()`: Close connection

**Example**: `examples/example_slack_integration/`

### 4. Validator Plugin
**Purpose**: Custom validation logic

**Base Class**: `ValidatorPlugin`

**Key Methods**:
- `validate_input(data)`: Validate data

### 5. Cache Provider Plugin
**Purpose**: Custom cache backends

**Base Class**: `CacheProviderPlugin`

**Key Methods**:
- `get(key)`: Get cached value
- `set(key, value, ttl)`: Set cached value
- `delete(key)`: Delete value
- `clear()`: Clear cache

### 6. Reporter Plugin
**Purpose**: Custom report formats

**Base Class**: `ReporterPlugin`

**Key Methods**:
- `generate_report(data, format)`: Generate report

## Example Plugins

### 1. Hello World (`example_hello_world/`)
**Type**: Command
**Purpose**: Simple greeting command
**Features**:
- Basic command structure
- Configuration usage
- Result formatting

**Files**:
- `plugin.json` - Manifest
- `plugin.py` - Implementation (128 lines)

### 2. Custom Analyzer (`example_custom_analyzer/`)
**Type**: Command
**Purpose**: Code complexity analysis
**Features**:
- AST-based analysis
- Cyclomatic complexity
- Import analysis
- Metrics collection

**Files**:
- `plugin.json` - Manifest
- `plugin.py` - Implementation (225 lines)

### 3. Slack Integration (`example_slack_integration/`)
**Type**: Integration
**Purpose**: Slack notifications
**Features**:
- Webhook integration
- Post-execution hooks
- Status notifications
- Error notifications

**Files**:
- `plugin.json` - Manifest
- `plugin.py` - Implementation (226 lines)

### 4. Domain Expert (`example_domain_expert/`)
**Type**: Agent
**Purpose**: Domain-specific analysis
**Features**:
- Multiple domains (web, data, security, performance, general)
- Configurable specialization
- Agent profile generation
- Domain-specific recommendations

**Files**:
- `plugin.json` - Manifest
- `plugin.py` - Implementation (294 lines)

### 5. Security Scanner (`example_security_scanner/`)
**Type**: Command
**Purpose**: Security vulnerability scanning
**Features**:
- Pattern-based detection
- Multiple vulnerability types
- Severity classification
- SQL injection detection
- XSS detection
- Secret detection

**Files**:
- `plugin.json` - Manifest
- `plugin.py` - Implementation (214 lines)

### Placeholder Plugins (Directories Created)
- `example_deployment/` - Deployment automation
- `example_language_support/` - Language-specific agent
- `example_jira_integration/` - JIRA integration
- `example_metrics_export/` - Metrics export
- `example_license_checker/` - License compliance
- `example_dependency_audit/` - Dependency vulnerabilities

## API Reference

### CommandAPI (`api/command_api.py`)

**Helper Functions**:
```python
# Create metadata
metadata = CommandAPI.create_metadata(
    name="my-plugin",
    version="1.0.0",
    description="My plugin",
    author="Author"
)

# Success result
result = CommandAPI.success_result(
    plugin_name="my-plugin",
    data={"key": "value"},
    message="Success"
)

# Error result
result = CommandAPI.error_result(
    plugin_name="my-plugin",
    error="Error message"
)

# Parse flags
flags = CommandAPI.parse_flags(['--flag', 'value', 'arg'])

# File operations
content = CommandAPI.read_file(path)
CommandAPI.write_file(path, content)
data = CommandAPI.read_json(path)
CommandAPI.write_json(path, data)
```

### AgentAPI (`api/agent_api.py`)

**Helper Functions**:
```python
# Create agent profile
profile = AgentAPI.create_agent_profile(
    capabilities=['analysis'],
    specializations=['domain'],
    languages=['python'],
    frameworks=['custom'],
    priority=8
)

# Success result
result = AgentAPI.success_result(
    plugin_name="agent",
    findings=["Finding 1"],
    recommendations=["Recommendation 1"]
)
```

## Development Guide

### Quick Start

1. **Create Plugin Directory**
```bash
mkdir -p ~/.claude/commands/plugins/my-plugin
cd ~/.claude/commands/plugins/my-plugin
```

2. **Create Manifest** (`plugin.json`)
```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "type": "command",
  "description": "My custom plugin",
  "author": "Your Name"
}
```

3. **Implement Plugin** (`plugin.py`)
```python
from plugins.core import CommandPlugin
from plugins.api import CommandAPI

class MyPlugin(CommandPlugin):
    def load(self):
        return True

    def execute(self, context):
        return CommandAPI.success_result(
            self.metadata.name,
            {"message": "Hello!"}
        )

    def get_command_info(self):
        return {"name": "my-plugin"}
```

4. **Test Plugin**
```bash
python3 plugin.py
```

### Plugin Lifecycle

1. **Discovery**: Plugin discovered from directory
2. **Loading**: Manifest read, plugin class loaded
3. **Validation**: Manifest and code validated
4. **Initialization**: `load()` method called
5. **Registration**: Plugin registered in manager
6. **Execution**: `execute()` method called when used
7. **Cleanup**: `cleanup()` method called on unload

### Hook Registration

```python
from plugins.core import HookType

def load(self):
    # Register hooks
    self.register_hook(HookType.PRE_EXECUTION, self.pre_hook)
    self.register_hook(HookType.POST_EXECUTION, self.post_hook)
    return True

def pre_hook(self, context, data):
    # Pre-execution logic
    return data

def post_hook(self, context, data):
    # Post-execution logic
    return data
```

### Configuration

```python
def load(self):
    # Get configuration
    setting = self.get_config('setting', 'default')

    # Validate required config
    if not self.get_config('required_setting'):
        self.logger.error("Required setting missing")
        return False

    return True
```

### State Management

```python
def execute(self, context):
    # Get state
    counter = self.get_state('counter', 0)

    # Update state
    self.set_state('counter', counter + 1)

    return result
```

## Testing

### Unit Tests

Located in `tests/test_plugin_manager.py`:
- `TestPluginBase`: Base plugin functionality
- `TestPluginDiscovery`: Discovery system
- `TestHookRegistry`: Hook system
- `TestPluginResult`: Result objects

**Run Tests**:
```bash
cd /Users/b80985/.claude/commands/plugins
python3 tests/test_plugin_manager.py
```

### Example Plugin Tests

```bash
# Test hello world
python3 examples/example_hello_world/plugin.py

# Test custom analyzer
python3 examples/example_custom_analyzer/plugin.py

# Test domain expert
python3 examples/example_domain_expert/plugin.py

# Test security scanner
python3 examples/example_security_scanner/plugin.py
```

### Integration Tests

```python
from plugins.core import PluginManager

# Initialize plugin system
manager = PluginManager()
manager.initialize()

# Get plugin
plugin = manager.get_plugin('my-plugin')

# Test execution
context = PluginContext(...)
result = plugin.execute(context)
assert result.success
```

## File Structure

```
plugins/
├── core/                           # Core plugin system (1,982 lines)
│   ├── __init__.py                # Module exports (42 lines)
│   ├── plugin_base.py             # Base classes (564 lines)
│   ├── plugin_hooks.py            # Hook system (433 lines)
│   ├── plugin_manager.py          # Plugin management (603 lines)
│   └── plugin_validator.py        # Validation (340 lines)
├── api/                            # Plugin APIs (455 lines)
│   ├── __init__.py                # Module exports (8 lines)
│   ├── agent_api.py               # Agent API (115 lines)
│   └── command_api.py             # Command API (332 lines)
├── examples/                       # Example plugins (1,087 lines)
│   ├── example_hello_world/       # Simple command (128 lines)
│   ├── example_custom_analyzer/   # Code analyzer (225 lines)
│   ├── example_slack_integration/ # Slack integration (226 lines)
│   ├── example_domain_expert/     # Domain agent (294 lines)
│   ├── example_security_scanner/  # Security scanner (214 lines)
│   └── [6 more directories]       # Ready for implementation
├── templates/                      # Plugin templates
│   ├── command_plugin_template/
│   ├── agent_plugin_template/
│   └── integration_plugin_template/
├── utils/                          # Utilities (90 lines)
│   ├── __init__.py                # Module exports (4 lines)
│   └── plugin_loader.py           # Dynamic loading (86 lines)
├── config/                         # Configuration
│   └── plugin_manifest_schema.json # JSON schema (80 lines)
├── docs/                           # Documentation (505 lines)
│   └── PLUGIN_DEVELOPMENT_GUIDE.md # Dev guide (505 lines)
├── tests/                          # Tests (266 lines)
│   └── test_plugin_manager.py     # Unit tests (266 lines)
├── README.md                       # Main documentation (325 lines)
├── PLUGIN_SYSTEM_SUMMARY.md        # Summary (415 lines)
└── PLUGIN_INDEX.md                 # This file

Total: 5,125+ lines of code across 25+ files
```

## Statistics

### Code Distribution
- Core System: ~1,982 lines (38.6%)
- APIs: ~455 lines (8.9%)
- Examples: ~1,087 lines (21.2%)
- Tests: ~266 lines (5.2%)
- Utils: ~90 lines (1.8%)
- Documentation: ~1,245 lines (24.3%)

### Plugin Examples
- 5 complete, working example plugins
- 6 directories ready for additional plugins
- Examples cover all major plugin types

### Features Implemented
✅ 6 plugin types supported
✅ Auto-discovery from directories
✅ pip-installable plugin support
✅ 11 hook types
✅ Security validation
✅ Configuration management
✅ State management
✅ Hot-reloading
✅ Comprehensive API
✅ Test framework
✅ Documentation

## Usage Examples

### Load and Execute Plugin

```python
from plugins.core import PluginManager, PluginContext
from pathlib import Path

# Initialize
manager = PluginManager()
manager.initialize()

# Get plugin
plugin = manager.get_plugin('hello-world')

# Create context
context = PluginContext(
    plugin_name='hello-world',
    command_name='hello-world',
    work_dir=Path.cwd(),
    config={'args': ['World']},
    framework_version='2.0.0'
)

# Execute
result = plugin.execute(context)
print(result.data)  # {'message': 'Hello, World!', 'greeted': 'World'}
```

### Register Hook

```python
from plugins.core import HookRegistry, HookType, PluginContext

registry = HookRegistry()

def my_hook(context, data):
    print("Hook executed!")
    return data

registry.register_hook('my-plugin', HookType.PRE_EXECUTION, my_hook)
```

### Create Simple Plugin

```python
from plugins.api import create_command_plugin

def my_execute(self, context):
    return CommandAPI.success_result(
        self.metadata.name,
        {"message": "Works!"}
    )

MyPlugin = create_command_plugin(
    name='simple',
    version='1.0.0',
    description='Simple plugin',
    author='Me',
    execute_func=my_execute
)
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory
2. **Read Dev Guide**: See `docs/PLUGIN_DEVELOPMENT_GUIDE.md`
3. **Create Plugin**: Use templates or examples as starting point
4. **Test Plugin**: Run directly or via manager
5. **Share Plugin**: Publish as pip package or share directory

## Support Resources

- **Main Documentation**: `README.md`
- **Development Guide**: `docs/PLUGIN_DEVELOPMENT_GUIDE.md`
- **System Summary**: `PLUGIN_SYSTEM_SUMMARY.md`
- **Example Code**: `examples/` directory
- **Test Code**: `tests/` directory
- **API Reference**: `api/` directory

## Version

Plugin System Version: 1.0.0
Framework Version: 2.0.0
Last Updated: 2025-09-29