# Claude Code Plugin System

Extensible plugin architecture for the command executor framework.

## Overview

The plugin system enables custom extensions to the Claude Code command executor framework through:

- **Command Plugins**: Add new slash commands
- **Agent Plugins**: Add custom analysis agents
- **Integration Plugins**: Connect to external services
- **Validator Plugins**: Add custom validation logic
- **Cache Provider Plugins**: Implement custom cache backends
- **Reporter Plugins**: Add custom report formats

## Quick Start

### 1. Hello World Plugin

Create a simple command plugin:

```bash
# Create plugin directory
mkdir -p ~/.claude/commands/plugins/my-plugin
cd ~/.claude/commands/plugins/my-plugin

# Create manifest
cat > plugin.json << 'EOF'
{
  "name": "my-plugin",
  "version": "1.0.0",
  "type": "command",
  "description": "My first plugin",
  "author": "Your Name"
}
EOF

# Create plugin code
cat > plugin.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.plugin_base import CommandPlugin, PluginContext
from api.command_api import CommandAPI

class MyPlugin(CommandPlugin):
    def load(self):
        return True

    def execute(self, context):
        return CommandAPI.success_result(
            self.metadata.name,
            {"message": "Hello from my plugin!"}
        )

    def get_command_info(self):
        return {
            "name": "my-plugin",
            "description": "My first plugin"
        }
EOF

# Test it
python plugin.py
```

### 2. Load Plugins in Framework

```python
from core.plugin_manager import PluginManager

# Initialize plugin system
manager = PluginManager()
manager.initialize()

# Get plugin
plugin = manager.get_plugin('my-plugin')

# Execute plugin
context = PluginContext(...)
result = plugin.execute(context)
print(result.data)
```

## Directory Structure

```
plugins/
├── core/                      # Core plugin system
│   ├── plugin_base.py        # Base classes
│   ├── plugin_manager.py     # Plugin management
│   ├── plugin_hooks.py       # Hook system
│   ├── plugin_registry.py    # Plugin registry
│   └── plugin_validator.py   # Validation
├── api/                       # Plugin APIs
│   ├── command_api.py        # Command plugin API
│   ├── agent_api.py          # Agent plugin API
│   └── ...
├── examples/                  # Example plugins
│   ├── example_hello_world/
│   ├── example_custom_analyzer/
│   ├── example_slack_integration/
│   └── ...
├── templates/                 # Plugin templates
│   ├── command_plugin_template/
│   ├── agent_plugin_template/
│   └── integration_plugin_template/
├── utils/                     # Utilities
│   ├── plugin_loader.py
│   └── plugin_packager.py
├── config/                    # Configuration
│   ├── plugin_manifest_schema.json
│   └── plugins.json
├── docs/                      # Documentation
│   ├── PLUGIN_DEVELOPMENT_GUIDE.md
│   ├── PLUGIN_API_REFERENCE.md
│   └── PLUGIN_EXAMPLES.md
└── tests/                     # Tests
    ├── test_plugin_manager.py
    └── test_example_plugins.py
```

## Plugin Types

### Command Plugins

Add custom commands to the framework.

**Example:** Custom deployment command

```python
class DeployPlugin(CommandPlugin):
    def execute(self, context):
        # Deploy logic
        return CommandAPI.success_result(
            self.metadata.name,
            {"deployed": True}
        )
```

### Agent Plugins

Add custom agents to the multi-agent system.

**Example:** Security analysis agent

```python
class SecurityAgentPlugin(AgentPlugin):
    def analyze(self, context):
        # Security analysis
        return {
            'findings': ['No vulnerabilities found'],
            'recommendations': []
        }
```

### Integration Plugins

Connect to external services.

**Example:** Slack notifications

```python
class SlackPlugin(IntegrationPlugin):
    def send(self, data):
        # Send to Slack
        return True
```

## Example Plugins

The `examples/` directory contains fully functional example plugins:

1. **hello-world**: Simple command plugin
2. **custom-analyzer**: Code analysis with metrics
3. **deployment**: Deployment automation
4. **domain-expert**: Custom agent for domain analysis
5. **slack-integration**: Slack notifications
6. **jira-integration**: JIRA issue tracking
7. **metrics-export**: Export metrics to Prometheus
8. **security-scanner**: Security vulnerability scanning
9. **license-checker**: License compliance checking
10. **dependency-audit**: Dependency vulnerability scanning

## Plugin Features

### Configuration

Plugins can be configured through:

1. **Plugin manifest** (`plugin.json`)
2. **Global config** (`~/.claude/commands/plugins/config/plugins.json`)
3. **Runtime config** (passed at initialization)

### Hooks

Plugins can register hooks to extend framework behavior:

```python
def load(self):
    self.register_hook(HookType.PRE_EXECUTION, self.pre_exec_hook)
    self.register_hook(HookType.POST_EXECUTION, self.post_exec_hook)
    return True

def pre_exec_hook(self, context, data):
    # Called before execution
    return data
```

Available hooks:
- `PRE_EXECUTION`: Before command execution
- `POST_EXECUTION`: After command execution
- `PRE_VALIDATION`: Before validation
- `POST_VALIDATION`: After validation
- `PRE_AGENT`: Before agent execution
- `POST_AGENT`: After agent execution
- `ON_ERROR`: On error
- `ON_SUCCESS`: On success

### State Management

Plugins can maintain state:

```python
def execute(self, context):
    # Get state
    counter = self.get_state('counter', 0)

    # Update state
    self.set_state('counter', counter + 1)

    return result
```

### Dependency Management

Plugins can declare dependencies:

```json
{
  "dependencies": ["requests>=2.28.0", "pyyaml"]
}
```

## Documentation

- **[Development Guide](docs/PLUGIN_DEVELOPMENT_GUIDE.md)**: Complete guide to plugin development
- **[API Reference](docs/PLUGIN_API_REFERENCE.md)**: API documentation
- **[Examples](docs/PLUGIN_EXAMPLES.md)**: Plugin examples with explanations
- **[Best Practices](docs/PLUGIN_BEST_PRACTICES.md)**: Best practices and patterns
- **[Security](docs/PLUGIN_SECURITY.md)**: Security considerations

## Testing

Run plugin tests:

```bash
# Run all tests
python -m pytest plugins/tests/

# Run specific test
python plugins/tests/test_plugin_manager.py

# Test example plugin
python plugins/examples/example_hello_world/plugin.py
```

## Installation Methods

### Method 1: Directory Plugin

Place plugin in one of:
- `~/.claude/commands/plugins/`
- `~/.claude/plugins/`
- `./plugins/` (project-local)

### Method 2: pip-Installable

Create `setup.py` with entry point:

```python
setup(
    name='claude-code-my-plugin',
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

## Security

Plugins run with security features:

1. **Sandboxing**: Isolated execution environment
2. **Permissions**: Required permissions system
3. **Validation**: Plugin validation before loading
4. **Code signing**: Optional code signing (coming soon)

## Contributing

Contributions are welcome! To contribute:

1. Create a plugin following the development guide
2. Add tests
3. Add documentation
4. Submit a pull request

## Support

- **Issues**: Report bugs or request features
- **Discussions**: Ask questions
- **Examples**: Check `plugins/examples/`

## License

Plugin system is part of the Claude Code Framework.

Individual plugins may have different licenses - check each plugin's manifest.