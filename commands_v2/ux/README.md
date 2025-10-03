# UX Enhancement System

Comprehensive user experience enhancements for the Claude Code Command Executor Framework.

## Overview

The UX system provides beautiful, informative, and intelligent user interactions including:

- **Progress Tracking** - Rich progress bars, spinners, and live dashboards
- **Error Formatting** - Beautiful error messages with actionable suggestions
- **Command Recommendations** - ML-based intelligent command suggestions
- **Enhanced CLI** - Smart command-line interface with autocomplete
- **Accessibility** - Support for screen readers and color-blind modes

## Quick Start

### Progress Tracking

```python
from ux.progress.progress_tracker import ProgressTracker

tracker = ProgressTracker()

with tracker.live_progress():
    task = tracker.add_task("Processing files", total=100)

    for i in range(100):
        tracker.update(task, advance=1)
        # Do work...

    tracker.complete(task)
```

### Error Formatting

```python
from ux.errors.error_formatter import ErrorFormatter

formatter = ErrorFormatter()

try:
    # Some code that fails
    result = process_data()
except Exception as e:
    formatted = formatter.format_exception(e)
    formatter.print_error(formatted)
```

### Command Recommendations

```python
from ux.recommendations.command_recommender import CommandRecommender

recommender = CommandRecommender()

# Get recommendations based on context
recommendations = recommender.recommend(
    context={"project_type": "python"},
    recent_commands=["git add", "git commit"],
    goal="improve code quality"
)

for rec in recommendations:
    print(f"{rec.command} - {rec.reason}")
```

## Architecture

### Directory Structure

```
ux/
├── progress/           # Progress tracking system
│   ├── progress_tracker.py
│   ├── live_dashboard.py
│   └── step_tracker.py
├── errors/            # Error formatting and recovery
│   ├── error_formatter.py
│   ├── error_suggestions.py
│   └── error_recovery.py
├── recommendations/   # Command recommendation system
│   ├── command_recommender.py
│   ├── workflow_suggester.py
│   └── flag_recommender.py
├── core/             # UX framework
│   ├── ux_manager.py
│   ├── interactive_mode.py
│   └── output_formatter.py
├── cli/              # CLI enhancements
│   ├── smart_cli.py
│   └── command_builder.py
├── examples/         # Example scripts
│   ├── progress_example.py
│   ├── error_example.py
│   └── recommendation_example.py
└── docs/            # Documentation
    ├── PROGRESS_GUIDE.md
    ├── ERROR_GUIDE.md
    └── RECOMMENDATION_GUIDE.md
```

## Features

### 1. Progress Tracking

#### Basic Progress Bars

```python
tracker = ProgressTracker()

with tracker.live_progress():
    task = tracker.add_task("Processing", total=100)
    for i in range(100):
        tracker.update(task, advance=1)
```

#### Hierarchical Progress

```python
parent = tracker.add_task("Parent Task", total=3)
child1 = tracker.add_task("Child 1", total=50, parent_id=parent)
child2 = tracker.add_task("Child 2", total=50, parent_id=parent)
```

#### Step Tracking

```python
from ux.progress.step_tracker import StepTracker

tracker = StepTracker("Operation Name")
tracker.add_step("analyze", "Analyze code")
tracker.add_step("optimize", "Optimize code")
tracker.add_step("test", "Run tests")

tracker.start_step("analyze")
# Do work...
tracker.complete_step("analyze")
```

#### Live Dashboard

```python
from ux.progress.live_dashboard import LiveDashboard

dashboard = LiveDashboard()

with dashboard.live():
    dashboard.update_command("optimize", "Running")
    dashboard.add_agent("Scientific Agent")
    # Do work...
    dashboard.remove_agent("Scientific Agent")
```

### 2. Error Handling

#### Beautiful Error Messages

Errors are automatically formatted with:
- Color-coded severity levels
- Context information (file, line, function)
- Code snippets with syntax highlighting
- Actionable suggestions
- Stack traces (when enabled)
- Documentation links

#### Error Suggestions

```python
from ux.errors.error_suggestions import suggest_fixes

suggestions = suggest_fixes(
    "No module named 'numpy'",
    category=ErrorCategory.DEPENDENCY
)

for sugg in suggestions:
    print(f"{sugg.title}: {sugg.command}")
```

#### Error Recovery

```python
from ux.errors.error_recovery import retry

@retry(max_attempts=3)
def unstable_operation():
    return api.call()
```

### 3. Command Recommendations

#### Context-Based

Recommendations based on:
- Project type (Python, JavaScript, etc.)
- Project structure (tests, CI, docs)
- Recent commands
- Usage patterns

#### Goal-Based

```python
recommendations = recommender.recommend(
    goal="improve code quality"
)
# Returns: /check-code-quality, /refactor-clean, etc.
```

#### Workflow Suggestions

```python
workflows = recommender.get_workflow_suggestions(
    goal="optimize performance"
)
# Returns: [[/optimize, /run-tests, /update-docs], ...]
```

### 4. UX Configuration

#### Configuration File

Create `~/.claude/ux_config.json`:

```json
{
  "theme": "dark",
  "animations": true,
  "progress_style": "bar",
  "verbosity": 1,
  "confirmation_prompts": true,
  "error_suggestions": true,
  "command_recommendations": true,
  "tutorial_mode": false,
  "accessibility_mode": false,
  "color_enabled": true,
  "output_format": "text"
}
```

#### Programmatic Configuration

```python
from ux.core.ux_manager import UXManager, ThemeMode, VerbosityLevel

ux = UXManager()
ux.config.theme = ThemeMode.DARK
ux.config.verbosity = VerbosityLevel.VERBOSE
ux.config.animations_enabled = True
ux.save_config()
```

## Integration

### With Command Executor

```python
from ux.progress.progress_tracker import get_global_tracker
from ux.errors.error_formatter import ErrorFormatter

class CommandExecutor:
    def __init__(self):
        self.tracker = get_global_tracker()
        self.error_formatter = ErrorFormatter()

    def execute(self, command):
        task = self.tracker.add_task(f"Executing {command}")

        try:
            # Execute command...
            self.tracker.complete(task)
        except Exception as e:
            formatted = self.error_formatter.format_exception(e)
            self.error_formatter.print_error(formatted)
```

### With Agents

```python
from ux.progress.live_dashboard import get_global_dashboard

class Agent:
    def execute(self):
        dashboard = get_global_dashboard()
        dashboard.add_agent(self.name)

        try:
            # Do work...
            pass
        finally:
            dashboard.remove_agent(self.name)
```

## Examples

Run the example scripts to see UX features in action:

```bash
# Progress tracking examples
python ux/examples/progress_example.py

# Error handling examples
python ux/examples/error_example.py

# Command recommendation examples
python ux/examples/recommendation_example.py
```

## Dependencies

Required:
- `rich` - Beautiful terminal output
- `click` - Enhanced CLI
- `psutil` - System resource monitoring

Optional:
- `prompt_toolkit` - Interactive features
- `scikit-learn` - ML-based recommendations

Install all dependencies:

```bash
pip install rich click psutil prompt-toolkit scikit-learn
```

## Accessibility

The UX system supports:

### Screen Readers

```python
ux.config.accessibility_mode = True
```

This disables:
- Animations
- Live updates
- Color-only information

### Color-Blind Mode

```python
ux.config.color_enabled = False
```

Uses patterns and symbols instead of colors.

### Quiet Mode

```python
ux.config.verbosity = VerbosityLevel.QUIET
```

Minimal output for scripting.

## Performance

The UX system is designed to be fast:

- **No slowdown** - UX operations happen asynchronously
- **Efficient rendering** - Only updates changed elements
- **Caching** - Reuses formatted output
- **Optional** - Can be completely disabled

Disable for maximum performance:

```python
tracker = ProgressTracker(enabled=False)
formatter = ErrorFormatter(enabled=False)
```

## Customization

### Custom Themes

```python
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green"
})

console = Console(theme=custom_theme)
```

### Custom Error Patterns

```python
from ux.errors.error_suggestions import ErrorSuggestionEngine

engine = ErrorSuggestionEngine()
engine.patterns.append(ErrorPattern(
    pattern=r"custom error pattern",
    category=ErrorCategory.CUSTOM,
    suggestions=[...]
))
```

### Custom Recommendations

```python
from ux.recommendations.command_recommender import CommandRecommender

recommender = CommandRecommender()

# Override recommendation logic
def custom_recommend(context):
    # Your logic here
    return recommendations

recommender.recommend = custom_recommend
```

## Testing

Run tests:

```bash
pytest ux/tests/
```

Run specific test suite:

```bash
pytest ux/tests/test_progress.py
pytest ux/tests/test_errors.py
pytest ux/tests/test_recommendations.py
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Support

- Documentation: https://docs.claude-commands.dev/ux
- Issues: https://github.com/org/repo/issues
- Discussions: https://github.com/org/repo/discussions

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.