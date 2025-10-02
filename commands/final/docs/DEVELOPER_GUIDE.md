# Developer Guide - Claude Code Command Executor Framework

> Complete guide for contributors and plugin developers

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Development Setup](#development-setup)
4. [Contributing](#contributing)
5. [Plugin Development](#plugin-development)
6. [Workflow Development](#workflow-development)
7. [Testing](#testing)
8. [Code Standards](#code-standards)

---

## System Architecture

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   Claude Code CLI Interface                    │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────┴──────────────────────────────────┐
│              Command Executor Framework (Core)                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Command Registry & Dispatcher                            │ │
│  └──────────────────────────┬───────────────────────────────┘ │
│                             │                                  │
│  ┌──────────────────────────┴───────────────────────────────┐ │
│  │  Agent System (23 Specialized Agents)                    │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  Orchestrator (Coordinates all agents)             │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────┬──────────┬──────────┬─────────────────┐   │ │
│  │  │  Core    │Scientific│  AI/ML   │  Engineering    │   │ │
│  │  │  (3)     │  (4)     │  (3)     │   + Domain (13)│   │ │
│  │  └──────────┴──────────┴──────────┴─────────────────┘   │ │
│  └────────────────────────────────────────────────────────┘  │
│                             │                                  │
│  ┌──────────────┬───────────┴──────────┬────────────────┐   │
│  │  Workflow    │   Plugin System      │   UX System    │   │
│  │  Engine      │   (Extensibility)    │   (Phase 6)    │   │
│  └──────────────┴──────────────────────┴────────────────┘   │
└────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┴──────────────────────────────────┐
│  Integration Layer (Git, GitHub, CI/CD, IDEs, Tools)          │
└───────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
User Command → Command Dispatcher → Agent Selection → Task Execution
                    ↓                      ↓                ↓
              Workflow Engine        Orchestrator      Multi-Agent
                    ↓                      ↓            Execution
              Plugin System          Result Synthesis       ↓
                    ↓                      ↓           Output & UX
              Configuration         Cache Management        ↓
                    ↓                      ↓           User Feedback
              Integration Layer      Error Handling
```

---

## Core Components

### 1. Command Executor Framework

Located in: `/Users/b80985/.claude/commands/executors/`

**Purpose:** Core execution engine that handles all command processing

**Key Classes:**

```python
# executors/base_executor.py
class BaseExecutor:
    """Base class for all command executors"""

    def execute(self, args: CommandArgs) -> ExecutionResult:
        """Main execution method"""
        pass

    def validate_args(self, args: CommandArgs) -> bool:
        """Validate command arguments"""
        pass

    def select_agents(self, args: CommandArgs) -> List[Agent]:
        """Select appropriate agents for task"""
        pass

# executors/command_registry.py
class CommandRegistry:
    """Registry of all available commands"""

    def register(self, command: Command) -> None:
        """Register a new command"""
        pass

    def get(self, name: str) -> Command:
        """Get command by name"""
        pass

# executors/dispatcher.py
class CommandDispatcher:
    """Dispatches commands to appropriate executors"""

    def dispatch(self, command: str, args: dict) -> ExecutionResult:
        """Dispatch command for execution"""
        pass
```

**Execution Flow:**

1. **Command Reception** - CLI receives command
2. **Parsing** - Arguments parsed and validated
3. **Agent Selection** - Appropriate agents selected
4. **Execution** - Command executed with selected agents
5. **Result Processing** - Results synthesized and formatted
6. **Output** - Results presented to user

### 2. Agent System

Located in: `/Users/b80985/.claude/commands/ai_features/agents/`

**Purpose:** 23 specialized AI agents providing domain expertise

**Key Classes:**

```python
# ai_features/agents/base_agent.py
class BaseAgent:
    """Base class for all agents"""

    name: str
    expertise: List[str]
    capabilities: List[str]

    def analyze(self, context: Context) -> Analysis:
        """Analyze given context"""
        pass

    def suggest(self, analysis: Analysis) -> Suggestions:
        """Provide suggestions based on analysis"""
        pass

    def implement(self, suggestions: Suggestions) -> Implementation:
        """Implement suggestions"""
        pass

# ai_features/agents/orchestrator.py
class OrchestratorAgent(BaseAgent):
    """Coordinates multiple agents"""

    def coordinate(self, agents: List[Agent], task: Task) -> Coordination:
        """Coordinate agent collaboration"""
        pass

    def synthesize(self, results: List[Result]) -> SynthesizedResult:
        """Synthesize results from multiple agents"""
        pass

# ai_features/agents/agent_selector.py
class AgentSelector:
    """Intelligent agent selection"""

    def select(self, task: Task, strategy: str = "auto") -> List[Agent]:
        """Select agents for task"""
        pass

    def optimize(self, agents: List[Agent], constraints: Constraints) -> List[Agent]:
        """Optimize agent selection"""
        pass
```

**Agent Categories:**

```python
# Agent registry
AGENT_REGISTRY = {
    "core": [OrchestratorAgent, QualityAssuranceAgent, DevOpsAgent],
    "scientific": [ScientificComputingAgent, PerformanceEngineerAgent,
                   GPUSpecialistAgent, ResearchScientistAgent],
    "ai_ml": [AIMLEngineerAgent, JAXSpecialistAgent, ModelOptimizationAgent],
    "engineering": [BackendEngineerAgent, FrontendEngineerAgent,
                    SecurityEngineerAgent, DatabaseEngineerAgent,
                    CloudArchitectAgent],
    "domain": [PythonExpertAgent, JuliaExpertAgent, JavaScriptExpertAgent,
               DocumentationAgent, CodeReviewerAgent, RefactoringAgent,
               TestingAgent, QuantumComputingAgent]
}
```

### 3. Workflow Engine

Located in: `/Users/b80985/.claude/commands/workflows/`

**Purpose:** Execute complex multi-step workflows

**Key Classes:**

```python
# workflows/workflow_engine.py
class WorkflowEngine:
    """Executes workflow definitions"""

    def load_workflow(self, path: str) -> Workflow:
        """Load workflow from YAML"""
        pass

    def execute(self, workflow: Workflow) -> WorkflowResult:
        """Execute workflow"""
        pass

    def handle_error(self, error: WorkflowError, step: Step) -> ErrorHandling:
        """Handle workflow errors"""
        pass

# workflows/workflow_parser.py
class WorkflowParser:
    """Parses YAML workflow definitions"""

    def parse(self, yaml_content: str) -> Workflow:
        """Parse workflow YAML"""
        pass

    def validate(self, workflow: Workflow) -> ValidationResult:
        """Validate workflow structure"""
        pass

# workflows/step_executor.py
class StepExecutor:
    """Executes individual workflow steps"""

    def execute_step(self, step: Step, context: Context) -> StepResult:
        """Execute single step"""
        pass

    def check_conditions(self, step: Step, context: Context) -> bool:
        """Check step conditions"""
        pass
```

### 4. Plugin System

Located in: `/Users/b80985/.claude/commands/plugins/`

**Purpose:** Extensibility through plugins

**Key Classes:**

```python
# plugins/plugin_manager.py
class PluginManager:
    """Manages plugin lifecycle"""

    def load_plugin(self, path: str) -> Plugin:
        """Load plugin from path"""
        pass

    def enable_plugin(self, name: str) -> None:
        """Enable plugin"""
        pass

    def disable_plugin(self, name: str) -> None:
        """Disable plugin"""
        pass

# plugins/plugin_base.py
class Plugin:
    """Base class for plugins"""

    name: str
    version: str
    description: str

    def initialize(self) -> None:
        """Initialize plugin"""
        pass

    def register_commands(self) -> List[Command]:
        """Register plugin commands"""
        pass

    def register_agents(self) -> List[Agent]:
        """Register plugin agents"""
        pass
```

### 5. UX System (Phase 6)

Located in: `/Users/b80985/.claude/commands/ux/`

**Purpose:** Rich user experience with animations and progress tracking

**Key Classes:**

```python
# ux/rich_console.py
class RichConsole:
    """Enhanced console output"""

    def display_progress(self, task: Task, progress: float) -> None:
        """Display progress bar"""
        pass

    def display_results(self, results: Results) -> None:
        """Display formatted results"""
        pass

# ux/animations.py
class AnimationSystem:
    """Console animations"""

    def animate_thinking(self, message: str) -> None:
        """Animate thinking process"""
        pass

    def animate_progress(self, steps: List[str]) -> None:
        """Animate workflow progress"""
        pass
```

---

## Development Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Git
git --version

# Required tools
pip install poetry  # Package management
pip install pre-commit  # Git hooks
```

### Clone and Setup

```bash
# Clone repository (adjust path as needed)
cd ~/.claude/commands

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Project Structure

```
.claude/commands/
├── executors/              # Execution framework
│   ├── base_executor.py
│   ├── command_registry.py
│   ├── dispatcher.py
│   └── implementations/    # Command implementations
├── ai_features/            # AI and agents
│   ├── agents/            # 23 agent implementations
│   ├── reasoning/         # AI reasoning
│   └── analysis/          # Code analysis
├── workflows/             # Workflow system
│   ├── engine/            # Workflow engine
│   ├── definitions/       # Workflow YAML files
│   └── templates/         # Workflow templates
├── plugins/               # Plugin system
│   ├── core/             # Core plugin functionality
│   ├── registry/         # Plugin registry
│   └── examples/         # Example plugins
├── ux/                   # User experience
│   ├── console/          # Console UI
│   ├── animations/       # Animations
│   └── progress/         # Progress tracking
├── cicd/                 # CI/CD integration
├── validation/           # Validation tools
├── docs/                 # Additional documentation
├── final/                # Phase 7 documentation
│   ├── docs/             # Master documentation
│   ├── tutorials/        # Tutorial library
│   ├── examples/         # Code examples
│   └── release/          # Release materials
└── tests/                # Test suites
    ├── unit/
    ├── integration/
    └── e2e/
```

---

## Contributing

### Contribution Workflow

1. **Fork and Clone**
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/claude-commands.git
cd claude-commands
```

2. **Create Branch**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bugfix branch
git checkout -b bugfix/issue-number
```

3. **Make Changes**
```bash
# Make your changes
# Follow code standards (see below)
# Add tests
# Update documentation
```

4. **Test**
```bash
# Run tests
pytest

# Run quality checks
/check-code-quality --auto-fix

# Run all validation
/run-all-tests --coverage
```

5. **Commit**
```bash
# Commit with conventional commit message
git add .
git commit -m "feat: add new feature"
# Or use /commit --ai-message
```

6. **Push and PR**
```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Commit Message Format

Follow Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
```
feat(agents): add quantum computing agent

fix(executor): resolve parallel execution bug

docs(tutorial): add scientific computing tutorial

test(workflow): add workflow engine tests
```

---

## Plugin Development

### Plugin Structure

```
my-plugin/
├── __init__.py           # Plugin entry point
├── plugin.yml            # Plugin metadata
├── commands/             # Custom commands
│   └── my_command.py
├── agents/               # Custom agents
│   └── my_agent.py
├── workflows/            # Custom workflows
│   └── my_workflow.yml
├── tests/                # Plugin tests
│   └── test_plugin.py
└── README.md             # Plugin documentation
```

### Creating a Plugin

#### 1. Plugin Entry Point

```python
# my_plugin/__init__.py
from claude_commands.plugin import Plugin, command, agent, workflow

class MyPlugin(Plugin):
    """My custom plugin"""

    name = "my-plugin"
    version = "1.0.0"
    description = "My custom plugin for specific functionality"
    author = "Your Name"

    def initialize(self):
        """Initialize plugin"""
        self.config = self.load_config()
        self.logger = self.get_logger()

    def cleanup(self):
        """Cleanup on plugin disable"""
        pass

    # Register custom command
    @command("my-custom-command")
    def my_command(self, args):
        """
        My custom command implementation

        Args:
            args: Command arguments

        Returns:
            Command execution result
        """
        # Implementation
        return {
            "status": "success",
            "message": "Command executed",
            "data": {}
        }

    # Register custom agent
    @agent("MyCustomAgent")
    def my_agent(self):
        """
        My custom agent

        Returns:
            Agent configuration
        """
        return {
            "name": "MyCustomAgent",
            "expertise": ["custom-domain"],
            "capabilities": [
                "analyze-custom-patterns",
                "suggest-custom-improvements"
            ]
        }

    # Register custom workflow
    @workflow("my-workflow")
    def my_workflow(self):
        """
        My custom workflow

        Returns:
            Workflow definition
        """
        return {
            "name": "my-workflow",
            "steps": [
                {"command": "my-custom-command", "args": {}}
            ]
        }
```

#### 2. Plugin Metadata

```yaml
# plugin.yml
name: my-plugin
version: 1.0.0
description: My custom plugin
author: Your Name
email: your.email@example.com
license: MIT
homepage: https://github.com/username/my-plugin

# Dependencies
dependencies:
  claude-commands: ">=1.0.0"
  other-package: ">=2.0.0"

# Capabilities
commands:
  - my-custom-command

agents:
  - MyCustomAgent

workflows:
  - my-workflow

# Configuration
config:
  default_setting: value

# Permissions
permissions:
  - read_files
  - write_files
  - execute_commands
```

#### 3. Custom Command Implementation

```python
# commands/my_command.py
from claude_commands.executor import BaseExecutor
from claude_commands.types import CommandArgs, ExecutionResult

class MyCommandExecutor(BaseExecutor):
    """Custom command executor"""

    def execute(self, args: CommandArgs) -> ExecutionResult:
        """Execute custom command"""

        # Validate arguments
        if not self.validate_args(args):
            return ExecutionResult(
                success=False,
                error="Invalid arguments"
            )

        # Select agents
        agents = self.select_agents(args)

        # Execute with agents
        results = []
        for agent in agents:
            result = agent.execute(args)
            results.append(result)

        # Synthesize results
        final_result = self.synthesize_results(results)

        return ExecutionResult(
            success=True,
            data=final_result,
            agents_used=[a.name for a in agents]
        )

    def validate_args(self, args: CommandArgs) -> bool:
        """Validate command arguments"""
        required_args = ["target"]
        return all(arg in args for arg in required_args)

    def select_agents(self, args: CommandArgs) -> List[Agent]:
        """Select appropriate agents"""
        # Custom agent selection logic
        return [MyCustomAgent()]
```

#### 4. Custom Agent Implementation

```python
# agents/my_agent.py
from claude_commands.agents import BaseAgent

class MyCustomAgent(BaseAgent):
    """Custom agent implementation"""

    name = "MyCustomAgent"
    expertise = ["custom-domain", "specialized-task"]
    capabilities = [
        "analyze-custom-patterns",
        "suggest-improvements",
        "implement-fixes"
    ]

    def analyze(self, context):
        """Analyze given context"""
        # Custom analysis logic
        findings = []

        for file in context.files:
            # Analyze file
            issues = self.analyze_file(file)
            findings.extend(issues)

        return {
            "findings": findings,
            "recommendations": self.generate_recommendations(findings)
        }

    def suggest(self, analysis):
        """Generate suggestions"""
        suggestions = []

        for finding in analysis["findings"]:
            suggestion = self.create_suggestion(finding)
            suggestions.append(suggestion)

        return suggestions

    def implement(self, suggestions):
        """Implement suggestions"""
        implementations = []

        for suggestion in suggestions:
            if suggestion.auto_fixable:
                impl = self.auto_fix(suggestion)
                implementations.append(impl)

        return implementations
```

### Plugin Testing

```python
# tests/test_plugin.py
import pytest
from my_plugin import MyPlugin

@pytest.fixture
def plugin():
    """Plugin fixture"""
    plugin = MyPlugin()
    plugin.initialize()
    return plugin

def test_plugin_initialization(plugin):
    """Test plugin initializes correctly"""
    assert plugin.name == "my-plugin"
    assert plugin.version == "1.0.0"

def test_custom_command(plugin):
    """Test custom command"""
    result = plugin.my_command({"target": "test"})
    assert result["status"] == "success"

def test_custom_agent(plugin):
    """Test custom agent"""
    agent_config = plugin.my_agent()
    assert agent_config["name"] == "MyCustomAgent"
    assert "custom-domain" in agent_config["expertise"]
```

### Publishing Plugin

```bash
# Package plugin
python setup.py sdist bdist_wheel

# Test locally
pip install -e .

# Publish to PyPI
twine upload dist/*

# Or publish to Claude Commands registry
claude-commands publish my-plugin
```

---

## Workflow Development

### Workflow YAML Structure

```yaml
# workflow-template.yml
name: My Custom Workflow
description: Description of workflow purpose
version: 1.0.0
author: Your Name

# Parameters
parameters:
  target:
    type: string
    required: true
    description: Target directory or file
  coverage:
    type: integer
    default: 90
    description: Target test coverage percentage

# Steps
steps:
  # Step 1: Analysis
  - name: analyze
    command: check-code-quality
    args:
      language: ${language}
      target: ${target}
    on_failure: abort
    timeout: 300

  # Step 2: Fix issues
  - name: fix
    command: refactor-clean
    args:
      target: ${target}
      implement: true
    depends_on: [analyze]
    on_failure: continue

  # Step 3: Generate tests
  - name: test-gen
    command: generate-tests
    args:
      coverage: ${coverage}
      target: ${target}
    depends_on: [fix]
    condition: ${fix.success}

  # Step 4: Run tests
  - name: test-run
    command: run-all-tests
    args:
      auto-fix: true
      coverage: true
    depends_on: [test-gen]

  # Step 5: Verify
  - name: verify
    command: double-check
    args:
      description: "workflow completed successfully"
    depends_on: [test-run]

# Success criteria
success_criteria:
  - all_steps_passed: true
  - coverage_met: ${coverage}

# Outputs
outputs:
  quality_report: ${analyze.output}
  test_results: ${test-run.output}
  coverage: ${test-run.coverage}
```

### Advanced Workflow Features

#### Conditional Execution

```yaml
steps:
  - name: optimize
    command: optimize
    condition: ${analyze.performance_issues} > 0
    args:
      category: performance
```

#### Parallel Steps

```yaml
parallel:
  - name: quality-check
    command: check-code-quality
  - name: security-scan
    command: check-code-quality
    args:
      focus: security
  - name: type-check
    command: check-code-quality
    args:
      focus: types
```

#### Error Handling

```yaml
steps:
  - name: risky-operation
    command: some-command
    on_failure: retry
    retry:
      max_attempts: 3
      backoff: exponential
      initial_delay: 5
```

#### Dynamic Parameters

```yaml
steps:
  - name: analyze
    command: check-code-quality
    args:
      target: ${env.TARGET_DIR}
      language: ${detect_language()}
```

---

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_executors.py
│   ├── test_agents.py
│   └── test_workflows.py
├── integration/       # Integration tests
│   ├── test_commands.py
│   └── test_workflows.py
├── e2e/              # End-to-end tests
│   └── test_scenarios.py
├── fixtures/         # Test fixtures
└── conftest.py       # Pytest configuration
```

### Writing Tests

#### Unit Tests

```python
# tests/unit/test_agents.py
import pytest
from claude_commands.agents import QualityAssuranceAgent

class TestQualityAssuranceAgent:
    """Test QA agent"""

    @pytest.fixture
    def agent(self):
        """Agent fixture"""
        return QualityAssuranceAgent()

    def test_analyze(self, agent):
        """Test analysis capability"""
        context = {"files": ["test.py"]}
        result = agent.analyze(context)
        assert "findings" in result
        assert "recommendations" in result

    def test_suggest(self, agent):
        """Test suggestion generation"""
        analysis = {"findings": [{"type": "style", "severity": "low"}]}
        suggestions = agent.suggest(analysis)
        assert len(suggestions) > 0
```

#### Integration Tests

```python
# tests/integration/test_commands.py
import pytest
from claude_commands.dispatcher import CommandDispatcher

class TestCommandIntegration:
    """Integration tests for commands"""

    @pytest.fixture
    def dispatcher(self):
        """Dispatcher fixture"""
        return CommandDispatcher()

    def test_quality_workflow(self, dispatcher):
        """Test complete quality workflow"""
        # Step 1: Check quality
        result1 = dispatcher.dispatch("check-code-quality", {
            "target": "tests/fixtures/sample_code",
            "auto_fix": True
        })
        assert result1.success

        # Step 2: Generate tests
        result2 = dispatcher.dispatch("generate-tests", {
            "target": "tests/fixtures/sample_code",
            "coverage": 90
        })
        assert result2.success

        # Step 3: Run tests
        result3 = dispatcher.dispatch("run-all-tests", {
            "auto_fix": True
        })
        assert result3.success
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_agents.py

# Run with coverage
pytest --cov=claude_commands --cov-report=html

# Run specific test
pytest tests/unit/test_agents.py::TestQualityAssuranceAgent::test_analyze

# Run with verbose output
pytest -v

# Run with debugging
pytest --pdb
```

---

## Code Standards

### Python Style

Follow PEP 8 with these specifics:

```python
# Type hints required
def analyze_code(file_path: str, options: Dict[str, Any]) -> Analysis:
    """
    Analyze code quality

    Args:
        file_path: Path to file to analyze
        options: Analysis options

    Returns:
        Analysis results

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    pass

# Docstrings required for all public functions/classes
class QualityAnalyzer:
    """
    Code quality analyzer

    Attributes:
        config: Analyzer configuration
        metrics: Quality metrics
    """

    def __init__(self, config: Config):
        """Initialize analyzer"""
        self.config = config

# Use descriptive variable names
user_count = get_user_count()  # Good
uc = get_user_count()  # Bad

# Constants in CAPS
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300

# Use type hints
from typing import List, Dict, Optional, Union

def process_files(
    file_paths: List[str],
    options: Optional[Dict[str, Any]] = None
) -> List[ProcessedFile]:
    pass
```

### Documentation Standards

```python
# Module docstring
"""
Module for code quality analysis.

This module provides comprehensive code quality analysis including:
- Style checking
- Complexity analysis
- Type checking
- Security scanning
"""

# Class docstring
class Analyzer:
    """
    Code analyzer with multi-metric support.

    This class provides analysis across multiple dimensions:
    quality, performance, security, and maintainability.

    Attributes:
        config: Analyzer configuration
        metrics: Available metrics

    Example:
        >>> analyzer = Analyzer(config)
        >>> result = analyzer.analyze("file.py")
        >>> print(result.score)
        85
    """

# Function docstring
def analyze_file(file_path: str, metrics: List[str]) -> AnalysisResult:
    """
    Analyze single file with specified metrics.

    Args:
        file_path: Path to file to analyze
        metrics: List of metric names to compute

    Returns:
        AnalysisResult containing scores and findings

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If metrics list is empty

    Example:
        >>> result = analyze_file("test.py", ["complexity", "style"])
        >>> result.score
        85
    """
```

### Error Handling

```python
# Use specific exceptions
class AnalysisError(Exception):
    """Error during code analysis"""
    pass

# Proper error handling
try:
    result = analyze_file(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise
except AnalysisError as e:
    logger.error(f"Analysis failed: {e}")
    return default_result()
finally:
    cleanup()

# Context managers
with open(file_path) as f:
    content = f.read()
```

### Testing Standards

```python
# Descriptive test names
def test_analyzer_handles_empty_file():
    """Test analyzer correctly handles empty file"""
    pass

def test_analyzer_detects_security_issues():
    """Test analyzer detects common security issues"""
    pass

# Use fixtures
@pytest.fixture
def sample_code():
    """Sample code for testing"""
    return """
    def example():
        return 42
    """

# Use parametrize for multiple inputs
@pytest.mark.parametrize("input,expected", [
    ("valid.py", True),
    ("invalid", False),
    ("test.py", True),
])
def test_file_validation(input, expected):
    """Test file validation"""
    assert is_valid_file(input) == expected
```

---

## Additional Resources

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Architecture](../ARCHITECTURE.md)** - Detailed architecture
- **[Contributing](../CONTRIBUTING.md)** - Contribution guidelines
- **[Plugin Development Guide](../../plugins/docs/PLUGIN_DEVELOPMENT_GUIDE.md)** - Detailed plugin guide
- **[Workflow Guide](../../workflows/TEMPLATE_GUIDE.md)** - Workflow development

---

**Version**: 1.0.0 | **Last Updated**: September 2025 | **Status**: Production Ready