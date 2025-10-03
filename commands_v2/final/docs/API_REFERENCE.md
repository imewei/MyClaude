# API Reference - Claude Code Command Executor Framework

> Complete API documentation for developers

## Overview

This reference documents all public APIs for the Claude Code Command Executor Framework. Use these APIs for plugin development, custom integrations, and extending the system.

---

## Table of Contents

1. [Executor Framework API](#executor-framework-api)
2. [Agent System API](#agent-system-api)
3. [Workflow API](#workflow-api)
4. [Plugin API](#plugin-api)
5. [UX System API](#ux-system-api)
6. [Utility APIs](#utility-apis)

---

## Executor Framework API

### BaseExecutor

Base class for all command executors.

```python
from claude_commands.executor import BaseExecutor

class BaseExecutor:
    """Base class for command executors"""

    def execute(self, args: CommandArgs) -> ExecutionResult:
        """
        Execute command with given arguments.

        Args:
            args: Command arguments and options

        Returns:
            ExecutionResult: Result of execution including status, data, errors

        Raises:
            ExecutionError: If execution fails critically
        """

    def validate_args(self, args: CommandArgs) -> ValidationResult:
        """
        Validate command arguments.

        Args:
            args: Arguments to validate

        Returns:
            ValidationResult: Validation result with errors if any
        """

    def select_agents(
        self,
        args: CommandArgs,
        strategy: str = "auto"
    ) -> List[Agent]:
        """
        Select appropriate agents for task.

        Args:
            args: Command arguments
            strategy: Selection strategy ("auto", "explicit", "intelligent")

        Returns:
            List[Agent]: Selected agents
        """

    def synthesize_results(
        self,
        results: List[AgentResult]
    ) -> SynthesizedResult:
        """
        Synthesize results from multiple agents.

        Args:
            results: Results from individual agents

        Returns:
            SynthesizedResult: Unified result
        """
```

### CommandRegistry

Registry for all available commands.

```python
from claude_commands.executor import CommandRegistry

class CommandRegistry:
    """Command registry"""

    @staticmethod
    def register(
        name: str,
        executor: Type[BaseExecutor],
        metadata: CommandMetadata
    ) -> None:
        """
        Register a new command.

        Args:
            name: Command name
            executor: Executor class
            metadata: Command metadata
        """

    @staticmethod
    def get(name: str) -> Command:
        """
        Get command by name.

        Args:
            name: Command name

        Returns:
            Command: Command instance

        Raises:
            CommandNotFoundError: If command doesn't exist
        """

    @staticmethod
    def list_commands() -> List[str]:
        """Get list of all registered commands"""

    @staticmethod
    def unregister(name: str) -> None:
        """Unregister command"""
```

### CommandDispatcher

Dispatches commands to appropriate executors.

```python
from claude_commands.executor import CommandDispatcher

class CommandDispatcher:
    """Command dispatcher"""

    def dispatch(
        self,
        command: str,
        args: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Dispatch command for execution.

        Args:
            command: Command name
            args: Command arguments

        Returns:
            ExecutionResult: Execution result

        Raises:
            CommandNotFoundError: If command doesn't exist
            ExecutionError: If execution fails
        """

    def validate(self, command: str, args: Dict[str, Any]) -> bool:
        """Validate command and arguments"""

    def get_metadata(self, command: str) -> CommandMetadata:
        """Get command metadata"""
```

---

## Agent System API

### BaseAgent

Base class for all agents.

```python
from claude_commands.agents import BaseAgent

class BaseAgent:
    """Base class for agents"""

    name: str
    expertise: List[str]
    capabilities: List[str]

    def analyze(self, context: Context) -> Analysis:
        """
        Analyze given context.

        Args:
            context: Analysis context including files, config, etc.

        Returns:
            Analysis: Analysis results with findings and metrics
        """

    def suggest(self, analysis: Analysis) -> List[Suggestion]:
        """
        Generate suggestions based on analysis.

        Args:
            analysis: Analysis results

        Returns:
            List[Suggestion]: Suggested improvements
        """

    def implement(
        self,
        suggestions: List[Suggestion]
    ) -> Implementation:
        """
        Implement suggestions.

        Args:
            suggestions: Suggestions to implement

        Returns:
            Implementation: Implementation results
        """

    def evaluate(self, implementation: Implementation) -> Evaluation:
        """Evaluate implementation quality"""
```

### OrchestratorAgent

Coordinates multiple agents.

```python
from claude_commands.agents import OrchestratorAgent

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent for multi-agent coordination"""

    def coordinate(
        self,
        agents: List[Agent],
        task: Task
    ) -> Coordination:
        """
        Coordinate multiple agents.

        Args:
            agents: Agents to coordinate
            task: Task to execute

        Returns:
            Coordination: Coordination plan and execution
        """

    def synthesize(
        self,
        results: List[AgentResult]
    ) -> SynthesizedResult:
        """
        Synthesize results from multiple agents.

        Args:
            results: Individual agent results

        Returns:
            SynthesizedResult: Unified result
        """

    def optimize_execution(
        self,
        agents: List[Agent],
        constraints: Constraints
    ) -> ExecutionPlan:
        """Optimize agent execution order and parallelization"""
```

### AgentSelector

Intelligent agent selection.

```python
from claude_commands.agents import AgentSelector

class AgentSelector:
    """Agent selection system"""

    def select(
        self,
        task: Task,
        strategy: str = "auto",
        constraints: Optional[Constraints] = None
    ) -> List[Agent]:
        """
        Select agents for task.

        Args:
            task: Task to execute
            strategy: Selection strategy
            constraints: Optional constraints

        Returns:
            List[Agent]: Selected agents
        """

    def select_by_expertise(
        self,
        required_expertise: List[str]
    ) -> List[Agent]:
        """Select agents by expertise"""

    def select_by_capability(
        self,
        required_capabilities: List[str]
    ) -> List[Agent]:
        """Select agents by capability"""
```

---

## Workflow API

### WorkflowEngine

Executes workflow definitions.

```python
from claude_commands.workflow import WorkflowEngine

class WorkflowEngine:
    """Workflow execution engine"""

    def load_workflow(self, path: str) -> Workflow:
        """
        Load workflow from file.

        Args:
            path: Path to workflow YAML file

        Returns:
            Workflow: Parsed workflow

        Raises:
            WorkflowParseError: If parsing fails
        """

    def execute(
        self,
        workflow: Workflow,
        context: Optional[Context] = None
    ) -> WorkflowResult:
        """
        Execute workflow.

        Args:
            workflow: Workflow to execute
            context: Optional execution context

        Returns:
            WorkflowResult: Execution result
        """

    def validate(self, workflow: Workflow) -> ValidationResult:
        """Validate workflow definition"""

    def handle_error(
        self,
        error: WorkflowError,
        step: Step
    ) -> ErrorHandling:
        """Handle workflow execution error"""
```

### Workflow

Workflow definition.

```python
from claude_commands.workflow import Workflow

class Workflow:
    """Workflow definition"""

    name: str
    version: str
    parameters: Dict[str, Parameter]
    steps: List[Step]
    success_criteria: List[Criterion]
    outputs: Dict[str, Output]

    def add_step(self, step: Step) -> None:
        """Add step to workflow"""

    def remove_step(self, name: str) -> None:
        """Remove step from workflow"""

    def get_step(self, name: str) -> Step:
        """Get step by name"""

    def validate(self) -> ValidationResult:
        """Validate workflow structure"""

    def to_yaml(self) -> str:
        """Export workflow to YAML"""

    @staticmethod
    def from_yaml(yaml_str: str) -> 'Workflow':
        """Create workflow from YAML"""
```

### Step

Workflow step definition.

```python
from claude_commands.workflow import Step

class Step:
    """Workflow step"""

    name: str
    command: str
    args: Dict[str, Any]
    depends_on: List[str]
    condition: Optional[str]
    on_failure: str  # "abort", "continue", "retry"
    timeout: int
    retry: Optional[RetryConfig]

    def execute(self, context: Context) -> StepResult:
        """Execute step"""

    def check_condition(self, context: Context) -> bool:
        """Check if step should execute"""

    def handle_failure(self, error: Exception) -> ErrorHandling:
        """Handle step failure"""
```

---

## Plugin API

### Plugin

Base class for plugins.

```python
from claude_commands.plugin import Plugin

class Plugin:
    """Base class for plugins"""

    name: str
    version: str
    description: str
    author: str

    def initialize(self) -> None:
        """Initialize plugin"""

    def cleanup(self) -> None:
        """Cleanup on disable"""

    def register_commands(self) -> List[Command]:
        """Register plugin commands"""

    def register_agents(self) -> List[Agent]:
        """Register plugin agents"""

    def register_workflows(self) -> List[Workflow]:
        """Register plugin workflows"""

    def get_config(self) -> PluginConfig:
        """Get plugin configuration"""

    def set_config(self, config: PluginConfig) -> None:
        """Set plugin configuration"""
```

### PluginManager

Manages plugin lifecycle.

```python
from claude_commands.plugin import PluginManager

class PluginManager:
    """Plugin manager"""

    def load_plugin(self, path: str) -> Plugin:
        """Load plugin from path"""

    def unload_plugin(self, name: str) -> None:
        """Unload plugin"""

    def enable_plugin(self, name: str) -> None:
        """Enable plugin"""

    def disable_plugin(self, name: str) -> None:
        """Disable plugin"""

    def list_plugins(self) -> List[PluginInfo]:
        """List all plugins"""

    def get_plugin(self, name: str) -> Plugin:
        """Get plugin by name"""

    def update_plugin(self, name: str) -> None:
        """Update plugin to latest version"""
```

### Decorators

Plugin decorators for registering components.

```python
from claude_commands.plugin import command, agent, workflow

@command("command-name")
def my_command(args: Dict[str, Any]) -> ExecutionResult:
    """Command implementation"""
    pass

@agent("AgentName")
class MyAgent(BaseAgent):
    """Agent implementation"""
    pass

@workflow("workflow-name")
def my_workflow() -> Workflow:
    """Workflow definition"""
    pass
```

---

## UX System API

### RichConsole

Enhanced console output.

```python
from claude_commands.ux import RichConsole

class RichConsole:
    """Rich console for enhanced output"""

    def display_progress(
        self,
        task: Task,
        progress: float,
        message: str
    ) -> None:
        """Display progress bar"""

    def display_results(
        self,
        results: Results,
        format: str = "table"
    ) -> None:
        """Display formatted results"""

    def display_tree(self, data: Dict[str, Any]) -> None:
        """Display data as tree"""

    def display_table(
        self,
        data: List[Dict],
        columns: List[str]
    ) -> None:
        """Display data as table"""

    def prompt(
        self,
        message: str,
        choices: Optional[List[str]] = None
    ) -> str:
        """Interactive prompt"""

    def confirm(self, message: str) -> bool:
        """Confirmation prompt"""
```

### AnimationSystem

Console animations.

```python
from claude_commands.ux import AnimationSystem

class AnimationSystem:
    """Animation system"""

    def animate_thinking(self, message: str) -> None:
        """Animate thinking process"""

    def animate_progress(
        self,
        steps: List[str],
        current: int
    ) -> None:
        """Animate workflow progress"""

    def animate_spinner(self, message: str) -> None:
        """Display spinner animation"""

    def animate_success(self, message: str) -> None:
        """Display success animation"""

    def animate_error(self, message: str) -> None:
        """Display error animation"""
```

---

## Utility APIs

### FileAnalyzer

File and codebase analysis.

```python
from claude_commands.utils import FileAnalyzer

class FileAnalyzer:
    """File analysis utilities"""

    @staticmethod
    def analyze_file(path: str) -> FileAnalysis:
        """Analyze single file"""

    @staticmethod
    def analyze_directory(
        path: str,
        recursive: bool = True
    ) -> DirectoryAnalysis:
        """Analyze directory"""

    @staticmethod
    def detect_language(path: str) -> str:
        """Detect programming language"""

    @staticmethod
    def extract_ast(path: str) -> AST:
        """Extract abstract syntax tree"""

    @staticmethod
    def analyze_complexity(path: str) -> ComplexityMetrics:
        """Analyze code complexity"""
```

### CacheManager

Cache management.

```python
from claude_commands.utils import CacheManager

class CacheManager:
    """Cache management"""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache"""

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry"""

    def clear(self) -> None:
        """Clear entire cache"""

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
```

### Logger

Logging utilities.

```python
from claude_commands.utils import Logger

class Logger:
    """Logging utilities"""

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get logger instance"""

    @staticmethod
    def log_execution(
        command: str,
        args: Dict,
        result: ExecutionResult
    ) -> None:
        """Log command execution"""

    @staticmethod
    def log_agent_activity(
        agent: Agent,
        activity: str,
        details: Dict
    ) -> None:
        """Log agent activity"""
```

---

## Type Definitions

### Common Types

```python
from typing import TypedDict, Literal, Union
from dataclasses import dataclass

# Command Arguments
class CommandArgs(TypedDict, total=False):
    target: str
    language: str
    auto_fix: bool
    agents: str
    parallel: bool
    # ... other common args

# Execution Result
@dataclass
class ExecutionResult:
    success: bool
    data: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    agents_used: List[str]
    execution_time: float

# Analysis Result
@dataclass
class Analysis:
    findings: List[Finding]
    metrics: Dict[str, float]
    recommendations: List[Recommendation]

# Agent Result
@dataclass
class AgentResult:
    agent_name: str
    analysis: Analysis
    suggestions: List[Suggestion]
    implementation: Optional[Implementation]
```

---

## Examples

### Custom Executor

```python
from claude_commands.executor import BaseExecutor
from claude_commands.types import CommandArgs, ExecutionResult

class MyExecutor(BaseExecutor):
    """Custom executor"""

    def execute(self, args: CommandArgs) -> ExecutionResult:
        # Validate
        if not self.validate_args(args):
            return ExecutionResult(
                success=False,
                errors=["Invalid arguments"]
            )

        # Select agents
        agents = self.select_agents(args)

        # Execute
        results = []
        for agent in agents:
            result = agent.analyze(args)
            suggestions = agent.suggest(result)
            if args.get("implement"):
                impl = agent.implement(suggestions)
                results.append(impl)

        # Synthesize
        final = self.synthesize_results(results)

        return ExecutionResult(
            success=True,
            data=final,
            agents_used=[a.name for a in agents]
        )
```

### Custom Agent

```python
from claude_commands.agents import BaseAgent

class MyCustomAgent(BaseAgent):
    """Custom agent"""

    name = "MyCustomAgent"
    expertise = ["custom-domain"]
    capabilities = ["analyze", "suggest", "implement"]

    def analyze(self, context):
        # Analysis logic
        findings = self.find_issues(context)
        metrics = self.compute_metrics(context)

        return Analysis(
            findings=findings,
            metrics=metrics,
            recommendations=self.generate_recommendations(findings)
        )

    def suggest(self, analysis):
        # Suggestion logic
        suggestions = []
        for finding in analysis.findings:
            suggestion = self.create_suggestion(finding)
            suggestions.append(suggestion)
        return suggestions

    def implement(self, suggestions):
        # Implementation logic
        implementations = []
        for suggestion in suggestions:
            if suggestion.auto_fixable:
                impl = self.apply_fix(suggestion)
                implementations.append(impl)
        return Implementation(changes=implementations)
```

### Custom Workflow

```python
from claude_commands.workflow import Workflow, Step

def create_custom_workflow():
    """Create custom workflow"""

    workflow = Workflow(
        name="custom-workflow",
        version="1.0.0"
    )

    # Add steps
    workflow.add_step(Step(
        name="analyze",
        command="check-code-quality",
        args={"auto_fix": True}
    ))

    workflow.add_step(Step(
        name="test",
        command="run-all-tests",
        args={"auto_fix": True},
        depends_on=["analyze"]
    ))

    workflow.add_step(Step(
        name="verify",
        command="double-check",
        args={"description": "workflow complete"},
        depends_on=["test"]
    ))

    return workflow
```

---

## Error Handling

### Common Exceptions

```python
from claude_commands.exceptions import (
    CommandNotFoundError,
    ExecutionError,
    ValidationError,
    AgentError,
    WorkflowError,
    PluginError
)

# Usage
try:
    result = dispatcher.dispatch("command", args)
except CommandNotFoundError:
    # Handle command not found
    pass
except ExecutionError as e:
    # Handle execution error
    logger.error(f"Execution failed: {e}")
except ValidationError as e:
    # Handle validation error
    logger.error(f"Validation failed: {e}")
```

---

## Configuration

### Config API

```python
from claude_commands.config import Config

# Load config
config = Config.load()

# Get value
value = config.get("agents.strategy")

# Set value
config.set("agents.strategy", "intelligent")

# Save config
config.save()

# Load from file
config = Config.from_file("~/.claude-commands/config.yml")
```

---

**Version**: 1.0.0 | **Last Updated**: September 2025 | **Status**: Production Ready

For more information, see:
- **[Developer Guide](DEVELOPER_GUIDE.md)**
- **[User Guide](USER_GUIDE.md)**
- **[Plugin Development Guide](../../plugins/docs/PLUGIN_DEVELOPMENT_GUIDE.md)**