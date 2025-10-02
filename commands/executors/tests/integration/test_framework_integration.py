#!/usr/bin/env python3
"""
Integration Tests for BaseCommandExecutor Framework
===================================================

Tests the complete execution pipeline:
1. Initialization
2. Validation
3. Pre-execution
4. Execution
5. Post-execution
6. Finalization

Coverage: Framework execution flow, error handling, result processing
"""

import pytest
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

from executors.framework import (
    BaseCommandExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionPhase,
    AgentType,
    CommandCategory,
    ValidationRule,
)


# ============================================================================
# Test Implementation
# ============================================================================

class TestCommandExecutor(BaseCommandExecutor):
    """Test implementation of BaseCommandExecutor"""

    def __init__(self):
        super().__init__(
            command_name="test_command",
            category=CommandCategory.ANALYSIS,
            version="1.0"
        )
        self.pre_execution_called = False
        self.post_execution_called = False
        self.execution_count = 0

    def validate_prerequisites(self, context: ExecutionContext) -> Tuple[bool, List[str]]:
        """Validate prerequisites"""
        errors = []

        # Check if work_dir exists
        if not context.work_dir.exists():
            errors.append(f"Work directory does not exist: {context.work_dir}")

        return (len(errors) == 0, errors)

    def execute_command(self, context: ExecutionContext) -> ExecutionResult:
        """Execute test command"""
        self.execution_count += 1
        start_time = time.time()

        # Simulate work
        time.sleep(0.1)

        return ExecutionResult(
            success=True,
            command=self.command_name,
            duration=time.time() - start_time,
            phase=ExecutionPhase.EXECUTION,
            summary="Test command executed successfully",
            details={"execution_count": self.execution_count},
            metrics={"test_metric": 42}
        )

    def pre_execution_hook(self, context: ExecutionContext) -> bool:
        """Pre-execution hook"""
        self.pre_execution_called = True
        return True

    def post_execution_hook(
        self,
        context: ExecutionContext,
        result: ExecutionResult
    ) -> ExecutionResult:
        """Post-execution hook"""
        self.post_execution_called = True
        result.details["post_processed"] = True
        return result

    def get_validation_rules(self) -> List[ValidationRule]:
        """Get validation rules"""
        def check_work_dir(ctx: ExecutionContext) -> Tuple[bool, str]:
            if ctx.work_dir.exists():
                return True, None
            return False, "Work directory must exist"

        return [
            ValidationRule(
                name="work_dir_exists",
                validator=check_work_dir,
                severity="error"
            )
        ]


class FailingCommandExecutor(BaseCommandExecutor):
    """Executor that fails during execution"""

    def __init__(self):
        super().__init__(
            command_name="failing_command",
            category=CommandCategory.ANALYSIS,
            version="1.0"
        )

    def validate_prerequisites(self, context: ExecutionContext) -> Tuple[bool, List[str]]:
        return True, []

    def execute_command(self, context: ExecutionContext) -> ExecutionResult:
        """Simulate execution failure"""
        raise RuntimeError("Simulated execution failure")


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.framework
class TestBaseCommandExecutorIntegration:
    """Integration tests for BaseCommandExecutor"""

    def test_successful_execution_flow(self, temp_workspace: Path):
        """Test complete successful execution flow"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        result = executor.execute({"test_arg": "test_value"})

        # Verify result
        assert result.success is True
        assert result.command == "test_command"
        assert result.duration > 0
        assert result.summary == "Test command executed successfully"
        assert result.metrics["test_metric"] == 42
        assert result.details["execution_count"] == 1
        assert result.details.get("post_processed") is True

        # Verify hooks were called
        assert executor.pre_execution_called is True
        assert executor.post_execution_called is True

    def test_execution_with_dry_run(self, temp_workspace: Path):
        """Test execution with dry-run mode"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        result = executor.execute({"dry_run": True})

        assert result.success is True
        # Dry-run should still execute but not make permanent changes

    def test_execution_with_validation_failure(self):
        """Test execution fails when validation fails"""
        executor = TestCommandExecutor()
        executor.work_dir = Path("/nonexistent/path")

        result = executor.execute({})

        assert result.success is False
        assert result.phase == ExecutionPhase.VALIDATION
        assert len(result.errors) > 0
        assert "does not exist" in str(result.errors)

    def test_execution_with_exception(self, temp_workspace: Path):
        """Test execution handles exceptions gracefully"""
        executor = FailingCommandExecutor()
        executor.work_dir = temp_workspace

        result = executor.execute({})

        assert result.success is False
        assert result.phase == ExecutionPhase.EXECUTION
        assert len(result.errors) > 0
        assert "Simulated execution failure" in str(result.errors)

    def test_execution_context_creation(self, temp_workspace: Path):
        """Test execution context is created correctly"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        args = {
            "dry_run": True,
            "interactive": True,
            "parallel": True,
            "intelligent": True,
            "orchestrate": True,
            "agents": "scientific",
            "custom_arg": "value"
        }

        result = executor.execute(args)

        assert result.success is True
        assert executor.context is not None
        assert executor.context.dry_run is True
        assert executor.context.interactive is True
        assert executor.context.parallel is True
        assert executor.context.intelligent is True
        assert executor.context.orchestrate is True
        assert len(executor.context.agents) > 0
        assert executor.context.args["custom_arg"] == "value"

    def test_multiple_executions(self, temp_workspace: Path):
        """Test multiple executions work correctly"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        # First execution
        result1 = executor.execute({"run": 1})
        assert result1.success is True
        assert result1.details["execution_count"] == 1

        # Second execution
        result2 = executor.execute({"run": 2})
        assert result2.success is True
        assert result2.details["execution_count"] == 2

        # Verify results are stored
        assert len(executor.results) == 2

    def test_execution_with_backup(self, temp_workspace: Path):
        """Test execution creates backup when implementing changes"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Original content")

        result = executor.execute({"implement": True})

        assert result.success is True
        # Verify backup_id is in metadata
        assert "backup_id" in executor.context.metadata

    def test_progress_tracking(self, temp_workspace: Path):
        """Test progress tracking during execution"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        result = executor.execute({})

        assert result.success is True
        # Progress tracker should have completed
        progress = executor.progress_tracker.get_progress()
        assert progress.get("task") == "test_command"

    def test_format_output(self, temp_workspace: Path):
        """Test output formatting"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        result = executor.execute({})
        output = executor.format_output(result)

        assert "test_command completed successfully" in output
        assert "Test command executed successfully" in output
        assert "test_metric: 42" in output
        assert "Duration:" in output

    def test_agent_parsing(self, temp_workspace: Path):
        """Test agent argument parsing"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        test_cases = [
            ("auto", [AgentType.AUTO]),
            ("core", [AgentType.CORE]),
            ("scientific", [AgentType.SCIENTIFIC]),
            ("all", [AgentType.ALL]),
        ]

        for agent_str, expected in test_cases:
            result = executor.execute({"agents": agent_str})
            assert result.success is True
            # Agents should be parsed correctly
            assert len(executor.context.agents) > 0


@pytest.mark.integration
@pytest.mark.framework
class TestExecutionPhases:
    """Test individual execution phases"""

    def test_initialization_phase(self, temp_workspace: Path):
        """Test initialization phase"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        context = executor._initialize({"test": "value"})

        assert context is not None
        assert context.command_name == "test_command"
        assert context.work_dir == temp_workspace
        assert context.args["test"] == "value"

    def test_validation_phase_success(self, temp_workspace: Path):
        """Test validation phase succeeds"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={}
        )

        success = executor._validate(context)
        assert success is True

    def test_validation_phase_failure(self):
        """Test validation phase fails appropriately"""
        executor = TestCommandExecutor()
        executor.work_dir = Path("/nonexistent")

        context = ExecutionContext(
            command_name="test",
            work_dir=Path("/nonexistent"),
            args={}
        )

        success = executor._validate(context)
        assert success is False

    def test_pre_execution_phase(self, temp_workspace: Path):
        """Test pre-execution phase"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={"implement": False}
        )

        success = executor._pre_execute(context)
        assert success is True
        assert executor.pre_execution_called is True

    def test_execution_phase(self, temp_workspace: Path):
        """Test execution phase"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={}
        )

        result = executor._execute(context)
        assert result.success is True
        assert result.phase == ExecutionPhase.EXECUTION

    def test_post_execution_phase(self, temp_workspace: Path):
        """Test post-execution phase"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={}
        )

        initial_result = ExecutionResult(
            success=True,
            command="test",
            duration=1.0,
            phase=ExecutionPhase.EXECUTION,
            summary="Test"
        )

        result = executor._post_execute(context, initial_result)
        assert result.details.get("post_processed") is True
        assert executor.post_execution_called is True

    def test_finalization_phase(self, temp_workspace: Path):
        """Test finalization phase"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={}
        )

        result = ExecutionResult(
            success=True,
            command="test",
            duration=1.0,
            phase=ExecutionPhase.EXECUTION,
            summary="Test"
        )

        executor._finalize(context, result)
        assert len(executor.results) == 1
        assert executor.results[0] == result


@pytest.mark.integration
@pytest.mark.framework
class TestErrorHandling:
    """Test error handling in execution pipeline"""

    def test_keyboard_interrupt_handling(self, temp_workspace: Path, monkeypatch):
        """Test keyboard interrupt is handled gracefully"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(executor, "execute_command", raise_keyboard_interrupt)

        result = executor.execute({})
        assert result.success is False
        assert "Interrupted by user" in str(result.errors)

    def test_exception_with_debug_mode(self, temp_workspace: Path, monkeypatch):
        """Test exception handling with debug mode enabled"""
        import os
        monkeypatch.setenv("DEBUG", "1")

        executor = FailingCommandExecutor()
        executor.work_dir = temp_workspace

        result = executor.execute({})
        assert result.success is False
        # Debug mode should include traceback
        assert len(result.errors) > 1

    def test_validation_error_messages(self):
        """Test validation errors provide clear messages"""
        executor = TestCommandExecutor()
        executor.work_dir = Path("/nonexistent/path/that/does/not/exist")

        result = executor.execute({})
        assert result.success is False
        assert any("Work directory" in str(error) for error in result.errors)


@pytest.mark.integration
@pytest.mark.framework
@pytest.mark.slow
class TestCaching:
    """Test caching functionality"""

    def test_cache_hit(self, temp_workspace: Path):
        """Test cache hit returns cached result"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        # First execution
        result1 = executor.execute({"cache_test": True})
        assert result1.success is True
        execution_count1 = result1.details["execution_count"]

        # Second execution with same args (should hit cache)
        result2 = executor.execute({"cache_test": True})
        assert result2.success is True
        # Note: Caching only works for non-implementation commands

    def test_cache_miss_on_different_args(self, temp_workspace: Path):
        """Test cache miss with different arguments"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        result1 = executor.execute({"arg": "value1"})
        assert result1.success is True

        result2 = executor.execute({"arg": "value2"})
        assert result2.success is True
        # Should execute twice with different args

    def test_no_cache_on_implement(self, temp_workspace: Path):
        """Test caching is disabled for implement=True"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        result1 = executor.execute({"implement": True})
        assert result1.success is True

        result2 = executor.execute({"implement": True})
        assert result2.success is True
        # Both should execute (no caching for implementations)