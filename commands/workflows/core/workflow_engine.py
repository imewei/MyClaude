#!/usr/bin/env python3
"""
Workflow Engine - Main orchestrator for executing command workflows

This module provides the core workflow execution engine that:
- Parses workflow YAML definitions
- Manages workflow state and context
- Executes workflows with proper ordering
- Handles dependencies between commands
- Provides error handling and rollback
- Tracks progress and aggregates results
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from .workflow_parser import WorkflowParser
from .dependency_resolver import DependencyResolver
from .command_composer import CommandComposer


logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class StepResult:
    """Result of a workflow step execution"""
    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """Shared context across workflow execution"""
    workflow_name: str
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, StepResult] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    rollback_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_name: str
    status: WorkflowStatus
    steps: List[StepResult]
    context: WorkflowContext
    duration: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """
    Main workflow execution engine

    Orchestrates workflow execution with dependency resolution,
    error handling, progress tracking, and result aggregation.
    """

    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = False,
        max_retries: int = 3,
        parallel_limit: int = 5
    ):
        """
        Initialize workflow engine

        Args:
            dry_run: If True, simulate execution without running commands
            verbose: Enable verbose logging
            max_retries: Maximum retry attempts for failed steps
            parallel_limit: Maximum parallel step execution
        """
        self.dry_run = dry_run
        self.verbose = verbose
        self.max_retries = max_retries
        self.parallel_limit = parallel_limit

        self.parser = WorkflowParser()
        self.resolver = DependencyResolver()
        self.composer = CommandComposer()

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def execute_workflow(
        self,
        workflow_path: Path,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[WorkflowContext] = None
    ) -> WorkflowResult:
        """
        Execute a workflow from a YAML file

        Args:
            workflow_path: Path to workflow YAML file
            variables: Optional variables to override workflow defaults
            context: Optional existing context to continue from

        Returns:
            WorkflowResult with execution details
        """
        logger.info(f"Starting workflow execution: {workflow_path}")
        start_time = time.time()

        try:
            # Parse workflow definition
            workflow_def = self.parser.parse_workflow(workflow_path)

            # Create or update context
            if context is None:
                context = WorkflowContext(
                    workflow_name=workflow_def['workflow']['name'],
                    variables={**workflow_def.get('variables', {}), **(variables or {})}
                )
            else:
                context.variables.update(variables or {})

            # Validate workflow
            validation_errors = self.parser.validate_workflow(workflow_def)
            if validation_errors:
                raise ValueError(f"Workflow validation failed: {validation_errors}")

            # Resolve dependencies and get execution order
            steps = workflow_def['steps']
            execution_order = self.resolver.resolve_execution_order(steps)

            logger.info(f"Execution order: {execution_order}")

            # Execute workflow
            workflow_status = WorkflowStatus.RUNNING
            step_results = []

            for step_batch in execution_order:
                # Execute steps in parallel if multiple in batch
                if len(step_batch) > 1:
                    batch_results = await self._execute_parallel_steps(
                        step_batch, workflow_def, context
                    )
                else:
                    batch_results = [
                        await self._execute_step(step_batch[0], workflow_def, context)
                    ]

                step_results.extend(batch_results)

                # Check for failures
                for result in batch_results:
                    if result.status == StepStatus.FAILED:
                        error_handling = self._get_step_error_handling(
                            result.step_id, workflow_def
                        )

                        if error_handling == "stop":
                            logger.error(f"Step {result.step_id} failed, stopping workflow")
                            workflow_status = WorkflowStatus.FAILED

                            # Rollback if configured
                            if self._should_rollback(workflow_def):
                                await self._rollback_workflow(context)
                                workflow_status = WorkflowStatus.ROLLED_BACK

                            break
                        elif error_handling == "rollback":
                            logger.warning(f"Step {result.step_id} failed, rolling back")
                            await self._rollback_workflow(context)
                            workflow_status = WorkflowStatus.ROLLED_BACK
                            break
                        else:  # continue
                            logger.warning(f"Step {result.step_id} failed, continuing")

                if workflow_status in (WorkflowStatus.FAILED, WorkflowStatus.ROLLED_BACK):
                    break

            # Determine final status
            if workflow_status == WorkflowStatus.RUNNING:
                if all(r.status == StepStatus.COMPLETED for r in step_results):
                    workflow_status = WorkflowStatus.COMPLETED
                else:
                    workflow_status = WorkflowStatus.FAILED

            duration = time.time() - start_time

            return WorkflowResult(
                workflow_name=context.workflow_name,
                status=workflow_status,
                steps=step_results,
                context=context,
                duration=duration,
                metadata={
                    'total_steps': len(step_results),
                    'successful_steps': sum(1 for r in step_results if r.status == StepStatus.COMPLETED),
                    'failed_steps': sum(1 for r in step_results if r.status == StepStatus.FAILED),
                    'skipped_steps': sum(1 for r in step_results if r.status == StepStatus.SKIPPED),
                }
            )

        except Exception as e:
            logger.exception(f"Workflow execution failed: {e}")
            duration = time.time() - start_time

            return WorkflowResult(
                workflow_name=workflow_path.name,
                status=WorkflowStatus.FAILED,
                steps=[],
                context=context or WorkflowContext(workflow_name=workflow_path.name),
                duration=duration,
                error=str(e)
            )

    async def _execute_step(
        self,
        step_id: str,
        workflow_def: Dict[str, Any],
        context: WorkflowContext
    ) -> StepResult:
        """
        Execute a single workflow step

        Args:
            step_id: Step identifier
            workflow_def: Complete workflow definition
            context: Workflow context

        Returns:
            StepResult with execution details
        """
        step = self._get_step_by_id(step_id, workflow_def)
        logger.info(f"Executing step: {step_id}")

        start_time = time.time()

        try:
            # Check condition
            if not self._evaluate_condition(step, context):
                logger.info(f"Step {step_id} condition not met, skipping")
                return StepResult(
                    step_id=step_id,
                    status=StepStatus.SKIPPED,
                    duration=time.time() - start_time
                )

            # Substitute variables
            step = self._substitute_variables(step, context)

            # Execute command
            if self.dry_run:
                logger.info(f"DRY RUN: Would execute {step['command']} with flags {step.get('flags', [])}")
                result = {"output": "Dry run - no actual execution", "success": True}
            else:
                result = await self.composer.execute_command(
                    command=step['command'],
                    flags=step.get('flags', []),
                    input_data=step.get('input'),
                    context=context
                )

            # Store result
            step_result = StepResult(
                step_id=step_id,
                status=StepStatus.COMPLETED if result.get('success', True) else StepStatus.FAILED,
                output=result.get('output'),
                duration=time.time() - start_time,
                metadata=result.get('metadata', {})
            )

            context.results[step_id] = step_result

            # Store rollback action if configured
            if step.get('rollback_command'):
                context.rollback_actions.append({
                    'step_id': step_id,
                    'command': step['rollback_command'],
                    'flags': step.get('rollback_flags', [])
                })

            return step_result

        except Exception as e:
            logger.exception(f"Step {step_id} failed: {e}")

            # Retry logic
            retry_config = step.get('retry', {})
            max_attempts = retry_config.get('max_attempts', self.max_retries)

            if step.get('_retry_count', 0) < max_attempts:
                logger.info(f"Retrying step {step_id} (attempt {step.get('_retry_count', 0) + 1}/{max_attempts})")
                step['_retry_count'] = step.get('_retry_count', 0) + 1

                # Exponential backoff
                if retry_config.get('backoff') == 'exponential':
                    await asyncio.sleep(2 ** step['_retry_count'])

                return await self._execute_step(step_id, workflow_def, context)

            return StepResult(
                step_id=step_id,
                status=StepStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time
            )

    async def _execute_parallel_steps(
        self,
        step_ids: List[str],
        workflow_def: Dict[str, Any],
        context: WorkflowContext
    ) -> List[StepResult]:
        """
        Execute multiple steps in parallel

        Args:
            step_ids: List of step identifiers
            workflow_def: Complete workflow definition
            context: Workflow context

        Returns:
            List of StepResults
        """
        logger.info(f"Executing {len(step_ids)} steps in parallel")

        # Create tasks
        tasks = [
            self._execute_step(step_id, workflow_def, context)
            for step_id in step_ids
        ]

        # Execute with concurrency limit
        results = []
        for i in range(0, len(tasks), self.parallel_limit):
            batch = tasks[i:i + self.parallel_limit]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Parallel step execution error: {result}")
                    results.append(StepResult(
                        step_id="unknown",
                        status=StepStatus.FAILED,
                        error=str(result)
                    ))
                else:
                    results.append(result)

        return results

    async def _rollback_workflow(self, context: WorkflowContext):
        """
        Rollback workflow by executing rollback actions in reverse order

        Args:
            context: Workflow context with rollback actions
        """
        logger.warning("Rolling back workflow")

        for action in reversed(context.rollback_actions):
            try:
                logger.info(f"Rolling back step: {action['step_id']}")

                if not self.dry_run:
                    await self.composer.execute_command(
                        command=action['command'],
                        flags=action.get('flags', []),
                        context=context
                    )

                # Update step status
                if action['step_id'] in context.results:
                    context.results[action['step_id']].status = StepStatus.ROLLED_BACK

            except Exception as e:
                logger.error(f"Rollback failed for step {action['step_id']}: {e}")

    def _get_step_by_id(self, step_id: str, workflow_def: Dict[str, Any]) -> Dict[str, Any]:
        """Get step definition by ID"""
        for step in workflow_def['steps']:
            if step['id'] == step_id:
                return step
        raise ValueError(f"Step not found: {step_id}")

    def _get_step_error_handling(self, step_id: str, workflow_def: Dict[str, Any]) -> str:
        """Get error handling strategy for step"""
        step = self._get_step_by_id(step_id, workflow_def)
        return step.get('on_error', 'stop')

    def _should_rollback(self, workflow_def: Dict[str, Any]) -> bool:
        """Check if workflow should rollback on error"""
        return workflow_def.get('workflow', {}).get('rollback_on_error', False)

    def _evaluate_condition(self, step: Dict[str, Any], context: WorkflowContext) -> bool:
        """
        Evaluate step condition

        Supports conditions like:
        - "step1.success"
        - "step1.quality_score > 80"
        - "has_performance_issues"
        """
        condition = step.get('condition')
        if not condition:
            return True

        try:
            # Simple success check
            if '.' in condition and condition.endswith('.success'):
                step_id = condition.split('.')[0]
                return (
                    step_id in context.results and
                    context.results[step_id].status == StepStatus.COMPLETED
                )

            # Variable check
            if condition in context.variables:
                return bool(context.variables[condition])

            # More complex evaluation (simplified - would use proper expression parser)
            # For now, default to True
            logger.warning(f"Complex condition evaluation not fully implemented: {condition}")
            return True

        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False

    def _substitute_variables(
        self,
        step: Dict[str, Any],
        context: WorkflowContext
    ) -> Dict[str, Any]:
        """
        Substitute variables in step definition

        Supports ${variable} syntax
        """
        import json

        # Convert to JSON and back for deep copy
        step_str = json.dumps(step)

        # Substitute variables
        for var_name, var_value in context.variables.items():
            step_str = step_str.replace(f"${{{var_name}}}", str(var_value))

        return json.loads(step_str)

    def get_workflow_status(self, context: WorkflowContext) -> Dict[str, Any]:
        """
        Get current workflow status

        Args:
            context: Workflow context

        Returns:
            Status dictionary with progress information
        """
        total_steps = len(context.results)
        completed = sum(1 for r in context.results.values() if r.status == StepStatus.COMPLETED)
        failed = sum(1 for r in context.results.values() if r.status == StepStatus.FAILED)
        running = sum(1 for r in context.results.values() if r.status == StepStatus.RUNNING)

        return {
            'workflow_name': context.workflow_name,
            'total_steps': total_steps,
            'completed': completed,
            'failed': failed,
            'running': running,
            'progress': completed / total_steps if total_steps > 0 else 0,
            'elapsed_time': (datetime.now() - context.start_time).total_seconds()
        }