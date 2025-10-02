#!/usr/bin/env python3
"""
Workflow Executor - High-level workflow execution interface

This module provides workflow execution management:
- Execute workflow steps
- Track progress
- Handle errors
- Generate reports
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.workflow_engine import WorkflowEngine, WorkflowResult, WorkflowStatus


logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    High-level workflow execution interface
    """

    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = False,
        log_file: Optional[Path] = None
    ):
        """
        Initialize workflow executor

        Args:
            dry_run: If True, simulate execution
            verbose: Enable verbose logging
            log_file: Optional log file path
        """
        self.dry_run = dry_run
        self.verbose = verbose
        self.log_file = log_file

        self.engine = WorkflowEngine(
            dry_run=dry_run,
            verbose=verbose
        )

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        handlers = [logging.StreamHandler()]

        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))

        level = logging.DEBUG if self.verbose else logging.INFO

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

    async def execute(
        self,
        workflow_path: Path,
        variables: Optional[Dict[str, Any]] = None,
        track_progress: bool = True
    ) -> WorkflowResult:
        """
        Execute workflow with progress tracking

        Args:
            workflow_path: Path to workflow YAML
            variables: Optional variable overrides
            track_progress: Enable progress tracking

        Returns:
            WorkflowResult
        """
        logger.info(f"Executing workflow: {workflow_path}")

        if track_progress:
            # Execute with progress callback
            result = await self._execute_with_progress(
                workflow_path,
                variables
            )
        else:
            # Execute without progress tracking
            result = await self.engine.execute_workflow(
                workflow_path,
                variables
            )

        # Log result
        self._log_result(result)

        # Generate report if requested
        if self.verbose:
            self._generate_report(result)

        return result

    async def _execute_with_progress(
        self,
        workflow_path: Path,
        variables: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Execute workflow with progress tracking

        Args:
            workflow_path: Path to workflow YAML
            variables: Optional variable overrides

        Returns:
            WorkflowResult
        """
        # Start execution
        task = asyncio.create_task(
            self.engine.execute_workflow(workflow_path, variables)
        )

        # Track progress
        while not task.done():
            await asyncio.sleep(1)

            # Get current status (would need context access)
            # For now, just log that we're waiting
            logger.debug("Workflow execution in progress...")

        return await task

    def _log_result(self, result: WorkflowResult):
        """
        Log workflow result

        Args:
            result: WorkflowResult
        """
        status_str = result.status.value.upper()

        if result.status == WorkflowStatus.COMPLETED:
            logger.info(
                f"Workflow '{result.workflow_name}' completed successfully "
                f"in {result.duration:.2f}s"
            )
        elif result.status == WorkflowStatus.FAILED:
            logger.error(
                f"Workflow '{result.workflow_name}' failed after "
                f"{result.duration:.2f}s: {result.error}"
            )
        else:
            logger.warning(
                f"Workflow '{result.workflow_name}' ended with status "
                f"{status_str} after {result.duration:.2f}s"
            )

    def _generate_report(self, result: WorkflowResult):
        """
        Generate execution report

        Args:
            result: WorkflowResult
        """
        report_lines = [
            "",
            "=" * 60,
            f"Workflow Execution Report: {result.workflow_name}",
            "=" * 60,
            f"Status: {result.status.value}",
            f"Duration: {result.duration:.2f}s",
            f"Total Steps: {result.metadata.get('total_steps', 0)}",
            f"Successful: {result.metadata.get('successful_steps', 0)}",
            f"Failed: {result.metadata.get('failed_steps', 0)}",
            f"Skipped: {result.metadata.get('skipped_steps', 0)}",
            "",
            "Step Results:",
            "-" * 60
        ]

        for step_result in result.steps:
            report_lines.append(
                f"  [{step_result.status.value}] {step_result.step_id} "
                f"({step_result.duration:.2f}s)"
            )

            if step_result.error:
                report_lines.append(f"    Error: {step_result.error}")

        report_lines.append("=" * 60)

        logger.info("\n".join(report_lines))

    def save_result(
        self,
        result: WorkflowResult,
        output_path: Path
    ):
        """
        Save workflow result to file

        Args:
            result: WorkflowResult
            output_path: Output file path
        """
        data = {
            'workflow_name': result.workflow_name,
            'status': result.status.value,
            'duration': result.duration,
            'error': result.error,
            'metadata': result.metadata,
            'steps': [
                {
                    'step_id': step.step_id,
                    'status': step.status.value,
                    'duration': step.duration,
                    'error': step.error,
                    'timestamp': step.timestamp.isoformat()
                }
                for step in result.steps
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved result to {output_path}")

    def load_result(self, input_path: Path) -> Dict[str, Any]:
        """
        Load workflow result from file

        Args:
            input_path: Input file path

        Returns:
            Result data
        """
        with open(input_path, 'r') as f:
            return json.load(f)

    async def execute_batch(
        self,
        workflow_paths: List[Path],
        variables: Optional[Dict[str, Any]] = None,
        parallel: bool = False
    ) -> List[WorkflowResult]:
        """
        Execute multiple workflows

        Args:
            workflow_paths: List of workflow paths
            variables: Optional variable overrides
            parallel: Execute in parallel if True

        Returns:
            List of WorkflowResults
        """
        logger.info(f"Executing {len(workflow_paths)} workflows")

        if parallel:
            tasks = [
                self.execute(path, variables, track_progress=False)
                for path in workflow_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to failed results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Workflow {workflow_paths[i]} failed: {result}")
                    # Create a failed result
                    from ..core.workflow_engine import WorkflowResult, WorkflowStatus, WorkflowContext
                    final_results.append(WorkflowResult(
                        workflow_name=workflow_paths[i].stem,
                        status=WorkflowStatus.FAILED,
                        steps=[],
                        context=WorkflowContext(workflow_name=workflow_paths[i].stem),
                        duration=0.0,
                        error=str(result)
                    ))
                else:
                    final_results.append(result)

            return final_results
        else:
            results = []
            for path in workflow_paths:
                result = await self.execute(path, variables)
                results.append(result)

            return results

    def get_execution_summary(
        self,
        results: List[WorkflowResult]
    ) -> Dict[str, Any]:
        """
        Get summary of multiple workflow executions

        Args:
            results: List of WorkflowResults

        Returns:
            Summary dictionary
        """
        total = len(results)
        completed = sum(1 for r in results if r.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == WorkflowStatus.FAILED)
        total_duration = sum(r.duration for r in results)

        return {
            'total_workflows': total,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total if total > 0 else 0,
            'total_duration': total_duration,
            'average_duration': total_duration / total if total > 0 else 0,
            'total_steps': sum(
                r.metadata.get('total_steps', 0) for r in results
            ),
            'successful_steps': sum(
                r.metadata.get('successful_steps', 0) for r in results
            ),
            'failed_steps': sum(
                r.metadata.get('failed_steps', 0) for r in results
            )
        }

    async def resume_workflow(
        self,
        workflow_path: Path,
        checkpoint_path: Path
    ) -> WorkflowResult:
        """
        Resume workflow from checkpoint

        Args:
            workflow_path: Path to workflow YAML
            checkpoint_path: Path to checkpoint file

        Returns:
            WorkflowResult
        """
        logger.info(f"Resuming workflow from checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint_data = self.load_result(checkpoint_path)

        # TODO: Implement checkpoint resume logic
        # For now, just execute normally
        logger.warning("Checkpoint resume not fully implemented yet")

        return await self.execute(workflow_path)