#!/usr/bin/env python3
"""
Command Composer - Chains and composes commands together

This module provides command composition:
- Chain commands together
- Pass output from one command to next
- Handle conditional execution
- Manage parallel command execution
- Coordinate shared context
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CommandComposer:
    """
    Composes and chains commands together with shared context
    """

    def __init__(self, commands_dir: Optional[Path] = None):
        """
        Initialize command composer

        Args:
            commands_dir: Directory containing command scripts
        """
        self.commands_dir = commands_dir or Path.home() / ".claude" / "commands"

    async def execute_command(
        self,
        command: str,
        flags: List[str],
        input_data: Any = None,
        context: Optional[Any] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Execute a single command

        Args:
            command: Command name (e.g., 'check-code-quality')
            flags: List of command flags
            input_data: Optional input data for command
            context: Optional workflow context
            timeout: Command timeout in seconds

        Returns:
            Dictionary with execution result
        """
        logger.info(f"Executing command: {command} {' '.join(flags)}")

        try:
            # Build command
            cmd_parts = [command] + flags

            if input_data:
                cmd_parts.append(str(input_data))

            # Find command script
            script_path = self._find_command_script(command)

            if not script_path:
                logger.warning(f"Command script not found: {command}, using direct execution")
                result = await self._execute_direct(cmd_parts, timeout)
            else:
                result = await self._execute_script(script_path, flags, input_data, timeout)

            return {
                'success': result.get('returncode', 0) == 0,
                'output': result.get('stdout', ''),
                'error': result.get('stderr', ''),
                'metadata': {
                    'command': command,
                    'flags': flags,
                    'returncode': result.get('returncode', 0)
                }
            }

        except asyncio.TimeoutError:
            logger.error(f"Command timeout: {command}")
            return {
                'success': False,
                'output': '',
                'error': f"Command timed out after {timeout} seconds",
                'metadata': {'command': command, 'flags': flags}
            }

        except Exception as e:
            logger.exception(f"Command execution failed: {e}")
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'metadata': {'command': command, 'flags': flags}
            }

    async def execute_chain(
        self,
        commands: List[Dict[str, Any]],
        context: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a chain of commands, passing output from one to next

        Args:
            commands: List of command definitions
            context: Optional workflow context

        Returns:
            List of execution results
        """
        logger.info(f"Executing command chain of {len(commands)} commands")

        results = []
        previous_output = None

        for cmd_def in commands:
            command = cmd_def['command']
            flags = cmd_def.get('flags', [])
            input_data = cmd_def.get('input', previous_output)

            result = await self.execute_command(
                command=command,
                flags=flags,
                input_data=input_data,
                context=context
            )

            results.append(result)

            # Stop chain if command failed and continue_on_error is False
            if not result['success'] and not cmd_def.get('continue_on_error', False):
                logger.warning(f"Command chain stopped at {command} due to failure")
                break

            # Pass output to next command
            previous_output = result['output']

        return results

    async def execute_parallel(
        self,
        commands: List[Dict[str, Any]],
        context: Optional[Any] = None,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple commands in parallel

        Args:
            commands: List of command definitions
            context: Optional workflow context
            max_concurrent: Maximum concurrent executions

        Returns:
            List of execution results
        """
        logger.info(f"Executing {len(commands)} commands in parallel")

        # Create tasks
        tasks = []
        for cmd_def in commands:
            task = self.execute_command(
                command=cmd_def['command'],
                flags=cmd_def.get('flags', []),
                input_data=cmd_def.get('input'),
                context=context
            )
            tasks.append(task)

        # Execute with concurrency limit
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Parallel execution error: {result}")
                    results.append({
                        'success': False,
                        'output': '',
                        'error': str(result),
                        'metadata': {}
                    })
                else:
                    results.append(result)

        return results

    async def execute_conditional(
        self,
        command: str,
        flags: List[str],
        condition: str,
        context: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute command conditionally

        Args:
            command: Command name
            flags: Command flags
            condition: Condition expression
            context: Workflow context

        Returns:
            Execution result if condition met, None otherwise
        """
        if not self._evaluate_condition(condition, context):
            logger.info(f"Condition not met for {command}: {condition}")
            return None

        return await self.execute_command(command, flags, context=context)

    def _evaluate_condition(self, condition: str, context: Any) -> bool:
        """
        Evaluate condition expression

        Simple condition evaluation. For production, use a proper
        expression parser.

        Args:
            condition: Condition string
            context: Workflow context

        Returns:
            True if condition met
        """
        if not context:
            return True

        try:
            # Simple evaluations
            if hasattr(context, 'results'):
                # Check step success
                if '.' in condition and condition.endswith('.success'):
                    step_id = condition.split('.')[0]
                    return (
                        step_id in context.results and
                        context.results[step_id].status.value == 'completed'
                    )

            if hasattr(context, 'variables'):
                # Check variable value
                if condition in context.variables:
                    return bool(context.variables[condition])

            # Default to True for unrecognized conditions
            logger.warning(f"Could not evaluate condition: {condition}")
            return True

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    async def _execute_script(
        self,
        script_path: Path,
        flags: List[str],
        input_data: Any,
        timeout: int
    ) -> Dict[str, Any]:
        """
        Execute command script

        Args:
            script_path: Path to command script
            flags: Command flags
            input_data: Input data
            timeout: Timeout in seconds

        Returns:
            Execution result
        """
        cmd = [str(script_path)] + flags

        if input_data:
            cmd.append(str(input_data))

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore')
            }

        except Exception as e:
            logger.exception(f"Script execution failed: {e}")
            raise

    async def _execute_direct(
        self,
        cmd_parts: List[str],
        timeout: int
    ) -> Dict[str, Any]:
        """
        Execute command directly

        Args:
            cmd_parts: Command parts
            timeout: Timeout in seconds

        Returns:
            Execution result
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore')
            }

        except Exception as e:
            logger.exception(f"Direct execution failed: {e}")
            raise

    def _find_command_script(self, command: str) -> Optional[Path]:
        """
        Find command script file

        Args:
            command: Command name

        Returns:
            Path to script or None
        """
        # Try different extensions
        for ext in ['.py', '.sh', '']:
            script_path = self.commands_dir / f"{command}{ext}"
            if script_path.exists():
                return script_path

        return None

    def compose_pipeline(
        self,
        steps: List[Dict[str, Any]]
    ) -> str:
        """
        Generate shell pipeline from steps

        Args:
            steps: List of step definitions

        Returns:
            Shell pipeline string
        """
        pipeline_parts = []

        for step in steps:
            command = step['command']
            flags = ' '.join(step.get('flags', []))
            input_data = step.get('input', '')

            cmd_str = f"{command} {flags} {input_data}".strip()
            pipeline_parts.append(cmd_str)

        return ' | '.join(pipeline_parts)

    async def execute_with_retry(
        self,
        command: str,
        flags: List[str],
        max_retries: int = 3,
        backoff: str = 'exponential',
        context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute command with retry logic

        Args:
            command: Command name
            flags: Command flags
            max_retries: Maximum retry attempts
            backoff: Backoff strategy ('linear' or 'exponential')
            context: Workflow context

        Returns:
            Execution result
        """
        for attempt in range(max_retries):
            result = await self.execute_command(
                command=command,
                flags=flags,
                context=context
            )

            if result['success']:
                return result

            if attempt < max_retries - 1:
                # Calculate backoff delay
                if backoff == 'exponential':
                    delay = 2 ** attempt
                else:  # linear
                    delay = attempt + 1

                logger.info(f"Retry {attempt + 1}/{max_retries} after {delay}s")
                await asyncio.sleep(delay)

        return result

    def transform_output(
        self,
        output: str,
        transformation: str
    ) -> Any:
        """
        Transform command output

        Args:
            output: Command output
            transformation: Transformation type

        Returns:
            Transformed output
        """
        transformations = {
            'json': lambda x: json.loads(x),
            'lines': lambda x: x.strip().split('\n'),
            'upper': lambda x: x.upper(),
            'lower': lambda x: x.lower(),
            'strip': lambda x: x.strip(),
        }

        transform_fn = transformations.get(transformation)
        if not transform_fn:
            logger.warning(f"Unknown transformation: {transformation}")
            return output

        try:
            return transform_fn(output)
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            return output