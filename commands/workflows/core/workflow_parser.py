#!/usr/bin/env python3
"""
Workflow Parser - YAML workflow definition parser and validator

This module provides workflow parsing and validation:
- Parse workflow YAML files
- Validate workflow structure
- Check for circular dependencies
- Verify command existence
- Validate flag combinations
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


logger = logging.getLogger(__name__)


class WorkflowParser:
    """
    Parser and validator for workflow YAML definitions
    """

    # List of all available commands
    AVAILABLE_COMMANDS = {
        'check-code-quality',
        'optimize',
        'run-all-tests',
        'generate-tests',
        'refactor-clean',
        'update-docs',
        'commit',
        'ci-setup',
        'debug',
        'explain-code',
        'fix-commit-errors',
        'fix-github-issue',
        'multi-agent-optimize',
        'double-check',
        'adopt-code',
        'clean-codebase',
        'reflection',
        'think-ultra'
    }

    # Required workflow fields
    REQUIRED_WORKFLOW_FIELDS = {'name', 'description', 'version'}

    # Required step fields
    REQUIRED_STEP_FIELDS = {'id', 'command'}

    # Valid step fields
    VALID_STEP_FIELDS = {
        'id', 'command', 'flags', 'input', 'depends_on',
        'condition', 'parallel', 'on_error', 'retry',
        'rollback_command', 'rollback_flags', 'timeout',
        'description'
    }

    # Valid error handling strategies
    VALID_ERROR_HANDLERS = {'continue', 'stop', 'rollback'}

    def parse_workflow(self, workflow_path: Path) -> Dict[str, Any]:
        """
        Parse workflow YAML file

        Args:
            workflow_path: Path to workflow YAML file

        Returns:
            Parsed workflow definition

        Raises:
            ValueError: If parsing fails
        """
        logger.info(f"Parsing workflow: {workflow_path}")

        try:
            with open(workflow_path, 'r') as f:
                workflow_def = yaml.safe_load(f)

            if not workflow_def:
                raise ValueError("Empty workflow file")

            # Ensure required top-level keys exist
            if 'workflow' not in workflow_def:
                raise ValueError("Missing 'workflow' section")

            if 'steps' not in workflow_def:
                raise ValueError("Missing 'steps' section")

            logger.info(f"Successfully parsed workflow: {workflow_def['workflow']['name']}")
            return workflow_def

        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error: {e}")
        except FileNotFoundError:
            raise ValueError(f"Workflow file not found: {workflow_path}")
        except Exception as e:
            raise ValueError(f"Failed to parse workflow: {e}")

    def validate_workflow(self, workflow_def: Dict[str, Any]) -> List[str]:
        """
        Validate workflow definition

        Args:
            workflow_def: Parsed workflow definition

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate workflow metadata
        workflow_meta = workflow_def.get('workflow', {})
        for field in self.REQUIRED_WORKFLOW_FIELDS:
            if field not in workflow_meta:
                errors.append(f"Missing required workflow field: {field}")

        # Validate steps
        steps = workflow_def.get('steps', [])
        if not steps:
            errors.append("Workflow has no steps")
            return errors

        step_ids = set()
        for idx, step in enumerate(steps):
            # Check required fields
            for field in self.REQUIRED_STEP_FIELDS:
                if field not in step:
                    errors.append(f"Step {idx}: Missing required field '{field}'")

            # Check step ID uniqueness
            step_id = step.get('id')
            if step_id:
                if step_id in step_ids:
                    errors.append(f"Duplicate step ID: {step_id}")
                step_ids.add(step_id)

            # Validate command
            command = step.get('command')
            if command and command not in self.AVAILABLE_COMMANDS:
                errors.append(f"Step {step_id}: Unknown command '{command}'")

            # Validate dependencies
            depends_on = step.get('depends_on', [])
            if depends_on:
                for dep in depends_on:
                    if dep not in step_ids and dep not in [s.get('id') for s in steps]:
                        errors.append(f"Step {step_id}: Unknown dependency '{dep}'")

            # Validate error handling
            on_error = step.get('on_error')
            if on_error and on_error not in self.VALID_ERROR_HANDLERS:
                errors.append(
                    f"Step {step_id}: Invalid error handler '{on_error}'. "
                    f"Must be one of: {self.VALID_ERROR_HANDLERS}"
                )

            # Validate retry configuration
            retry = step.get('retry')
            if retry:
                if not isinstance(retry, dict):
                    errors.append(f"Step {step_id}: 'retry' must be a dictionary")
                else:
                    if 'max_attempts' in retry and not isinstance(retry['max_attempts'], int):
                        errors.append(f"Step {step_id}: 'max_attempts' must be an integer")
                    if 'backoff' in retry and retry['backoff'] not in ['linear', 'exponential']:
                        errors.append(f"Step {step_id}: Invalid backoff strategy '{retry['backoff']}'")

            # Check for invalid fields
            for field in step.keys():
                if field not in self.VALID_STEP_FIELDS:
                    errors.append(f"Step {step_id}: Unknown field '{field}'")

        # Check for circular dependencies
        circular_deps = self._check_circular_dependencies(steps)
        if circular_deps:
            errors.append(f"Circular dependencies detected: {circular_deps}")

        if errors:
            logger.warning(f"Workflow validation found {len(errors)} errors")
        else:
            logger.info("Workflow validation successful")

        return errors

    def _check_circular_dependencies(self, steps: List[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Check for circular dependencies in workflow steps

        Args:
            steps: List of step definitions

        Returns:
            List of steps involved in circular dependency, or None
        """
        # Build dependency graph
        graph = {}
        for step in steps:
            step_id = step['id']
            graph[step_id] = step.get('depends_on', [])

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    cycle = has_cycle(neighbor, path[:])
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            rec_stack.remove(node)
            return None

        for node in graph:
            if node not in visited:
                cycle = has_cycle(node, [])
                if cycle:
                    return cycle

        return None

    def validate_command_flags(
        self,
        command: str,
        flags: List[str]
    ) -> List[str]:
        """
        Validate command flag combinations

        Args:
            command: Command name
            flags: List of flags

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Command-specific validation rules
        flag_rules = {
            'check-code-quality': {
                'mutually_exclusive': [
                    ['--auto-fix', '--dry-run']
                ],
                'requires': {
                    '--language': ['python', 'julia', 'jax', 'auto']
                }
            },
            'optimize': {
                'mutually_exclusive': [
                    ['--implement', '--dry-run']
                ],
                'requires': {
                    '--category': ['all', 'algorithm', 'memory', 'io', 'concurrency']
                }
            },
            'run-all-tests': {
                'mutually_exclusive': [
                    ['--unit', '--integration', '--all']
                ]
            }
        }

        if command not in flag_rules:
            return errors

        rules = flag_rules[command]

        # Check mutually exclusive flags
        for exclusive_group in rules.get('mutually_exclusive', []):
            found = [flag for flag in exclusive_group if flag in flags]
            if len(found) > 1:
                errors.append(
                    f"Command '{command}': Flags {found} are mutually exclusive"
                )

        # Check flag value requirements
        for flag_prefix, valid_values in rules.get('requires', {}).items():
            matching_flags = [f for f in flags if f.startswith(flag_prefix)]
            for flag in matching_flags:
                if '=' in flag:
                    value = flag.split('=')[1]
                    if value not in valid_values:
                        errors.append(
                            f"Command '{command}': Invalid value '{value}' for flag '{flag_prefix}'. "
                            f"Valid values: {valid_values}"
                        )

        return errors

    def get_workflow_metadata(self, workflow_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract workflow metadata

        Args:
            workflow_def: Parsed workflow definition

        Returns:
            Metadata dictionary
        """
        workflow = workflow_def.get('workflow', {})
        steps = workflow_def.get('steps', [])

        return {
            'name': workflow.get('name'),
            'description': workflow.get('description'),
            'version': workflow.get('version'),
            'author': workflow.get('author'),
            'tags': workflow.get('tags', []),
            'total_steps': len(steps),
            'commands_used': list(set(step.get('command') for step in steps)),
            'has_parallel_steps': any(step.get('parallel') for step in steps),
            'has_conditional_steps': any(step.get('condition') for step in steps),
            'max_depth': self._calculate_max_depth(steps)
        }

    def _calculate_max_depth(self, steps: List[Dict[str, Any]]) -> int:
        """
        Calculate maximum dependency depth

        Args:
            steps: List of step definitions

        Returns:
            Maximum depth
        """
        # Build dependency graph
        graph = {}
        for step in steps:
            step_id = step['id']
            graph[step_id] = step.get('depends_on', [])

        # Calculate depth for each node
        depths = {}

        def get_depth(node: str) -> int:
            if node in depths:
                return depths[node]

            deps = graph.get(node, [])
            if not deps:
                depths[node] = 0
                return 0

            max_dep_depth = max(get_depth(dep) for dep in deps)
            depths[node] = max_dep_depth + 1
            return depths[node]

        for node in graph:
            get_depth(node)

        return max(depths.values()) if depths else 0

    def parse_workflow_template(
        self,
        template_path: Path,
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse workflow template with variable substitution

        Args:
            template_path: Path to workflow template
            variables: Variables to substitute

        Returns:
            Parsed and substituted workflow definition
        """
        workflow_def = self.parse_workflow(template_path)

        # Substitute variables in YAML
        import json
        workflow_str = json.dumps(workflow_def)

        for var_name, var_value in variables.items():
            workflow_str = workflow_str.replace(f"${{{var_name}}}", str(var_value))

        return json.loads(workflow_str)

    def get_step_summary(self, step: Dict[str, Any]) -> str:
        """
        Generate human-readable step summary

        Args:
            step: Step definition

        Returns:
            Summary string
        """
        parts = [
            f"Step: {step['id']}",
            f"Command: {step['command']}"
        ]

        if step.get('flags'):
            parts.append(f"Flags: {' '.join(step['flags'])}")

        if step.get('depends_on'):
            parts.append(f"Depends on: {', '.join(step['depends_on'])}")

        if step.get('condition'):
            parts.append(f"Condition: {step['condition']}")

        return " | ".join(parts)