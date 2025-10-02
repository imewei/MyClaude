#!/usr/bin/env python3
"""
Workflow Validator - Validates workflow definitions

This module provides comprehensive workflow validation:
- Validate workflow structure
- Check command compatibility
- Verify flag combinations
- Detect potential issues
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


logger = logging.getLogger(__name__)


class WorkflowValidator:
    """
    Comprehensive workflow validation
    """

    # Command compatibility matrix
    COMMAND_COMPATIBILITY = {
        'check-code-quality': {
            'before': [],
            'after': ['refactor-clean', 'clean-codebase', 'optimize'],
            'parallel_with': ['run-all-tests']
        },
        'optimize': {
            'before': ['check-code-quality', 'debug'],
            'after': ['run-all-tests', 'generate-tests'],
            'parallel_with': []
        },
        'run-all-tests': {
            'before': ['generate-tests'],
            'after': ['commit', 'update-docs'],
            'parallel_with': ['check-code-quality']
        },
        'commit': {
            'before': ['run-all-tests', 'check-code-quality'],
            'after': [],
            'parallel_with': []
        }
    }

    # Flag compatibility rules
    FLAG_COMPATIBILITY = {
        'mutually_exclusive': [
            ['--auto-fix', '--dry-run'],
            ['--implement', '--dry-run'],
            ['--unit', '--integration', '--all']
        ],
        'requires_together': [
            ['--orchestrate', '--agents']
        ]
    }

    def __init__(self):
        """Initialize workflow validator"""
        self.errors = []
        self.warnings = []
        self.suggestions = []

    def validate_workflow(
        self,
        workflow_path: Path,
        strict: bool = False
    ) -> Dict[str, List[str]]:
        """
        Validate workflow comprehensively

        Args:
            workflow_path: Path to workflow YAML
            strict: If True, fail on warnings

        Returns:
            Dictionary with errors, warnings, and suggestions
        """
        logger.info(f"Validating workflow: {workflow_path}")

        self.errors = []
        self.warnings = []
        self.suggestions = []

        try:
            # Load workflow
            with open(workflow_path, 'r') as f:
                workflow_def = yaml.safe_load(f)

            # Structural validation
            self._validate_structure(workflow_def)

            # Semantic validation
            self._validate_semantics(workflow_def)

            # Command validation
            self._validate_commands(workflow_def)

            # Dependency validation
            self._validate_dependencies(workflow_def)

            # Flag validation
            self._validate_flags(workflow_def)

            # Performance validation
            self._validate_performance(workflow_def)

            # Best practices
            self._check_best_practices(workflow_def)

        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
        except FileNotFoundError:
            self.errors.append(f"Workflow file not found: {workflow_path}")
        except Exception as e:
            self.errors.append(f"Validation error: {e}")

        result = {
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'valid': len(self.errors) == 0 and (not strict or len(self.warnings) == 0)
        }

        logger.info(
            f"Validation complete: {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings, {len(self.suggestions)} suggestions"
        )

        return result

    def _validate_structure(self, workflow_def: Dict[str, Any]):
        """Validate workflow structure"""
        # Check required sections
        if 'workflow' not in workflow_def:
            self.errors.append("Missing 'workflow' section")
            return

        if 'steps' not in workflow_def:
            self.errors.append("Missing 'steps' section")
            return

        # Validate workflow metadata
        workflow = workflow_def['workflow']
        required_fields = ['name', 'description', 'version']

        for field in required_fields:
            if field not in workflow:
                self.errors.append(f"Missing required workflow field: {field}")

        # Validate steps
        steps = workflow_def['steps']
        if not steps:
            self.errors.append("Workflow has no steps")

        if not isinstance(steps, list):
            self.errors.append("Steps must be a list")

    def _validate_semantics(self, workflow_def: Dict[str, Any]):
        """Validate workflow semantics"""
        steps = workflow_def.get('steps', [])

        # Check step IDs are unique
        step_ids = [step.get('id') for step in steps if 'id' in step]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [sid for sid in step_ids if step_ids.count(sid) > 1]
            self.errors.append(f"Duplicate step IDs: {set(duplicates)}")

        # Check each step has required fields
        for idx, step in enumerate(steps):
            if 'id' not in step:
                self.errors.append(f"Step {idx}: Missing 'id' field")

            if 'command' not in step:
                self.errors.append(f"Step {step.get('id', idx)}: Missing 'command' field")

    def _validate_commands(self, workflow_def: Dict[str, Any]):
        """Validate command usage and compatibility"""
        steps = workflow_def.get('steps', [])

        for step in steps:
            command = step.get('command')
            if not command:
                continue

            # Check command compatibility
            step_id = step.get('id')
            depends_on = step.get('depends_on', [])

            if command in self.COMMAND_COMPATIBILITY:
                compat = self.COMMAND_COMPATIBILITY[command]

                # Check if dependencies make sense
                for dep_id in depends_on:
                    dep_step = self._get_step_by_id(dep_id, steps)
                    if dep_step:
                        dep_command = dep_step.get('command')

                        # Warn if unusual ordering
                        if (compat['before'] and
                            dep_command not in compat['before']):
                            self.warnings.append(
                                f"Step {step_id}: Unusual to run {command} "
                                f"after {dep_command}"
                            )

    def _validate_dependencies(self, workflow_def: Dict[str, Any]):
        """Validate step dependencies"""
        steps = workflow_def.get('steps', [])
        step_ids = {step.get('id') for step in steps}

        for step in steps:
            step_id = step.get('id')
            depends_on = step.get('depends_on', [])

            # Check dependencies exist
            for dep in depends_on:
                if dep not in step_ids:
                    self.errors.append(
                        f"Step {step_id}: Unknown dependency '{dep}'"
                    )

            # Check for self-dependency
            if step_id in depends_on:
                self.errors.append(
                    f"Step {step_id}: Cannot depend on itself"
                )

        # Check for circular dependencies
        if self._has_circular_deps(steps):
            self.errors.append("Circular dependencies detected")

    def _validate_flags(self, workflow_def: Dict[str, Any]):
        """Validate command flags"""
        steps = workflow_def.get('steps', [])

        for step in steps:
            flags = step.get('flags', [])
            if not flags:
                continue

            # Check mutually exclusive flags
            for exclusive_group in self.FLAG_COMPATIBILITY['mutually_exclusive']:
                found = [flag for flag in flags if any(
                    flag.startswith(ex) for ex in exclusive_group
                )]
                if len(found) > 1:
                    self.errors.append(
                        f"Step {step.get('id')}: Mutually exclusive flags: {found}"
                    )

            # Check required flag combinations
            for required_group in self.FLAG_COMPATIBILITY['requires_together']:
                has_any = any(
                    any(flag.startswith(req) for flag in flags)
                    for req in required_group
                )
                has_all = all(
                    any(flag.startswith(req) for flag in flags)
                    for req in required_group
                )

                if has_any and not has_all:
                    self.warnings.append(
                        f"Step {step.get('id')}: Flags {required_group} "
                        f"work best together"
                    )

    def _validate_performance(self, workflow_def: Dict[str, Any]):
        """Validate workflow performance characteristics"""
        steps = workflow_def.get('steps', [])

        # Check for parallelization opportunities
        parallel_candidates = []

        for step in steps:
            depends_on = step.get('depends_on', [])

            # Find steps with same dependencies (can run in parallel)
            for other_step in steps:
                if step == other_step:
                    continue

                other_depends = other_step.get('depends_on', [])
                if set(depends_on) == set(other_depends):
                    if not step.get('parallel') and not other_step.get('parallel'):
                        parallel_candidates.append(
                            (step.get('id'), other_step.get('id'))
                        )

        if parallel_candidates:
            self.suggestions.append(
                f"Consider parallelizing steps with same dependencies: "
                f"{parallel_candidates[:3]}"
            )

        # Check workflow depth
        max_depth = self._calculate_max_depth(steps)
        if max_depth > 10:
            self.warnings.append(
                f"Deep workflow ({max_depth} levels) may take long to execute"
            )

    def _check_best_practices(self, workflow_def: Dict[str, Any]):
        """Check workflow best practices"""
        workflow = workflow_def.get('workflow', {})
        steps = workflow_def.get('steps', [])

        # Check for author
        if 'author' not in workflow:
            self.suggestions.append("Consider adding 'author' field")

        # Check for tags
        if not workflow.get('tags'):
            self.suggestions.append("Consider adding tags for discoverability")

        # Check for error handling
        steps_with_error_handling = [
            s for s in steps if 'on_error' in s or 'retry' in s
        ]

        if len(steps_with_error_handling) < len(steps) * 0.5:
            self.suggestions.append(
                "Consider adding error handling (on_error, retry) to more steps"
            )

        # Check for rollback commands
        critical_steps = [s for s in steps if s.get('command') in [
            'commit', 'refactor-clean', 'ci-setup'
        ]]

        steps_with_rollback = [
            s for s in critical_steps if 'rollback_command' in s
        ]

        if len(steps_with_rollback) < len(critical_steps):
            self.suggestions.append(
                "Consider adding rollback commands to critical steps"
            )

        # Check for commit step
        has_commit = any(s.get('command') == 'commit' for s in steps)
        if not has_commit:
            self.suggestions.append(
                "Consider adding a commit step to save changes"
            )

    def _has_circular_deps(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for circular dependencies"""
        graph = {}
        for step in steps:
            step_id = step['id']
            graph[step_id] = set(step.get('depends_on', []))

        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def _calculate_max_depth(self, steps: List[Dict[str, Any]]) -> int:
        """Calculate maximum dependency depth"""
        graph = {}
        for step in steps:
            step_id = step['id']
            graph[step_id] = set(step.get('depends_on', []))

        depths = {}

        def get_depth(node: str) -> int:
            if node in depths:
                return depths[node]

            deps = graph.get(node, set())
            if not deps:
                depths[node] = 0
                return 0

            max_dep_depth = max(get_depth(dep) for dep in deps)
            depths[node] = max_dep_depth + 1
            return depths[node]

        for node in graph:
            get_depth(node)

        return max(depths.values()) if depths else 0

    def _get_step_by_id(
        self,
        step_id: str,
        steps: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get step by ID"""
        for step in steps:
            if step.get('id') == step_id:
                return step
        return None