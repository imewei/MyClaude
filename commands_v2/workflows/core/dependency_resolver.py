#!/usr/bin/env python3
"""
Dependency Resolver - Resolves command dependencies and execution order

This module provides dependency resolution:
- Build execution DAG (directed acyclic graph)
- Topological sort for execution order
- Detect circular dependencies
- Group parallel-executable steps
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    Resolves dependencies between workflow steps and determines execution order
    """

    def resolve_execution_order(self, steps: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Resolve execution order for workflow steps

        Returns steps grouped by execution level, where steps in each level
        can be executed in parallel.

        Args:
            steps: List of step definitions

        Returns:
            List of lists, where each inner list contains step IDs that can
            be executed in parallel

        Example:
            Input: [
                {'id': 'A', 'depends_on': []},
                {'id': 'B', 'depends_on': ['A']},
                {'id': 'C', 'depends_on': ['A']},
                {'id': 'D', 'depends_on': ['B', 'C']}
            ]
            Output: [['A'], ['B', 'C'], ['D']]
        """
        logger.info(f"Resolving execution order for {len(steps)} steps")

        # Build dependency graph
        graph = self._build_dependency_graph(steps)

        # Check for cycles
        if self._has_cycle(graph):
            raise ValueError("Circular dependency detected in workflow")

        # Perform topological sort with level grouping
        execution_order = self._topological_sort_levels(graph)

        logger.info(f"Execution order: {execution_order}")
        return execution_order

    def _build_dependency_graph(
        self,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """
        Build dependency graph from steps

        Args:
            steps: List of step definitions

        Returns:
            Dictionary mapping step ID to set of dependencies
        """
        graph = {}

        for step in steps:
            step_id = step['id']
            depends_on = step.get('depends_on', [])

            # Convert to set for efficient lookup
            graph[step_id] = set(depends_on) if depends_on else set()

        return graph

    def _has_cycle(self, graph: Dict[str, Set[str]]) -> bool:
        """
        Check if dependency graph has cycles using DFS

        Args:
            graph: Dependency graph

        Returns:
            True if cycle detected, False otherwise
        """
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Visit all dependencies
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Check all nodes
        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def _topological_sort_levels(
        self,
        graph: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """
        Perform topological sort and group by execution level

        Uses Kahn's algorithm to produce level-based ordering.

        Args:
            graph: Dependency graph

        Returns:
            List of levels, each containing step IDs that can execute in parallel
        """
        # Calculate in-degree for each node
        in_degree = {node: 0 for node in graph}

        for node in graph:
            for dependency in graph[node]:
                in_degree[dependency] = in_degree.get(dependency, 0)

        for node in graph:
            for dependency in graph[node]:
                in_degree[node] += 1

        # Find all nodes with in-degree 0
        queue = deque([node for node, degree in in_degree.items() if degree == 0])

        levels = []
        while queue:
            # All nodes at current level (can execute in parallel)
            current_level = []

            # Process all nodes at this level
            for _ in range(len(queue)):
                node = queue.popleft()
                current_level.append(node)

                # Reduce in-degree for nodes that depend on this one
                for other_node in graph:
                    if node in graph[other_node]:
                        in_degree[other_node] -= 1
                        if in_degree[other_node] == 0:
                            queue.append(other_node)

            levels.append(current_level)

        # Check if all nodes were processed
        if sum(len(level) for level in levels) != len(graph):
            raise ValueError("Unable to resolve execution order - possible cycle")

        return levels

    def get_dependency_chain(
        self,
        step_id: str,
        graph: Dict[str, Set[str]]
    ) -> List[str]:
        """
        Get complete dependency chain for a step

        Args:
            step_id: Step identifier
            graph: Dependency graph

        Returns:
            List of step IDs in dependency chain (including step_id)
        """
        chain = []
        visited = set()

        def dfs(node: str):
            if node in visited:
                return
            visited.add(node)

            # Visit dependencies first
            for dep in graph.get(node, set()):
                dfs(dep)

            chain.append(node)

        dfs(step_id)
        return chain

    def get_parallel_groups(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """
        Get groups of steps that can be executed in parallel

        Args:
            steps: List of step definitions

        Returns:
            List of groups, each containing step IDs that can run in parallel
        """
        # Check for explicit parallel flag
        parallel_steps = [step for step in steps if step.get('parallel', False)]

        if not parallel_steps:
            return []

        # Group by shared dependencies
        groups = []
        processed = set()

        for step in parallel_steps:
            if step['id'] in processed:
                continue

            step_deps = set(step.get('depends_on', []))

            # Find other steps with same dependencies
            group = [step['id']]

            for other_step in parallel_steps:
                if other_step['id'] == step['id'] or other_step['id'] in processed:
                    continue

                other_deps = set(other_step.get('depends_on', []))

                # Can run in parallel if same dependencies
                if step_deps == other_deps:
                    group.append(other_step['id'])
                    processed.add(other_step['id'])

            groups.append(group)
            processed.add(step['id'])

        return groups

    def validate_dependencies(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Validate step dependencies

        Args:
            steps: List of step definitions

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Create set of valid step IDs
        valid_ids = {step['id'] for step in steps}

        # Check each step's dependencies
        for step in steps:
            step_id = step['id']
            depends_on = step.get('depends_on', [])

            for dep in depends_on:
                if dep not in valid_ids:
                    errors.append(
                        f"Step '{step_id}' depends on unknown step '{dep}'"
                    )

                # Check for self-dependency
                if dep == step_id:
                    errors.append(
                        f"Step '{step_id}' cannot depend on itself"
                    )

        # Check for circular dependencies
        graph = self._build_dependency_graph(steps)
        if self._has_cycle(graph):
            cycle = self._find_cycle(graph)
            errors.append(
                f"Circular dependency detected: {' -> '.join(cycle)}"
            )

        return errors

    def _find_cycle(self, graph: Dict[str, Set[str]]) -> List[str]:
        """
        Find a cycle in the dependency graph

        Args:
            graph: Dependency graph

        Returns:
            List of step IDs forming a cycle
        """
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return path

        return []

    def get_critical_path(
        self,
        steps: List[Dict[str, Any]],
        durations: Dict[str, float]
    ) -> Tuple[List[str], float]:
        """
        Calculate critical path through workflow

        Args:
            steps: List of step definitions
            durations: Estimated duration for each step

        Returns:
            Tuple of (critical path step IDs, total duration)
        """
        graph = self._build_dependency_graph(steps)

        # Calculate earliest start time for each step
        earliest_start = {}
        earliest_finish = {}

        def calculate_earliest(node: str) -> float:
            if node in earliest_finish:
                return earliest_finish[node]

            deps = graph.get(node, set())
            if not deps:
                earliest_start[node] = 0
            else:
                earliest_start[node] = max(
                    calculate_earliest(dep) for dep in deps
                )

            duration = durations.get(node, 0)
            earliest_finish[node] = earliest_start[node] + duration

            return earliest_finish[node]

        # Calculate for all nodes
        for node in graph:
            calculate_earliest(node)

        # Find node with maximum earliest finish
        if not earliest_finish:
            return [], 0.0

        end_node = max(earliest_finish, key=earliest_finish.get)
        total_duration = earliest_finish[end_node]

        # Backtrack to find critical path
        critical_path = []
        current = end_node

        while current:
            critical_path.append(current)

            deps = graph.get(current, set())
            if not deps:
                break

            # Find dependency on critical path
            current = max(
                deps,
                key=lambda d: earliest_finish.get(d, 0)
            )

        critical_path.reverse()

        return critical_path, total_duration

    def optimize_execution_order(
        self,
        steps: List[Dict[str, Any]],
        priorities: Dict[str, int]
    ) -> List[List[str]]:
        """
        Optimize execution order based on priorities

        Args:
            steps: List of step definitions
            priorities: Priority for each step (higher = earlier execution)

        Returns:
            Optimized execution order
        """
        # Get base execution order
        execution_order = self.resolve_execution_order(steps)

        # Sort each level by priority
        for level in execution_order:
            level.sort(
                key=lambda step_id: priorities.get(step_id, 0),
                reverse=True
            )

        return execution_order