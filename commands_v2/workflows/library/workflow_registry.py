#!/usr/bin/env python3
"""
Workflow Registry - Manages available workflows

This module provides workflow registration and discovery:
- Register available workflows
- Discover workflows from directory
- Load workflow definitions
- Workflow metadata management
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


logger = logging.getLogger(__name__)


class WorkflowRegistry:
    """
    Central registry for available workflows
    """

    def __init__(self, workflows_dir: Optional[Path] = None):
        """
        Initialize workflow registry

        Args:
            workflows_dir: Base directory containing workflows
        """
        self.workflows_dir = workflows_dir or Path.home() / ".claude" / "commands" / "workflows"
        self.templates_dir = self.workflows_dir / "templates"
        self.custom_dir = self.workflows_dir / "custom"

        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._discover_workflows()

    def _discover_workflows(self):
        """Discover all available workflows"""
        logger.info("Discovering workflows...")

        # Discover template workflows
        if self.templates_dir.exists():
            for workflow_file in self.templates_dir.glob("*.yaml"):
                self._register_workflow(workflow_file, category="template")

        # Discover custom workflows
        if self.custom_dir.exists():
            for workflow_file in self.custom_dir.glob("*.yaml"):
                self._register_workflow(workflow_file, category="custom")

        logger.info(f"Discovered {len(self._workflows)} workflows")

    def _register_workflow(self, workflow_path: Path, category: str = "custom"):
        """
        Register a workflow

        Args:
            workflow_path: Path to workflow YAML file
            category: Workflow category
        """
        try:
            with open(workflow_path, 'r') as f:
                workflow_def = yaml.safe_load(f)

            if not workflow_def or 'workflow' not in workflow_def:
                logger.warning(f"Invalid workflow: {workflow_path}")
                return

            workflow_meta = workflow_def['workflow']
            workflow_name = workflow_meta.get('name', workflow_path.stem)

            self._workflows[workflow_name] = {
                'path': workflow_path,
                'category': category,
                'name': workflow_name,
                'description': workflow_meta.get('description', ''),
                'version': workflow_meta.get('version', '1.0'),
                'author': workflow_meta.get('author', 'Unknown'),
                'tags': workflow_meta.get('tags', []),
                'steps_count': len(workflow_def.get('steps', [])),
                'has_parallel': any(
                    step.get('parallel', False) for step in workflow_def.get('steps', [])
                ),
                'commands_used': list(set(
                    step.get('command') for step in workflow_def.get('steps', [])
                    if 'command' in step
                ))
            }

            logger.debug(f"Registered workflow: {workflow_name}")

        except Exception as e:
            logger.error(f"Failed to register workflow {workflow_path}: {e}")

    def list_workflows(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available workflows

        Args:
            category: Filter by category
            tag: Filter by tag

        Returns:
            List of workflow metadata
        """
        workflows = list(self._workflows.values())

        if category:
            workflows = [w for w in workflows if w['category'] == category]

        if tag:
            workflows = [w for w in workflows if tag in w['tags']]

        return sorted(workflows, key=lambda w: w['name'])

    def get_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow metadata by name

        Args:
            name: Workflow name

        Returns:
            Workflow metadata or None
        """
        return self._workflows.get(name)

    def get_workflow_path(self, name: str) -> Optional[Path]:
        """
        Get workflow file path

        Args:
            name: Workflow name

        Returns:
            Path to workflow file or None
        """
        workflow = self.get_workflow(name)
        return workflow['path'] if workflow else None

    def search_workflows(self, query: str) -> List[Dict[str, Any]]:
        """
        Search workflows by name, description, or tags

        Args:
            query: Search query

        Returns:
            List of matching workflows
        """
        query = query.lower()
        results = []

        for workflow in self._workflows.values():
            if (
                query in workflow['name'].lower() or
                query in workflow['description'].lower() or
                any(query in tag.lower() for tag in workflow['tags'])
            ):
                results.append(workflow)

        return sorted(results, key=lambda w: w['name'])

    def get_workflows_by_command(self, command: str) -> List[Dict[str, Any]]:
        """
        Get workflows that use a specific command

        Args:
            command: Command name

        Returns:
            List of workflows using the command
        """
        return [
            workflow for workflow in self._workflows.values()
            if command in workflow['commands_used']
        ]

    def get_workflows_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get workflows with a specific tag

        Args:
            tag: Tag name

        Returns:
            List of workflows with the tag
        """
        return [
            workflow for workflow in self._workflows.values()
            if tag in workflow['tags']
        ]

    def register_custom_workflow(
        self,
        workflow_path: Path,
        copy_to_custom: bool = True
    ) -> bool:
        """
        Register a custom workflow

        Args:
            workflow_path: Path to workflow file
            copy_to_custom: If True, copy to custom workflows directory

        Returns:
            True if registered successfully
        """
        try:
            if copy_to_custom:
                # Ensure custom directory exists
                self.custom_dir.mkdir(parents=True, exist_ok=True)

                # Copy workflow
                import shutil
                dest_path = self.custom_dir / workflow_path.name
                shutil.copy(workflow_path, dest_path)
                workflow_path = dest_path

            self._register_workflow(workflow_path, category="custom")
            return True

        except Exception as e:
            logger.error(f"Failed to register custom workflow: {e}")
            return False

    def unregister_workflow(self, name: str) -> bool:
        """
        Unregister a workflow

        Args:
            name: Workflow name

        Returns:
            True if unregistered successfully
        """
        if name in self._workflows:
            del self._workflows[name]
            return True
        return False

    def reload_workflows(self):
        """Reload all workflows from disk"""
        self._workflows.clear()
        self._discover_workflows()

    def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get workflow statistics

        Returns:
            Statistics dictionary
        """
        workflows = list(self._workflows.values())

        return {
            'total_workflows': len(workflows),
            'template_workflows': len([w for w in workflows if w['category'] == 'template']),
            'custom_workflows': len([w for w in workflows if w['category'] == 'custom']),
            'workflows_with_parallel': len([w for w in workflows if w['has_parallel']]),
            'average_steps': sum(w['steps_count'] for w in workflows) / len(workflows) if workflows else 0,
            'most_used_commands': self._get_most_used_commands(),
            'common_tags': self._get_common_tags()
        }

    def _get_most_used_commands(self) -> List[tuple]:
        """Get most frequently used commands across workflows"""
        command_counts = {}

        for workflow in self._workflows.values():
            for command in workflow['commands_used']:
                command_counts[command] = command_counts.get(command, 0) + 1

        return sorted(
            command_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

    def _get_common_tags(self) -> List[tuple]:
        """Get most common tags across workflows"""
        tag_counts = {}

        for workflow in self._workflows.values():
            for tag in workflow['tags']:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return sorted(
            tag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

    def export_registry(self, output_path: Path):
        """
        Export registry to JSON file

        Args:
            output_path: Output file path
        """
        import json

        with open(output_path, 'w') as f:
            json.dump(
                {name: {**meta, 'path': str(meta['path'])}
                 for name, meta in self._workflows.items()},
                f,
                indent=2
            )

        logger.info(f"Exported registry to {output_path}")

    def import_registry(self, input_path: Path):
        """
        Import registry from JSON file

        Args:
            input_path: Input file path
        """
        import json

        with open(input_path, 'r') as f:
            data = json.load(f)

        for name, meta in data.items():
            workflow_path = Path(meta['path'])
            if workflow_path.exists():
                self._register_workflow(workflow_path, meta['category'])

        logger.info(f"Imported registry from {input_path}")