#!/usr/bin/env python3
"""
Workflow CLI - Command-line interface for workflow management

Provides commands for:
- Listing available workflows
- Running workflows
- Creating workflows from templates
- Validating workflows
- Managing workflow registry
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.core.workflow_engine import WorkflowEngine
from workflows.library.workflow_registry import WorkflowRegistry
from workflows.library.workflow_validator import WorkflowValidator
from workflows.library.workflow_executor import WorkflowExecutor


def list_workflows(args):
    """List available workflows"""
    registry = WorkflowRegistry()

    workflows = registry.list_workflows(
        category=args.category,
        tag=args.tag
    )

    if not workflows:
        print("No workflows found")
        return

    print(f"\nAvailable Workflows ({len(workflows)}):\n")
    print("-" * 80)

    for workflow in workflows:
        print(f"Name: {workflow['name']}")
        print(f"  Description: {workflow['description']}")
        print(f"  Category: {workflow['category']}")
        print(f"  Version: {workflow['version']}")
        print(f"  Steps: {workflow['steps_count']}")

        if workflow['tags']:
            print(f"  Tags: {', '.join(workflow['tags'])}")

        if workflow['commands_used']:
            print(f"  Commands: {', '.join(workflow['commands_used'][:5])}")

        print()


def run_workflow(args):
    """Run a workflow"""
    registry = WorkflowRegistry()

    # Get workflow path
    workflow_path = None

    if Path(args.workflow).exists():
        workflow_path = Path(args.workflow)
    else:
        workflow_path = registry.get_workflow_path(args.workflow)

    if not workflow_path:
        print(f"Error: Workflow not found: {args.workflow}")
        return 1

    # Parse variables
    variables = {}
    if args.var:
        for var_def in args.var:
            key, value = var_def.split('=', 1)
            variables[key] = value

    # Create executor
    executor = WorkflowExecutor(
        dry_run=args.dry_run,
        verbose=args.verbose,
        log_file=Path(args.log) if args.log else None
    )

    # Execute workflow
    print(f"Executing workflow: {workflow_path.name}")

    if args.dry_run:
        print("DRY RUN MODE - No actual execution\n")

    result = asyncio.run(executor.execute(
        workflow_path=workflow_path,
        variables=variables,
        track_progress=not args.no_progress
    ))

    # Save result if requested
    if args.output:
        executor.save_result(result, Path(args.output))

    # Return exit code based on result
    return 0 if result.status.value == 'completed' else 1


def validate_workflow(args):
    """Validate a workflow"""
    workflow_path = Path(args.workflow)

    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_path}")
        return 1

    validator = WorkflowValidator()
    result = validator.validate_workflow(
        workflow_path,
        strict=args.strict
    )

    print(f"\nValidation Results for: {workflow_path.name}\n")
    print("=" * 60)

    if result['errors']:
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"  ❌ {error}")

    if result['warnings']:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for warning in result['warnings']:
            print(f"  ⚠️  {warning}")

    if result['suggestions']:
        print(f"\nSuggestions ({len(result['suggestions'])}):")
        for suggestion in result['suggestions']:
            print(f"  💡 {suggestion}")

    print("\n" + "=" * 60)

    if result['valid']:
        print("✅ Workflow is valid")
        return 0
    else:
        print("❌ Workflow validation failed")
        return 1


def create_workflow(args):
    """Create a workflow from template"""
    registry = WorkflowRegistry()

    # Get template path
    template_path = registry.get_workflow_path(args.template)

    if not template_path:
        print(f"Error: Template not found: {args.template}")
        return 1

    # Create output path
    output_path = Path(args.output) if args.output else Path(f"{args.name}.yaml")

    # Parse variables
    variables = {}
    if args.var:
        for var_def in args.var:
            key, value = var_def.split('=', 1)
            variables[key] = value

    # Load and customize template
    import yaml

    with open(template_path, 'r') as f:
        workflow_def = yaml.safe_load(f)

    # Update workflow metadata
    workflow_def['workflow']['name'] = args.name

    if args.description:
        workflow_def['workflow']['description'] = args.description

    # Apply variable overrides
    if variables:
        workflow_def.setdefault('variables', {}).update(variables)

    # Write new workflow
    with open(output_path, 'w') as f:
        yaml.dump(workflow_def, f, default_flow_style=False, sort_keys=False)

    print(f"Created workflow: {output_path}")

    # Register if requested
    if args.register:
        if registry.register_custom_workflow(output_path, copy_to_custom=True):
            print(f"Registered workflow: {args.name}")

    return 0


def search_workflows(args):
    """Search workflows"""
    registry = WorkflowRegistry()

    results = registry.search_workflows(args.query)

    if not results:
        print(f"No workflows found matching: {args.query}")
        return

    print(f"\nSearch Results ({len(results)}):\n")

    for workflow in results:
        print(f"Name: {workflow['name']}")
        print(f"  Description: {workflow['description']}")
        print(f"  Category: {workflow['category']}")
        print()


def workflow_info(args):
    """Show workflow information"""
    registry = WorkflowRegistry()

    workflow = registry.get_workflow(args.workflow)

    if not workflow:
        print(f"Error: Workflow not found: {args.workflow}")
        return 1

    print(f"\nWorkflow Information\n")
    print("=" * 60)
    print(f"Name: {workflow['name']}")
    print(f"Description: {workflow['description']}")
    print(f"Version: {workflow['version']}")
    print(f"Author: {workflow['author']}")
    print(f"Category: {workflow['category']}")
    print(f"Steps: {workflow['steps_count']}")

    if workflow['tags']:
        print(f"Tags: {', '.join(workflow['tags'])}")

    print(f"Has Parallel Steps: {workflow['has_parallel']}")

    print(f"\nCommands Used ({len(workflow['commands_used'])}):")
    for cmd in workflow['commands_used']:
        print(f"  - {cmd}")

    print(f"\nWorkflow Path: {workflow['path']}")

    return 0


def registry_stats(args):
    """Show registry statistics"""
    registry = WorkflowRegistry()

    stats = registry.get_workflow_stats()

    print("\nWorkflow Registry Statistics\n")
    print("=" * 60)
    print(f"Total Workflows: {stats['total_workflows']}")
    print(f"Template Workflows: {stats['template_workflows']}")
    print(f"Custom Workflows: {stats['custom_workflows']}")
    print(f"Workflows with Parallel Steps: {stats['workflows_with_parallel']}")
    print(f"Average Steps per Workflow: {stats['average_steps']:.1f}")

    if stats['most_used_commands']:
        print(f"\nMost Used Commands:")
        for cmd, count in stats['most_used_commands'][:5]:
            print(f"  {cmd}: {count}")

    if stats['common_tags']:
        print(f"\nCommon Tags:")
        for tag, count in stats['common_tags'][:5]:
            print(f"  {tag}: {count}")

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Workflow CLI - Manage and execute command workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List available workflows')
    list_parser.add_argument('--category', help='Filter by category')
    list_parser.add_argument('--tag', help='Filter by tag')
    list_parser.set_defaults(func=list_workflows)

    # Run command
    run_parser = subparsers.add_parser('run', help='Run a workflow')
    run_parser.add_argument('workflow', help='Workflow name or path')
    run_parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    run_parser.add_argument('--var', action='append', help='Variable override (key=value)')
    run_parser.add_argument('--output', '-o', help='Save result to file')
    run_parser.add_argument('--log', help='Log file path')
    run_parser.add_argument('--no-progress', action='store_true', help='Disable progress tracking')
    run_parser.set_defaults(func=run_workflow)

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a workflow')
    validate_parser.add_argument('workflow', help='Workflow path')
    validate_parser.add_argument('--strict', action='store_true', help='Fail on warnings')
    validate_parser.set_defaults(func=validate_workflow)

    # Create command
    create_parser = subparsers.add_parser('create', help='Create workflow from template')
    create_parser.add_argument('name', help='New workflow name')
    create_parser.add_argument('--template', '-t', required=True, help='Template name')
    create_parser.add_argument('--description', '-d', help='Workflow description')
    create_parser.add_argument('--var', action='append', help='Variable override (key=value)')
    create_parser.add_argument('--output', '-o', help='Output path')
    create_parser.add_argument('--register', action='store_true', help='Register after creation')
    create_parser.set_defaults(func=create_workflow)

    # Search command
    search_parser = subparsers.add_parser('search', help='Search workflows')
    search_parser.add_argument('query', help='Search query')
    search_parser.set_defaults(func=search_workflows)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show workflow information')
    info_parser.add_argument('workflow', help='Workflow name')
    info_parser.set_defaults(func=workflow_info)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show registry statistics')
    stats_parser.set_defaults(func=registry_stats)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())