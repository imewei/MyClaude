#!/usr/bin/env python3
"""
Integration Workflow Generator

Creates integration workflow documentation by:
- Identifying common plugin combinations
- Generating workflow documentation templates
- Creating integration examples
- Documenting multi-plugin use cases

Author: Technical Writer / Systems Architect
Part of: Plugin Review and Optimization - Task Group 0.4
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
import sys


@dataclass
class WorkflowStep:
    """Single step in a workflow"""
    step_number: int
    plugin: str
    agent: str = ""
    command: str = ""
    skill: str = ""
    description: str = ""
    expected_output: str = ""


@dataclass
class IntegrationWorkflow:
    """Complete workflow integrating multiple plugins"""
    name: str
    description: str
    category: str
    plugins: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    example_use_cases: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)


class WorkflowGenerator:
    """Generates integration workflow documentation"""

    # Predefined workflow templates for common scenarios
    WORKFLOW_TEMPLATES = {
        "scientific-computing-full-stack": {
            "name": "Scientific Computing Full-Stack Workflow",
            "description": "End-to-end scientific computing workflow from problem formulation to publication",
            "category": "scientific-computing",
            "plugins": ["julia-development", "python-development", "hpc-computing",
                       "data-visualization", "research-methodology", "code-documentation"],
            "steps": [
                {"plugin": "research-methodology", "description": "Define research question and methodology"},
                {"plugin": "julia-development", "description": "Implement computational model"},
                {"plugin": "hpc-computing", "description": "Optimize for parallel computing"},
                {"plugin": "python-development", "description": "Create data analysis pipeline"},
                {"plugin": "data-visualization", "description": "Generate publication-quality figures"},
                {"plugin": "code-documentation", "description": "Document code and methods"},
            ]
        },
        "julia-sciml-bayesian": {
            "name": "Julia SciML + Bayesian Analysis Workflow",
            "description": "Solve differential equations with uncertainty quantification",
            "category": "scientific-computing",
            "plugins": ["julia-development", "jax-implementation", "hpc-computing"],
            "steps": [
                {"plugin": "julia-development", "command": "sciml-setup",
                 "description": "Set up SciML project with ODE/PDE problem"},
                {"plugin": "julia-development", "agent": "turing-pro",
                 "description": "Define Bayesian model for parameter inference"},
                {"plugin": "hpc-computing", "description": "Parallelize MCMC sampling"},
                {"plugin": "jax-implementation", "description": "Alternative JAX implementation for comparison"},
            ]
        },
        "machine-learning-pipeline": {
            "name": "Machine Learning Development Pipeline",
            "description": "Complete ML workflow from data to deployment",
            "category": "development",
            "plugins": ["python-development", "machine-learning", "deep-learning",
                       "unit-testing", "cicd-automation", "observability-monitoring"],
            "steps": [
                {"plugin": "python-development", "description": "Set up Python project structure"},
                {"plugin": "machine-learning", "description": "Develop ML model and training pipeline"},
                {"plugin": "deep-learning", "description": "Implement neural network architectures"},
                {"plugin": "unit-testing", "command": "test-generate",
                 "description": "Generate tests for ML components"},
                {"plugin": "cicd-automation", "command": "workflow-automate",
                 "description": "Set up CI/CD for model training and deployment"},
                {"plugin": "observability-monitoring", "description": "Add monitoring and logging"},
            ]
        },
        "full-stack-web-app": {
            "name": "Full-Stack Web Application Workflow",
            "description": "Build and deploy a complete web application",
            "category": "development",
            "plugins": ["frontend-mobile-development", "backend-development", "python-development",
                       "unit-testing", "git-pr-workflows", "cicd-automation"],
            "steps": [
                {"plugin": "frontend-mobile-development", "description": "Build React/TypeScript frontend"},
                {"plugin": "backend-development", "description": "Create FastAPI backend"},
                {"plugin": "python-development", "description": "Implement business logic"},
                {"plugin": "unit-testing", "description": "Write comprehensive tests"},
                {"plugin": "git-pr-workflows", "description": "Create feature branch and PR"},
                {"plugin": "cicd-automation", "description": "Deploy to production"},
            ]
        },
        "molecular-dynamics-workflow": {
            "name": "Molecular Dynamics Simulation Workflow",
            "description": "Run and analyze molecular simulations",
            "category": "scientific-computing",
            "plugins": ["molecular-simulation", "python-development", "hpc-computing",
                       "statistical-physics", "data-visualization"],
            "steps": [
                {"plugin": "molecular-simulation", "description": "Set up MD simulation"},
                {"plugin": "hpc-computing", "description": "Optimize for GPU acceleration"},
                {"plugin": "python-development", "description": "Create analysis scripts"},
                {"plugin": "statistical-physics", "description": "Calculate thermodynamic properties"},
                {"plugin": "data-visualization", "description": "Visualize trajectories and results"},
            ]
        },
        "code-quality-workflow": {
            "name": "Code Quality Assurance Workflow",
            "description": "Comprehensive code review and quality improvement",
            "category": "quality",
            "plugins": ["comprehensive-review", "quality-engineering", "unit-testing",
                       "debugging-toolkit", "code-documentation", "codebase-cleanup"],
            "steps": [
                {"plugin": "comprehensive-review", "command": "full-review",
                 "description": "Perform multi-perspective code review"},
                {"plugin": "quality-engineering", "command": "double-check",
                 "description": "Run comprehensive validation"},
                {"plugin": "unit-testing", "command": "run-all-tests",
                 "description": "Execute full test suite"},
                {"plugin": "debugging-toolkit", "description": "Debug failing tests"},
                {"plugin": "code-documentation", "command": "update-docs",
                 "description": "Update documentation"},
                {"plugin": "codebase-cleanup", "description": "Clean up code structure"},
            ]
        },
        "jax-development-workflow": {
            "name": "JAX Scientific Computing Workflow",
            "description": "Develop high-performance scientific code with JAX",
            "category": "scientific-computing",
            "plugins": ["jax-implementation", "python-development", "hpc-computing",
                       "deep-learning", "unit-testing"],
            "steps": [
                {"plugin": "jax-implementation", "description": "Implement JAX transformations"},
                {"plugin": "python-development", "description": "Set up project structure"},
                {"plugin": "hpc-computing", "description": "Configure GPU acceleration"},
                {"plugin": "deep-learning", "description": "Build neural networks with Flax"},
                {"plugin": "unit-testing", "description": "Test JIT-compiled functions"},
            ]
        },
        "ci-cd-testing-workflow": {
            "name": "CI/CD Testing Pipeline",
            "description": "Automated testing and deployment pipeline",
            "category": "devops",
            "plugins": ["git-pr-workflows", "unit-testing", "cicd-automation",
                       "quality-engineering", "observability-monitoring"],
            "steps": [
                {"plugin": "git-pr-workflows", "description": "Create PR with changes"},
                {"plugin": "unit-testing", "description": "Run automated tests"},
                {"plugin": "quality-engineering", "description": "Validate code quality"},
                {"plugin": "cicd-automation", "description": "Deploy to staging"},
                {"plugin": "observability-monitoring", "description": "Monitor deployment"},
            ]
        },
    }

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = plugins_dir
        self.workflows: List[IntegrationWorkflow] = []
        self.plugin_metadata: Dict[str, dict] = {}
        self._load_plugin_metadata()

    def _load_plugin_metadata(self):
        """Load metadata from all plugins"""
        plugin_dirs = [d for d in self.plugins_dir.iterdir() if d.is_dir()]

        for plugin_dir in sorted(plugin_dirs):
            plugin_json = plugin_dir / "plugin.json"
            if plugin_json.exists():
                try:
                    with open(plugin_json) as f:
                        data = json.load(f)
                    plugin_name = data.get("name", plugin_dir.name)
                    self.plugin_metadata[plugin_name] = data
                except Exception as e:
                    print(f"⚠️  Warning: Error loading {plugin_json}: {e}")

    def generate_workflows(self) -> List[IntegrationWorkflow]:
        """Generate all predefined workflows"""
        print("🔍 Generating integration workflows...")

        for workflow_id, template in self.WORKFLOW_TEMPLATES.items():
            workflow = self._build_workflow_from_template(template)
            if workflow:
                self.workflows.append(workflow)

        # Generate custom workflows based on plugin analysis
        self._generate_custom_workflows()

        return self.workflows

    def _build_workflow_from_template(self, template: dict) -> IntegrationWorkflow:
        """Build a workflow from a template"""
        # Validate that all plugins exist
        missing_plugins = [
            p for p in template["plugins"]
            if p not in self.plugin_metadata
        ]

        if missing_plugins:
            print(f"⚠️  Warning: Workflow '{template['name']}' references "
                  f"missing plugins: {', '.join(missing_plugins)}")
            return None

        workflow = IntegrationWorkflow(
            name=template["name"],
            description=template["description"],
            category=template["category"],
            plugins=template["plugins"]
        )

        # Build steps
        for i, step_template in enumerate(template.get("steps", []), 1):
            plugin_name = step_template["plugin"]
            plugin_data = self.plugin_metadata.get(plugin_name, {})

            step = WorkflowStep(
                step_number=i,
                plugin=plugin_name,
                agent=step_template.get("agent", ""),
                command=step_template.get("command", ""),
                skill=step_template.get("skill", ""),
                description=step_template.get("description", ""),
            )

            # Enrich step with plugin details
            if step.agent and 'agents' in plugin_data:
                agent_data = next(
                    (a for a in plugin_data['agents'] if a.get('name') == step.agent),
                    None
                )
                if agent_data:
                    step.description += f" (Agent: {agent_data.get('description', '')})"

            if step.command and 'commands' in plugin_data:
                cmd_data = next(
                    (c for c in plugin_data['commands'] if c.get('name') == step.command),
                    None
                )
                if cmd_data:
                    step.description += f" (Command: {cmd_data.get('description', '')})"

            workflow.steps.append(step)

        return workflow

    def _generate_custom_workflows(self):
        """Generate custom workflows based on plugin categories"""
        # Group plugins by category
        by_category = defaultdict(list)
        for plugin_name, data in self.plugin_metadata.items():
            category = data.get("category", "uncategorized")
            by_category[category].append(plugin_name)

        # Generate category-specific workflows
        for category, plugins in by_category.items():
            if len(plugins) >= 3:  # Need at least 3 plugins for a workflow
                workflow = self._create_category_workflow(category, plugins)
                if workflow:
                    self.workflows.append(workflow)

    def _create_category_workflow(
        self, category: str, plugins: List[str]
    ) -> IntegrationWorkflow:
        """Create a generic workflow for a category"""
        workflow = IntegrationWorkflow(
            name=f"{category.replace('-', ' ').title()} Integration Workflow",
            description=f"Integrate multiple {category} plugins for comprehensive workflow",
            category=category,
            plugins=plugins[:5]  # Limit to 5 plugins
        )

        # Create generic steps
        for i, plugin in enumerate(workflow.plugins, 1):
            plugin_data = self.plugin_metadata.get(plugin, {})
            description = plugin_data.get("description", f"Use {plugin}")

            step = WorkflowStep(
                step_number=i,
                plugin=plugin,
                description=description[:100]  # Truncate long descriptions
            )
            workflow.steps.append(step)

        return workflow

    def generate_report(self, output_path: Path = None) -> str:
        """Generate comprehensive workflow documentation"""
        lines = []

        # Header
        lines.append("# Plugin Integration Workflows")
        lines.append("")
        lines.append("Common multi-plugin workflows for Claude Code marketplace")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Workflows:** {len(self.workflows)}")
        by_category = defaultdict(int)
        for workflow in self.workflows:
            by_category[workflow.category] += 1
        for category in sorted(by_category.keys()):
            lines.append(f"  - {category}: {by_category[category]} workflows")
        lines.append("")

        # Table of Contents
        lines.append("## Table of Contents")
        lines.append("")
        for i, workflow in enumerate(self.workflows, 1):
            anchor = workflow.name.lower().replace(" ", "-").replace("/", "-")
            lines.append(f"{i}. [{workflow.name}](#{anchor})")
        lines.append("")

        # Workflows by category
        workflows_by_category = defaultdict(list)
        for workflow in self.workflows:
            workflows_by_category[workflow.category].append(workflow)

        for category in sorted(workflows_by_category.keys()):
            lines.append(f"## {category.replace('-', ' ').title()} Workflows")
            lines.append("")

            for workflow in workflows_by_category[category]:
                self._add_workflow_section(lines, workflow)

        report = "\n".join(lines)

        # Write to file if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding='utf-8')
            print(f"✅ Workflow documentation saved to: {output_path}")

        return report

    def _add_workflow_section(self, lines: List[str], workflow: IntegrationWorkflow):
        """Add a workflow section to the report"""
        lines.append(f"### {workflow.name}")
        lines.append("")
        lines.append(f"**Description:** {workflow.description}")
        lines.append("")

        # Plugins involved
        lines.append("**Plugins:**")
        for plugin in workflow.plugins:
            plugin_data = self.plugin_metadata.get(plugin, {})
            desc = plugin_data.get("description", "")
            if desc:
                lines.append(f"- `{plugin}` - {desc[:80]}")
            else:
                lines.append(f"- `{plugin}`")
        lines.append("")

        # Workflow steps
        if workflow.steps:
            lines.append("**Workflow Steps:**")
            lines.append("")
            for step in workflow.steps:
                lines.append(f"{step.step_number}. **{step.plugin}**")
                if step.agent:
                    lines.append(f"   - Agent: `{step.agent}`")
                if step.command:
                    lines.append(f"   - Command: `{step.command}`")
                if step.skill:
                    lines.append(f"   - Skill: `{step.skill}`")
                lines.append(f"   - {step.description}")
                if step.expected_output:
                    lines.append(f"   - Expected: {step.expected_output}")
                lines.append("")

        # Example usage
        lines.append("**Example Usage:**")
        lines.append("")
        lines.append("```bash")
        for step in workflow.steps[:5]:  # First 5 steps
            if step.command:
                lines.append(f"# Step {step.step_number}: {step.description[:60]}")
                lines.append(f"/{step.command}")
            else:
                lines.append(f"# Step {step.step_number}: Use {step.plugin}")
        lines.append("```")
        lines.append("")

        lines.append("---")
        lines.append("")

    def export_workflows_json(self, output_path: Path):
        """Export workflows as JSON"""
        workflows_data = [
            {
                "name": w.name,
                "description": w.description,
                "category": w.category,
                "plugins": w.plugins,
                "steps": [
                    {
                        "step": s.step_number,
                        "plugin": s.plugin,
                        "agent": s.agent,
                        "command": s.command,
                        "skill": s.skill,
                        "description": s.description,
                        "expected_output": s.expected_output
                    }
                    for s in w.steps
                ],
                "prerequisites": w.prerequisites,
                "example_use_cases": w.example_use_cases,
                "expected_outcomes": w.expected_outcomes
            }
            for w in self.workflows
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(workflows_data, f, indent=2)

        print(f"✅ Workflows exported to: {output_path}")

    def generate_workflow_template(self, name: str, plugins: List[str]) -> str:
        """Generate a blank workflow template"""
        lines = []
        lines.append(f"# {name}")
        lines.append("")
        lines.append("**Description:** [Describe the workflow purpose and goals]")
        lines.append("")
        lines.append("**Category:** [scientific-computing|development|devops|quality]")
        lines.append("")
        lines.append("**Plugins:**")
        for plugin in plugins:
            lines.append(f"- `{plugin}`")
        lines.append("")
        lines.append("**Workflow Steps:**")
        lines.append("")
        for i, plugin in enumerate(plugins, 1):
            lines.append(f"{i}. **{plugin}**")
            lines.append("   - Agent: [agent-name or leave blank]")
            lines.append("   - Command: [/command-name or leave blank]")
            lines.append("   - Description: [What to do in this step]")
            lines.append("")
        lines.append("**Prerequisites:**")
        lines.append("- [List any prerequisites]")
        lines.append("")
        lines.append("**Expected Outcomes:**")
        lines.append("- [List expected results]")
        lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate integration workflow documentation for Claude Code plugins"
    )
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=Path.cwd() / "plugins",
        help="Path to plugins directory (default: ./plugins)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/integration-workflows.md"),
        help="Output file for workflow documentation (default: reports/integration-workflows.md)"
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Export workflows as JSON"
    )
    parser.add_argument(
        "--generate-template",
        nargs="+",
        metavar="PLUGIN",
        help="Generate a blank workflow template for specified plugins"
    )

    args = parser.parse_args()

    # Validate plugins directory
    if not args.plugins_dir.exists():
        print(f"❌ Error: Plugins directory not found: {args.plugins_dir}")
        sys.exit(1)

    # Generate template if requested
    if args.generate_template:
        generator = WorkflowGenerator(args.plugins_dir)
        template = generator.generate_workflow_template(
            "Custom Workflow",
            args.generate_template
        )
        print(template)
        return 0

    # Create generator and generate workflows
    generator = WorkflowGenerator(args.plugins_dir)
    workflows = generator.generate_workflows()

    # Generate report
    print("\n📊 Generating workflow documentation...")
    generator.generate_report(args.output)

    # Export JSON if requested
    if args.export_json:
        generator.export_workflows_json(args.export_json)

    # Print summary
    print("\n✅ Generation complete!")
    print(f"   Workflows created: {len(workflows)}")
    print(f"   Documentation saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
