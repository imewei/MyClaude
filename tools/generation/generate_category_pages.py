#!/usr/bin/env python3
"""
Generate category landing pages for Sphinx documentation.

Task Group 5.2 & 5.3: Create and populate category landing pages.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class CategoryPageGenerator:
    """Generate category landing pages with plugin listings"""

    def __init__(self, plugins_dir: Path, categories_dir: Path):
        self.plugins_dir = plugins_dir
        self.categories_dir = categories_dir
        self.category_data = self._collect_category_data()

    def _collect_category_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect all plugins organized by category"""
        category_plugins = defaultdict(list)

        for plugin_dir in sorted(self.plugins_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue

            plugin_json = plugin_dir / "plugin.json"
            if not plugin_json.exists():
                continue

            with open(plugin_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            category = data.get('category', 'tools')
            plugin_info = {
                'dir_name': plugin_dir.name,
                'name': data.get('name', plugin_dir.name),
                'description': data.get('description', ''),
                'agents': data.get('agents', []),
                'commands': data.get('commands', []),
                'skills': data.get('skills', []),
                'keywords': data.get('keywords', []),
            }
            category_plugins[category].append(plugin_info)

        return category_plugins

    def _generate_category_page(
        self,
        category_name: str,
        display_name: str,
        description: str,
        plugins: List[Dict[str, Any]]
    ) -> str:
        """Generate RST content for a category page"""

        # Calculate statistics
        total_agents = sum(len(p['agents']) for p in plugins)
        total_commands = sum(len(p['commands']) for p in plugins)
        total_skills = sum(len(p['skills']) for p in plugins)

        # Build RST content
        lines = []

        # Title
        lines.append(display_name)
        lines.append("=" * len(display_name))
        lines.append("")

        # Overview
        lines.append(description)
        lines.append("")

        # Statistics
        lines.append("Statistics")
        lines.append("-" * 10)
        lines.append("")
        lines.append(f"- **Plugins:** {len(plugins)}")
        lines.append(f"- **Total Agents:** {total_agents}")
        lines.append(f"- **Total Commands:** {total_commands}")
        lines.append(f"- **Total Skills:** {total_skills}")
        lines.append("")

        # Plugin descriptions
        lines.append("Plugins in This Category")
        lines.append("-" * 24)
        lines.append("")

        for plugin in sorted(plugins, key=lambda p: p['name']):
            lines.append(f"**{plugin['name']}**")
            lines.append("")
            if plugin['description']:
                lines.append(plugin['description'])
                lines.append("")
            lines.append(f"- Agents: {len(plugin['agents'])}")
            lines.append(f"- Commands: {len(plugin['commands'])}")
            lines.append(f"- Skills: {len(plugin['skills'])}")
            lines.append("")

        # Common use cases
        lines.append("Common Use Cases")
        lines.append("-" * 16)
        lines.append("")
        use_cases = self._generate_use_cases(category_name, plugins)
        lines.extend(use_cases)
        lines.append("")

        # Integration patterns
        lines.append("Integration Patterns")
        lines.append("-" * 19)
        lines.append("")
        integration_patterns = self._generate_integration_patterns(category_name, plugins)
        lines.extend(integration_patterns)
        lines.append("")

        # TOC tree
        lines.append("Plugin Documentation")
        lines.append("-" * 20)
        lines.append("")
        lines.append(".. toctree::")
        lines.append("   :maxdepth: 1")
        lines.append("   :caption: Plugins")
        lines.append("")

        for plugin in sorted(plugins, key=lambda p: p['name']):
            lines.append(f"   /plugins/{plugin['dir_name']}")

        lines.append("")

        return "\n".join(lines)

    def _generate_use_cases(self, category: str, plugins: List[Dict[str, Any]]) -> List[str]:
        """Generate common use cases for the category"""
        use_cases_map = {
            'scientific-computing': [
                "- High-performance numerical simulations",
                "- Scientific machine learning and differential equations",
                "- Data visualization and analysis for research",
                "- Bayesian inference and probabilistic programming",
            ],
            'development': [
                "- Full-stack application development",
                "- API design and microservices architecture",
                "- Code migration and framework modernization",
                "- Package and library development",
                "- Systems programming and performance optimization",
            ],
            'devops': [
                "- Continuous integration and deployment",
                "- Git workflow automation and PR management",
                "- Infrastructure monitoring and observability",
                "- Performance tracking and alerting",
            ],
            'ai-ml': [
                "- Deep learning model development and training",
                "- Machine learning pipeline automation",
                "- MLOps and model deployment",
                "- Neural architecture design and optimization",
            ],
            'tools': [
                "- Development workflow automation",
                "- Code quality analysis and improvement",
                "- Documentation generation",
                "- Testing and quality engineering",
                "- Agent orchestration and coordination",
            ],
            'orchestration': [
                "- End-to-end full-stack workflow coordination",
                "- Multi-layer application development",
                "- Frontend-backend-database integration",
            ],
            'quality': [
                "- Comprehensive code review and analysis",
                "- Security auditing and vulnerability detection",
                "- Multi-perspective code quality assessment",
            ],
            'developer-tools': [
                "- Command-line tool design and development",
                "- Developer automation and productivity",
            ],
            'dev-tools': [
                "- Interactive debugging and troubleshooting",
                "- Developer experience optimization",
            ],
        }

        return use_cases_map.get(category, ["- General development and automation tasks"])

    def _generate_integration_patterns(
        self,
        category: str,
        plugins: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate integration patterns for the category"""
        patterns = []

        if category == 'scientific-computing':
            patterns.extend([
                "Plugins in this category integrate well with:",
                "",
                "- **HPC and GPU Computing:** Combine Julia development with high-performance computing for large-scale simulations",
                "- **Data Visualization:** Use visualization plugins to analyze and present scientific results",
                "- **AI/ML:** Integration with JAX and deep learning for physics-informed machine learning",
            ])
        elif category == 'development':
            patterns.extend([
                "Plugins in this category integrate well with:",
                "",
                "- **Testing and Quality:** Combine development plugins with testing frameworks for robust applications",
                "- **DevOps:** Integration with CI/CD pipelines for automated deployment",
                "- **LLM Applications:** Build intelligent applications with LLM development plugins",
            ])
        elif category == 'devops':
            patterns.extend([
                "Plugins in this category integrate well with:",
                "",
                "- **Development:** Automate deployment for applications built with development plugins",
                "- **Monitoring:** Track application performance with observability tools",
                "- **Infrastructure:** Deploy to cloud platforms with infrastructure plugins",
            ])
        elif category == 'ai-ml':
            patterns.extend([
                "Plugins in this category integrate well with:",
                "",
                "- **Scientific Computing:** Apply deep learning to scientific problems",
                "- **Development:** Build production ML applications",
                "- **Data Visualization:** Visualize model performance and results",
            ])
        else:
            patterns.extend([
                "Plugins in this category can be combined with plugins from other categories",
                "to create comprehensive development workflows. See :doc:`/integration-map`",
                "for detailed integration patterns.",
            ])

        return patterns

    def generate_all_categories(self):
        """Generate all category pages"""
        self.categories_dir.mkdir(parents=True, exist_ok=True)

        category_info = {
            'scientific-computing': {
                'display_name': 'Scientific Computing',
                'description': (
                    'Plugins for scientific computing, numerical analysis, and research workflows. '
                    'Includes support for Julia, SciML, high-performance computing, and data visualization.'
                ),
            },
            'development': {
                'display_name': 'Development',
                'description': (
                    'General-purpose development plugins covering backend, frontend, systems programming, '
                    'and application development across multiple languages and frameworks.'
                ),
            },
            'devops': {
                'display_name': 'DevOps',
                'description': (
                    'DevOps automation plugins for CI/CD, Git workflows, infrastructure management, '
                    'and system observability.'
                ),
            },
            'ai-ml': {
                'display_name': 'AI & Machine Learning',
                'description': (
                    'Artificial intelligence and machine learning plugins for deep learning, '
                    'model training, and MLOps workflows.'
                ),
            },
            'tools': {
                'display_name': 'Tools',
                'description': (
                    'General-purpose tools and utilities for development automation, code quality, '
                    'testing, documentation, and workflow coordination.'
                ),
            },
            'orchestration': {
                'display_name': 'Orchestration',
                'description': (
                    'Workflow orchestration and coordination plugins for managing complex '
                    'multi-layer applications and development processes.'
                ),
            },
            'quality': {
                'display_name': 'Quality Engineering',
                'description': (
                    'Code quality, review, and security analysis plugins for maintaining '
                    'high-quality codebases.'
                ),
            },
            'developer-tools': {
                'display_name': 'Developer Tools',
                'description': (
                    'Command-line tools and developer utilities for automation and productivity.'
                ),
            },
            'dev-tools': {
                'display_name': 'Development Tools',
                'description': (
                    'Interactive development tools for debugging and developer experience optimization.'
                ),
            },
        }

        generated_files = []

        for category, plugins in self.category_data.items():
            if not plugins:
                continue

            info = category_info.get(category, {
                'display_name': category.replace('-', ' ').title(),
                'description': f'Plugins in the {category} category.',
            })

            rst_content = self._generate_category_page(
                category,
                info['display_name'],
                info['description'],
                plugins
            )

            output_file = self.categories_dir / f"{category}.rst"
            output_file.write_text(rst_content, encoding='utf-8')
            generated_files.append(output_file)

            print(f"✓ Generated {output_file.name} ({len(plugins)} plugins)")

        return generated_files


def main():
    """Main entry point"""
    repo_root = Path(__file__).parent.parent
    plugins_dir = repo_root / "plugins"
    categories_dir = repo_root / "docs" / "categories"

    generator = CategoryPageGenerator(plugins_dir, categories_dir)
    files = generator.generate_all_categories()

    print(f"\n✓ Generated {len(files)} category pages")


if __name__ == "__main__":
    main()
