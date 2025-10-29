#!/usr/bin/env python3
"""
Sphinx Documentation Generator

Generates comprehensive Sphinx RST documentation for Claude Code plugins by:
- Extracting metadata from plugin.json files
- Generating RST files with standardized sections
- Detecting cross-plugin integrations with bidirectional references
- Creating integration matrix and workflow patterns
- Following Sphinx conventions

Reuses logic from:
- plugin-review-script.py: plugin.json reading and validation
- metadata-validator.py: schema validation and parsing
- dependency-mapper.py: cross-reference detection
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import argparse


@dataclass
class PluginMetadata:
    """Extracted plugin metadata"""
    name: str
    version: str
    description: str
    author: str
    license: str
    category: str
    keywords: List[str] = field(default_factory=list)
    agents: List[Dict[str, Any]] = field(default_factory=list)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)


@dataclass
class IntegrationReference:
    """Represents a reference between two plugins"""
    source_plugin: str
    target_plugin: str
    reference_type: str  # 'integration', 'workflow', 'documentation', 'related'
    context: str
    file_path: str


class SphinxDocGenerator:
    """Main Sphinx documentation generator with comprehensive integration mapping"""

    def __init__(self, plugins_dir: Path, verbose: bool = False):
        self.plugins_dir = Path(plugins_dir)
        self.verbose = verbose
        self.all_plugins: Set[str] = set()
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self.integration_map: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.integration_references: List[IntegrationReference] = []
        self.integration_patterns: Dict[str, List[str]] = {}

    def discover_plugins(self, plugin_filter: Optional[str] = None) -> List[Path]:
        """Discover all plugin directories"""
        plugins = []

        for plugin_dir in sorted(self.plugins_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue

            if plugin_filter and plugin_filter not in plugin_dir.name:
                continue

            plugin_json = plugin_dir / "plugin.json"
            if plugin_json.exists():
                plugins.append(plugin_dir)
                self.all_plugins.add(plugin_dir.name)

        if self.verbose:
            print(f"📦 Discovered {len(plugins)} plugins")

        return plugins

    def extract_plugin_metadata(self, plugin_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from plugin.json (reusing plugin-review-script patterns)"""
        plugin_json_path = plugin_path / "plugin.json"

        if not plugin_json_path.exists():
            if self.verbose:
                print(f"⚠️  Warning: No plugin.json found in {plugin_path}")
            return None

        try:
            with open(plugin_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Extract core fields
            result = {
                "name": metadata.get("name", plugin_path.name),
                "version": metadata.get("version", "unknown"),
                "description": metadata.get("description", ""),
                "author": self._format_author(metadata.get("author", "Unknown")),
                "license": metadata.get("license", "Unknown"),
                "category": metadata.get("category", "uncategorized"),
                "keywords": metadata.get("keywords", []),
                "agents": metadata.get("agents", []),
                "commands": metadata.get("commands", []),
                "skills": metadata.get("skills", []),
                "prerequisites": metadata.get("prerequisites", []),
                "integration_points": metadata.get("integration_points", [])
            }

            return result

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ Error: Invalid JSON in {plugin_json_path}: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"❌ Error reading {plugin_json_path}: {e}")
            return None

    def _format_author(self, author: Any) -> str:
        """Format author field (handle string or object)"""
        if isinstance(author, str):
            return author
        elif isinstance(author, dict):
            name = author.get("name", "Unknown")
            url = author.get("url", "")
            if url:
                return f"{name} <{url}>"
            return name
        return "Unknown"

    def detect_integrations(self) -> Dict[str, Set[str]]:
        """
        Comprehensive integration scanning across all 31 plugins

        Task 3.2: Scan all plugin README.md files for plugin name references
        - Detect references in: descriptions, usage examples, integration sections
        - Build bidirectional reference map
        - Categorize reference types
        """
        if self.verbose:
            print("🔍 Detecting plugin integrations...")

        integration_map = defaultdict(set)

        for plugin_name in self.all_plugins:
            plugin_path = self.plugins_dir / plugin_name

            # Scan README for references
            readme_path = plugin_path / "README.md"
            if readme_path.exists():
                references = self._scan_file_for_references(readme_path, plugin_name)
                for target, ref_type, context in references:
                    integration_map[plugin_name].add(target)

            # Scan agent documentation
            agents_dir = plugin_path / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.md"):
                    references = self._scan_file_for_references(agent_file, plugin_name)
                    for target, ref_type, context in references:
                        integration_map[plugin_name].add(target)

            # Scan command documentation
            commands_dir = plugin_path / "commands"
            if commands_dir.exists():
                for command_file in commands_dir.glob("*.md"):
                    references = self._scan_file_for_references(command_file, plugin_name)
                    for target, ref_type, context in references:
                        integration_map[plugin_name].add(target)

            # Scan skill documentation
            skills_dir = plugin_path / "skills"
            if skills_dir.exists():
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir():
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.exists():
                            references = self._scan_file_for_references(skill_file, plugin_name)
                            for target, ref_type, context in references:
                                integration_map[plugin_name].add(target)

        self.integration_map = integration_map

        if self.verbose:
            total_refs = sum(len(targets) for targets in integration_map.values())
            print(f"✅ Found {total_refs} integration references")

        return dict(integration_map)

    def _scan_file_for_references(
        self,
        file_path: Path,
        source_plugin: str
    ) -> List[Tuple[str, str, str]]:
        """
        Scan file for references to other plugins

        Returns list of (target_plugin, reference_type, context) tuples
        Uses dependency-mapper.py _extract_documentation_references() pattern
        """
        references = []

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                for target_plugin in self.all_plugins:
                    if target_plugin == source_plugin:
                        continue

                    # Check for plugin name mentions using patterns
                    patterns = [
                        rf'\b{re.escape(target_plugin)}\b',
                        rf'\b{re.escape(target_plugin)}\s+plugin\b',
                        rf'\[.*?{re.escape(target_plugin)}.*?\]',
                    ]

                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            # Determine reference type
                            ref_type = self._determine_reference_type(line, target_plugin)

                            # Extract context
                            context = line.strip()
                            if len(context) > 100:
                                context = context[:97] + "..."

                            # Record reference
                            ref = IntegrationReference(
                                source_plugin=source_plugin,
                                target_plugin=target_plugin,
                                reference_type=ref_type,
                                context=context,
                                file_path=str(file_path.relative_to(self.plugins_dir))
                            )
                            self.integration_references.append(ref)

                            references.append((target_plugin, ref_type, context))
                            break  # Avoid duplicate references from same line

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Warning: Error scanning {file_path}: {e}")

        return references

    def _determine_reference_type(self, line: str, target_plugin: str) -> str:
        """
        Determine reference type using _determine_reference_type() pattern
        from dependency-mapper.py
        """
        line_lower = line.lower()

        # Check for specific contexts
        if any(word in line_lower for word in ['integrate', 'integration', 'combine', 'work with', 'together with']):
            return 'integration'
        elif any(word in line_lower for word in ['workflow', 'pipeline', 'process', 'chain']):
            return 'workflow'
        elif any(word in line_lower for word in ['agent', 'expert', 'specialist']):
            return 'agent'
        elif any(word in line_lower for word in ['command', 'slash-command', '/']):
            return 'command'
        elif any(word in line_lower for word in ['skill', 'pattern', 'technique']):
            return 'skill'
        elif any(word in line_lower for word in ['see also', 'related', 'similar', 'like']):
            return 'related'
        else:
            return 'documentation'

    def build_reverse_dependencies(self) -> Dict[str, Set[str]]:
        """
        Build bidirectional reference map

        Task 3.2: Build reverse dependency map (A references B, B referenced by A)
        """
        reverse_deps = defaultdict(set)

        for source_plugin, target_plugins in self.integration_map.items():
            for target_plugin in target_plugins:
                reverse_deps[target_plugin].add(source_plugin)

        self.reverse_dependencies = reverse_deps

        if self.verbose:
            plugins_referenced = len([p for p, refs in reverse_deps.items() if refs])
            print(f"📊 Built reverse dependencies: {plugins_referenced} plugins are referenced by others")

        return dict(reverse_deps)

    def identify_integration_patterns(self) -> Dict[str, List[str]]:
        """
        Create integration pattern detector

        Task 3.3: Identify common multi-plugin workflows
        - Group plugins by shared keywords (e.g., "HPC", "scientific", "ML")
        - Detect category-based integration patterns
        - Store patterns for quick-start guide generation
        """
        if self.verbose:
            print("🔍 Identifying integration patterns...")

        patterns = {}

        # Load all plugin metadata
        for plugin_name in self.all_plugins:
            plugin_path = self.plugins_dir / plugin_name
            metadata = self.extract_plugin_metadata(plugin_path)
            if metadata:
                self.plugin_metadata[plugin_name] = metadata

        # Category-based patterns
        category_groups = defaultdict(list)
        for plugin_name, metadata in self.plugin_metadata.items():
            category = metadata.get("category", "uncategorized")
            category_groups[category].append(plugin_name)

        # Add category patterns with 3+ plugins
        for category, plugins in category_groups.items():
            if len(plugins) >= 3:
                pattern_name = f"{category.replace('-', ' ').title()} Workflow"
                patterns[pattern_name] = sorted(plugins[:5])  # Top 5

        # Keyword-based patterns
        keyword_groups = defaultdict(list)
        for plugin_name, metadata in self.plugin_metadata.items():
            keywords = metadata.get("keywords", [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_groups[keyword_lower].append(plugin_name)

        # Identify common multi-plugin workflows based on keywords
        important_keywords = ["hpc", "scientific", "ml", "machine-learning", "gpu",
                             "parallel", "cloud", "kubernetes", "docker", "testing",
                             "api", "rest", "microservices", "frontend", "backend"]

        for keyword in important_keywords:
            if keyword in keyword_groups and len(keyword_groups[keyword]) >= 2:
                pattern_name = f"{keyword.upper() if keyword == 'hpc' or keyword == 'ml' else keyword.title()} Integration Pattern"
                patterns[pattern_name] = sorted(keyword_groups[keyword][:5])

        # Specific workflow patterns (examples from spec)
        # Julia + HPC + GPU pattern
        julia_like = [p for p in self.all_plugins if 'julia' in p.lower()]
        hpc_like = [p for p in self.all_plugins if 'hpc' in p.lower()]
        gpu_like = [p for p in self.all_plugins if 'gpu' in p.lower()]

        if julia_like and hpc_like:
            patterns["Scientific Computing HPC Workflow"] = julia_like + hpc_like + gpu_like

        # Python + API + Testing pattern
        python_like = [p for p in self.all_plugins if 'python' in p.lower()]
        api_like = [p for p in self.all_plugins if 'api' in p.lower()]
        testing_like = [p for p in self.all_plugins if 'test' in p.lower()]

        if python_like and (api_like or testing_like):
            patterns["Development & Testing Workflow"] = python_like + api_like + testing_like

        self.integration_patterns = patterns

        if self.verbose:
            print(f"✅ Identified {len(patterns)} integration patterns")

        return patterns
    def extract_code_blocks_from_readme(self, plugin_path: Path) -> List[Dict[str, Any]]:
        """
        Task 4.2: Extract code blocks from plugin README.md

        Parse existing plugin README.md files and identify code blocks
        (markdown fenced blocks ```language)
        """
        readme_path = plugin_path / "README.md"
        if not readme_path.exists():
            return []

        try:
            content = readme_path.read_text(encoding='utf-8')
            return self._parse_markdown_code_blocks(content)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Warning: Error reading README from {plugin_path}: {e}")
            return []

    def _parse_markdown_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse markdown content and extract code blocks with context

        Task 4.2: Identify code blocks (markdown fenced blocks ```language)
        - Extract usage examples, installation instructions, API examples
        - Preserve code language tags for Sphinx highlighting
        """
        code_blocks = []

        # Pattern to match fenced code blocks with optional language
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) if match.group(1) else ''
            code = match.group(2).strip()

            # Try to extract context (heading before the code block)
            context = self._extract_code_context(content, match.start())

            code_blocks.append({
                'language': language,
                'code': code,
                'context': context,
                'position': match.start()
            })

        return code_blocks

    def _extract_code_context(self, content: str, code_position: int) -> str:
        """Extract context (heading/paragraph) before a code block"""
        # Find the text before the code block
        before_code = content[:code_position]
        lines = before_code.split('\n')

        # Look for the last heading or paragraph before code
        context = ""
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()

            # Found a heading
            if line.startswith('#'):
                context = line.lstrip('#').strip()
                break

            # Found non-empty line (potential context)
            if line and not line.startswith('```'):
                # Don't use very long lines as context
                if len(line) < 100:
                    context = line
                break

        return context

    def map_code_blocks_to_sections(
        self,
        code_blocks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map code examples to appropriate plugin sections

        Task 4.2: Map examples to appropriate plugin sections
        (Commands, Skills, Usage Examples)
        """
        sections = {
            'installation': [],
            'usage': [],
            'commands': [],
            'skills': [],
            'configuration': [],
            'advanced': [],
            'examples': []
        }

        for block in code_blocks:
            context_lower = block['context'].lower()

            # Classify based on context
            if any(word in context_lower for word in ['install', 'setup', 'requirement']):
                sections['installation'].append(block)
            elif any(word in context_lower for word in ['command', '/']) :
                sections['commands'].append(block)
            elif any(word in context_lower for word in ['skill', 'pattern', 'technique']):
                sections['skills'].append(block)
            elif any(word in context_lower for word in ['config', 'setting']):
                sections['configuration'].append(block)
            elif any(word in context_lower for word in ['advanced', 'expert', 'complex']):
                sections['advanced'].append(block)
            elif any(word in context_lower for word in ['usage', 'example', 'quick start', 'getting started']):
                sections['usage'].append(block)
            else:
                # Default to examples
                sections['examples'].append(block)

        return sections

    def generate_rst_code_block(
        self,
        code_block: Dict[str, Any],
        include_context: bool = True
    ) -> str:
        """
        Generate RST code-block directive from code block data

        Task 4.2: Preserve code language tags for Sphinx highlighting
        """
        lines = []

        # Add context if available and requested
        if include_context and code_block.get('context'):
            lines.append(code_block['context'])
            lines.append('')

        # Add code-block directive
        language = code_block.get('language', '')
        if language:
            lines.append(f".. code-block:: {language}")
        else:
            lines.append(".. code-block::")
        lines.append('')

        # Add code with proper indentation (3 spaces for RST)
        code = code_block['code']
        for code_line in code.split('\n'):
            lines.append(f"   {code_line}")

        lines.append('')

        return '\n'.join(lines)



    def generate_integration_matrix(self) -> str:
        """
        Generate integration matrix (docs/integration-map.rst)

        Task 3.4: Create table showing all plugin-to-plugin relationships
        - Include columns: Plugin A, Plugin B, Integration Type, Reference Count
        - Add category grouping for readability
        - Include bidirectional references
        """
        lines = []

        # Header
        lines.append("Integration Map")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Cross-plugin integration relationships for Claude Code marketplace plugins.")
        lines.append("")

        # Summary statistics
        total_integrations = sum(len(targets) for targets in self.integration_map.values())
        plugins_with_deps = len([p for p, deps in self.integration_map.items() if deps])
        plugins_referenced = len([p for p, refs in self.reverse_dependencies.items() if refs])

        lines.append("Summary")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"- **Total Plugins:** {len(self.all_plugins)}")
        lines.append(f"- **Total Integration Points:** {total_integrations}")
        lines.append(f"- **Plugins with Dependencies:** {plugins_with_deps}")
        lines.append(f"- **Plugins Referenced by Others:** {plugins_referenced}")
        lines.append("")

        # Integration matrix by category
        lines.append("Integration Matrix by Category")
        lines.append("-" * 80)
        lines.append("")

        # Group plugins by category
        category_groups = defaultdict(list)
        for plugin_name in self.all_plugins:
            metadata = self.plugin_metadata.get(plugin_name, {})
            category = metadata.get("category", "uncategorized")
            category_groups[category].append(plugin_name)

        for category in sorted(category_groups.keys()):
            plugins = sorted(category_groups[category])

            lines.append(f"{category.replace('-', ' ').title()}")
            lines.append("~" * 40)
            lines.append("")

            for plugin in plugins:
                targets = self.integration_map.get(plugin, set())
                referenced_by = self.reverse_dependencies.get(plugin, set())

                if targets or referenced_by:
                    lines.append(f"**{plugin}**")
                    lines.append("")

                    if targets:
                        lines.append("   *Integrates with:*")
                        lines.append("")
                        for target in sorted(targets):
                            # Count references
                            ref_count = len([r for r in self.integration_references
                                           if r.source_plugin == plugin and r.target_plugin == target])
                            # Get reference types
                            ref_types = set([r.reference_type for r in self.integration_references
                                           if r.source_plugin == plugin and r.target_plugin == target])
                            types_str = ", ".join(sorted(ref_types))

                            lines.append(f"   - :doc:`/plugins/{target}` ({ref_count} refs, {types_str})")
                        lines.append("")

                    if referenced_by:
                        lines.append("   *Referenced by:*")
                        lines.append("")
                        for source in sorted(referenced_by):
                            lines.append(f"   - :doc:`/plugins/{source}`")
                        lines.append("")

            lines.append("")

        # Integration patterns
        if self.integration_patterns:
            lines.append("Common Integration Patterns")
            lines.append("-" * 80)
            lines.append("")
            lines.append("Identified multi-plugin workflow patterns:")
            lines.append("")

            for pattern_name, plugin_list in sorted(self.integration_patterns.items()):
                lines.append(f"**{pattern_name}**")
                lines.append("")
                for plugin in plugin_list[:5]:  # Limit to 5
                    if plugin in self.all_plugins:
                        lines.append(f"- :doc:`/plugins/{plugin}`")
                lines.append("")

        return "\n".join(lines)

    def generate_plugin_rst(self, plugin_path: Path, output_dir: Path) -> str:
        """Generate RST documentation for a plugin with enhanced integration sections"""
        metadata = self.extract_plugin_metadata(plugin_path)
        if not metadata:
            return ""

        plugin_name = metadata["name"]

        # Build RST content
        rst_lines = []

        # Title
        title = self._format_title(plugin_name)
        rst_lines.append(title)
        rst_lines.append("=" * len(title))
        rst_lines.append("")

        # Module directive with synopsis
        rst_lines.append(f".. module:: {plugin_name}")
        rst_lines.append(f"   :synopsis: {metadata['description']}")
        rst_lines.append("")

        # Description section
        rst_lines.extend(self._generate_description_section(metadata))

        # Agents section
        if metadata["agents"]:
            rst_lines.extend(self._generate_agents_section(metadata["agents"]))

        # Commands section
        if metadata["commands"]:
            rst_lines.extend(self._generate_commands_section(metadata["commands"]))

        # Skills section
        if metadata["skills"]:
            rst_lines.extend(self._generate_skills_section(metadata["skills"]))

        # Usage Examples section (placeholder)
        rst_lines.extend(self._generate_usage_examples_section(plugin_path))

        # Integration section (enhanced with Task 3.5 implementation)
        rst_lines.extend(self._generate_integration_section(plugin_name))

        # See Also section
        rst_lines.extend(self._generate_see_also_section(plugin_name, metadata["category"]))

        # References section
        rst_lines.extend(self._generate_references_section())

        rst_content = "\n".join(rst_lines)

        # Write to file
        output_file = output_dir / f"{plugin_name}.rst"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rst_content)

        if self.verbose:
            print(f"✅ Generated RST for {plugin_name}")

        return rst_content

    def _format_title(self, plugin_name: str) -> str:
        """Format plugin name as title"""
        return plugin_name.replace("-", " ").title()

    def _generate_description_section(self, metadata: Dict[str, Any]) -> List[str]:
        """Generate Description section with metadata table"""
        lines = []

        lines.append("Description")
        lines.append("-----------")
        lines.append("")
        lines.append(metadata["description"])
        lines.append("")

        # Metadata table
        lines.append("**Metadata:**")
        lines.append("")
        lines.append(f"- **Version:** {metadata['version']}")
        lines.append(f"- **Category:** {metadata['category']}")
        lines.append(f"- **License:** {metadata['license']}")
        lines.append(f"- **Author:** {metadata['author']}")

        if metadata["keywords"]:
            keywords_str = ", ".join(metadata["keywords"])
            lines.append(f"- **Keywords:** {keywords_str}")

        lines.append("")

        return lines

    def _generate_agents_section(self, agents: List[Dict[str, Any]]) -> List[str]:
        """Generate Agents section with agent directives"""
        lines = []

        lines.append("Agents")
        lines.append("------")
        lines.append("")

        for agent in agents:
            agent_name = agent.get("name", "unknown")
            agent_desc = agent.get("description", "No description")
            agent_status = agent.get("status", "unknown")
            agent_expertise = agent.get("expertise", [])

            lines.append(f".. agent:: {agent_name}")
            lines.append("")
            lines.append(f"   {agent_desc}")
            lines.append("")
            lines.append(f"   **Status:** {agent_status}")

            if agent_expertise:
                expertise_str = ", ".join(agent_expertise)
                lines.append(f"   **Expertise:** {expertise_str}")

            lines.append("")

        return lines

    def _generate_commands_section(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Generate Commands section with command directives and code blocks"""
        lines = []

        lines.append("Commands")
        lines.append("--------")
        lines.append("")

        for command in commands:
            command_name = command.get("name", "unknown")
            command_desc = command.get("description", "No description")
            command_status = command.get("status", "unknown")
            command_priority = command.get("priority")

            lines.append(f".. command:: {command_name}")
            lines.append("")
            lines.append(f"   {command_desc}")
            lines.append("")
            lines.append(f"   **Status:** {command_status}")

            if command_priority is not None:
                lines.append(f"   **Priority:** {command_priority}")

            lines.append("")
            lines.append("   Usage Example:")
            lines.append("")
            lines.append("   .. code-block:: bash")
            lines.append("")
            lines.append(f"      {command_name}")
            lines.append("")

        return lines

    def _generate_skills_section(self, skills: List[Dict[str, Any]]) -> List[str]:
        """Generate Skills section with skill directives and code blocks"""
        lines = []

        lines.append("Skills")
        lines.append("------")
        lines.append("")

        for skill in skills:
            skill_name = skill.get("name", "unknown")
            skill_desc = skill.get("description", "No description")
            skill_status = skill.get("status", "active")

            lines.append(f".. skill:: {skill_name}")
            lines.append("")
            lines.append(f"   {skill_desc}")
            lines.append("")
            lines.append(f"   **Status:** {skill_status}")
            lines.append("")

        return lines

    def _generate_usage_examples_section(self, plugin_path: Path) -> List[str]:
        """
        Generate Usage Examples section with code blocks from README

        Task 4.4: Populate Usage Examples sections
        - Extract code examples from existing README.md
        - Convert to RST code-block directives with language tags
        - Add context/explanation above each code block
        """
        lines = []

        lines.append("Usage Examples")
        lines.append("--------------")
        lines.append("")

        # Extract code blocks from README
        code_blocks = self.extract_code_blocks_from_readme(plugin_path)

        if not code_blocks:
            lines.append("*No code examples available in README.*")
            lines.append("")
            return lines

        # Map code blocks to sections
        sections = self.map_code_blocks_to_sections(code_blocks)

        # Generate RST for each section with examples
        section_order = ['installation', 'usage', 'commands', 'skills', 'configuration', 'advanced', 'examples']
        section_titles = {
            'installation': 'Installation',
            'usage': 'Basic Usage',
            'commands': 'Command Examples',
            'skills': 'Skill Patterns',
            'configuration': 'Configuration',
            'advanced': 'Advanced Usage',
            'examples': 'Additional Examples'
        }

        for section_key in section_order:
            section_blocks = sections.get(section_key, [])
            if section_blocks:
                # Add subsection header
                section_title = section_titles[section_key]
                lines.append(section_title)
                lines.append("~" * len(section_title))
                lines.append("")

                # Add code blocks
                for block in section_blocks[:3]:  # Limit to 3 examples per section
                    rst_code = self.generate_rst_code_block(block, include_context=True)
                    lines.append(rst_code)

        # If no sections had content, show message
        if all(not sections.get(key) for key in section_order):
            lines.append("*Code examples found but could not be categorized.*")
            lines.append("")

        return lines

    def _generate_integration_section(self, plugin_name: str) -> List[str]:
        """
        Generate Integration section with comprehensive integration data

        Task 3.5: Update sphinx-doc-generator.py with integration sections
        - Add "Integrates With" list with cross-references using :doc: directive
        - Add "Common Workflows" subsection with detected patterns
        - Add "Referenced By" list showing reverse dependencies
        """
        lines = []

        lines.append("Integration")
        lines.append("-----------")
        lines.append("")

        # Check if we have integration data
        integrations = self.integration_map.get(plugin_name, set())
        referenced_by = self.reverse_dependencies.get(plugin_name, set())

        has_content = False

        # Integrates With section
        if integrations:
            has_content = True
            lines.append("**Integrates With:**")
            lines.append("")
            lines.append("This plugin integrates with the following plugins:")
            lines.append("")

            for target in sorted(integrations):
                # Get reference types for this integration
                ref_types = set([r.reference_type for r in self.integration_references
                               if r.source_plugin == plugin_name and r.target_plugin == target])
                if ref_types:
                    types_str = f" ({', '.join(sorted(ref_types))})"
                else:
                    types_str = ""

                lines.append(f"- :doc:`/plugins/{target}`{types_str}")
            lines.append("")

        # Referenced By section (reverse dependencies)
        if referenced_by:
            has_content = True
            lines.append("**Referenced By:**")
            lines.append("")
            lines.append("This plugin is referenced by:")
            lines.append("")

            for source in sorted(referenced_by):
                lines.append(f"- :doc:`/plugins/{source}`")
            lines.append("")

        # Common Workflows section
        related_patterns = []
        for pattern_name, plugin_list in self.integration_patterns.items():
            if plugin_name in plugin_list:
                related_patterns.append((pattern_name, plugin_list))

        if related_patterns:
            has_content = True
            lines.append("**Common Workflows:**")
            lines.append("")
            lines.append("This plugin is part of the following workflow patterns:")
            lines.append("")

            for pattern_name, plugin_list in related_patterns:
                lines.append(f"- **{pattern_name}**: ", )
                other_plugins = [p for p in plugin_list if p != plugin_name and p in self.all_plugins]
                if other_plugins:
                    lines[-1] += ", ".join(f":doc:`/plugins/{p}`" for p in other_plugins[:3])
                lines.append("")

        if not has_content:
            lines.append("*No integration information available for this plugin.*")
            lines.append("")

        return lines

    def _generate_see_also_section(self, plugin_name: str, category: str) -> List[str]:
        """Generate See Also section"""
        lines = []

        lines.append("See Also")
        lines.append("--------")
        lines.append("")
        lines.append(f"- :doc:`/categories/{category}`")
        lines.append("- :doc:`/integration-map`")
        lines.append("")

        return lines

    def _generate_references_section(self) -> List[str]:
        """Generate References section"""
        lines = []

        lines.append("References")
        lines.append("----------")
        lines.append("")
        lines.append("*External resources and links will be added as available.*")
        lines.append("")

        return lines

    def generate_all_plugins(self, output_dir: Path, plugin_filter: Optional[str] = None):
        """Generate RST documentation for all plugins"""
        plugins = self.discover_plugins(plugin_filter)

        if not plugins:
            print("⚠️  No plugins found")
            return

        # First pass: discover all plugins for integration detection
        if self.verbose:
            print("First pass: discovering plugins...")

        # Second pass: detect integrations
        self.detect_integrations()

        # Build reverse dependencies
        self.build_reverse_dependencies()

        # Identify integration patterns
        self.identify_integration_patterns()

        # Third pass: generate documentation
        if self.verbose:
            print(f"\nGenerating RST documentation for {len(plugins)} plugins...")

        success_count = 0
        for plugin_path in plugins:
            try:
                self.generate_plugin_rst(plugin_path, output_dir)
                success_count += 1
            except Exception as e:
                print(f"❌ Error generating docs for {plugin_path.name}: {e}")

        print(f"\n✅ Successfully generated {success_count}/{len(plugins)} plugin documentation files")

    def generate_integration_map_file(self, output_path: Path):
        """Generate the integration-map.rst file"""
        matrix_content = self.generate_integration_matrix()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(matrix_content)

        if self.verbose:
            print(f"✅ Generated integration map: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate Sphinx RST documentation for Claude Code plugins with integration mapping"
    )
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=Path.cwd() / "plugins",
        help="Path to plugins directory (default: ./plugins)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "docs" / "plugins",
        help="Output directory for RST files (default: ./docs/plugins)"
    )
    parser.add_argument(
        "--plugin-filter",
        type=str,
        help="Filter plugins by name substring"
    )
    parser.add_argument(
        "--generate-integration-map",
        action="store_true",
        help="Generate integration-map.rst file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate plugins directory
    if not args.plugins_dir.exists():
        print(f"❌ Error: Plugins directory not found: {args.plugins_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate documentation
    generator = SphinxDocGenerator(args.plugins_dir, verbose=args.verbose)
    generator.generate_all_plugins(args.output_dir, args.plugin_filter)

    # Generate integration map if requested
    if args.generate_integration_map:
        docs_dir = args.output_dir.parent
        integration_map_path = docs_dir / "integration-map.rst"
        generator.generate_integration_map_file(integration_map_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
