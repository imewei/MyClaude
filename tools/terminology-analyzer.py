#!/usr/bin/env python3
"""
Terminology Consistency Analyzer

Analyzes terminology across plugins by:
- Extracting technical terms from all plugins
- Identifying terminology variations
- Mapping synonyms and inconsistencies
- Suggesting standardization

Author: Technical Writer / Systems Architect
Part of: Plugin Review and Optimization - Task Group 0.4
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import sys


@dataclass
class TermOccurrence:
    """Single occurrence of a term"""
    term: str
    plugin: str
    file_path: str
    line_number: int
    context: str
    normalized_term: str = ""


@dataclass
class TermVariation:
    """Represents variations of the same concept"""
    canonical_term: str
    variations: Set[str] = field(default_factory=set)
    occurrences: List[TermOccurrence] = field(default_factory=list)
    plugin_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class TerminologyReport:
    """Complete terminology analysis report"""
    total_terms: int = 0
    unique_terms: int = 0
    variations_found: Dict[str, TermVariation] = field(default_factory=dict)
    inconsistencies: List[Tuple[str, List[str]]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class TerminologyAnalyzer:
    """Analyzes and standardizes terminology across plugins"""

    # Known technical term patterns and their canonical forms
    TERM_PATTERNS = {
        # Programming concepts
        r'\b(optimization|optimisation)\b': 'optimization',
        r'\b(parallelization|parallelisation)\b': 'parallelization',
        r'\b(serialization|serialisation)\b': 'serialization',
        r'\b(visualization|visualisation)\b': 'visualization',

        # Hyphenation variations
        r'\b(multi[\s-]?threading)\b': 'multithreading',
        r'\b(multi[\s-]?processing)\b': 'multiprocessing',
        r'\b(multi[\s-]?platform)\b': 'multiplatform',
        r'\b(cross[\s-]?platform)\b': 'cross-platform',
        r'\b(real[\s-]?time)\b': 'real-time',
        r'\b(high[\s-]?performance)\b': 'high-performance',
        r'\b(end[\s-]?to[\s-]?end)\b': 'end-to-end',

        # Testing terminology
        r'\b(unit[\s-]?test(?:ing)?)\b': 'unit testing',
        r'\b(integration[\s-]?test(?:ing)?)\b': 'integration testing',
        r'\b(e2e|end[\s-]?to[\s-]?end)[\s-]?test(?:ing)?\b': 'end-to-end testing',

        # CI/CD terminology
        r'\b(CI[\s/]?CD|continuous[\s-]?integration|continuous[\s-]?deployment)\b': 'CI/CD',
        r'\b(devops|dev[\s-]?ops)\b': 'DevOps',

        # Scientific computing
        r'\b(ODE|ordinary[\s-]?differential[\s-]?equation)s?\b': 'ODE',
        r'\b(PDE|partial[\s-]?differential[\s-]?equation)s?\b': 'PDE',
        r'\b(SDE|stochastic[\s-]?differential[\s-]?equation)s?\b': 'SDE',
        r'\b(MCMC|Markov[\s-]?chain[\s-]?Monte[\s-]?Carlo)\b': 'MCMC',
        r'\b(ML|machine[\s-]?learning)\b': 'machine learning',
        r'\b(DL|deep[\s-]?learning)\b': 'deep learning',
        r'\b(AI|artificial[\s-]?intelligence)\b': 'AI',
        r'\b(GPU|graphics[\s-]?processing[\s-]?unit)s?\b': 'GPU',
        r'\b(CPU|central[\s-]?processing[\s-]?unit)s?\b': 'CPU',
        r'\b(HPC|high[\s-]?performance[\s-]?computing)\b': 'HPC',

        # Framework/library names
        r'\b(sciml|SciML)\b': 'SciML',
        r'\b(jax|JAX)\b': 'JAX',
        r'\b(numpy|NumPy)\b': 'NumPy',
        r'\b(scipy|SciPy)\b': 'SciPy',
        r'\b(pytorch|PyTorch)\b': 'PyTorch',
        r'\b(tensorflow|TensorFlow)\b': 'TensorFlow',

        # Code/documentation variations
        r'\b(code[\s-]?base|codebase)\b': 'codebase',
        r'\b(work[\s-]?flow|workflow)\b': 'workflow',
        r'\b(data[\s-]?set|dataset)\b': 'dataset',
        r'\b(name[\s-]?space|namespace)\b': 'namespace',
    }

    # Common synonyms that should be standardized
    SYNONYM_GROUPS = {
        'optimize': ['optimize', 'optimise', 'optimization', 'optimisation'],
        'visualize': ['visualize', 'visualise', 'visualization', 'visualisation', 'plot', 'graph'],
        'parallel': ['parallel', 'concurrent', 'multithreaded', 'distributed'],
        'configuration': ['configuration', 'config', 'settings', 'options'],
        'documentation': ['documentation', 'docs', 'readme'],
        'repository': ['repository', 'repo'],
        'implementation': ['implementation', 'impl'],
        'specification': ['specification', 'spec'],
        'framework': ['framework', 'library', 'package'],
    }

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = plugins_dir
        self.terms: List[TermOccurrence] = []
        self.term_counts = Counter()
        self.report = TerminologyReport()

    def analyze_all_plugins(self) -> TerminologyReport:
        """Analyze terminology across all plugins"""
        print("🔍 Analyzing terminology consistency...")

        # Step 1: Extract all terms
        self._extract_terms_from_plugins()

        # Step 2: Normalize and identify variations
        self._identify_variations()

        # Step 3: Find inconsistencies
        self._find_inconsistencies()

        # Step 4: Generate recommendations
        self._generate_recommendations()

        return self.report

    def _extract_terms_from_plugins(self):
        """Extract technical terms from all plugin files"""
        plugin_dirs = [d for d in self.plugins_dir.iterdir() if d.is_dir()]

        for plugin_dir in sorted(plugin_dirs):
            plugin_name = plugin_dir.name

            # Extract from plugin.json
            plugin_json = plugin_dir / "plugin.json"
            if plugin_json.exists():
                self._extract_from_json(plugin_json, plugin_name)

            # Extract from README
            readme = plugin_dir / "README.md"
            if readme.exists():
                self._extract_from_markdown(readme, plugin_name)

            # Extract from agent documentation
            agents_dir = plugin_dir / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.md"):
                    self._extract_from_markdown(agent_file, plugin_name)

            # Extract from command documentation
            commands_dir = plugin_dir / "commands"
            if commands_dir.exists():
                for command_file in commands_dir.glob("*.md"):
                    self._extract_from_markdown(command_file, plugin_name)

            # Extract from skill documentation
            skills_dir = plugin_dir / "skills"
            if skills_dir.exists():
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir():
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.exists():
                            self._extract_from_markdown(skill_file, plugin_name)

    def _extract_from_json(self, file_path: Path, plugin_name: str):
        """Extract terms from JSON files"""
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Extract from description
            if 'description' in data:
                self._extract_terms_from_text(
                    data['description'], file_path, plugin_name, 0, "description"
                )

            # Extract from keywords
            if 'keywords' in data:
                for keyword in data['keywords']:
                    self._add_term(
                        keyword, plugin_name, str(file_path), 0, f"keyword: {keyword}"
                    )

        except Exception as e:
            print(f"⚠️  Warning: Error extracting from {file_path}: {e}")

    def _extract_from_markdown(self, file_path: Path, plugin_name: str):
        """Extract terms from markdown files"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                # Skip code blocks and links
                if line.startswith('```') or line.startswith('http'):
                    continue

                self._extract_terms_from_text(
                    line, file_path, plugin_name, line_num, line.strip()[:100]
                )

        except Exception as e:
            print(f"⚠️  Warning: Error extracting from {file_path}: {e}")

    def _extract_terms_from_text(
        self, text: str, file_path: Path, plugin_name: str,
        line_num: int, context: str
    ):
        """Extract technical terms from text using patterns"""
        for pattern, canonical in self.TERM_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group(0)
                self._add_term(term, plugin_name, str(file_path), line_num, context)

    def _add_term(
        self, term: str, plugin: str, file_path: str,
        line_num: int, context: str
    ):
        """Add a term occurrence"""
        # Normalize the term
        normalized = self._normalize_term(term)

        occurrence = TermOccurrence(
            term=term,
            plugin=plugin,
            file_path=file_path,
            line_number=line_num,
            context=context,
            normalized_term=normalized
        )

        self.terms.append(occurrence)
        self.term_counts[normalized] += 1

    def _normalize_term(self, term: str) -> str:
        """Normalize a term to its canonical form"""
        term_lower = term.lower().strip()

        # Check against patterns
        for pattern, canonical in self.TERM_PATTERNS.items():
            if re.match(pattern, term_lower):
                return canonical

        # Check against synonym groups
        for canonical, synonyms in self.SYNONYM_GROUPS.items():
            if term_lower in [s.lower() for s in synonyms]:
                return canonical

        return term_lower

    def _identify_variations(self):
        """Identify variations of the same term"""
        # Group terms by normalized form
        term_groups = defaultdict(list)
        for occurrence in self.terms:
            term_groups[occurrence.normalized_term].append(occurrence)

        # Create variations for terms with multiple forms
        for normalized, occurrences in term_groups.items():
            if len(occurrences) < 2:
                continue

            # Find unique variations
            variations = set(occ.term for occ in occurrences)
            if len(variations) > 1:
                variation = TermVariation(
                    canonical_term=normalized,
                    variations=variations,
                    occurrences=occurrences
                )

                # Count usage per plugin
                for occ in occurrences:
                    variation.plugin_usage[occ.plugin] += 1

                self.report.variations_found[normalized] = variation

        self.report.total_terms = len(self.terms)
        self.report.unique_terms = len(term_groups)

    def _find_inconsistencies(self):
        """Find terminology inconsistencies"""
        # Check for variations used within same plugin
        for canonical, variation in self.report.variations_found.items():
            # Find plugins using multiple variations
            plugin_variations = defaultdict(set)
            for occ in variation.occurrences:
                plugin_variations[occ.plugin].add(occ.term)

            for plugin, terms in plugin_variations.items():
                if len(terms) > 1:
                    self.report.inconsistencies.append(
                        (plugin, [canonical, list(terms)])
                    )

    def _generate_recommendations(self):
        """Generate standardization recommendations"""
        recommendations = []

        # High-impact standardizations
        high_impact = [
            (canonical, var)
            for canonical, var in self.report.variations_found.items()
            if len(var.variations) >= 3 or len(var.plugin_usage) >= 5
        ]

        if high_impact:
            recommendations.append(
                f"**High Priority**: Standardize {len(high_impact)} terms with "
                f"multiple variations across many plugins"
            )

        # Plugin-specific inconsistencies
        plugins_with_issues = set(p for p, _ in self.report.inconsistencies)
        if plugins_with_issues:
            recommendations.append(
                f"**Fix Internal Inconsistencies**: {len(plugins_with_issues)} plugins "
                f"use multiple variations of the same term internally"
            )

        # Suggest glossary
        if len(self.report.variations_found) > 10:
            recommendations.append(
                "**Create Terminology Glossary**: Document canonical forms for "
                "commonly used technical terms across all plugins"
            )

        # Framework naming
        recommendations.append(
            "**Standardize Framework Names**: Use official capitalization for "
            "frameworks (e.g., 'SciML', 'JAX', 'NumPy', 'PyTorch')"
        )

        # Hyphenation
        recommendations.append(
            "**Consistent Hyphenation**: Standardize hyphenation for compound "
            "terms (e.g., 'high-performance', 'multi-threading', 'cross-platform')"
        )

        self.report.recommendations = recommendations

    def generate_report(self, output_path: Path = None) -> str:
        """Generate comprehensive terminology report"""
        lines = []

        # Header
        lines.append("# Terminology Consistency Analysis")
        lines.append("")
        lines.append("Analysis of terminology usage and consistency across Claude Code plugins")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Term Occurrences:** {self.report.total_terms:,}")
        lines.append(f"- **Unique Normalized Terms:** {self.report.unique_terms:,}")
        lines.append(f"- **Terms with Variations:** {len(self.report.variations_found)}")
        lines.append(f"- **Plugins with Inconsistencies:** {len(set(p for p, _ in self.report.inconsistencies))}")
        lines.append("")

        # Top terms
        lines.append("## Most Common Technical Terms")
        lines.append("")
        lines.append("| Term | Occurrences | Variations |")
        lines.append("|------|-------------|------------|")

        top_terms = self.term_counts.most_common(30)
        for term, count in top_terms:
            variations = len(self.report.variations_found.get(term, TermVariation("")).variations)
            var_indicator = f"{variations} forms" if variations > 1 else "—"
            lines.append(f"| `{term}` | {count} | {var_indicator} |")
        lines.append("")

        # Variations found
        if self.report.variations_found:
            lines.append("## Terminology Variations")
            lines.append("")
            lines.append("Terms with multiple forms found across plugins:")
            lines.append("")

            # Sort by number of variations and usage count
            sorted_variations = sorted(
                self.report.variations_found.items(),
                key=lambda x: (-len(x[1].variations), -len(x[1].occurrences))
            )

            for canonical, variation in sorted_variations[:50]:  # Top 50
                lines.append(f"### {canonical}")
                lines.append("")
                lines.append(f"**Canonical form:** `{canonical}`")
                lines.append("")
                lines.append(f"**Variations found ({len(variation.variations)}):**")
                for var in sorted(variation.variations):
                    lines.append(f"- `{var}`")
                lines.append("")
                lines.append(f"**Usage across plugins ({len(variation.plugin_usage)} plugins):**")
                plugin_usage_sorted = sorted(
                    variation.plugin_usage.items(),
                    key=lambda x: -x[1]
                )
                for plugin, count in plugin_usage_sorted[:10]:  # Top 10 plugins
                    lines.append(f"- `{plugin}`: {count} occurrences")
                if len(variation.plugin_usage) > 10:
                    lines.append(f"- _(... and {len(variation.plugin_usage) - 10} more plugins)_")
                lines.append("")

        # Inconsistencies
        if self.report.inconsistencies:
            lines.append("## Internal Inconsistencies")
            lines.append("")
            lines.append("Plugins using multiple variations of the same term:")
            lines.append("")

            for plugin, (canonical, variations) in self.report.inconsistencies:
                lines.append(f"**{plugin}**")
                lines.append(f"- Term: `{canonical}`")
                lines.append(f"- Uses: {', '.join(f'`{v}`' for v in variations)}")
                lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        for rec in self.report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        # Standardization guide
        lines.append("## Proposed Standardization Guide")
        lines.append("")
        lines.append("### Framework and Library Names")
        lines.append("")
        lines.append("| Canonical Form | Avoid |")
        lines.append("|----------------|-------|")
        lines.append("| SciML | sciml, SCIML |")
        lines.append("| JAX | jax, Jax |")
        lines.append("| NumPy | numpy, Numpy |")
        lines.append("| SciPy | scipy, Scipy |")
        lines.append("| PyTorch | pytorch, Pytorch |")
        lines.append("| TensorFlow | tensorflow, Tensorflow |")
        lines.append("")

        lines.append("### Compound Terms")
        lines.append("")
        lines.append("| Canonical Form | Avoid |")
        lines.append("|----------------|-------|")
        lines.append("| high-performance | high performance, highperformance |")
        lines.append("| multi-threading | multithreading, multi threading |")
        lines.append("| cross-platform | crossplatform, cross platform |")
        lines.append("| end-to-end | end to end, endtoend |")
        lines.append("| real-time | realtime, real time |")
        lines.append("")

        lines.append("### Acronyms")
        lines.append("")
        lines.append("| Canonical Form | Full Form | Avoid |")
        lines.append("|----------------|-----------|-------|")
        lines.append("| ODE | Ordinary Differential Equation | ode |")
        lines.append("| PDE | Partial Differential Equation | pde |")
        lines.append("| SDE | Stochastic Differential Equation | sde |")
        lines.append("| MCMC | Markov Chain Monte Carlo | mcmc |")
        lines.append("| HPC | High-Performance Computing | hpc |")
        lines.append("| CI/CD | Continuous Integration/Deployment | CI-CD, CICD |")
        lines.append("")

        lines.append("### British vs American Spelling")
        lines.append("")
        lines.append("| Preferred (American) | Avoid (British) |")
        lines.append("|---------------------|-----------------|")
        lines.append("| optimization | optimisation |")
        lines.append("| visualization | visualisation |")
        lines.append("| parallelization | parallelisation |")
        lines.append("| serialization | serialisation |")
        lines.append("")

        report = "\n".join(lines)

        # Write to file if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding='utf-8')
            print(f"✅ Terminology analysis saved to: {output_path}")

        return report

    def export_glossary(self, output_path: Path):
        """Export standardized glossary as JSON"""
        glossary = {
            "canonical_terms": {
                canonical: {
                    "variations": list(var.variations),
                    "usage_count": len(var.occurrences),
                    "plugin_count": len(var.plugin_usage)
                }
                for canonical, var in self.report.variations_found.items()
            },
            "inconsistencies": [
                {
                    "plugin": plugin,
                    "term": canonical,
                    "variations_used": variations
                }
                for plugin, (canonical, variations) in self.report.inconsistencies
            ],
            "recommendations": self.report.recommendations
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(glossary, f, indent=2)

        print(f"✅ Glossary exported to: {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze terminology consistency across Claude Code plugins"
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
        default=Path("reports/terminology-analysis.md"),
        help="Output file for analysis report (default: reports/terminology-analysis.md)"
    )
    parser.add_argument(
        "--export-glossary",
        type=Path,
        help="Export standardized glossary as JSON"
    )

    args = parser.parse_args()

    # Validate plugins directory
    if not args.plugins_dir.exists():
        print(f"❌ Error: Plugins directory not found: {args.plugins_dir}")
        sys.exit(1)

    # Create analyzer and analyze
    analyzer = TerminologyAnalyzer(args.plugins_dir)
    report = analyzer.analyze_all_plugins()

    # Generate report
    print(f"\n📊 Generating terminology analysis...")
    analyzer.generate_report(args.output)

    # Export glossary if requested
    if args.export_glossary:
        analyzer.export_glossary(args.export_glossary)

    # Print summary
    print(f"\n✅ Analysis complete!")
    print(f"   Total terms: {report.total_terms:,}")
    print(f"   Unique terms: {report.unique_terms:,}")
    print(f"   Variations found: {len(report.variations_found)}")
    print(f"   Inconsistencies: {len(report.inconsistencies)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
