#!/usr/bin/env python3
"""
Command Suggestion Analyzer for Plugin Triggering Patterns

Analyzes command relevance in different contexts, validates suggestion timing,
and evaluates priority ranking accuracy.

Usage:
    python3 tools/command_analyzer.py
    python3 tools/command_analyzer.py --plugins-dir /path/to/plugins
    python3 tools/command_analyzer.py --corpus-dir /path/to/test-corpus
    python3 tools/command_analyzer.py --plugin julia-development
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Set, Optional, Tuple


@dataclass
class Command:
    """Represents a plugin command."""

    name: str
    description: str
    status: str
    priority: int = 5
    plugin_name: str = ""
    keywords: Set[str] = field(default_factory=set)


@dataclass
class CommandContext:
    """Context information for command suggestion."""

    file_path: Optional[Path] = None
    file_extension: Optional[str] = None
    file_content: Optional[str] = None
    directory_structure: List[str] = field(default_factory=list)
    project_type: Optional[str] = None
    keywords_present: Set[str] = field(default_factory=set)


@dataclass
class CommandSuggestionResult:
    """Result of command suggestion analysis."""

    command_name: str
    plugin_name: str
    context_name: str
    is_relevant: bool
    relevance_score: float
    timing_appropriate: bool
    priority_correct: bool
    matching_keywords: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class CommandAnalysisMetrics:
    """Aggregated command suggestion metrics."""

    total_suggestions: int = 0
    relevant_suggestions: int = 0
    irrelevant_suggestions: int = 0
    timing_appropriate: int = 0
    timing_inappropriate: int = 0
    priority_correct: int = 0
    priority_incorrect: int = 0

    @property
    def relevance_accuracy(self) -> float:
        """Percentage of relevant suggestions."""
        if self.total_suggestions == 0:
            return 0.0
        return self.relevant_suggestions / self.total_suggestions * 100

    @property
    def timing_accuracy(self) -> float:
        """Percentage of appropriately timed suggestions."""
        if self.total_suggestions == 0:
            return 0.0
        return self.timing_appropriate / self.total_suggestions * 100

    @property
    def priority_accuracy(self) -> float:
        """Percentage of correctly prioritized suggestions."""
        if self.total_suggestions == 0:
            return 0.0
        return self.priority_correct / self.total_suggestions * 100


class CommandSuggestionAnalyzer:
    """Analyzes command suggestion accuracy and relevance."""

    def __init__(self, plugins_dir: str, corpus_dir: str):
        self.plugins_dir = Path(plugins_dir)
        self.corpus_dir = Path(corpus_dir)
        self.commands: List[Command] = []
        self.results: List[CommandSuggestionResult] = []

    def load_commands(self, specific_plugin: Optional[str] = None) -> None:
        """Load commands from all plugins."""
        print("Loading plugin commands...")

        plugin_dirs = []
        if specific_plugin:
            plugin_path = self.plugins_dir / specific_plugin
            if plugin_path.exists():
                plugin_dirs.append(plugin_path)
        else:
            plugin_dirs = [p for p in self.plugins_dir.iterdir() if p.is_dir()]

        for plugin_dir in plugin_dirs:
            plugin_json = plugin_dir / "plugin.json"
            if not plugin_json.exists():
                continue

            try:
                data = json.loads(plugin_json.read_text())
                plugin_name = data["name"]

                if "commands" not in data or not data["commands"]:
                    continue

                for cmd_data in data["commands"]:
                    command = Command(
                        name=cmd_data["name"],
                        description=cmd_data.get("description", ""),
                        status=cmd_data.get("status", "active"),
                        priority=cmd_data.get("priority", 5),
                        plugin_name=plugin_name
                    )

                    # Extract keywords from command name and description
                    command.keywords = self._extract_keywords(
                        f"{command.name} {command.description}"
                    )

                    # Add plugin keywords
                    if "keywords" in data:
                        command.keywords.update(kw.lower() for kw in data["keywords"])

                    self.commands.append(command)

                print(f"  ✓ {plugin_name}: {len([c for c in self.commands if c.plugin_name == plugin_name])} commands")

            except Exception as e:
                print(f"  ✗ Error loading {plugin_dir.name}: {e}")

        print(f"\nLoaded {len(self.commands)} total commands")

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Remove slash prefix from commands
        text = text.replace("/", " ")

        # Split on non-alphanumeric characters
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Filter out common words
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are"
        }

        return {w for w in words if w not in stopwords and len(w) > 2}

    def analyze_context(self, sample_path: Path) -> List[CommandContext]:
        """Analyze contexts within a sample project."""
        contexts = []

        # Get all relevant files
        for file_path in sample_path.rglob("*"):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in {".py", ".jl", ".js", ".ts", ".rs", ".cpp", ".c", ".go",
                          ".java", ".toml", ".json", ".yaml", ".yml", ".md"}:
                continue

            try:
                content = file_path.read_text(errors="ignore")
                context = CommandContext(
                    file_path=file_path,
                    file_extension=ext,
                    file_content=content,
                    keywords_present=self._extract_keywords(content)
                )

                # Infer project type from directory structure
                context.directory_structure = [
                    str(p.relative_to(sample_path))
                    for p in sample_path.rglob("*")
                    if p.is_dir()
                ]

                contexts.append(context)

            except Exception:
                pass

        return contexts

    def calculate_relevance_score(
        self, command: Command, context: CommandContext
    ) -> Tuple[float, List[str]]:
        """Calculate relevance score for command in given context."""
        score = 0.0
        matching_keywords = []

        # Check keyword overlap
        keyword_overlap = command.keywords & context.keywords_present
        if keyword_overlap:
            score += min(len(keyword_overlap) * 0.2, 0.6)
            matching_keywords.extend(list(keyword_overlap)[:5])

        # Check file extension relevance
        ext_map = {
            ".py": {"python", "fastapi", "django", "pytest"},
            ".jl": {"julia", "sciml", "turing", "jump"},
            ".js": {"javascript", "node", "react", "jest"},
            ".ts": {"typescript", "node", "react"},
            ".rs": {"rust", "cargo"},
            ".cpp": {"cpp", "c++", "cmake"},
        }

        if context.file_extension in ext_map:
            ext_keywords = ext_map[context.file_extension]
            if command.keywords & ext_keywords:
                score += 0.3

        # Check command name relevance
        cmd_name_lower = command.name.lower()

        # Setup/scaffold commands are relevant at project start
        if any(term in cmd_name_lower for term in ["setup", "scaffold", "init", "create"]):
            # Check if context suggests early stage
            if context.file_content and len(context.file_content) < 500:
                score += 0.2

        # Test commands are relevant for test files
        if "test" in cmd_name_lower:
            if context.file_path and "test" in str(context.file_path).lower():
                score += 0.3

        # Optimize commands are relevant for performance contexts
        if "optimize" in cmd_name_lower or "profile" in cmd_name_lower:
            perf_keywords = {"benchmark", "profile", "performance", "optimize", "speed"}
            if context.keywords_present & perf_keywords:
                score += 0.3

        # Debug commands are relevant for error contexts
        if "debug" in cmd_name_lower:
            debug_keywords = {"error", "exception", "debug", "trace", "stack"}
            if context.keywords_present & debug_keywords:
                score += 0.3

        # CI/CD commands are relevant for workflow contexts
        if "ci" in cmd_name_lower or "workflow" in cmd_name_lower:
            if any("workflow" in p or ".github" in p for p in context.directory_structure):
                score += 0.4

        return min(score, 1.0), matching_keywords

    def assess_timing(
        self, command: Command, context: CommandContext, relevance_score: float
    ) -> bool:
        """Assess if command suggestion timing is appropriate."""
        cmd_name = command.name.lower()

        # Setup/scaffold commands should be suggested early
        if any(term in cmd_name for term in ["setup", "scaffold", "init"]):
            # Appropriate if project is new/small
            return not context.directory_structure or len(context.directory_structure) < 5

        # Testing commands appropriate when tests exist or are being written
        if "test" in cmd_name:
            return (context.file_path and "test" in str(context.file_path).lower()) or \
                   any("test" in d for d in context.directory_structure)

        # Optimization commands appropriate for established projects
        if "optimize" in cmd_name or "profile" in cmd_name:
            return len(context.directory_structure) > 3

        # CI/CD commands appropriate when workflows are present or being set up
        if "ci" in cmd_name or "workflow" in cmd_name:
            return any("workflow" in p or ".github" in p for p in context.directory_structure)

        # For other commands, timing is appropriate if relevance is high
        return relevance_score >= 0.4

    def assess_priority(
        self, command: Command, relevance_score: float
    ) -> bool:
        """Assess if command priority is correctly set."""
        # High priority (1-3) should have high relevance
        if command.priority <= 3:
            return relevance_score >= 0.5

        # Medium priority (4-6) should have medium relevance
        if command.priority <= 6:
            return 0.3 <= relevance_score < 0.7

        # Low priority (7-10) can have any relevance
        return True

    def test_command_suggestions(self) -> None:
        """Test command suggestions across all contexts."""
        print("\nTesting command suggestions...")

        # Load test corpus
        samples = []
        sample_dirs = [d for d in self.corpus_dir.iterdir() if d.is_dir()]

        for sample_dir in sample_dirs:
            metadata_file = sample_dir / "metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                metadata["path"] = sample_dir
                samples.append(metadata)

        print(f"  Testing against {len(samples)} samples...")

        for sample in samples:
            print(f"  Analyzing {sample['name']}...")
            contexts = self.analyze_context(sample["path"])

            for context in contexts[:5]:  # Limit contexts per sample
                for command in self.commands:
                    # Calculate relevance
                    relevance_score, matching_keywords = self.calculate_relevance_score(
                        command, context
                    )

                    is_relevant = relevance_score >= 0.3

                    # Assess timing
                    timing_appropriate = self.assess_timing(
                        command, context, relevance_score
                    )

                    # Assess priority
                    priority_correct = self.assess_priority(command, relevance_score)

                    # Create result
                    result = CommandSuggestionResult(
                        command_name=command.name,
                        plugin_name=command.plugin_name,
                        context_name=f"{sample['name']}/{context.file_path.name if context.file_path else 'unknown'}",
                        is_relevant=is_relevant,
                        relevance_score=relevance_score,
                        timing_appropriate=timing_appropriate,
                        priority_correct=priority_correct,
                        matching_keywords=matching_keywords,
                        explanation=self._generate_explanation(
                            command, context, relevance_score
                        )
                    )

                    self.results.append(result)

        print(f"\n  Analyzed {len(self.results)} command suggestions")

    def _generate_explanation(
        self, command: Command, context: CommandContext, score: float
    ) -> str:
        """Generate explanation for relevance score."""
        if score >= 0.7:
            return "Highly relevant - strong keyword match and context alignment"
        elif score >= 0.4:
            return "Moderately relevant - partial keyword match or context alignment"
        elif score >= 0.2:
            return "Low relevance - minimal keyword match"
        else:
            return "Not relevant - no significant keyword or context match"

    def calculate_metrics(self) -> CommandAnalysisMetrics:
        """Calculate aggregated command analysis metrics."""
        metrics = CommandAnalysisMetrics()

        for result in self.results:
            metrics.total_suggestions += 1

            if result.is_relevant:
                metrics.relevant_suggestions += 1
            else:
                metrics.irrelevant_suggestions += 1

            if result.timing_appropriate:
                metrics.timing_appropriate += 1
            else:
                metrics.timing_inappropriate += 1

            if result.priority_correct:
                metrics.priority_correct += 1
            else:
                metrics.priority_incorrect += 1

        return metrics

    def generate_report(self) -> str:
        """Generate command suggestion analysis report."""
        metrics = self.calculate_metrics()

        report = f"""# Command Suggestion Analysis Report

**Analysis Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Commands:** {len(self.commands)}
**Total Suggestions Analyzed:** {len(self.results)}

## Summary Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Relevance Accuracy | {metrics.relevance_accuracy:.1f}% | >80% | {self._status_icon(metrics.relevance_accuracy > 80)} |
| Timing Accuracy | {metrics.timing_accuracy:.1f}% | >85% | {self._status_icon(metrics.timing_accuracy > 85)} |
| Priority Accuracy | {metrics.priority_accuracy:.1f}% | >90% | {self._status_icon(metrics.priority_accuracy > 90)} |

## Suggestion Breakdown

- **Relevant Suggestions:** {metrics.relevant_suggestions} ({metrics.relevance_accuracy:.1f}%)
- **Irrelevant Suggestions:** {metrics.irrelevant_suggestions}
- **Timing Appropriate:** {metrics.timing_appropriate} ({metrics.timing_accuracy:.1f}%)
- **Timing Inappropriate:** {metrics.timing_inappropriate}
- **Priority Correct:** {metrics.priority_correct} ({metrics.priority_accuracy:.1f}%)
- **Priority Incorrect:** {metrics.priority_incorrect}

## Command Performance

"""

        # Analyze per-command metrics
        command_stats = {}
        for result in self.results:
            key = f"{result.plugin_name}/{result.command_name}"
            if key not in command_stats:
                command_stats[key] = {
                    "total": 0,
                    "relevant": 0,
                    "timing_ok": 0,
                    "priority_ok": 0,
                    "avg_score": 0.0
                }

            stats = command_stats[key]
            stats["total"] += 1
            if result.is_relevant:
                stats["relevant"] += 1
            if result.timing_appropriate:
                stats["timing_ok"] += 1
            if result.priority_correct:
                stats["priority_ok"] += 1
            stats["avg_score"] += result.relevance_score

        report += "### Top Performing Commands\n\n"
        report += "| Command | Plugin | Relevance | Timing | Priority | Avg Score |\n"
        report += "|---------|--------|-----------|--------|----------|----------|\n"

        # Sort by average score
        sorted_commands = sorted(
            command_stats.items(),
            key=lambda x: x[1]["avg_score"] / x[1]["total"],
            reverse=True
        )

        for key, stats in sorted_commands[:10]:
            plugin_name, cmd_name = key.split("/", 1)
            relevance_pct = stats["relevant"] / stats["total"] * 100
            timing_pct = stats["timing_ok"] / stats["total"] * 100
            priority_pct = stats["priority_ok"] / stats["total"] * 100
            avg_score = stats["avg_score"] / stats["total"]

            report += f"| {cmd_name} | {plugin_name} | {relevance_pct:.0f}% | "
            report += f"{timing_pct:.0f}% | {priority_pct:.0f}% | {avg_score:.2f} |\n"

        report += "\n### Commands Needing Improvement\n\n"

        # Find commands with poor performance
        poor_commands = [
            (key, stats) for key, stats in sorted_commands
            if stats["relevant"] / stats["total"] < 0.5
        ]

        if poor_commands:
            report += "| Command | Plugin | Issue | Relevance |\n"
            report += "|---------|--------|-------|----------|\n"

            for key, stats in poor_commands[:10]:
                plugin_name, cmd_name = key.split("/", 1)
                relevance_pct = stats["relevant"] / stats["total"] * 100
                issue = "Low relevance"
                if stats["timing_ok"] / stats["total"] < 0.5:
                    issue = "Poor timing"
                if stats["priority_ok"] / stats["total"] < 0.5:
                    issue = "Incorrect priority"

                report += f"| {cmd_name} | {plugin_name} | {issue} | {relevance_pct:.0f}% |\n"
        else:
            report += "No commands with poor performance detected.\n"

        report += """
## Sample Analysis

### High Relevance Suggestions

"""

        high_relevance = [r for r in self.results if r.relevance_score >= 0.7]
        for result in high_relevance[:10]:
            report += f"- **{result.command_name}** ({result.plugin_name})\n"
            report += f"  - Context: {result.context_name}\n"
            report += f"  - Score: {result.relevance_score:.2f}\n"
            report += f"  - Keywords: {', '.join(result.matching_keywords[:5])}\n\n"

        report += """
### Low Relevance Suggestions (Potential Issues)

"""

        low_relevance = [r for r in self.results if r.is_relevant and r.relevance_score < 0.4]
        for result in low_relevance[:10]:
            report += f"- **{result.command_name}** ({result.plugin_name})\n"
            report += f"  - Context: {result.context_name}\n"
            report += f"  - Score: {result.relevance_score:.2f}\n"
            report += f"  - Issue: {result.explanation}\n\n"

        report += """
## Overall Assessment

"""

        if metrics.relevance_accuracy > 80 and metrics.timing_accuracy > 85:
            report += "**Status:** ✅ EXCELLENT - Command suggestions are highly accurate\n\n"
        elif metrics.relevance_accuracy > 60 and metrics.timing_accuracy > 70:
            report += "**Status:** ⚠️ GOOD - Command suggestions are generally accurate but have room for improvement\n\n"
        else:
            report += "**Status:** ❌ NEEDS IMPROVEMENT - Command suggestions need optimization\n\n"

        report += "Command suggestion system is "
        if metrics.relevance_accuracy > 80:
            report += "performing well with high relevance accuracy. "
        else:
            report += "showing room for improvement in relevance detection. "

        if metrics.timing_accuracy < 85:
            report += "\n\n**Recommendation:** Review timing logic for commands to ensure "
            report += "suggestions appear at appropriate points in the development workflow."

        if metrics.priority_accuracy < 90:
            report += "\n\n**Recommendation:** Review command priority assignments to better "
            report += "reflect their importance in different contexts."

        return report

    def _status_icon(self, condition: bool) -> str:
        """Return status icon based on condition."""
        return "✅" if condition else "❌"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze command suggestion relevance and timing"
    )
    parser.add_argument(
        "--plugins-dir",
        default="plugins",
        help="Path to plugins directory (default: plugins)"
    )
    parser.add_argument(
        "--corpus-dir",
        default="test-corpus",
        help="Path to test corpus directory (default: test-corpus)"
    )
    parser.add_argument(
        "--plugin",
        help="Analyze specific plugin only"
    )
    parser.add_argument(
        "--output",
        default="reports/command-analysis.md",
        help="Output report file (default: reports/command-analysis.md)"
    )

    args = parser.parse_args()

    # Resolve paths
    plugins_dir = Path(args.plugins_dir).absolute()
    corpus_dir = Path(args.corpus_dir).absolute()

    if not plugins_dir.exists():
        print(f"Error: Plugins directory not found: {plugins_dir}")
        return 1

    if not corpus_dir.exists():
        print(f"Error: Test corpus directory not found: {corpus_dir}")
        print("Run test_corpus_generator.py first to create test samples")
        return 1

    # Run analysis
    analyzer = CommandSuggestionAnalyzer(str(plugins_dir), str(corpus_dir))
    analyzer.load_commands(args.plugin)
    analyzer.test_command_suggestions()

    # Generate report
    report = analyzer.generate_report()
    print("\n" + "=" * 70)
    print(report)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n✓ Report saved to: {output_path.absolute()}")

    # Determine exit code
    metrics = analyzer.calculate_metrics()
    if metrics.relevance_accuracy > 80 and metrics.timing_accuracy > 85:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
