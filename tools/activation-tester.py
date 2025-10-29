#!/usr/bin/env python3
"""
Activation Accuracy Tester for Plugin Triggering Patterns

Tests plugin activation accuracy against test corpus samples, measuring false positive
and false negative rates. Target: <5% for both metrics.

Usage:
    python3 tools/activation-tester.py
    python3 tools/activation-tester.py --plugins-dir /path/to/plugins
    python3 tools/activation-tester.py --corpus-dir /path/to/test-corpus
    python3 tools/activation-tester.py --plugin julia-development
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


@dataclass
class PluginTriggeringPatterns:
    """Triggering patterns extracted from a plugin."""

    name: str
    keywords: Set[str] = field(default_factory=set)
    file_extensions: Set[str] = field(default_factory=set)
    directory_patterns: Set[str] = field(default_factory=set)
    content_patterns: Set[str] = field(default_factory=set)
    category: Optional[str] = None


@dataclass
class ActivationTestResult:
    """Result of activation test for a sample."""

    sample_name: str
    sample_category: str
    expected_plugins: List[str]
    activated_plugins: List[str]
    expected_trigger: bool
    actual_trigger: bool
    true_positive: bool = False
    true_negative: bool = False
    false_positive: bool = False
    false_negative: bool = False
    confidence_score: float = 0.0
    matching_patterns: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ActivationMetrics:
    """Aggregated activation accuracy metrics."""

    total_samples: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        """Overall accuracy percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total_samples * 100

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator * 100

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator * 100

    @property
    def f1_score(self) -> float:
        """F1 score: harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def false_positive_rate(self) -> float:
        """False positive rate: FP / (FP + TN)."""
        denominator = self.false_positives + self.true_negatives
        if denominator == 0:
            return 0.0
        return self.false_positives / denominator * 100

    @property
    def false_negative_rate(self) -> float:
        """False negative rate: FN / (FN + TP)."""
        denominator = self.false_negatives + self.true_positives
        if denominator == 0:
            return 0.0
        return self.false_negatives / denominator * 100


class PluginActivationTester:
    """Tests plugin activation accuracy using test corpus."""

    def __init__(self, plugins_dir: str, corpus_dir: str):
        self.plugins_dir = Path(plugins_dir)
        self.corpus_dir = Path(corpus_dir)
        self.plugins: Dict[str, PluginTriggeringPatterns] = {}
        self.results: List[ActivationTestResult] = []

    def load_plugins(self, specific_plugin: Optional[str] = None) -> None:
        """Load plugin triggering patterns."""
        print("Loading plugin triggering patterns...")

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
                patterns = self._extract_patterns(data, plugin_dir)
                self.plugins[patterns.name] = patterns
                print(f"  ✓ {patterns.name}: {len(patterns.keywords)} keywords, "
                      f"{len(patterns.file_extensions)} extensions")
            except Exception as e:
                print(f"  ✗ Error loading {plugin_dir.name}: {e}")

        print(f"\nLoaded {len(self.plugins)} plugins")

    def _extract_patterns(
        self, data: Dict, plugin_dir: Path
    ) -> PluginTriggeringPatterns:
        """Extract triggering patterns from plugin metadata."""
        patterns = PluginTriggeringPatterns(name=data["name"])

        # Extract keywords
        if "keywords" in data:
            patterns.keywords = set(kw.lower() for kw in data["keywords"])

        # Extract category
        if "category" in data:
            patterns.category = data["category"]

        # Infer file extensions from keywords
        extension_map = {
            "julia": {".jl"},
            "python": {".py"},
            "javascript": {".js", ".jsx"},
            "typescript": {".ts", ".tsx"},
            "rust": {".rs"},
            "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".h"},
            "c": {".c", ".h"},
            "go": {".go"},
            "java": {".java"},
            "csharp": {".cs"},
            "ruby": {".rb"},
            "php": {".php"},
        }

        for keyword in patterns.keywords:
            if keyword in extension_map:
                patterns.file_extensions.update(extension_map[keyword])

        # Infer directory patterns from keywords
        directory_map = {
            "testing": {"test", "tests", "__tests__"},
            "cicd": {".github/workflows", ".gitlab-ci", "jenkins"},
            "docker": {"docker", "dockerfile"},
            "kubernetes": {"k8s", "kubernetes"},
        }

        for keyword in patterns.keywords:
            if keyword in directory_map:
                patterns.directory_patterns.update(directory_map[keyword])

        # Extract content patterns from keywords
        patterns.content_patterns = set(patterns.keywords)

        # Read agent files for additional patterns
        agents_dir = plugin_dir / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.md"):
                try:
                    content = agent_file.read_text().lower()
                    # Extract triggering criteria
                    if "triggering criteria" in content or "use this agent when" in content:
                        # Extract terms from criteria section
                        lines = content.split("\n")
                        in_criteria = False
                        for line in lines:
                            if "triggering criteria" in line or "use this agent when" in line:
                                in_criteria = True
                            elif in_criteria and line.startswith("#"):
                                break
                            elif in_criteria:
                                # Extract technical terms
                                words = re.findall(r'\b[a-z]+(?:\.[a-z]+)+\b', line)
                                patterns.content_patterns.update(words)
                except Exception:
                    pass

        return patterns

    def load_test_corpus(self) -> List[Dict]:
        """Load test corpus samples."""
        print("\nLoading test corpus...")

        samples = []
        sample_dirs = [d for d in self.corpus_dir.iterdir() if d.is_dir()]

        for sample_dir in sample_dirs:
            metadata_file = sample_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                metadata = json.loads(metadata_file.read_text())
                metadata["path"] = sample_dir
                samples.append(metadata)
            except Exception as e:
                print(f"  ✗ Error loading {sample_dir.name}: {e}")

        print(f"  Loaded {len(samples)} test samples")
        return samples

    def test_file_extension_matching(
        self, sample_path: Path, plugin: PluginTriggeringPatterns
    ) -> Tuple[bool, List[str]]:
        """Test if plugin should activate based on file extensions."""
        if not plugin.file_extensions:
            return False, []

        matching_files = []
        for file_path in sample_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in plugin.file_extensions:
                    matching_files.append(str(file_path.relative_to(sample_path)))

        return len(matching_files) > 0, matching_files

    def test_directory_pattern_matching(
        self, sample_path: Path, plugin: PluginTriggeringPatterns
    ) -> Tuple[bool, List[str]]:
        """Test if plugin should activate based on directory patterns."""
        if not plugin.directory_patterns:
            return False, []

        matching_dirs = []
        for dir_path in sample_path.rglob("*"):
            if dir_path.is_dir():
                dir_name = dir_path.name.lower()
                for pattern in plugin.directory_patterns:
                    if pattern.lower() in str(dir_path).lower():
                        matching_dirs.append(str(dir_path.relative_to(sample_path)))
                        break

        return len(matching_dirs) > 0, matching_dirs

    def test_content_pattern_matching(
        self, sample_path: Path, plugin: PluginTriggeringPatterns
    ) -> Tuple[bool, List[str]]:
        """Test if plugin should activate based on content patterns."""
        if not plugin.content_patterns:
            return False, []

        matching_patterns = []
        match_count = 0

        for file_path in sample_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in {
                ".py", ".jl", ".js", ".ts", ".rs", ".cpp", ".c", ".go", ".java",
                ".toml", ".json", ".yaml", ".yml", ".md", ".txt"
            }:
                try:
                    content = file_path.read_text(errors="ignore").lower()
                    for pattern in plugin.content_patterns:
                        if pattern in content:
                            match_count += 1
                            if pattern not in matching_patterns:
                                matching_patterns.append(pattern)
                except Exception:
                    pass

        # Require multiple pattern matches for activation
        threshold = max(2, len(plugin.content_patterns) * 0.1)
        return match_count >= threshold, matching_patterns

    def test_sample(self, sample: Dict) -> ActivationTestResult:
        """Test plugin activation for a single sample."""
        sample_path = sample["path"]
        result = ActivationTestResult(
            sample_name=sample["name"],
            sample_category=sample["category"],
            expected_plugins=sample["expected_plugins"],
            activated_plugins=[],
            expected_trigger=sample["expected_trigger"]
        )

        # Test each plugin
        for plugin_name, plugin in self.plugins.items():
            confidence_scores = []
            matching_patterns = []

            # Test file extensions
            ext_match, ext_files = self.test_file_extension_matching(sample_path, plugin)
            if ext_match:
                confidence_scores.append(0.4)
                matching_patterns.extend([f"ext:{f}" for f in ext_files[:3]])

            # Test directory patterns
            dir_match, dir_patterns = self.test_directory_pattern_matching(sample_path, plugin)
            if dir_match:
                confidence_scores.append(0.3)
                matching_patterns.extend([f"dir:{d}" for d in dir_patterns[:3]])

            # Test content patterns
            content_match, content_patterns = self.test_content_pattern_matching(sample_path, plugin)
            if content_match:
                confidence_scores.append(0.3)
                matching_patterns.extend([f"content:{p}" for p in content_patterns[:5]])

            # Calculate overall confidence
            confidence = sum(confidence_scores) / 1.0 if confidence_scores else 0.0

            # Activate if confidence is high enough
            if confidence >= 0.3:  # 30% threshold
                result.activated_plugins.append(plugin_name)
                result.matching_patterns[plugin_name] = matching_patterns

        # Determine if activation occurred
        result.actual_trigger = len(result.activated_plugins) > 0

        # Calculate true/false positives/negatives
        if result.expected_trigger and result.actual_trigger:
            result.true_positive = True
        elif not result.expected_trigger and not result.actual_trigger:
            result.true_negative = True
        elif not result.expected_trigger and result.actual_trigger:
            result.false_positive = True
        elif result.expected_trigger and not result.actual_trigger:
            result.false_negative = True

        # Calculate confidence score based on expected plugin match
        if result.expected_plugins:
            matched = len(set(result.activated_plugins) & set(result.expected_plugins))
            result.confidence_score = matched / len(result.expected_plugins)
        else:
            result.confidence_score = 1.0 if result.true_negative else 0.0

        return result

    def run_tests(self) -> None:
        """Run activation tests on all corpus samples."""
        print("\nRunning activation tests...")

        samples = self.load_test_corpus()

        for i, sample in enumerate(samples, 1):
            print(f"  Testing {i}/{len(samples)}: {sample['name']}...", end=" ")
            result = self.test_sample(sample)
            self.results.append(result)

            status = "✓" if (result.true_positive or result.true_negative) else "✗"
            print(status)

    def calculate_metrics(self) -> ActivationMetrics:
        """Calculate aggregated activation metrics."""
        metrics = ActivationMetrics()

        for result in self.results:
            metrics.total_samples += 1
            if result.true_positive:
                metrics.true_positives += 1
            if result.true_negative:
                metrics.true_negatives += 1
            if result.false_positive:
                metrics.false_positives += 1
            if result.false_negative:
                metrics.false_negatives += 1

        return metrics

    def generate_report(self) -> str:
        """Generate activation accuracy report."""
        metrics = self.calculate_metrics()

        report = f"""# Activation Accuracy Test Report

**Test Date:** {self._get_timestamp()}
**Total Plugins:** {len(self.plugins)}
**Total Samples:** {len(self.results)}

## Summary Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | {metrics.accuracy:.1f}% | >90% | {self._status_icon(metrics.accuracy > 90)} |
| Precision | {metrics.precision:.1f}% | >90% | {self._status_icon(metrics.precision > 90)} |
| Recall | {metrics.recall:.1f}% | >90% | {self._status_icon(metrics.recall > 90)} |
| F1 Score | {metrics.f1_score:.1f}% | >90% | {self._status_icon(metrics.f1_score > 90)} |
| False Positive Rate | {metrics.false_positive_rate:.1f}% | <5% | {self._status_icon(metrics.false_positive_rate < 5)} |
| False Negative Rate | {metrics.false_negative_rate:.1f}% | <5% | {self._status_icon(metrics.false_negative_rate < 5)} |

## Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | TP: {metrics.true_positives} | FN: {metrics.false_negatives} |
| **Actual Negative** | FP: {metrics.false_positives} | TN: {metrics.true_negatives} |

## Detailed Results

### True Positives ({metrics.true_positives})

Samples that correctly triggered plugins:

"""

        for result in self.results:
            if result.true_positive:
                report += f"- **{result.sample_name}** ({result.sample_category})\n"
                report += f"  - Expected: {', '.join(result.expected_plugins)}\n"
                report += f"  - Activated: {', '.join(result.activated_plugins)}\n"
                report += f"  - Confidence: {result.confidence_score:.1%}\n\n"

        report += f"""
### False Positives ({metrics.false_positives})

Samples that triggered plugins but shouldn't have:

"""

        for result in self.results:
            if result.false_positive:
                report += f"- **{result.sample_name}** ({result.sample_category})\n"
                report += f"  - Expected: No activation\n"
                report += f"  - Activated: {', '.join(result.activated_plugins)}\n"
                report += f"  - Problematic plugins: {', '.join(result.activated_plugins)}\n\n"

        report += f"""
### False Negatives ({metrics.false_negatives})

Samples that should have triggered plugins but didn't:

"""

        for result in self.results:
            if result.false_negative:
                report += f"- **{result.sample_name}** ({result.sample_category})\n"
                report += f"  - Expected: {', '.join(result.expected_plugins)}\n"
                report += f"  - Activated: None\n"
                report += f"  - Missing plugins: {', '.join(result.expected_plugins)}\n\n"

        report += f"""
## Plugin Performance

"""

        # Calculate per-plugin metrics
        plugin_stats = {}
        for result in self.results:
            for plugin in result.expected_plugins:
                if plugin not in plugin_stats:
                    plugin_stats[plugin] = {"expected": 0, "activated": 0}
                plugin_stats[plugin]["expected"] += 1

            for plugin in result.activated_plugins:
                if plugin not in plugin_stats:
                    plugin_stats[plugin] = {"expected": 0, "activated": 0}
                plugin_stats[plugin]["activated"] += 1

        report += "| Plugin | Expected | Activated | Accuracy |\n"
        report += "|--------|----------|-----------|----------|\n"

        for plugin, stats in sorted(plugin_stats.items()):
            expected = stats["expected"]
            activated = stats["activated"]
            accuracy = (min(expected, activated) / max(expected, activated) * 100
                       if max(expected, activated) > 0 else 0)
            report += f"| {plugin} | {expected} | {activated} | {accuracy:.1f}% |\n"

        report += f"""
## Overall Assessment

"""

        if metrics.false_positive_rate < 5 and metrics.false_negative_rate < 5:
            report += "**Status:** ✅ EXCELLENT - Both FP and FN rates below 5% target\n\n"
        elif metrics.false_positive_rate < 10 and metrics.false_negative_rate < 10:
            report += "**Status:** ⚠️ GOOD - FP and FN rates acceptable but could be improved\n\n"
        else:
            report += "**Status:** ❌ NEEDS IMPROVEMENT - FP or FN rates above acceptable threshold\n\n"

        report += f"Plugin activation patterns are "
        if metrics.accuracy > 90:
            report += "highly accurate and meet quality standards. "
        elif metrics.accuracy > 75:
            report += "generally accurate but have room for improvement. "
        else:
            report += "need significant optimization. "

        if metrics.false_positive_rate >= 5:
            report += f"\n\n**Action Required:** High false positive rate ({metrics.false_positive_rate:.1f}%) "
            report += "indicates plugins are activating too broadly. Review triggering patterns."

        if metrics.false_negative_rate >= 5:
            report += f"\n\n**Action Required:** High false negative rate ({metrics.false_negative_rate:.1f}%) "
            report += "indicates plugins are not activating when needed. Expand triggering patterns."

        return report

    def _status_icon(self, condition: bool) -> str:
        """Return status icon based on condition."""
        return "✅" if condition else "❌"

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="Test plugin activation accuracy against test corpus"
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
        help="Test specific plugin only"
    )
    parser.add_argument(
        "--output",
        default="reports/activation-accuracy.md",
        help="Output report file (default: reports/activation-accuracy.md)"
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
        print(f"Run test-corpus-generator.py first to create test samples")
        return 1

    # Run tests
    tester = PluginActivationTester(str(plugins_dir), str(corpus_dir))
    tester.load_plugins(args.plugin)
    tester.run_tests()

    # Generate report
    report = tester.generate_report()
    print("\n" + "=" * 70)
    print(report)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n✓ Report saved to: {output_path.absolute()}")

    # Determine exit code based on metrics
    metrics = tester.calculate_metrics()
    if metrics.false_positive_rate < 5 and metrics.false_negative_rate < 5:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
