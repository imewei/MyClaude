#!/usr/bin/env python3
"""
Skill Application Validator for Plugin Triggering Patterns

Tests skill pattern matching, validates skill recommendations,
and checks for over-triggering issues.

Usage:
    python3 tools/skill_validator.py
    python3 tools/skill_validator.py --plugins-dir /path/to/plugins
    python3 tools/skill_validator.py --corpus-dir /path/to/test-corpus
    python3 tools/skill_validator.py --plugin julia-development
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


@dataclass
class Skill:
    """Represents a plugin skill."""

    name: str
    description: str
    status: str
    plugin_name: str
    keywords: Set[str] = field(default_factory=set)
    patterns: Set[str] = field(default_factory=set)
    file_patterns: Set[str] = field(default_factory=set)


@dataclass
class SkillContext:
    """Context for skill application."""

    file_path: Path
    file_extension: str
    file_content: str
    imports: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    classes: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    patterns_found: Set[str] = field(default_factory=set)


@dataclass
class SkillApplicationResult:
    """Result of skill application test."""

    skill_name: str
    plugin_name: str
    context_path: str
    should_apply: bool
    did_apply: bool
    confidence_score: float
    true_positive: bool = False
    true_negative: bool = False
    false_positive: bool = False
    false_negative: bool = False
    matching_patterns: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class SkillValidationMetrics:
    """Aggregated skill validation metrics."""

    total_tests: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        """Overall accuracy percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total_tests * 100

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
    def over_trigger_rate(self) -> float:
        """False positive rate (over-triggering)."""
        denominator = self.false_positives + self.true_negatives
        if denominator == 0:
            return 0.0
        return self.false_positives / denominator * 100

    @property
    def under_trigger_rate(self) -> float:
        """False negative rate (under-triggering)."""
        denominator = self.false_negatives + self.true_positives
        if denominator == 0:
            return 0.0
        return self.false_negatives / denominator * 100


class SkillApplicationValidator:
    """Validates skill pattern matching and application."""

    def __init__(self, plugins_dir: str, corpus_dir: str):
        self.plugins_dir = Path(plugins_dir)
        self.corpus_dir = Path(corpus_dir)
        self.skills: List[Skill] = []
        self.results: List[SkillApplicationResult] = []

    def load_skills(self, specific_plugin: Optional[str] = None) -> None:
        """Load skills from all plugins."""
        print("Loading plugin skills...")

        plugin_dirs = self._get_plugin_dirs(specific_plugin)

        for plugin_dir in plugin_dirs:
            self._load_skills_from_plugin(plugin_dir)

        print(f"\nLoaded {len(self.skills)} total skills")

    def _get_plugin_dirs(self, specific_plugin: Optional[str]) -> List[Path]:
        """Get list of plugin directories to process."""
        if specific_plugin:
            plugin_path = self.plugins_dir / specific_plugin
            if plugin_path.exists():
                return [plugin_path]
            return []

        return [p for p in self.plugins_dir.iterdir() if p.is_dir()]

    def _load_skills_from_plugin(self, plugin_dir: Path) -> None:
        """Load skills from a single plugin directory."""
        plugin_json = plugin_dir / "plugin.json"
        if not plugin_json.exists():
            return

        try:
            data = json.loads(plugin_json.read_text())
            plugin_name = data["name"]

            if "skills" not in data or not data["skills"]:
                return

            for skill_data in data["skills"]:
                self._add_skill(skill_data, plugin_name, plugin_dir)

            count = len([s for s in self.skills if s.plugin_name == plugin_name])
            print(f"  ✓ {plugin_name}: {count} skills")

        except Exception as e:
            print(f"  ✗ Error loading {plugin_dir.name}: {e}")

    def _add_skill(self, skill_data: Dict, plugin_name: str, plugin_dir: Path) -> None:
        """Create and add a skill object from data."""
        skill = Skill(
            name=skill_data["name"],
            description=skill_data.get("description", ""),
            status=skill_data.get("status", "active"),
            plugin_name=plugin_name
        )

        # Extract keywords from skill name and description
        text = f"{skill.name} {skill.description}".lower()
        skill.keywords = self._extract_keywords(text)

        # Extract technical patterns
        skill.patterns = self._extract_patterns(text)

        # Try to load skill documentation for more patterns
        self._enrich_skill_from_docs(skill, plugin_dir)

        # Infer file patterns from skill name
        skill.file_patterns = self._infer_file_patterns(skill.name)

        self.skills.append(skill)

    def _enrich_skill_from_docs(self, skill: Skill, plugin_dir: Path) -> None:
        """Enrich skill keywords and patterns from documentation."""
        skill_file = plugin_dir / "skills" / f"{skill.name}.md"
        if skill_file.exists():
            try:
                skill_content = skill_file.read_text().lower()
                skill.keywords.update(self._extract_keywords(skill_content))
                skill.patterns.update(self._extract_patterns(skill_content))
            except Exception:
                pass

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Split on non-alphanumeric
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Filter stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are",
            "this", "that", "these", "those", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should"
        }

        return {w for w in words if w not in stopwords and len(w) > 2}

    def _extract_patterns(self, text: str) -> Set[str]:
        """Extract technical patterns from text."""
        patterns: set[str] = set()

        # Extract package/module patterns (e.g., DifferentialEquations.jl)
        packages = re.findall(r'\b[A-Z][a-zA-Z]+(?:\.[a-z]+)+\b', text)
        patterns.update(p.lower() for p in packages)

        # Extract API patterns (e.g., @code_warntype)
        api_patterns = re.findall(r'@\w+|:\w+', text)
        patterns.update(p.lower() for p in api_patterns)

        # Extract common function patterns
        func_patterns = re.findall(r'\b\w+\(\)', text)
        patterns.update(p.lower().replace('()', '') for p in func_patterns)

        return patterns

    def _infer_file_patterns(self, skill_name: str) -> Set[str]:
        """Infer file patterns from skill name."""
        patterns = set()

        # Language-specific skills
        if "julia" in skill_name:
            patterns.add(".jl")
        if "python" in skill_name:
            patterns.add(".py")
        if "javascript" in skill_name or "js" in skill_name:
            patterns.update({".js", ".jsx"})
        if "typescript" in skill_name or "ts" in skill_name:
            patterns.update({".ts", ".tsx"})
        if "rust" in skill_name:
            patterns.add(".rs")
        if "cpp" in skill_name or "c++" in skill_name:
            patterns.update({".cpp", ".hpp", ".cc", ".h"})

        # File type specific
        if "test" in skill_name:
            patterns.add("test")
        if "config" in skill_name:
            patterns.update({".toml", ".json", ".yaml", ".yml"})
        if "docker" in skill_name:
            patterns.add("dockerfile")

        return patterns

    def analyze_file_context(self, file_path: Path) -> SkillContext:
        """Analyze a file to extract context for skill matching."""
        content = file_path.read_text(errors="ignore")
        ext = file_path.suffix.lower()

        context = SkillContext(
            file_path=file_path,
            file_extension=ext,
            file_content=content
        )

        # Extract keywords
        context.keywords = self._extract_keywords(content)

        # Extract patterns
        context.patterns_found = self._extract_patterns(content)

        # Extract imports based on file type
        if ext == ".py":
            imports = re.findall(r'import\s+(\w+)|from\s+(\w+)', content)
            context.imports = {imp[0] or imp[1] for imp in imports}
        elif ext == ".jl":
            imports = re.findall(r'using\s+(\w+)|import\s+(\w+)', content)
            context.imports = {imp[0] or imp[1] for imp in imports}
        elif ext in {".js", ".ts", ".jsx", ".tsx"}:
            imports = re.findall(r'import.*from\s+[\'"](\w+)', content)
            context.imports = set(imports)

        # Extract function definitions
        if ext == ".py":
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            context.functions = set(functions)
        elif ext == ".jl":
            functions = re.findall(r'function\s+(\w+)\s*\(', content)
            context.functions = set(functions)
        elif ext in {".js", ".ts"}:
            functions = re.findall(r'function\s+(\w+)|const\s+(\w+)\s*=.*=>', content)
            context.functions = {f[0] or f[1] for f in functions}

        # Extract class definitions
        if ext in {".py", ".js", ".ts"}:
            classes = re.findall(r'class\s+(\w+)', content)
            context.classes = set(classes)

        return context

    def calculate_skill_match_score(
        self, skill: Skill, context: SkillContext
    ) -> Tuple[float, List[str]]:
        """Calculate how well a skill matches the given context."""
        score = 0.0
        matching_patterns = []

        # Check file extension match
        if skill.file_patterns:
            if context.file_extension in skill.file_patterns:
                score += 0.3
                matching_patterns.append(f"ext:{context.file_extension}")
            elif any(pattern in str(context.file_path).lower() for pattern in skill.file_patterns):
                score += 0.2
                matching_patterns.append("file_pattern")

        # Check keyword overlap
        keyword_overlap = skill.keywords & context.keywords
        if keyword_overlap:
            keyword_score = min(len(keyword_overlap) * 0.1, 0.4)
            score += keyword_score
            matching_patterns.extend([f"kw:{kw}" for kw in list(keyword_overlap)[:3]])

        # Check pattern overlap (imports, functions, etc.)
        pattern_overlap = skill.patterns & context.patterns_found
        if pattern_overlap:
            pattern_score = min(len(pattern_overlap) * 0.15, 0.4)
            score += pattern_score
            matching_patterns.extend([f"pat:{p}" for p in list(pattern_overlap)[:3]])

        # Check specific imports
        if skill.patterns & context.imports:
            score += 0.3
            matching_patterns.append("import_match")

        # Skill-specific rules
        skill_name_lower = skill.name.lower()

        # Testing skills need test-related context
        if "test" in skill_name_lower:
            if "test" in str(context.file_path).lower() or "test" in context.keywords:
                score += 0.2

        # Optimization skills need performance context
        if "optim" in skill_name_lower or "performance" in skill_name_lower:
            perf_keywords = {"benchmark", "profile", "performance", "speed", "optimize"}
            if context.keywords & perf_keywords:
                score += 0.2

        # Async skills need async context
        if "async" in skill_name_lower:
            async_keywords = {"async", "await", "asyncio", "task", "coroutine"}
            if context.keywords & async_keywords:
                score += 0.2

        # Package/module skills need related imports
        if "package" in skill_name_lower or "module" in skill_name_lower:
            pkg_keywords = {"import", "using", "require", "package", "module"}
            if context.keywords & pkg_keywords:
                score += 0.2

        return min(score, 1.0), matching_patterns

    def determine_expected_application(
        self, skill: Skill, context: SkillContext
    ) -> bool:
        """Determine if skill should apply to this context (ground truth)."""
        # This is a heuristic for generating ground truth
        # In practice, this would come from manual labeling

        score, _ = self.calculate_skill_match_score(skill, context)

        # Check plugin-context alignment
        plugin_context_match = False

        # Julia skills should apply to Julia files
        if skill.plugin_name == "julia-development":
            plugin_context_match = context.file_extension == ".jl"

        # Python skills should apply to Python files
        elif "python" in skill.plugin_name:
            plugin_context_match = context.file_extension == ".py"

        # JavaScript/TypeScript skills
        elif "javascript" in skill.plugin_name or "typescript" in skill.plugin_name:
            plugin_context_match = context.file_extension in {".js", ".ts", ".jsx", ".tsx"}

        # Rust skills
        elif "rust" in skill.plugin_name or "systems" in skill.plugin_name:
            plugin_context_match = context.file_extension in {".rs", ".toml"}

        # Generic skills (testing, CI/CD, etc.) can apply broadly
        else:
            plugin_context_match = True

        # Skill should apply if:
        # 1. Context matches plugin
        # 2. AND score is high enough
        return plugin_context_match and score >= 0.3

    def test_skill_application(self) -> None:
        """Test skill application across all contexts."""
        print("\nTesting skill applications...")

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

            # Get all code files
            code_extensions = {".py", ".jl", ".js", ".ts", ".jsx", ".tsx", ".rs",
                              ".cpp", ".c", ".go", ".java"}

            for file_path in sample["path"].rglob("*"):
                if not file_path.is_file() or file_path.suffix.lower() not in code_extensions:
                    continue

                try:
                    context = self.analyze_file_context(file_path)

                    for skill in self.skills:
                        # Calculate match score
                        score, matching_patterns = self.calculate_skill_match_score(
                            skill, context
                        )

                        # Determine expected and actual application
                        should_apply = self.determine_expected_application(skill, context)
                        did_apply = score >= 0.3  # Application threshold

                        # Create result
                        result = SkillApplicationResult(
                            skill_name=skill.name,
                            plugin_name=skill.plugin_name,
                            context_path=str(file_path.relative_to(sample["path"])),
                            should_apply=should_apply,
                            did_apply=did_apply,
                            confidence_score=score,
                            matching_patterns=matching_patterns,
                            explanation=self._generate_explanation(skill, context, score)
                        )

                        # Determine classification
                        if should_apply and did_apply:
                            result.true_positive = True
                        elif not should_apply and not did_apply:
                            result.true_negative = True
                        elif not should_apply and did_apply:
                            result.false_positive = True
                        elif should_apply and not did_apply:
                            result.false_negative = True

                        self.results.append(result)

                except Exception:
                    pass

        print(f"\n  Analyzed {len(self.results)} skill applications")

    def _generate_explanation(
        self, skill: Skill, context: SkillContext, score: float
    ) -> str:
        """Generate explanation for skill application decision."""
        if score >= 0.7:
            return "Strong match - skill highly relevant to context"
        elif score >= 0.5:
            return "Good match - skill relevant to context"
        elif score >= 0.3:
            return "Moderate match - skill partially relevant"
        else:
            return "Weak match - skill not relevant to context"

    def calculate_metrics(self) -> SkillValidationMetrics:
        """Calculate aggregated skill validation metrics."""
        metrics = SkillValidationMetrics()

        for result in self.results:
            metrics.total_tests += 1

            if result.true_positive:
                metrics.true_positives += 1
            elif result.true_negative:
                metrics.true_negatives += 1
            elif result.false_positive:
                metrics.false_positives += 1
            elif result.false_negative:
                metrics.false_negatives += 1

        return metrics

    def generate_report(self) -> str:
        """Generate skill validation report."""
        metrics = self.calculate_metrics()

        report = f"""# Skill Application Validation Report

**Validation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Skills:** {len(self.skills)}
**Total Applications Tested:** {len(self.results)}

## Summary Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | {metrics.accuracy:.1f}% | >90% | {self._status_icon(metrics.accuracy > 90)} |
| Precision | {metrics.precision:.1f}% | >85% | {self._status_icon(metrics.precision > 85)} |
| Recall | {metrics.recall:.1f}% | >85% | {self._status_icon(metrics.recall > 85)} |
| Over-Trigger Rate | {metrics.over_trigger_rate:.1f}% | <10% | {self._status_icon(metrics.over_trigger_rate < 10)} |
| Under-Trigger Rate | {metrics.under_trigger_rate:.1f}% | <10% | {self._status_icon(metrics.under_trigger_rate < 10)} |

## Classification Results

|  | Predicted Apply | Predicted Not Apply |
|--|-----------------|-------------------|
| **Should Apply** | TP: {metrics.true_positives} | FN: {metrics.false_negatives} |
| **Should Not Apply** | FP: {metrics.false_positives} | TN: {metrics.true_negatives} |

## Skill Performance Analysis

"""

        # Analyze per-skill performance
        skill_stats = {}
        for result in self.results:
            key = f"{result.plugin_name}/{result.skill_name}"
            if key not in skill_stats:
                skill_stats[key] = {
                    "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                    "total": 0, "avg_score": 0.0
                }

            stats = skill_stats[key]
            stats["total"] += 1
            stats["avg_score"] += result.confidence_score

            if result.true_positive:
                stats["tp"] += 1
            elif result.true_negative:
                stats["tn"] += 1
            elif result.false_positive:
                stats["fp"] += 1
            elif result.false_negative:
                stats["fn"] += 1

        report += "### Top Performing Skills\n\n"
        report += "| Skill | Plugin | Accuracy | Precision | Avg Score |\n"
        report += "|-------|--------|----------|-----------|----------|\n"

        sorted_skills = sorted(
            skill_stats.items(),
            key=lambda x: (x[1]["tp"] + x[1]["tn"]) / x[1]["total"],
            reverse=True
        )

        for key, stats in sorted_skills[:10]:
            plugin_name, skill_name = key.split("/", 1)
            accuracy = (stats["tp"] + stats["tn"]) / stats["total"] * 100
            precision = stats["tp"] / (stats["tp"] + stats["fp"]) * 100 if (stats["tp"] + stats["fp"]) > 0 else 0
            avg_score = stats["avg_score"] / stats["total"]

            report += f"| {skill_name} | {plugin_name} | {accuracy:.0f}% | "
            report += f"{precision:.0f}% | {avg_score:.2f} |\n"

        report += "\n### Skills with Over-Triggering Issues\n\n"

        # Find skills with high false positive rate
        over_trigger_skills = [
            (key, stats) for key, stats in sorted_skills
            if stats["fp"] / max(stats["fp"] + stats["tn"], 1) > 0.2
        ]

        if over_trigger_skills:
            report += "| Skill | Plugin | FP Rate | Issue |\n"
            report += "|-------|--------|---------|-------|\n"

            for key, stats in over_trigger_skills[:10]:
                plugin_name, skill_name = key.split("/", 1)
                fp_rate = stats["fp"] / max(stats["fp"] + stats["tn"], 1) * 100

                report += f"| {skill_name} | {plugin_name} | {fp_rate:.0f}% | "
                report += "Applying too broadly |\n"
        else:
            report += "No significant over-triggering issues detected.\n"

        report += "\n### Skills with Under-Triggering Issues\n\n"

        # Find skills with high false negative rate
        under_trigger_skills = [
            (key, stats) for key, stats in sorted_skills
            if stats["fn"] / max(stats["fn"] + stats["tp"], 1) > 0.2
        ]

        if under_trigger_skills:
            report += "| Skill | Plugin | FN Rate | Issue |\n"
            report += "|-------|--------|---------|-------|\n"

            for key, stats in under_trigger_skills[:10]:
                plugin_name, skill_name = key.split("/", 1)
                fn_rate = stats["fn"] / max(stats["fn"] + stats["tp"], 1) * 100

                report += f"| {skill_name} | {plugin_name} | {fn_rate:.0f}% | "
                report += "Not applying when needed |\n"
        else:
            report += "No significant under-triggering issues detected.\n"

        report += """
## Overall Assessment

"""

        if metrics.over_trigger_rate < 10 and metrics.under_trigger_rate < 10:
            report += "**Status:** ✅ EXCELLENT - Skill triggering is well-balanced\n\n"
        elif metrics.over_trigger_rate < 20 and metrics.under_trigger_rate < 20:
            report += "**Status:** ⚠️ GOOD - Skill triggering is acceptable but could be improved\n\n"
        else:
            report += "**Status:** ❌ NEEDS IMPROVEMENT - Skill triggering needs optimization\n\n"

        report += f"Skill pattern matching is performing at {metrics.accuracy:.1f}% accuracy. "

        if metrics.over_trigger_rate >= 10:
            report += f"\n\n**Action Required:** Over-trigger rate of {metrics.over_trigger_rate:.1f}% "
            report += "indicates skills are applying too broadly. Review and tighten matching patterns."

        if metrics.under_trigger_rate >= 10:
            report += f"\n\n**Action Required:** Under-trigger rate of {metrics.under_trigger_rate:.1f}% "
            report += "indicates skills are not applying when needed. Expand matching patterns and keywords."

        return report

    def _status_icon(self, condition: bool) -> str:
        """Return status icon based on condition."""
        return "✅" if condition else "❌"


def main():
    parser = argparse.ArgumentParser(
        description="Validate skill pattern matching and application"
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
        help="Validate specific plugin only"
    )
    parser.add_argument(
        "--output",
        default="reports/skill-validation.md",
        help="Output report file (default: reports/skill-validation.md)"
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

    # Run validation
    validator = SkillApplicationValidator(str(plugins_dir), str(corpus_dir))
    validator.load_skills(args.plugin)
    validator.test_skill_application()

    # Generate report
    report = validator.generate_report()
    print("\n" + "=" * 70)
    print(report)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n✓ Report saved to: {output_path.absolute()}")

    # Determine exit code
    metrics = validator.calculate_metrics()
    if metrics.over_trigger_rate < 10 and metrics.under_trigger_rate < 10:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
