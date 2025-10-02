"""
ML-based command recommendation system.

Provides intelligent command suggestions based on context, usage patterns,
and workflow detection.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import Counter, defaultdict
import re


@dataclass
class CommandRecommendation:
    """A recommended command."""
    command: str
    description: str
    confidence: float
    reason: str
    flags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    related_commands: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Project context for recommendations."""
    project_type: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    has_tests: bool = False
    has_ci: bool = False
    has_docs: bool = False
    file_count: int = 0
    recent_changes: List[str] = field(default_factory=list)


class CommandRecommender:
    """
    ML-based command recommendation system.

    Features:
    - Context detection (analyzes project state)
    - Usage pattern learning
    - Workflow detection
    - Next-command prediction
    - Confidence scoring
    - Alternative suggestions with explanations
    - Continuous learning from usage

    Example:
        recommender = CommandRecommender()

        # Get recommendations
        recommendations = recommender.recommend(
            context={"project_type": "python"},
            recent_commands=["git add .", "git commit"]
        )

        for rec in recommendations:
            print(f"{rec.command} - {rec.reason} (confidence: {rec.confidence})")
    """

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize command recommender.

        Args:
            history_file: Path to command history file
        """
        self.history_file = history_file or (Path.home() / ".claude" / "command_history.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        self.command_history: List[Dict[str, Any]] = []
        self.command_sequences: List[List[str]] = []
        self.command_contexts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self._load_history()

    def _load_history(self):
        """Load command history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.command_history = data.get("history", [])
                    self._build_patterns()
            except Exception:
                pass

    def _save_history(self):
        """Save command history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    "history": self.command_history[-1000:],  # Keep last 1000
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception:
            pass

    def _build_patterns(self):
        """Build pattern models from history."""
        # Extract command sequences
        current_sequence = []
        for entry in self.command_history:
            command = entry.get("command", "")
            if command:
                current_sequence.append(command)
                if len(current_sequence) > 5:  # Keep sequences of max 5
                    current_sequence.pop(0)

                # Store context for this command
                context = entry.get("context", {})
                self.command_contexts[command].append(context)

        # Build n-gram sequences
        for i in range(len(self.command_history) - 1):
            cmd1 = self.command_history[i].get("command", "")
            cmd2 = self.command_history[i + 1].get("command", "")
            if cmd1 and cmd2:
                self.command_sequences.append([cmd1, cmd2])

    def record_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
        success: bool = True
    ):
        """
        Record a command execution for learning.

        Args:
            command: Command that was executed
            context: Context information
            success: Whether command succeeded
        """
        entry = {
            "command": command,
            "context": context or {},
            "success": success,
            "timestamp": datetime.now().isoformat()
        }

        self.command_history.append(entry)

        # Update patterns
        if command not in self.command_contexts:
            self.command_contexts[command] = []
        self.command_contexts[command].append(context or {})

        # Save periodically
        if len(self.command_history) % 10 == 0:
            self._save_history()

    def detect_context(self, project_path: Optional[Path] = None) -> ProjectContext:
        """
        Detect project context.

        Args:
            project_path: Path to project directory

        Returns:
            Detected project context
        """
        context = ProjectContext()

        if not project_path:
            project_path = Path.cwd()

        if not project_path.exists():
            return context

        # Detect languages
        if list(project_path.glob("*.py")):
            context.languages.append("python")
        if list(project_path.glob("*.js")) or list(project_path.glob("*.ts")):
            context.languages.append("javascript")
        if list(project_path.glob("*.java")):
            context.languages.append("java")
        if list(project_path.glob("*.jl")):
            context.languages.append("julia")

        # Detect frameworks
        if (project_path / "requirements.txt").exists():
            context.frameworks.append("python")
        if (project_path / "package.json").exists():
            context.frameworks.append("node")
        if (project_path / "Cargo.toml").exists():
            context.frameworks.append("rust")

        # Detect tests
        context.has_tests = any([
            (project_path / "tests").exists(),
            (project_path / "test").exists(),
            list(project_path.glob("**/test_*.py")),
            list(project_path.glob("**/*_test.py"))
        ])

        # Detect CI
        context.has_ci = any([
            (project_path / ".github" / "workflows").exists(),
            (project_path / ".gitlab-ci.yml").exists(),
            (project_path / "Jenkinsfile").exists()
        ])

        # Detect docs
        context.has_docs = any([
            (project_path / "docs").exists(),
            (project_path / "README.md").exists(),
            (project_path / "documentation").exists()
        ])

        # File count
        try:
            context.file_count = sum(1 for _ in project_path.rglob("*") if _.is_file())
        except Exception:
            pass

        # Determine project type
        if "python" in context.languages:
            context.project_type = "python"
        elif "javascript" in context.languages:
            context.project_type = "javascript"
        elif "java" in context.languages:
            context.project_type = "java"

        return context

    def recommend(
        self,
        context: Optional[Dict[str, Any]] = None,
        recent_commands: Optional[List[str]] = None,
        goal: Optional[str] = None,
        limit: int = 5
    ) -> List[CommandRecommendation]:
        """
        Get command recommendations.

        Args:
            context: Current context
            recent_commands: Recently executed commands
            goal: User's goal (e.g., "improve code quality")
            limit: Maximum recommendations

        Returns:
            List of recommended commands
        """
        recommendations = []
        context = context or {}
        recent_commands = recent_commands or []

        # Get recommendations from different sources
        recommendations.extend(self._recommend_from_sequence(recent_commands))
        recommendations.extend(self._recommend_from_context(context))
        recommendations.extend(self._recommend_from_goal(goal))
        recommendations.extend(self._recommend_from_patterns(recent_commands))

        # Score and rank
        recommendations = self._score_recommendations(recommendations, context)

        # Remove duplicates
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec.command not in seen:
                seen.add(rec.command)
                unique_recommendations.append(rec)

        # Sort by confidence
        unique_recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return unique_recommendations[:limit]

    def _recommend_from_sequence(self, recent_commands: List[str]) -> List[CommandRecommendation]:
        """Recommend based on command sequence."""
        recommendations = []

        if not recent_commands:
            return recommendations

        last_command = recent_commands[-1]

        # Find what typically follows this command
        following_commands = Counter()
        for seq in self.command_sequences:
            if len(seq) >= 2 and seq[0] == last_command:
                following_commands[seq[1]] += 1

        # Create recommendations
        total = sum(following_commands.values())
        for command, count in following_commands.most_common(3):
            confidence = count / total if total > 0 else 0.5

            recommendations.append(CommandRecommendation(
                command=command,
                description=f"Often follows '{last_command}'",
                confidence=confidence,
                reason="Based on usage patterns"
            ))

        return recommendations

    def _recommend_from_context(self, context: Dict[str, Any]) -> List[CommandRecommendation]:
        """Recommend based on context."""
        recommendations = []

        project_type = context.get("project_type")

        if project_type == "python":
            recommendations.append(CommandRecommendation(
                command="/optimize",
                description="Optimize Python code performance",
                confidence=0.8,
                reason="Python project detected",
                flags=["--language=python"],
                examples=["/optimize --language=python src/"]
            ))

            recommendations.append(CommandRecommendation(
                command="/check-code-quality",
                description="Check Python code quality",
                confidence=0.75,
                reason="Python project detected",
                flags=["--language=python"],
                examples=["/check-code-quality --language=python"]
            ))

        # Test-related recommendations
        if context.get("has_tests"):
            recommendations.append(CommandRecommendation(
                command="/run-all-tests",
                description="Run all tests",
                confidence=0.85,
                reason="Test suite detected",
                examples=["/run-all-tests --coverage"]
            ))

        # CI-related recommendations
        if context.get("has_ci"):
            recommendations.append(CommandRecommendation(
                command="/fix-commit-errors",
                description="Fix CI/CD errors",
                confidence=0.7,
                reason="CI configuration detected",
                examples=["/fix-commit-errors --auto-fix"]
            ))

        return recommendations

    def _recommend_from_goal(self, goal: Optional[str]) -> List[CommandRecommendation]:
        """Recommend based on user's goal."""
        recommendations = []

        if not goal:
            return recommendations

        goal_lower = goal.lower()

        # Code quality goals
        if any(word in goal_lower for word in ["quality", "clean", "refactor", "improve"]):
            recommendations.append(CommandRecommendation(
                command="/refactor-clean",
                description="Refactor and clean code",
                confidence=0.9,
                reason=f"Matches goal: {goal}",
                examples=["/refactor-clean --scope=project"]
            ))

            recommendations.append(CommandRecommendation(
                command="/check-code-quality",
                description="Check code quality",
                confidence=0.85,
                reason=f"Matches goal: {goal}",
                examples=["/check-code-quality"]
            ))

        # Performance goals
        if any(word in goal_lower for word in ["fast", "speed", "performance", "optimize"]):
            recommendations.append(CommandRecommendation(
                command="/optimize",
                description="Optimize code performance",
                confidence=0.95,
                reason=f"Matches goal: {goal}",
                examples=["/optimize --implement"]
            ))

        # Documentation goals
        if any(word in goal_lower for word in ["document", "docs", "readme"]):
            recommendations.append(CommandRecommendation(
                command="/update-docs",
                description="Update documentation",
                confidence=0.95,
                reason=f"Matches goal: {goal}",
                examples=["/update-docs --type=all"]
            ))

        # Testing goals
        if any(word in goal_lower for word in ["test", "testing", "coverage"]):
            recommendations.append(CommandRecommendation(
                command="/generate-tests",
                description="Generate test suite",
                confidence=0.9,
                reason=f"Matches goal: {goal}",
                examples=["/generate-tests --coverage=80"]
            ))

        return recommendations

    def _recommend_from_patterns(self, recent_commands: List[str]) -> List[CommandRecommendation]:
        """Recommend based on detected patterns."""
        recommendations = []

        # Detect workflows
        if "git" in " ".join(recent_commands):
            recommendations.append(CommandRecommendation(
                command="/commit",
                description="Create smart commit",
                confidence=0.7,
                reason="Git workflow detected",
                flags=["--ai-message"],
                examples=["/commit --ai-message --push"]
            ))

        return recommendations

    def _score_recommendations(
        self,
        recommendations: List[CommandRecommendation],
        context: Dict[str, Any]
    ) -> List[CommandRecommendation]:
        """Score and adjust recommendation confidence."""
        for rec in recommendations:
            # Boost confidence for frequently used commands
            command_count = sum(1 for entry in self.command_history if entry.get("command") == rec.command)
            if command_count > 10:
                rec.confidence *= 1.2
            elif command_count > 5:
                rec.confidence *= 1.1

            # Cap at 1.0
            rec.confidence = min(1.0, rec.confidence)

        return recommendations

    def get_workflow_suggestions(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[List[str]]:
        """
        Get suggested workflows (command sequences).

        Args:
            goal: User's goal
            context: Current context

        Returns:
            List of command sequences (workflows)
        """
        workflows = []

        goal_lower = goal.lower()

        # Code quality workflow
        if "quality" in goal_lower:
            workflows.append([
                "/check-code-quality",
                "/refactor-clean --implement",
                "/run-all-tests",
                "/commit --ai-message"
            ])

        # Optimization workflow
        if "optim" in goal_lower or "performance" in goal_lower:
            workflows.append([
                "/optimize --analysis=comprehensive",
                "/optimize --implement",
                "/run-all-tests --benchmark",
                "/update-docs"
            ])

        # Documentation workflow
        if "document" in goal_lower:
            workflows.append([
                "/explain-code",
                "/update-docs --type=all",
                "/commit --ai-message"
            ])

        return workflows


# Global recommender instance
_global_recommender: Optional[CommandRecommender] = None


def get_global_recommender() -> CommandRecommender:
    """Get or create global recommender."""
    global _global_recommender
    if _global_recommender is None:
        _global_recommender = CommandRecommender()
    return _global_recommender