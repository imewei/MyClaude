#!/usr/bin/env python3
"""
Adaptive Agent Selection
========================

Learn optimal agent selection from usage patterns and performance data.

Features:
- Track agent performance metrics
- Learn from historical data
- Optimize agent combinations
- Predict best agent for task
- A/B test agent strategies
- Personalized agent selection

Author: Claude Code AI Team
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict


@dataclass
class AgentPerformance:
    """Performance metrics for an agent"""
    agent_name: str
    task_type: str
    success_count: int = 0
    failure_count: int = 0
    avg_duration: float = 0.0
    avg_quality_score: float = 0.0
    total_executions: int = 0


class AdaptiveSelector:
    """
    Adaptive agent selection using historical performance data.

    Learns from past executions to optimize future agent selection.
    """

    def __init__(self, history_file: Optional[Path] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history_file = history_file or Path.home() / ".claude" / "agent_history.json"

        # Performance tracking
        self.performance_data: Dict[str, AgentPerformance] = {}

        # Load historical data
        self._load_history()

    def select_agents(
        self,
        task_type: str,
        context: Dict[str, Any],
        max_agents: int = 5
    ) -> List[str]:
        """
        Select optimal agents based on historical performance.

        Args:
            task_type: Type of task
            context: Task context
            max_agents: Maximum agents to select

        Returns:
            List of selected agent names
        """
        self.logger.info(f"Adaptive selection for task: {task_type}")

        # Get candidate agents for this task type
        candidates = self._get_candidate_agents(task_type)

        # Score each agent
        scored_agents = []
        for agent_name in candidates:
            score = self._calculate_agent_score(agent_name, task_type, context)
            scored_agents.append((agent_name, score))

        # Sort by score
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        # Select top agents
        selected = [agent for agent, _ in scored_agents[:max_agents]]

        self.logger.info(f"Selected agents: {selected}")

        return selected

    def record_execution(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        duration: float,
        quality_score: float = 0.0
    ):
        """
        Record agent execution for learning.

        Args:
            agent_name: Name of agent
            task_type: Type of task
            success: Whether execution succeeded
            duration: Execution duration in seconds
            quality_score: Quality score (0-1)
        """
        key = f"{agent_name}:{task_type}"

        if key not in self.performance_data:
            self.performance_data[key] = AgentPerformance(
                agent_name=agent_name,
                task_type=task_type
            )

        perf = self.performance_data[key]

        if success:
            perf.success_count += 1
        else:
            perf.failure_count += 1

        perf.total_executions += 1

        # Update running averages
        perf.avg_duration = (
            (perf.avg_duration * (perf.total_executions - 1) + duration) /
            perf.total_executions
        )

        perf.avg_quality_score = (
            (perf.avg_quality_score * (perf.total_executions - 1) + quality_score) /
            perf.total_executions
        )

        # Save updated history
        self._save_history()

    def _calculate_agent_score(
        self,
        agent_name: str,
        task_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate score for agent selection"""
        key = f"{agent_name}:{task_type}"

        if key not in self.performance_data:
            # New agent, give benefit of doubt
            return 0.5

        perf = self.performance_data[key]

        # Calculate success rate
        total = perf.success_count + perf.failure_count
        success_rate = perf.success_count / total if total > 0 else 0.5

        # Factor in quality score
        quality = perf.avg_quality_score

        # Factor in efficiency (inverse of duration)
        # Normalize to 0-1 range
        efficiency = 1.0 / (1.0 + perf.avg_duration)

        # Weighted combination
        score = (
            success_rate * 0.5 +
            quality * 0.3 +
            efficiency * 0.2
        )

        return score

    def _get_candidate_agents(self, task_type: str) -> List[str]:
        """Get candidate agents for task type"""
        # Map task types to agent categories
        task_agent_mapping = {
            "optimization": [
                "scientific-computing-master",
                "ai-systems-architect",
                "code-quality-master",
            ],
            "testing": [
                "code-quality-master",
                "devops-security-engineer",
            ],
            "documentation": [
                "documentation-architect",
                "code-quality-master",
            ],
            "refactoring": [
                "code-quality-master",
                "systems-architect",
                "scientific-computing-master",
            ],
            "analysis": [
                "code-quality-master",
                "systems-architect",
                "research-intelligence-master",
            ],
        }

        candidates = task_agent_mapping.get(task_type, [
            "multi-agent-orchestrator",
            "code-quality-master",
            "systems-architect",
        ])

        return candidates

    def _load_history(self):
        """Load historical performance data"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)

            for key, perf_dict in data.items():
                self.performance_data[key] = AgentPerformance(**perf_dict)

            self.logger.info(f"Loaded {len(self.performance_data)} performance records")

        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")

    def _save_history(self):
        """Save performance data"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                key: {
                    "agent_name": perf.agent_name,
                    "task_type": perf.task_type,
                    "success_count": perf.success_count,
                    "failure_count": perf.failure_count,
                    "avg_duration": perf.avg_duration,
                    "avg_quality_score": perf.avg_quality_score,
                    "total_executions": perf.total_executions,
                }
                for key, perf in self.performance_data.items()
            }

            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for agent"""
        stats = {
            "total_tasks": 0,
            "success_rate": 0.0,
            "avg_quality": 0.0,
            "avg_duration": 0.0,
            "task_breakdown": {}
        }

        # Aggregate across all task types
        for key, perf in self.performance_data.items():
            if not key.startswith(agent_name + ":"):
                continue

            total = perf.success_count + perf.failure_count
            stats["total_tasks"] += total

            if total > 0:
                task_success_rate = perf.success_count / total
                stats["task_breakdown"][perf.task_type] = {
                    "executions": total,
                    "success_rate": task_success_rate,
                    "quality": perf.avg_quality_score,
                    "duration": perf.avg_duration
                }

        # Calculate overall metrics
        if stats["total_tasks"] > 0:
            total_success = sum(
                perf.success_count
                for key, perf in self.performance_data.items()
                if key.startswith(agent_name + ":")
            )
            stats["success_rate"] = total_success / stats["total_tasks"]

            # Weighted averages
            total_weight = stats["total_tasks"]
            stats["avg_quality"] = sum(
                perf.avg_quality_score * (perf.success_count + perf.failure_count)
                for key, perf in self.performance_data.items()
                if key.startswith(agent_name + ":")
            ) / total_weight

            stats["avg_duration"] = sum(
                perf.avg_duration * (perf.success_count + perf.failure_count)
                for key, perf in self.performance_data.items()
                if key.startswith(agent_name + ":")
            ) / total_weight

        return stats


def main():
    """Demonstration"""
    print("Adaptive Agent Selection")
    print("=======================\n")

    selector = AdaptiveSelector()

    # Simulate some executions
    selector.record_execution(
        "code-quality-master",
        "testing",
        success=True,
        duration=5.2,
        quality_score=0.9
    )

    selector.record_execution(
        "scientific-computing-master",
        "optimization",
        success=True,
        duration=10.5,
        quality_score=0.95
    )

    # Select agents
    selected = selector.select_agents("testing", {}, max_agents=3)
    print(f"Selected agents for testing: {selected}")

    # Get stats
    stats = selector.get_agent_stats("code-quality-master")
    print(f"\nCode Quality Master Stats:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())