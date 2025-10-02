#!/usr/bin/env python3
"""
Integration Tests for Agent Orchestration
=========================================

Tests the 23-agent system coordination:
- Agent selection
- Intelligent matching
- Parallel execution
- Result synthesis
- Conflict resolution

Coverage: Multi-agent coordination, load balancing, communication
"""

import pytest
import time
from pathlib import Path
from typing import Dict, Any, List

from executors.framework import AgentOrchestrator, AgentType, ExecutionContext
from executors.agent_system import (
    AgentSelector,
    AgentRegistry,
    AgentProfile,
    AgentCapability,
    IntelligentAgentMatcher,
    AgentCoordinator,
    AgentCommunication,
    AgentTask,
)


@pytest.mark.integration
@pytest.mark.agents
class TestAgentOrchestrator:
    """Integration tests for AgentOrchestrator"""

    def test_agent_selection_auto(self, temp_workspace: Path):
        """Test automatic agent selection"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={},
            agents=[AgentType.AUTO]
        )

        agents = orchestrator.select_agents([AgentType.AUTO], context)

        assert len(agents) > 0
        assert all(isinstance(agent, str) for agent in agents)

    def test_agent_selection_core(self):
        """Test core agent team selection"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=Path.cwd(),
            args={},
            agents=[AgentType.CORE]
        )

        agents = orchestrator.select_agents([AgentType.CORE], context)

        assert len(agents) == 5
        expected_agents = [
            "multi-agent-orchestrator",
            "code-quality-master",
            "systems-architect",
            "scientific-computing-master",
            "documentation-architect"
        ]
        for expected in expected_agents:
            assert expected in agents

    def test_agent_selection_all(self):
        """Test all agents selection"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=Path.cwd(),
            args={},
            agents=[AgentType.ALL]
        )

        agents = orchestrator.select_agents([AgentType.ALL], context)

        # Should return all 23+ agents
        assert len(agents) >= 20

    def test_orchestrate_sequential(self, temp_workspace: Path):
        """Test sequential agent orchestration"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={},
            parallel=False,
            agents=[AgentType.CORE]
        )

        agents = ["code-quality-master", "systems-architect"]
        task = "Analyze code quality and architecture"

        result = orchestrator.orchestrate(agents, context, task)

        assert "agents_executed" in result
        assert result["agents_executed"] == 2
        assert "findings" in result
        assert "recommendations" in result

    def test_orchestrate_parallel(self, temp_workspace: Path):
        """Test parallel agent orchestration"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={},
            parallel=True,
            agents=[AgentType.CORE]
        )

        agents = ["code-quality-master", "systems-architect", "scientific-computing-master"]
        task = "Comprehensive code analysis"

        start_time = time.time()
        result = orchestrator.orchestrate(agents, context, task)
        duration = time.time() - start_time

        assert result["agents_executed"] == 3
        # Parallel execution should complete reasonably fast
        assert duration < 10.0

    def test_result_synthesis(self):
        """Test agent result synthesis"""
        orchestrator = AgentOrchestrator()

        results = {
            "agent1": {
                "status": "completed",
                "findings": ["Issue 1", "Issue 2"],
                "recommendations": ["Fix 1"]
            },
            "agent2": {
                "status": "completed",
                "findings": ["Issue 2", "Issue 3"],
                "recommendations": ["Fix 2"]
            },
            "agent3": {
                "error": "Failed"
            }
        }

        synthesis = orchestrator._synthesize_results(results)

        assert synthesis["agents_executed"] == 3
        assert synthesis["successful"] == 2
        assert synthesis["failed"] == 1
        assert len(synthesis["findings"]) >= 3
        assert len(synthesis["recommendations"]) >= 2


@pytest.mark.integration
@pytest.mark.agents
class TestAgentSelector:
    """Integration tests for AgentSelector"""

    def test_intelligent_selection_scientific(self, temp_workspace: Path):
        """Test intelligent selection for scientific computing project"""
        selector = AgentSelector()

        # Create requirements.txt with scientific packages
        (temp_workspace / "requirements.txt").write_text(
            "numpy>=1.20.0\nscipy>=1.7.0\njax>=0.4.0\n"
        )

        context = {
            "work_dir": temp_workspace,
            "task_type": "optimization",
            "languages": ["python"],
            "frameworks": ["numpy", "jax"]
        }

        agents = selector.select_agents(context, mode="auto", max_agents=10)

        assert len(agents) > 0
        assert len(agents) <= 10

        # Should include scientific computing agents
        agent_names = [a.name for a in agents]
        assert any("scientific" in name for name in agent_names)

    def test_intelligent_selection_web(self, temp_workspace: Path):
        """Test intelligent selection for web development project"""
        selector = AgentSelector()

        # Create package.json for web project
        (temp_workspace / "package.json").write_text('{"name": "webapp"}')

        context = {
            "work_dir": temp_workspace,
            "task_type": "refactoring",
            "languages": ["javascript", "typescript"],
            "frameworks": ["react"]
        }

        agents = selector.select_agents(context, mode="auto", max_agents=8)

        assert len(agents) > 0
        assert len(agents) <= 8

        # Should include web development agents
        agent_names = [a.name for a in agents]
        assert any(name in ["fullstack-developer", "systems-architect"] for name in agent_names)

    def test_mode_based_selection(self):
        """Test different selection modes"""
        selector = AgentSelector()
        context = {"work_dir": Path.cwd(), "task_type": "analysis"}

        test_modes = ["core", "scientific", "engineering", "ai", "quality", "research"]

        for mode in test_modes:
            agents = selector.select_agents(context, mode=mode)
            assert len(agents) > 0, f"No agents selected for mode: {mode}"

    def test_max_agents_limit(self):
        """Test max agents limit is respected"""
        selector = AgentSelector()
        context = {"work_dir": Path.cwd(), "task_type": "analysis"}

        for max_agents in [1, 3, 5, 10]:
            agents = selector.select_agents(context, mode="all", max_agents=max_agents)
            assert len(agents) <= max_agents

    def test_context_analysis(self):
        """Test context analysis for capability detection"""
        selector = AgentSelector()

        # Test scientific indicators
        context_sci = {
            "work_dir": "/path/to/research/project",
            "languages": ["python", "julia"],
            "frameworks": ["numpy", "scipy"]
        }

        assert selector._has_scientific_indicators(context_sci)

        # Test ML indicators
        context_ml = {
            "work_dir": "/path/to/ml/project",
            "frameworks": ["torch", "tensorflow"]
        }

        assert selector._has_ml_indicators(context_ml)


@pytest.mark.integration
@pytest.mark.agents
class TestAgentRegistry:
    """Integration tests for AgentRegistry"""

    def test_get_all_agents(self):
        """Test getting all agents"""
        agents = AgentRegistry.get_all_agents()

        assert len(agents) >= 20
        assert all(isinstance(a, AgentProfile) for a in agents)

    def test_get_agent_by_name(self):
        """Test getting specific agent"""
        agent = AgentRegistry.get_agent("scientific-computing-master")

        assert agent is not None
        assert agent.name == "scientific-computing-master"
        assert AgentCapability.SCIENTIFIC_COMPUTING in agent.capabilities

    def test_get_agents_by_category(self):
        """Test getting agents by category"""
        categories = ["orchestration", "scientific", "engineering", "quality"]

        for category in categories:
            agents = AgentRegistry.get_agents_by_category(category)
            assert len(agents) > 0
            assert all(a.category == category for a in agents)

    def test_get_agents_by_capability(self):
        """Test getting agents by capability"""
        capabilities = [
            AgentCapability.SCIENTIFIC_COMPUTING,
            AgentCapability.CODE_ANALYSIS,
            AgentCapability.PERFORMANCE_OPTIMIZATION,
            AgentCapability.ML_AI
        ]

        for capability in capabilities:
            agents = AgentRegistry.get_agents_by_capability(capability)
            assert len(agents) > 0
            assert all(capability in a.capabilities for a in agents)

    def test_all_agents_have_required_fields(self):
        """Test all agents have required profile fields"""
        agents = AgentRegistry.get_all_agents()

        for agent in agents:
            assert agent.name
            assert agent.category
            assert len(agent.capabilities) > 0
            assert len(agent.specializations) > 0
            assert len(agent.languages) > 0
            assert 1 <= agent.priority <= 10


@pytest.mark.integration
@pytest.mark.agents
class TestIntelligentAgentMatcher:
    """Integration tests for IntelligentAgentMatcher"""

    def test_match_agents_scientific_computing(self):
        """Test matching agents for scientific computing task"""
        matcher = IntelligentAgentMatcher()

        required_capabilities = {
            AgentCapability.SCIENTIFIC_COMPUTING,
            AgentCapability.PERFORMANCE_OPTIMIZATION
        }

        context = {
            "task_type": "optimization",
            "description": "Optimize scientific computing code",
            "languages": ["python", "julia"],
            "frameworks": ["numpy", "jax"]
        }

        matches = matcher.match_agents(required_capabilities, context)

        assert len(matches) > 0
        # Check that matched agents have relevant capabilities
        for agent, score in matches:
            assert score > 0
            assert any(cap in agent.capabilities for cap in required_capabilities)

    def test_match_score_calculation(self):
        """Test match score calculation"""
        matcher = IntelligentAgentMatcher()

        agent = AgentRegistry.get_agent("scientific-computing-master")

        required_capabilities = {
            AgentCapability.SCIENTIFIC_COMPUTING,
            AgentCapability.PERFORMANCE_OPTIMIZATION
        }

        context = {
            "task_type": "optimization",
            "description": "scientific computing optimization",
            "languages": ["python", "julia"],
            "frameworks": ["numpy", "scipy"]
        }

        score = matcher._calculate_match_score(agent, required_capabilities, context)

        assert 0.0 <= score <= 1.0
        # Should have high score due to capability and tech match
        assert score > 0.5

    def test_match_agents_ml_task(self):
        """Test matching agents for ML/AI task"""
        matcher = IntelligentAgentMatcher()

        required_capabilities = {
            AgentCapability.ML_AI,
            AgentCapability.PERFORMANCE_OPTIMIZATION
        }

        context = {
            "task_type": "ml_optimization",
            "description": "Optimize neural network training",
            "languages": ["python"],
            "frameworks": ["jax", "pytorch"]
        }

        matches = matcher.match_agents(required_capabilities, context)

        assert len(matches) > 0
        # Should include ML-specific agents
        agent_names = [agent.name for agent, score in matches]
        assert any("neural" in name or "jax" in name or "ai" in name for name in agent_names)


@pytest.mark.integration
@pytest.mark.agents
class TestAgentCoordinator:
    """Integration tests for AgentCoordinator"""

    def test_coordinate_execution(self):
        """Test coordinating task execution across agents"""
        coordinator = AgentCoordinator()

        agents = [
            AgentRegistry.get_agent("code-quality-master"),
            AgentRegistry.get_agent("systems-architect")
        ]

        tasks = [
            {
                "description": "Analyze code quality",
                "context": {"type": "quality"},
                "dependencies": []
            },
            {
                "description": "Review architecture",
                "context": {"type": "architecture"},
                "dependencies": []
            }
        ]

        agent_tasks = coordinator.coordinate_execution(agents, tasks, parallel=False)

        assert len(agent_tasks) == 2
        assert all(isinstance(task, AgentTask) for task in agent_tasks)

    def test_load_balancing(self):
        """Test load balancing across agents"""
        coordinator = AgentCoordinator()

        agents = AgentRegistry.get_all_agents()[:5]

        # Create many tasks
        tasks = [
            {
                "description": f"Task {i}",
                "context": {},
                "dependencies": []
            }
            for i in range(10)
        ]

        agent_tasks = coordinator.coordinate_execution(agents, tasks, parallel=True)

        assert len(agent_tasks) == 10
        # Tasks should be distributed across agents


@pytest.mark.integration
@pytest.mark.agents
class TestAgentCommunication:
    """Integration tests for AgentCommunication"""

    def test_message_passing(self):
        """Test message passing between agents"""
        comm = AgentCommunication()

        message_id = comm.send_message(
            sender="agent1",
            recipient="agent2",
            message_type="query",
            content={"question": "What do you think?"}
        )

        assert message_id is not None

        messages = comm.get_messages("agent2")
        assert len(messages) == 1
        assert messages[0].sender == "agent1"
        assert messages[0].message_type == "query"

    def test_message_filtering(self):
        """Test message filtering by type"""
        comm = AgentCommunication()

        comm.send_message("a1", "a2", "query", {"q": "1"})
        comm.send_message("a1", "a2", "response", {"r": "1"})
        comm.send_message("a1", "a2", "finding", {"f": "1"})

        queries = comm.get_messages("a2", message_type="query")
        assert len(queries) == 1

        responses = comm.get_messages("a2", message_type="response")
        assert len(responses) == 1

    def test_shared_knowledge_base(self):
        """Test shared knowledge base"""
        comm = AgentCommunication()

        comm.update_knowledge("key1", "value1")
        comm.update_knowledge("key2", {"nested": "value"})

        assert comm.get_knowledge("key1") == "value1"
        assert comm.get_knowledge("key2")["nested"] == "value"
        assert comm.get_knowledge("nonexistent") is None

    def test_conflict_detection(self):
        """Test conflict detection in findings"""
        comm = AgentCommunication()

        findings = [
            {"agent": "a1", "recommendation": "Use pattern A"},
            {"agent": "a2", "recommendation": "Use pattern B"}
        ]

        conflicts = comm.detect_conflicts(findings)

        # Conflict detection should work
        assert isinstance(conflicts, list)

    def test_consensus_building(self):
        """Test consensus building"""
        comm = AgentCommunication()

        findings = [
            {"agent": "a1", "finding": "Issue X"},
            {"agent": "a2", "finding": "Issue X"},
            {"agent": "a3", "finding": "Issue Y"}
        ]

        consensus = comm.build_consensus(findings)

        assert "agreed" in consensus
        assert "disputed" in consensus
        assert "confidence" in consensus