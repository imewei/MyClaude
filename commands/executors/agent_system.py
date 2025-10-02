#!/usr/bin/env python3
"""
Advanced Agent System for Command Execution
===========================================

Multi-agent system with intelligent selection, coordination, and communication
for the 14-command Claude Code system.

Components:
- AgentSelector: Intelligent agent selection based on context
- IntelligentAgentMatcher: ML-inspired agent matching algorithm
- AgentCoordinator: Advanced coordination and load balancing
- AgentCommunication: Inter-agent message passing and shared knowledge

Author: Claude Code Framework
Version: 2.0
Last Updated: 2025-09-29
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib


# ============================================================================
# Agent Definitions
# ============================================================================

class AgentCapability(Enum):
    """Agent capabilities for matching"""
    CODE_ANALYSIS = "code_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SCIENTIFIC_COMPUTING = "scientific_computing"
    ML_AI = "ml_ai"
    ARCHITECTURE_DESIGN = "architecture_design"
    QUALITY_ASSURANCE = "quality_assurance"
    DOCUMENTATION = "documentation"
    DEVOPS = "devops"
    SECURITY = "security"
    TESTING = "testing"
    REFACTORING = "refactoring"
    RESEARCH = "research"
    DATA_SCIENCE = "data_science"
    VISUALIZATION = "visualization"
    DATABASE = "database"
    QUANTUM = "quantum"
    PARALLEL_COMPUTING = "parallel_computing"


@dataclass
class AgentProfile:
    """Complete agent profile with capabilities and metadata"""
    name: str
    category: str
    capabilities: List[AgentCapability]
    specializations: List[str]
    languages: List[str]
    frameworks: List[str]
    max_load: int = 10
    priority: int = 5  # 1-10, higher is higher priority
    description: str = ""


@dataclass
class AgentTask:
    """Task assigned to an agent"""
    task_id: str
    agent_name: str
    description: str
    context: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Message for inter-agent communication"""
    sender: str
    recipient: str
    message_type: str  # query, response, finding, recommendation, conflict
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: hashlib.md5(
        str(datetime.now()).encode()).hexdigest())


# ============================================================================
# Agent Registry
# ============================================================================

class AgentRegistry:
    """
    Central registry of all 23 agents with their profiles.

    23-Agent Personal Agent System:
    - Multi-Agent Orchestration (2 agents)
    - Scientific Computing & Research (8 agents)
    - Engineering & Architecture (4 agents)
    - Quality & Documentation (2 agents)
    - Domain Specialists (4 agents)
    - Scientific Domain Experts (3 agents)
    """

    # Complete agent profiles
    AGENTS = {
        # Multi-Agent Orchestration
        "multi-agent-orchestrator": AgentProfile(
            name="multi-agent-orchestrator",
            category="orchestration",
            capabilities=[
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.ARCHITECTURE_DESIGN,
            ],
            specializations=["workflow coordination", "agent management"],
            languages=["python", "javascript", "julia"],
            frameworks=["multi-agent systems"],
            priority=10,
            description="Coordinates multi-agent workflows and optimizes collaboration"
        ),
        "command-systems-engineer": AgentProfile(
            name="command-systems-engineer",
            category="orchestration",
            capabilities=[
                AgentCapability.ARCHITECTURE_DESIGN,
                AgentCapability.CODE_ANALYSIS,
            ],
            specializations=["command optimization", "system integration"],
            languages=["python", "bash"],
            frameworks=["CLI systems"],
            priority=8,
            description="Optimizes command system architecture and integration"
        ),

        # Scientific Computing & Research
        "scientific-computing-master": AgentProfile(
            name="scientific-computing-master",
            category="scientific",
            capabilities=[
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.PERFORMANCE_OPTIMIZATION,
                AgentCapability.PARALLEL_COMPUTING,
            ],
            specializations=["numerical computing", "HPC", "scientific algorithms"],
            languages=["python", "julia", "fortran", "c", "c++"],
            frameworks=["numpy", "scipy", "jax", "mpi", "openmp"],
            priority=10,
            description="Expert in scientific computing and HPC optimization"
        ),
        "research-intelligence-master": AgentProfile(
            name="research-intelligence-master",
            category="scientific",
            capabilities=[
                AgentCapability.RESEARCH,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.DATA_SCIENCE,
            ],
            specializations=["research methodology", "data analysis", "experimentation"],
            languages=["python", "julia", "r"],
            frameworks=["jupyter", "pandas", "matplotlib"],
            priority=9,
            description="Research analysis and experimental design expert"
        ),
        "jax-pro": AgentProfile(
            name="jax-pro",
            category="scientific",
            capabilities=[
                AgentCapability.ML_AI,
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.PERFORMANCE_OPTIMIZATION,
            ],
            specializations=["JAX", "automatic differentiation", "GPU optimization"],
            languages=["python"],
            frameworks=["jax", "flax", "optax"],
            priority=9,
            description="JAX and GPU-accelerated computing specialist"
        ),
        "neural-networks-master": AgentProfile(
            name="neural-networks-master",
            category="scientific",
            capabilities=[
                AgentCapability.ML_AI,
                AgentCapability.PERFORMANCE_OPTIMIZATION,
            ],
            specializations=["deep learning", "neural architectures", "training optimization"],
            languages=["python"],
            frameworks=["pytorch", "tensorflow", "jax"],
            priority=9,
            description="Deep learning and neural network optimization expert"
        ),
        "advanced-quantum-computing-expert": AgentProfile(
            name="advanced-quantum-computing-expert",
            category="scientific",
            capabilities=[
                AgentCapability.QUANTUM,
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.RESEARCH,
            ],
            specializations=["quantum algorithms", "quantum simulation", "quantum ML"],
            languages=["python", "julia"],
            frameworks=["qiskit", "cirq", "pennylane"],
            priority=8,
            description="Quantum computing algorithms and simulation specialist"
        ),
        "correlation-function-expert": AgentProfile(
            name="correlation-function-expert",
            category="scientific",
            capabilities=[
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.DATA_SCIENCE,
            ],
            specializations=["correlation functions", "statistical mechanics", "scattering"],
            languages=["python", "julia", "fortran"],
            frameworks=["numpy", "scipy"],
            priority=7,
            description="Correlation function and scattering analysis expert"
        ),
        "neutron-soft-matter-expert": AgentProfile(
            name="neutron-soft-matter-expert",
            category="scientific",
            capabilities=[
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.RESEARCH,
            ],
            specializations=["neutron scattering", "soft matter", "polymer physics"],
            languages=["python", "fortran"],
            frameworks=["sasview", "mantid"],
            priority=7,
            description="Neutron scattering and soft matter physics specialist"
        ),
        "nonequilibrium-stochastic-expert": AgentProfile(
            name="nonequilibrium-stochastic-expert",
            category="scientific",
            capabilities=[
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.RESEARCH,
            ],
            specializations=["stochastic processes", "nonequilibrium physics", "simulations"],
            languages=["python", "julia", "c++"],
            frameworks=["numpy", "scipy", "gillespie"],
            priority=7,
            description="Nonequilibrium and stochastic systems expert"
        ),

        # Engineering & Architecture
        "systems-architect": AgentProfile(
            name="systems-architect",
            category="engineering",
            capabilities=[
                AgentCapability.ARCHITECTURE_DESIGN,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.PERFORMANCE_OPTIMIZATION,
            ],
            specializations=["system design", "scalability", "patterns"],
            languages=["python", "javascript", "java", "go"],
            frameworks=["microservices", "cloud architecture"],
            priority=9,
            description="System architecture and design patterns expert"
        ),
        "ai-systems-architect": AgentProfile(
            name="ai-systems-architect",
            category="engineering",
            capabilities=[
                AgentCapability.ML_AI,
                AgentCapability.ARCHITECTURE_DESIGN,
                AgentCapability.PERFORMANCE_OPTIMIZATION,
            ],
            specializations=["ML systems", "MLOps", "AI infrastructure"],
            languages=["python"],
            frameworks=["mlflow", "kubeflow", "ray"],
            priority=9,
            description="AI/ML system architecture and MLOps specialist"
        ),
        "fullstack-developer": AgentProfile(
            name="fullstack-developer",
            category="engineering",
            capabilities=[
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.ARCHITECTURE_DESIGN,
            ],
            specializations=["web development", "APIs", "databases"],
            languages=["python", "javascript", "typescript"],
            frameworks=["react", "node", "django", "flask"],
            priority=8,
            description="Full-stack web development and API design expert"
        ),
        "devops-security-engineer": AgentProfile(
            name="devops-security-engineer",
            category="engineering",
            capabilities=[
                AgentCapability.DEVOPS,
                AgentCapability.SECURITY,
                AgentCapability.TESTING,
            ],
            specializations=["CI/CD", "security", "infrastructure"],
            languages=["python", "bash", "yaml"],
            frameworks=["docker", "kubernetes", "terraform"],
            priority=9,
            description="DevOps, security, and infrastructure automation expert"
        ),

        # Quality & Documentation
        "code-quality-master": AgentProfile(
            name="code-quality-master",
            category="quality",
            capabilities=[
                AgentCapability.QUALITY_ASSURANCE,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.REFACTORING,
            ],
            specializations=["code quality", "best practices", "static analysis"],
            languages=["python", "javascript", "java"],
            frameworks=["pylint", "black", "mypy"],
            priority=10,
            description="Code quality, standards, and best practices expert"
        ),
        "documentation-architect": AgentProfile(
            name="documentation-architect",
            category="quality",
            capabilities=[
                AgentCapability.DOCUMENTATION,
                AgentCapability.CODE_ANALYSIS,
            ],
            specializations=["technical writing", "API docs", "architecture docs"],
            languages=["markdown", "rst", "asciidoc"],
            frameworks=["sphinx", "mkdocs", "docusaurus"],
            priority=8,
            description="Technical documentation and knowledge architecture expert"
        ),

        # Domain Specialists
        "data-professional": AgentProfile(
            name="data-professional",
            category="domain",
            capabilities=[
                AgentCapability.DATA_SCIENCE,
                AgentCapability.CODE_ANALYSIS,
            ],
            specializations=["data engineering", "ETL", "data pipelines"],
            languages=["python", "sql"],
            frameworks=["pandas", "spark", "airflow"],
            priority=8,
            description="Data engineering and pipeline architecture specialist"
        ),
        "visualization-interface-master": AgentProfile(
            name="visualization-interface-master",
            category="domain",
            capabilities=[
                AgentCapability.VISUALIZATION,
                AgentCapability.CODE_ANALYSIS,
            ],
            specializations=["data visualization", "UI/UX", "dashboards"],
            languages=["python", "javascript"],
            frameworks=["matplotlib", "plotly", "d3", "react"],
            priority=7,
            description="Data visualization and interface design expert"
        ),
        "database-workflow-engineer": AgentProfile(
            name="database-workflow-engineer",
            category="domain",
            capabilities=[
                AgentCapability.DATABASE,
                AgentCapability.ARCHITECTURE_DESIGN,
            ],
            specializations=["database design", "query optimization", "workflows"],
            languages=["sql", "python"],
            frameworks=["postgresql", "mongodb", "redis"],
            priority=8,
            description="Database architecture and optimization specialist"
        ),
        "scientific-code-adoptor": AgentProfile(
            name="scientific-code-adoptor",
            category="domain",
            capabilities=[
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.REFACTORING,
            ],
            specializations=["legacy code", "modernization", "migration"],
            languages=["python", "fortran", "c", "julia"],
            frameworks=["numpy", "scipy", "jax"],
            priority=8,
            description="Scientific code modernization and migration expert"
        ),

        # Scientific Domain Experts
        "xray-soft-matter-expert": AgentProfile(
            name="xray-soft-matter-expert",
            category="scientific_domain",
            capabilities=[
                AgentCapability.SCIENTIFIC_COMPUTING,
                AgentCapability.RESEARCH,
            ],
            specializations=["x-ray scattering", "SAXS/WAXS", "soft matter"],
            languages=["python", "c"],
            frameworks=["pyFAI", "fabio"],
            priority=7,
            description="X-ray scattering and soft matter characterization expert"
        ),
    }

    @classmethod
    def get_agent(cls, name: str) -> Optional[AgentProfile]:
        """Get agent profile by name"""
        return cls.AGENTS.get(name)

    @classmethod
    def get_agents_by_category(cls, category: str) -> List[AgentProfile]:
        """Get all agents in a category"""
        return [agent for agent in cls.AGENTS.values()
                if agent.category == category]

    @classmethod
    def get_agents_by_capability(
        cls,
        capability: AgentCapability
    ) -> List[AgentProfile]:
        """Get all agents with a specific capability"""
        return [agent for agent in cls.AGENTS.values()
                if capability in agent.capabilities]

    @classmethod
    def get_all_agents(cls) -> List[AgentProfile]:
        """Get all agent profiles"""
        return list(cls.AGENTS.values())


# ============================================================================
# Agent Selector
# ============================================================================

class AgentSelector:
    """
    Intelligent agent selection based on context and requirements.

    Features:
    - Context-aware selection
    - Capability matching
    - Load balancing
    - Priority-based selection
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = AgentRegistry()

    def select_agents(
        self,
        context: Dict[str, Any],
        mode: str = "auto",
        max_agents: Optional[int] = None
    ) -> List[AgentProfile]:
        """
        Select optimal agents for task.

        Args:
            context: Task context with codebase info
            mode: Selection mode (auto, core, scientific, etc.)
            max_agents: Maximum number of agents to select

        Returns:
            List of selected agent profiles
        """
        if mode == "all":
            return self.registry.get_all_agents()

        if mode == "auto":
            return self._intelligent_selection(context, max_agents)

        # Mode-specific selection
        return self._mode_based_selection(mode, context, max_agents)

    def _intelligent_selection(
        self,
        context: Dict[str, Any],
        max_agents: Optional[int] = None
    ) -> List[AgentProfile]:
        """
        Intelligent agent selection based on context analysis.

        Args:
            context: Task context
            max_agents: Maximum agents to select

        Returns:
            Selected agents
        """
        self.logger.info("Performing intelligent agent selection")

        # Analyze context to determine required capabilities
        required_capabilities = self._analyze_context(context)

        # Match agents to capabilities
        matcher = IntelligentAgentMatcher()
        matched_agents = matcher.match_agents(required_capabilities, context)

        # Sort by priority and relevance
        sorted_agents = sorted(
            matched_agents,
            key=lambda x: (x[1], x[0].priority),
            reverse=True
        )

        # Select top agents
        selected = [agent for agent, score in sorted_agents]

        if max_agents:
            selected = selected[:max_agents]

        self.logger.info(f"Selected {len(selected)} agents: {[a.name for a in selected]}")

        return selected

    def _mode_based_selection(
        self,
        mode: str,
        context: Dict[str, Any],
        max_agents: Optional[int] = None
    ) -> List[AgentProfile]:
        """Select agents based on predefined mode"""
        mode_mapping = {
            "core": self._select_core_agents,
            "scientific": self._select_scientific_agents,
            "engineering": self._select_engineering_agents,
            "ai": self._select_ai_agents,
            "quality": self._select_quality_agents,
            "research": self._select_research_agents,
            "domain": self._select_domain_agents,
        }

        selector = mode_mapping.get(mode, self._select_core_agents)
        agents = selector(context)

        if max_agents:
            agents = agents[:max_agents]

        return agents

    def _analyze_context(self, context: Dict[str, Any]) -> Set[AgentCapability]:
        """Analyze context to determine required capabilities"""
        capabilities = set()

        # Default capabilities
        capabilities.add(AgentCapability.CODE_ANALYSIS)

        # Detect scientific computing
        if self._has_scientific_indicators(context):
            capabilities.add(AgentCapability.SCIENTIFIC_COMPUTING)
            capabilities.add(AgentCapability.PERFORMANCE_OPTIMIZATION)

        # Detect ML/AI
        if self._has_ml_indicators(context):
            capabilities.add(AgentCapability.ML_AI)

        # Detect testing needs
        if context.get("task_type") in ["testing", "quality"]:
            capabilities.add(AgentCapability.TESTING)
            capabilities.add(AgentCapability.QUALITY_ASSURANCE)

        # Detect documentation needs
        if context.get("task_type") == "documentation":
            capabilities.add(AgentCapability.DOCUMENTATION)

        # Detect optimization needs
        if context.get("task_type") == "optimization":
            capabilities.add(AgentCapability.PERFORMANCE_OPTIMIZATION)

        return capabilities

    def _has_scientific_indicators(self, context: Dict[str, Any]) -> bool:
        """Check for scientific computing indicators"""
        indicators = [
            "numpy", "scipy", "jax", "fortran", "mpi",
            "scientific", "research", "simulation"
        ]

        work_dir = context.get("work_dir", "")
        languages = context.get("languages", [])
        frameworks = context.get("frameworks", [])

        return any(
            indicator in str(work_dir).lower() or
            indicator in str(languages).lower() or
            indicator in str(frameworks).lower()
            for indicator in indicators
        )

    def _has_ml_indicators(self, context: Dict[str, Any]) -> bool:
        """Check for ML/AI indicators"""
        indicators = [
            "torch", "tensorflow", "jax", "ml", "neural",
            "model", "training", "deep learning"
        ]

        work_dir = context.get("work_dir", "")
        frameworks = context.get("frameworks", [])

        return any(
            indicator in str(work_dir).lower() or
            indicator in str(frameworks).lower()
            for indicator in indicators
        )

    # Mode-specific selectors
    def _select_core_agents(self, context: Dict[str, Any]) -> List[AgentProfile]:
        """Select core 5-agent team"""
        core_names = [
            "multi-agent-orchestrator",
            "code-quality-master",
            "systems-architect",
            "scientific-computing-master",
            "documentation-architect"
        ]
        return [self.registry.get_agent(name) for name in core_names
                if self.registry.get_agent(name)]

    def _select_scientific_agents(self, context: Dict[str, Any]) -> List[AgentProfile]:
        """Select scientific computing team"""
        return self.registry.get_agents_by_category("scientific")

    def _select_engineering_agents(self, context: Dict[str, Any]) -> List[AgentProfile]:
        """Select engineering team"""
        return self.registry.get_agents_by_category("engineering")

    def _select_ai_agents(self, context: Dict[str, Any]) -> List[AgentProfile]:
        """Select AI/ML team"""
        ai_names = [
            "ai-systems-architect",
            "neural-networks-master",
            "jax-pro",
            "scientific-computing-master",
            "research-intelligence-master"
        ]
        return [self.registry.get_agent(name) for name in ai_names
                if self.registry.get_agent(name)]

    def _select_quality_agents(self, context: Dict[str, Any]) -> List[AgentProfile]:
        """Select quality team"""
        return self.registry.get_agents_by_category("quality")

    def _select_research_agents(self, context: Dict[str, Any]) -> List[AgentProfile]:
        """Select research team"""
        return [agent for agent in self.registry.get_all_agents()
                if AgentCapability.RESEARCH in agent.capabilities]

    def _select_domain_agents(self, context: Dict[str, Any]) -> List[AgentProfile]:
        """Select domain specialists"""
        return self.registry.get_agents_by_category("domain")


# ============================================================================
# Intelligent Agent Matcher
# ============================================================================

class IntelligentAgentMatcher:
    """
    ML-inspired agent matching algorithm.

    Uses weighted scoring to match agents to requirements:
    - Capability match (40%)
    - Specialization match (30%)
    - Language/framework match (20%)
    - Priority (10%)
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def match_agents(
        self,
        required_capabilities: Set[AgentCapability],
        context: Dict[str, Any]
    ) -> List[Tuple[AgentProfile, float]]:
        """
        Match agents to requirements with scoring.

        Args:
            required_capabilities: Required capabilities
            context: Task context

        Returns:
            List of (agent, score) tuples
        """
        all_agents = AgentRegistry.get_all_agents()
        scored_agents = []

        for agent in all_agents:
            score = self._calculate_match_score(
                agent,
                required_capabilities,
                context
            )
            if score > 0:
                scored_agents.append((agent, score))

        return scored_agents

    def _calculate_match_score(
        self,
        agent: AgentProfile,
        required_capabilities: Set[AgentCapability],
        context: Dict[str, Any]
    ) -> float:
        """Calculate match score for an agent"""
        score = 0.0

        # Capability match (40%)
        if required_capabilities:
            capability_match = len(
                set(agent.capabilities) & required_capabilities
            ) / len(required_capabilities)
            score += capability_match * 0.4

        # Specialization match (30%)
        specialization_score = self._match_specializations(
            agent.specializations,
            context
        )
        score += specialization_score * 0.3

        # Language/framework match (20%)
        tech_score = self._match_technologies(agent, context)
        score += tech_score * 0.2

        # Priority (10%)
        priority_score = agent.priority / 10.0
        score += priority_score * 0.1

        return score

    def _match_specializations(
        self,
        specializations: List[str],
        context: Dict[str, Any]
    ) -> float:
        """Match agent specializations to context"""
        if not specializations:
            return 0.0

        task_type = context.get("task_type", "")
        description = context.get("description", "")

        matches = sum(
            1 for spec in specializations
            if spec.lower() in task_type.lower() or
            spec.lower() in description.lower()
        )

        return matches / len(specializations)

    def _match_technologies(
        self,
        agent: AgentProfile,
        context: Dict[str, Any]
    ) -> float:
        """Match agent languages/frameworks to context"""
        context_languages = set(context.get("languages", []))
        context_frameworks = set(context.get("frameworks", []))

        agent_languages = set(agent.languages)
        agent_frameworks = set(agent.frameworks)

        language_match = len(agent_languages & context_languages)
        framework_match = len(agent_frameworks & context_frameworks)

        total_match = language_match + framework_match
        total_possible = len(context_languages) + len(context_frameworks)

        if total_possible == 0:
            return 0.5  # Neutral score if no tech specified

        return total_match / total_possible


# ============================================================================
# Agent Coordinator
# ============================================================================

class AgentCoordinator:
    """
    Advanced agent coordination with load balancing and dependency management.

    Features:
    - Task dependency resolution
    - Load balancing across agents
    - Parallel execution planning
    - Conflict detection and resolution
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.active_tasks: Dict[str, AgentTask] = {}
        self.agent_loads: Dict[str, int] = {}

    def coordinate_execution(
        self,
        agents: List[AgentProfile],
        tasks: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[AgentTask]:
        """
        Coordinate task execution across agents.

        Args:
            agents: Available agents
            tasks: Tasks to execute
            parallel: Enable parallel execution

        Returns:
            List of agent tasks with assignments
        """
        self.logger.info(f"Coordinating {len(tasks)} tasks across {len(agents)} agents")

        # Create agent tasks
        agent_tasks = self._create_agent_tasks(tasks, agents)

        # Resolve dependencies
        execution_plan = self._resolve_dependencies(agent_tasks)

        # Load balance
        balanced_plan = self._load_balance(execution_plan, agents, parallel)

        return balanced_plan

    def _create_agent_tasks(
        self,
        tasks: List[Dict[str, Any]],
        agents: List[AgentProfile]
    ) -> List[AgentTask]:
        """Create agent tasks from task descriptions"""
        agent_tasks = []

        for i, task in enumerate(tasks):
            # Select best agent for task
            agent = self._select_agent_for_task(task, agents)

            agent_task = AgentTask(
                task_id=f"task_{i}",
                agent_name=agent.name,
                description=task.get("description", ""),
                context=task.get("context", {}),
                dependencies=task.get("dependencies", [])
            )

            agent_tasks.append(agent_task)

        return agent_tasks

    def _select_agent_for_task(
        self,
        task: Dict[str, Any],
        agents: List[AgentProfile]
    ) -> AgentProfile:
        """Select best agent for a specific task"""
        # Simple selection - choose least loaded agent with required capabilities
        # In real implementation, would use sophisticated matching

        if not agents:
            raise ValueError("No agents available")

        # For now, select first agent
        # TODO: Implement sophisticated matching
        return agents[0]

    def _resolve_dependencies(
        self,
        tasks: List[AgentTask]
    ) -> List[AgentTask]:
        """Resolve task dependencies for execution order"""
        # Topological sort of tasks based on dependencies
        # For framework, return as-is
        # TODO: Implement proper topological sort
        return tasks

    def _load_balance(
        self,
        tasks: List[AgentTask],
        agents: List[AgentProfile],
        parallel: bool
    ) -> List[AgentTask]:
        """Load balance tasks across agents"""
        if not parallel:
            return tasks

        # Redistribute tasks to balance load
        # TODO: Implement load balancing algorithm
        return tasks


# ============================================================================
# Agent Communication
# ============================================================================

class AgentCommunication:
    """
    Inter-agent message passing and shared knowledge system.

    Features:
    - Message passing between agents
    - Shared knowledge base
    - Conflict detection
    - Consensus building
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.messages: List[AgentMessage] = []
        self.knowledge_base: Dict[str, Any] = {}

    def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        content: Dict[str, Any]
    ) -> str:
        """
        Send message from one agent to another.

        Args:
            sender: Sender agent name
            recipient: Recipient agent name
            message_type: Type of message
            content: Message content

        Returns:
            Message ID
        """
        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content
        )

        self.messages.append(message)
        self.logger.debug(f"Message sent: {sender} -> {recipient} ({message_type})")

        return message.message_id

    def get_messages(
        self,
        recipient: str,
        message_type: Optional[str] = None
    ) -> List[AgentMessage]:
        """Get messages for an agent"""
        messages = [m for m in self.messages if m.recipient == recipient]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        return messages

    def update_knowledge(self, key: str, value: Any):
        """Update shared knowledge base"""
        self.knowledge_base[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }

    def get_knowledge(self, key: str) -> Optional[Any]:
        """Get value from shared knowledge base"""
        entry = self.knowledge_base.get(key)
        return entry["value"] if entry else None

    def detect_conflicts(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect conflicts in agent findings"""
        conflicts = []

        # Simple conflict detection - look for contradictory recommendations
        # TODO: Implement sophisticated conflict detection

        return conflicts

    def build_consensus(
        self,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build consensus from multiple agent findings"""
        consensus = {
            "agreed": [],
            "disputed": [],
            "confidence": 0.0
        }

        # Simple consensus - find common findings
        # TODO: Implement voting/confidence-based consensus

        return consensus


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Agent system demonstration"""
    print("Advanced Agent System")
    print("====================\n")

    # Demonstrate agent selection
    selector = AgentSelector()

    context = {
        "task_type": "optimization",
        "work_dir": "/path/to/scientific/project",
        "languages": ["python", "julia"],
        "frameworks": ["jax", "numpy"],
        "description": "Optimize scientific computing code"
    }

    print("Intelligent Agent Selection:")
    agents = selector.select_agents(context, mode="auto", max_agents=5)

    for i, agent in enumerate(agents, 1):
        print(f"\n{i}. {agent.name}")
        print(f"   Category: {agent.category}")
        print(f"   Capabilities: {[c.value for c in agent.capabilities]}")
        print(f"   Priority: {agent.priority}")

    print(f"\n\nTotal agents in registry: {len(AgentRegistry.get_all_agents())}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())