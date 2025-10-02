"""Advanced Applications for Optimal Control.

This module provides advanced optimal control applications:
1. Multi-objective optimization (Pareto fronts, scalarization methods)
2. Robust control under uncertainty (worst-case, distributional)
3. Stochastic optimal control (chance constraints, risk-aware)
4. Real-world case studies and applications

Author: Nonequilibrium Physics Agents
"""

from .multi_objective import (
    MultiObjectiveOptimizer,
    ParetoFront,
    WeightedSumMethod,
    EpsilonConstraintMethod,
    NormalBoundaryIntersection,
    NSGA2Optimizer,
)

from .robust_control import (
    RobustOptimizer,
    UncertaintySet,
    RobustMPC,
    MinMaxOptimizer,
    DistributionallyRobust,
    TubeBasedMPC,
)

from .stochastic_control import (
    StochasticOptimizer,
    ChanceConstrainedOptimizer,
    RiskAwareOptimizer,
    CVaROptimizer,
    StochasticMPC,
    ScenarioTreeOptimizer,
)

from .case_studies import (
    CartPoleStabilization,
    QuadrotorTrajectory,
    RobotArmControl,
    EnergySystemOptimization,
    PortfolioOptimization,
    ChemicalReactorControl,
)

__all__ = [
    # Multi-objective
    'MultiObjectiveOptimizer',
    'ParetoFront',
    'WeightedSumMethod',
    'EpsilonConstraintMethod',
    'NormalBoundaryIntersection',
    'NSGA2Optimizer',
    # Robust control
    'RobustOptimizer',
    'UncertaintySet',
    'RobustMPC',
    'MinMaxOptimizer',
    'DistributionallyRobust',
    'TubeBasedMPC',
    # Stochastic control
    'StochasticOptimizer',
    'ChanceConstrainedOptimizer',
    'RiskAwareOptimizer',
    'CVaROptimizer',
    'StochasticMPC',
    'ScenarioTreeOptimizer',
    # Case studies
    'CartPoleStabilization',
    'QuadrotorTrajectory',
    'RobotArmControl',
    'EnergySystemOptimization',
    'PortfolioOptimization',
    'ChemicalReactorControl',
]
