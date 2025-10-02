"""Tests for OptimizationAgent - streamlined version."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.optimization_agent import OptimizationAgent
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return OptimizationAgent(config={'tolerance': 1e-6})

# Initialization
def test_initialization_default():
    agent = OptimizationAgent()
    assert agent.metadata.name == "OptimizationAgent"
    assert agent.VERSION == "1.0.0"

def test_metadata(agent):
    metadata = agent.get_metadata()
    assert metadata.name == "OptimizationAgent"
    assert 'numpy' in metadata.dependencies

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 4
    names = [c.name for c in caps]
    assert 'minimize_unconstrained' in names

# Validation
def test_validate_unconstrained_valid(agent):
    data = {
        'problem_type': 'optimization_unconstrained',
        'objective': lambda x: x**2,
        'initial_guess': [0]
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_missing_objective(agent):
    data = {'problem_type': 'optimization_unconstrained', 'initial_guess': [0]}
    val = agent.validate_input(data)
    assert not val.valid

def test_validate_root_finding_valid(agent):
    data = {'problem_type': 'root_finding', 'function': lambda x: x**2 - 4, 'initial_guess': 1}
    val = agent.validate_input(data)
    assert val.valid

# Unconstrained optimization
def test_minimize_quadratic(agent):
    # min (x-2)^2 + (y-3)^2
    result = agent.execute({
        'problem_type': 'optimization_unconstrained',
        'objective': lambda x: (x[0]-2)**2 + (x[1]-3)**2,
        'initial_guess': [0, 0]
    })
    assert result.success
    x = result.data['solution']['x']
    assert np.allclose(x, [2, 3], atol=1e-4)

def test_minimize_rosenbrock(agent):
    # Rosenbrock: min (1-x)^2 + 100(y-x^2)^2
    result = agent.execute({
        'problem_type': 'optimization_unconstrained',
        'objective': lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
        'initial_guess': [0, 0],
        'method': 'BFGS'
    })
    assert result.success
    x = result.data['solution']['x']
    assert np.allclose(x, [1, 1], atol=1e-3)

# Root finding
def test_find_root_simple(agent):
    # Find root of x^2 - 4 = 0 (should be 2)
    result = agent.execute({
        'problem_type': 'root_finding',
        'function': lambda x: x**2 - 4,
        'initial_guess': 1.0,
        'method': 'newton'
    })
    assert result.success
    root = result.data['solution']['root']
    assert np.allclose(root, 2.0, atol=1e-6)

# Global optimization
def test_global_optimization(agent):
    # min x^2 + y^2 over [-5, 5]
    result = agent.execute({
        'problem_type': 'global_optimization',
        'objective': lambda x: x[0]**2 + x[1]**2,
        'bounds': [(-5, 5), (-5, 5)]
    })
    assert result.success
    x = result.data['solution']['x']
    assert np.allclose(x, [0, 0], atol=0.1)

# Caching
def test_caching(agent):
    agent.clear_cache()
    data = {
        'problem_type': 'optimization_unconstrained',
        'objective': lambda x: x[0]**2,
        'initial_guess': [1]
    }
    r1 = agent.execute_with_caching(data)
    r2 = agent.execute_with_caching(data)
    # Either r1 succeeds and r2 is cached, or both are cached (from previous run)
    assert r1.success or r1.status == AgentStatus.CACHED
    assert r2.status == AgentStatus.CACHED

# Provenance
def test_provenance(agent):
    result = agent.execute({
        'problem_type': 'optimization_unconstrained',
        'objective': lambda x: x[0]**2,
        'initial_guess': [1]
    })
    assert result.provenance is not None
    assert result.provenance.agent_name == "OptimizationAgent"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
