"""Tests for IntegrationAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.integration_agent import IntegrationAgent
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return IntegrationAgent(config={'tolerance': 1e-6})

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "IntegrationAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 2

# Validation
def test_validate_1d_valid(agent):
    data = {
        'problem_type': 'integrate_1d',
        'function': lambda x: x**2,
        'bounds': [0, 1]
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_missing_function(agent):
    data = {'problem_type': 'integrate_1d', 'bounds': [0, 1]}
    val = agent.validate_input(data)
    assert not val.valid

# 1D integration
def test_integrate_x_squared(agent):
    # ∫₀¹ x² dx = 1/3
    result = agent.execute({
        'problem_type': 'integrate_1d',
        'function': lambda x: x**2,
        'bounds': [0, 1]
    })
    assert result.success
    value = result.data['solution']['value']
    assert np.allclose(value, 1/3, atol=1e-6)

def test_integrate_sin(agent):
    # ∫₀ᵖⁱ sin(x) dx = 2
    result = agent.execute({
        'problem_type': 'integrate_1d',
        'function': np.sin,
        'bounds': [0, np.pi]
    })
    assert result.success
    value = result.data['solution']['value']
    assert np.allclose(value, 2.0, atol=1e-6)

# 2D integration
def test_integrate_2d(agent):
    # ∫₀¹∫₀¹ x*y dx dy = 1/4
    result = agent.execute({
        'problem_type': 'integrate_2d',
        'function': lambda y, x: x * y,  # Note: dblquad order
        'bounds': [[0, 1], [0, 1]]
    })
    assert result.success
    value = result.data['solution']['value']
    assert np.allclose(value, 0.25, atol=1e-6)

# Monte Carlo
def test_monte_carlo(agent):
    # Simple 2D integral
    result = agent.execute({
        'problem_type': 'monte_carlo',
        'function': lambda x: x[0]**2 + x[1]**2,
        'bounds': [[0, 1], [0, 1]],
        'n_samples': 10000
    })
    assert result.success
    # Exact: ∫₀¹∫₀¹ (x²+y²) dx dy = 2/3
    value = result.data['solution']['value']
    assert np.allclose(value, 2/3, atol=0.1)  # MC has larger error

# Provenance
def test_provenance(agent):
    result = agent.execute({
        'problem_type': 'integrate_1d',
        'function': lambda x: x,
        'bounds': [0, 1]
    })
    assert result.provenance is not None
    assert result.provenance.agent_name == "IntegrationAgent"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
