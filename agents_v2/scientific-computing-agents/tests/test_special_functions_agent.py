"""Tests for SpecialFunctionsAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.special_functions_agent import SpecialFunctionsAgent
from base_agent import AgentStatus

@pytest.fixture
def agent():
    return SpecialFunctionsAgent()

# Initialization
def test_initialization(agent):
    assert agent.metadata.name == "SpecialFunctionsAgent"
    assert agent.VERSION == "1.0.0"

def test_capabilities(agent):
    caps = agent.get_capabilities()
    assert len(caps) == 3

# Validation
def test_validate_special_function_valid(agent):
    data = {
        'problem_type': 'special_function',
        'function_type': 'bessel_j0',
        'x': [1.0]
    }
    val = agent.validate_input(data)
    assert val.valid

def test_validate_missing_function_type(agent):
    data = {'problem_type': 'special_function', 'x': [1.0]}
    val = agent.validate_input(data)
    assert not val.valid

# Special functions
def test_bessel_j0(agent):
    result = agent.execute({
        'problem_type': 'special_function',
        'function_type': 'bessel_j0',
        'x': [0.0]
    })
    assert result.success
    values = result.data['solution']['values']
    assert np.allclose(values, [1.0], atol=1e-10)

def test_erf(agent):
    result = agent.execute({
        'problem_type': 'special_function',
        'function_type': 'erf',
        'x': [0.0]
    })
    assert result.success
    values = result.data['solution']['values']
    assert np.allclose(values, [0.0], atol=1e-10)

def test_gamma(agent):
    # Gamma(5) = 4! = 24
    result = agent.execute({
        'problem_type': 'special_function',
        'function_type': 'gamma',
        'x': [5.0]
    })
    assert result.success
    values = result.data['solution']['values']
    assert np.allclose(values, [24.0], atol=1e-6)

# Transforms
def test_fft(agent):
    data = np.array([1.0, 0.0, 0.0, 0.0])
    result = agent.execute({
        'problem_type': 'transform',
        'transform_type': 'fft',
        'data': data
    })
    assert result.success
    transformed = result.data['solution']['transformed']
    assert len(transformed) == 4

def test_dct(agent):
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = agent.execute({
        'problem_type': 'transform',
        'transform_type': 'dct',
        'data': data
    })
    assert result.success
    transformed = result.data['solution']['transformed']
    assert len(transformed) == 4

# Orthogonal polynomials
def test_legendre(agent):
    result = agent.execute({
        'problem_type': 'orthogonal_polynomial',
        'polynomial_type': 'legendre',
        'degree': 2,
        'x': [0.0]
    })
    assert result.success
    values = result.data['solution']['values']
    # P_2(0) = -1/2
    assert np.allclose(values, [-0.5], atol=1e-10)

def test_chebyshev(agent):
    result = agent.execute({
        'problem_type': 'orthogonal_polynomial',
        'polynomial_type': 'chebyshev',
        'degree': 1,
        'x': [1.0]
    })
    assert result.success
    values = result.data['solution']['values']
    # T_1(1) = 1
    assert np.allclose(values, [1.0], atol=1e-10)

# Provenance
def test_provenance(agent):
    result = agent.execute({
        'problem_type': 'special_function',
        'function_type': 'erf',
        'x': [0.0]
    })
    assert result.provenance is not None
    assert result.provenance.agent_name == "SpecialFunctionsAgent"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
