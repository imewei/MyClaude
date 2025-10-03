"""Tests for 2D/3D PDE solvers in ODEPDESolverAgent."""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent
from base_agent import AgentStatus


@pytest.fixture
def agent():
    return ODEPDESolverAgent()


# ============================================================================
# 2D Heat Equation Tests
# ============================================================================

def test_2d_heat_basic(agent):
    """Test 2D heat equation with simple initial condition."""
    def initial_condition(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    result = agent.solve_pde_2d({
        'pde_type': 'heat',
        'domain': [[0, 1], [0, 1]],
        'nx': 20,
        'ny': 20,
        'alpha': 0.01,
        'initial_condition': initial_condition,
        't_span': (0, 0.1),
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success
    assert result.status == AgentStatus.SUCCESS
    assert 'solution' in result.data

    sol = result.data['solution']
    assert 'u' in sol
    assert 'x' in sol
    assert 'y' in sol
    assert 't' in sol
    assert sol['pde_type'] == 'heat'

    # Check solution shape
    assert sol['u'].shape == (20, 20)

    # Solution should decay (max should be less than initial max)
    initial_max = np.max(initial_condition(
        *np.meshgrid(sol['x'], sol['y'], indexing='ij')
    ))
    assert np.max(sol['u']) < initial_max


def test_2d_heat_analytical_comparison(agent):
    """Test 2D heat equation against analytical solution."""
    # For u_t = α∇²u with u(x,y,0) = sin(πx)sin(πy), BC u=0
    # Analytical: u(x,y,t) = exp(-2απ²t)sin(πx)sin(πy)

    alpha = 0.01
    t_final = 0.5

    def initial_condition(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    result = agent.solve_pde_2d({
        'pde_type': 'heat',
        'domain': [[0, 1], [0, 1]],
        'nx': 30,
        'ny': 30,
        'alpha': alpha,
        'initial_condition': initial_condition,
        't_span': (0, t_final),
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success

    sol = result.data['solution']
    X, Y = sol['X'], sol['Y']
    u_numerical = sol['u']

    # Analytical solution
    u_analytical = np.exp(-2 * alpha * np.pi**2 * t_final) * \
                   np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Compute error
    error = np.linalg.norm(u_numerical - u_analytical) / np.linalg.norm(u_analytical)

    # Error should be small (< 1%)
    assert error < 0.01, f"Error {error} too large"


# ============================================================================
# 2D Poisson Equation Tests
# ============================================================================

def test_2d_poisson_basic(agent):
    """Test 2D Poisson equation with constant source."""
    def source_term(X, Y):
        return -2.0 * np.ones_like(X)  # Constant source

    result = agent.solve_pde_2d({
        'pde_type': 'poisson',
        'domain': [[0, 1], [0, 1]],
        'nx': 30,
        'ny': 30,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success
    assert result.status == AgentStatus.SUCCESS

    sol = result.data['solution']
    assert 'u' in sol
    assert sol['pde_type'] == 'poisson'
    assert sol['u'].shape == (30, 30)

    # For constant negative source with zero BC, solution should be positive
    assert np.all(sol['u'] >= 0)

    # Maximum should be in interior
    assert np.max(sol['u']) > 0


def test_2d_poisson_laplacian_verification(agent):
    """Test that ∇²u ≈ f for Poisson solution."""
    def source_term(X, Y):
        # Use a smooth source term
        return -np.sin(np.pi * X) * np.sin(np.pi * Y)

    result = agent.solve_pde_2d({
        'pde_type': 'poisson',
        'domain': [[0, 1], [0, 1]],
        'nx': 40,
        'ny': 40,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success

    sol = result.data['solution']
    u = sol['u']
    x = sol['x']
    y = sol['y']
    X, Y = sol['X'], sol['Y']

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute Laplacian of u numerically
    laplacian = np.zeros_like(u)
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            d2u_dx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            d2u_dy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            laplacian[i, j] = d2u_dx2 + d2u_dy2

    # Compute source term on grid
    f = source_term(X, Y)

    # Compare interior points only
    residual = np.linalg.norm(laplacian[1:-1, 1:-1] - f[1:-1, 1:-1])

    # Residual should be very small
    assert residual < 1e-8, f"Residual {residual} too large"


def test_2d_poisson_symmetry(agent):
    """Test that symmetric source gives symmetric solution."""
    def source_term(X, Y):
        # Radially symmetric source
        r_squared = (X - 0.5)**2 + (Y - 0.5)**2
        return -np.exp(-r_squared / 0.01)

    result = agent.solve_pde_2d({
        'pde_type': 'poisson',
        'domain': [[0, 1], [0, 1]],
        'nx': 40,
        'ny': 40,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success

    sol = result.data['solution']
    u = sol['u']

    # Check symmetry about center
    # Compare u[i,j] with u[n-1-i, j] and u[i, n-1-j]
    n = u.shape[0]
    mid = n // 2

    # X-symmetry
    x_sym_error = np.mean(np.abs(u[:mid, :] - np.flip(u[mid:, :], axis=0)))
    # Y-symmetry
    y_sym_error = np.mean(np.abs(u[:, :mid] - np.flip(u[:, mid:], axis=1)))

    # Symmetry errors should be small
    assert x_sym_error < 0.01
    assert y_sym_error < 0.01


# ============================================================================
# 2D Wave Equation Tests
# ============================================================================

def test_2d_wave_basic(agent):
    """Test 2D wave equation with Gaussian pulse."""
    def initial_condition(X, Y):
        r_squared = (X - 0.5)**2 + (Y - 0.5)**2
        return np.exp(-r_squared / 0.01)

    def initial_velocity(X, Y):
        return np.zeros_like(X)

    result = agent.solve_pde_2d({
        'pde_type': 'wave',
        'domain': [[0, 1], [0, 1]],
        'nx': 30,
        'ny': 30,
        'wave_speed': 1.0,
        'initial_condition': initial_condition,
        'initial_velocity': initial_velocity,
        't_span': (0, 0.5),
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success
    assert result.status == AgentStatus.SUCCESS

    sol = result.data['solution']
    assert 'u' in sol
    assert 'v' in sol  # Velocity
    assert 'u_all' in sol  # Time history
    assert sol['pde_type'] == 'wave'

    # Check shapes
    assert sol['u'].shape == (30, 30)
    assert sol['v'].shape == (30, 30)
    assert len(sol['t']) > 1  # Multiple time steps


def test_2d_wave_energy_conservation(agent):
    """Test that wave equation approximately conserves energy."""
    def initial_condition(X, Y):
        return np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)

    result = agent.solve_pde_2d({
        'pde_type': 'wave',
        'domain': [[0, 1], [0, 1]],
        'nx': 40,
        'ny': 40,
        'wave_speed': 1.0,
        'initial_condition': initial_condition,
        't_span': (0, 1.0),
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success

    sol = result.data['solution']
    u_all = sol['u_all']
    v_all = sol['v_all']
    x = sol['x']
    y = sol['y']
    c = sol['wave_speed']

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dA = dx * dy

    # Compute energy at first and last time steps
    energies = []
    for k in [0, -1]:
        U_k = u_all[:, :, k]
        V_k = v_all[:, :, k]

        # Kinetic energy
        KE = 0.5 * np.sum(V_k**2) * dA

        # Potential energy (gradient of u)
        grad_u_x = np.diff(U_k, axis=0) / dx
        grad_u_y = np.diff(U_k, axis=1) / dy
        PE = 0.5 * c**2 * (np.sum(grad_u_x**2) + np.sum(grad_u_y**2)) * dA

        energies.append(KE + PE)

    # Energy should be approximately conserved (within 5%)
    energy_change = abs(energies[1] - energies[0]) / energies[0]
    assert energy_change < 0.05, f"Energy changed by {100*energy_change:.1f}%"


# ============================================================================
# 3D Poisson Equation Tests
# ============================================================================

def test_3d_poisson_basic(agent):
    """Test 3D Poisson equation with point source."""
    def source_term(x, y, z):
        # Point source at center (approximated as Gaussian)
        r_squared = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2
        return -np.exp(-r_squared / 0.01)

    result = agent.solve_poisson_3d({
        'domain': [[0, 1], [0, 1], [0, 1]],
        'nx': 15,
        'ny': 15,
        'nz': 15,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success
    assert result.status == AgentStatus.SUCCESS

    sol = result.data['solution']
    assert 'u' in sol
    assert sol['u'].shape == (15, 15, 15)

    # Solution should be positive (negative source)
    assert np.max(sol['u']) > 0
    # Maximum should be near center
    center_idx = 15 // 2
    assert sol['u'][center_idx, center_idx, center_idx] > np.mean(sol['u'])


def test_3d_poisson_charge_conservation(agent):
    """Test that total charge is conserved in 3D Poisson."""
    total_charge = -1.0
    sigma = 0.1

    def source_term(x, y, z):
        # Normalized 3D Gaussian with total integral = total_charge
        r_squared = x**2 + y**2 + z**2
        normalization = total_charge / (sigma**3 * (2*np.pi)**(3/2))
        return normalization * np.exp(-r_squared / (2 * sigma**2))

    result = agent.solve_poisson_3d({
        'domain': [[-1, 1], [-1, 1], [-1, 1]],
        'nx': 20,
        'ny': 20,
        'nz': 20,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success

    sol = result.data['solution']
    x, y, z = sol['x'], sol['y'], sol['z']

    # Compute grid spacing and volume element
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dV = dx * dy * dz

    # Create grid and compute source
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    F = source_term(X, Y, Z)

    # Integrate source term
    integrated_charge = np.sum(F) * dV

    # Should match total_charge within numerical error
    error = abs(integrated_charge - total_charge) / abs(total_charge)
    assert error < 0.05, f"Charge conservation error: {100*error:.1f}%"


def test_3d_poisson_symmetry(agent):
    """Test 3D Poisson with spherically symmetric source."""
    def source_term(x, y, z):
        r_squared = x**2 + y**2 + z**2
        return -np.exp(-r_squared / 0.05)

    result = agent.solve_poisson_3d({
        'domain': [[-1, 1], [-1, 1], [-1, 1]],
        'nx': 20,
        'ny': 20,
        'nz': 20,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success

    sol = result.data['solution']
    u = sol['u']
    x, y, z = sol['x'], sol['y'], sol['z']

    # Check that solution is symmetric
    # For spherically symmetric source, solution along any axis should match
    center_idx = len(x) // 2

    # Extract profiles along x, y, z axes
    u_x = u[center_idx:, center_idx, center_idx]
    u_y = u[center_idx, center_idx:, center_idx]
    u_z = u[center_idx, center_idx, center_idx:]

    # Profiles should be similar
    max_diff_xy = np.max(np.abs(u_x - u_y))
    max_diff_xz = np.max(np.abs(u_x - u_z))

    # Allow some numerical error
    assert max_diff_xy < 0.05
    assert max_diff_xz < 0.05


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_invalid_pde_type(agent):
    """Test that invalid PDE type raises error."""
    result = agent.solve_pde_2d({
        'pde_type': 'invalid_type',
        'domain': [[0, 1], [0, 1]],
        'nx': 10,
        'ny': 10
    })

    assert not result.success
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0


def test_wave_missing_initial_condition(agent):
    """Test that wave equation requires initial condition."""
    with pytest.raises(ValueError, match="initial_condition"):
        agent.solve_pde_2d({
            'pde_type': 'wave',
            'domain': [[0, 1], [0, 1]],
            'nx': 10,
            'ny': 10,
            'wave_speed': 1.0
        })


def test_heat_missing_initial_condition(agent):
    """Test that heat equation requires initial condition."""
    with pytest.raises(ValueError, match="initial_condition"):
        agent.solve_pde_2d({
            'pde_type': 'heat',
            'domain': [[0, 1], [0, 1]],
            'nx': 10,
            'ny': 10,
            'alpha': 0.01
        })


# ============================================================================
# Performance Tests
# ============================================================================

def test_2d_poisson_larger_grid(agent):
    """Test 2D Poisson on larger grid (performance check)."""
    def source_term(X, Y):
        return -np.ones_like(X)

    result = agent.solve_pde_2d({
        'pde_type': 'poisson',
        'domain': [[0, 1], [0, 1]],
        'nx': 100,
        'ny': 100,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success
    assert result.data['solution']['u'].shape == (100, 100)

    # Should complete reasonably quickly (< 1 second)
    exec_time = result.data.get('execution_time', 0)
    assert exec_time < 1.0, f"Took too long: {exec_time}s"


def test_3d_poisson_moderate_grid(agent):
    """Test 3D Poisson on moderate grid."""
    def source_term(x, y, z):
        return -1.0

    result = agent.solve_poisson_3d({
        'domain': [[0, 1], [0, 1], [0, 1]],
        'nx': 25,
        'ny': 25,
        'nz': 25,
        'source_term': source_term,
        'boundary_conditions': {'value': 0.0}
    })

    assert result.success
    assert result.data['solution']['u'].shape == (25, 25, 25)

    # Should handle 15,625 unknowns efficiently (< 10 seconds)
    exec_time = result.data.get('execution_time', 0)
    assert exec_time < 10.0, f"Took too long: {exec_time}s"
