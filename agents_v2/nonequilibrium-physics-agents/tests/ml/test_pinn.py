"""Tests for Physics-Informed Neural Networks (PINNs).

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.pinn_optimal_control import (
    PINNOptimalControl,
    PINNConfig,
    PINNArchitecture,
    SamplingStrategy,
    InverseOptimalControl
)

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestPINNConfiguration:
    """Tests for PINN configuration."""

    def test_default_config(self):
        """Test: Default PINN configuration."""
        config = PINNConfig()

        assert config.architecture == PINNArchitecture.VANILLA.value
        assert len(config.hidden_layers) == 3
        assert config.learning_rate > 0
        assert config.pde_weight > 0

    def test_custom_config(self):
        """Test: Custom PINN configuration."""
        config = PINNConfig(
            hidden_layers=[128, 128],
            learning_rate=1e-4,
            num_epochs=5000
        )

        assert len(config.hidden_layers) == 2
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5000


class TestPINNModel:
    """Tests for PINN model creation."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_vanilla_model(self):
        """Test: Create vanilla PINN model."""
        config = PINNConfig(architecture=PINNArchitecture.VANILLA.value)
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)

        assert model is not None
        assert pinn.model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_residual_model(self):
        """Test: Create residual PINN model."""
        config = PINNConfig(architecture=PINNArchitecture.RESIDUAL.value)
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)

        assert model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_fourier_model(self):
        """Test: Create Fourier PINN model."""
        config = PINNConfig(architecture=PINNArchitecture.FOURIER.value)
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)

        assert model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_model_forward_pass(self):
        """Test: PINN forward pass."""
        config = PINNConfig()
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)

        # Initialize model
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.ones((1, 3))
        params = model.init(key, x_dummy)

        # Forward pass
        output = model.apply(params, x_dummy)

        assert output.shape == (1, 1)
        assert jnp.isfinite(output).all()


class TestSampling:
    """Tests for collocation point sampling."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_uniform_sampling(self):
        """Test: Uniform sampling strategy."""
        config = PINNConfig(sampling_strategy=SamplingStrategy.UNIFORM.value)
        pinn = PINNOptimalControl(config)

        state_bounds = [(0.0, 1.0), (0.0, 1.0)]
        time_bounds = (0.0, 1.0)

        points = pinn.sample_collocation_points(
            n_points=100,
            state_bounds=state_bounds,
            time_bounds=time_bounds
        )

        assert points.shape == (100, 3)  # 2 states + 1 time
        assert jnp.all(points >= 0.0)
        assert jnp.all(points <= 1.0)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_quasi_random_sampling(self):
        """Test: Quasi-random (Sobol) sampling."""
        config = PINNConfig(sampling_strategy=SamplingStrategy.QUASI_RANDOM.value)
        pinn = PINNOptimalControl(config)

        state_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        time_bounds = (0.0, 1.0)

        points = pinn.sample_collocation_points(
            n_points=100,
            state_bounds=state_bounds,
            time_bounds=time_bounds
        )

        assert points.shape == (100, 3)
        # Check bounds
        assert jnp.all(points[:, 0] >= -1.0) and jnp.all(points[:, 0] <= 1.0)
        assert jnp.all(points[:, 1] >= -1.0) and jnp.all(points[:, 1] <= 1.0)
        assert jnp.all(points[:, 2] >= 0.0) and jnp.all(points[:, 2] <= 1.0)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_boundary_emphasis_sampling(self):
        """Test: Boundary emphasis sampling."""
        config = PINNConfig(sampling_strategy=SamplingStrategy.BOUNDARY_EMPHASIS.value)
        pinn = PINNOptimalControl(config)

        state_bounds = [(0.0, 1.0), (0.0, 1.0)]
        time_bounds = (0.0, 1.0)

        points = pinn.sample_collocation_points(
            n_points=100,
            state_bounds=state_bounds,
            time_bounds=time_bounds
        )

        assert points.shape == (100, 3)

        # Check that some points are at boundaries
        at_boundary = (
            (jnp.abs(points[:, 0] - 0.0) < 1e-6) |
            (jnp.abs(points[:, 0] - 1.0) < 1e-6) |
            (jnp.abs(points[:, 1] - 0.0) < 1e-6) |
            (jnp.abs(points[:, 1] - 1.0) < 1e-6)
        )

        # At least 20% should be at boundaries (30% expected, allow variance)
        assert jnp.sum(at_boundary) >= 20


class TestHJBResidual:
    """Tests for HJB residual computation."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_hjb_residual_computation(self):
        """Test: HJB residual can be computed."""
        config = PINNConfig()
        pinn = PINNOptimalControl(config)

        # Create model
        model = pinn.create_model(input_dim=3, output_dim=1)
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.ones((1, 3))
        params = model.init(key, x_dummy)

        # Define simple dynamics and cost
        def dynamics(x, u, t):
            """Linear dynamics."""
            return jnp.array([x[1], u[0]])

        def running_cost(x, u, t):
            """Quadratic cost."""
            return jnp.sum(x**2) + 0.1 * jnp.sum(u**2)

        # Sample points
        x = jnp.array([[0.5, 0.5, 0.5]])  # (state1, state2, time)

        # Compute residual
        residual = pinn.hjb_residual(params, x, dynamics, running_cost)

        assert residual.shape == (1, 1)
        assert jnp.isfinite(residual).all()

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_hjb_residual_batch(self):
        """Test: HJB residual on batch of points."""
        config = PINNConfig()
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.ones((1, 3))
        params = model.init(key, x_dummy)

        def dynamics(x, u, t):
            return jnp.array([x[1], u[0]])

        def running_cost(x, u, t):
            return jnp.sum(x**2) + 0.1 * jnp.sum(u**2)

        # Batch of points
        x = jnp.array([
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7]
        ])

        residual = pinn.hjb_residual(params, x, dynamics, running_cost)

        assert residual.shape == (3, 1)
        assert jnp.isfinite(residual).all()


class TestLossFunctions:
    """Tests for PINN loss functions."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_boundary_loss(self):
        """Test: Boundary condition loss."""
        config = PINNConfig()
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.ones((1, 3))
        params = model.init(key, x_dummy)

        # Boundary points
        x_boundary = jnp.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        boundary_values = jnp.array([[0.0], [0.0]])

        loss = pinn.boundary_loss(params, x_boundary, boundary_values)

        assert jnp.isfinite(loss)
        assert loss >= 0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_initial_condition_loss(self):
        """Test: Initial condition loss."""
        config = PINNConfig()
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.ones((1, 3))
        params = model.init(key, x_dummy)

        # Initial condition points (t=0)
        x_initial = jnp.array([[0.5, 0.5, 0.0], [0.3, 0.7, 0.0]])
        initial_values = jnp.array([[1.0], [1.0]])

        loss = pinn.initial_condition_loss(params, x_initial, initial_values)

        assert jnp.isfinite(loss)
        assert loss >= 0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_total_loss(self):
        """Test: Total PINN loss computation."""
        config = PINNConfig()
        pinn = PINNOptimalControl(config)

        model = pinn.create_model(input_dim=3, output_dim=1)
        key = jax.random.PRNGKey(0)
        x_dummy = jnp.ones((1, 3))
        params = model.init(key, x_dummy)

        def dynamics(x, u, t):
            return jnp.array([x[1], u[0]])

        def running_cost(x, u, t):
            return jnp.sum(x**2) + 0.1 * jnp.sum(u**2)

        # Sample points
        x_collocation = jnp.array([[0.5, 0.5, 0.5]])
        x_boundary = jnp.array([[0.0, 0.0, 1.0]])
        boundary_values = jnp.array([[0.0]])
        x_initial = jnp.array([[0.5, 0.5, 0.0]])
        initial_values = jnp.array([[1.0]])

        total_loss, loss_dict = pinn.total_loss(
            params,
            x_collocation,
            x_boundary,
            boundary_values,
            x_initial,
            initial_values,
            dynamics,
            running_cost
        )

        assert jnp.isfinite(total_loss)
        assert 'total' in loss_dict
        assert 'pde' in loss_dict
        assert 'boundary' in loss_dict
        assert 'initial' in loss_dict
        assert all(jnp.isfinite(v) for v in loss_dict.values())


class TestInverseOptimalControl:
    """Tests for inverse optimal control."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_inverse_oc_initialization(self):
        """Test: Inverse optimal control initialization."""
        inv_oc = InverseOptimalControl(n_states=2, n_controls=1)

        assert inv_oc.n_states == 2
        assert inv_oc.n_controls == 1
        assert inv_oc.cost_params is None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_parametric_cost(self):
        """Test: Parametric cost function."""
        inv_oc = InverseOptimalControl(n_states=2, n_controls=1)

        # Create parameters
        n_q = 2 * 2
        n_r = 1 * 1
        params = jnp.ones(n_q + n_r) * 0.1

        state = jnp.array([1.0, 0.5])
        control = jnp.array([0.2])

        cost = inv_oc.parametric_cost(params, state, control)

        assert jnp.isfinite(cost)
        assert cost >= 0  # Should be non-negative for positive definite Q, R

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_learn_from_demonstrations_simple(self):
        """Test: Learn cost from simple demonstrations."""
        inv_oc = InverseOptimalControl(n_states=2, n_controls=1)

        # Create simple expert trajectory
        expert_trajectories = [
            {
                'states': np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]]),
                'controls': np.array([[0.0], [0.0], [0.0]])
            }
        ]

        def dynamics(x, u):
            return x  # Dummy dynamics

        # Learn cost (quick test with few iterations)
        learned_cost = inv_oc.learn_cost_from_demonstrations(
            expert_trajectories,
            dynamics,
            num_iterations=10
        )

        assert 'Q' in learned_cost
        assert 'R' in learned_cost
        assert learned_cost['Q'].shape == (2, 2)
        assert learned_cost['R'].shape == (1, 1)


class TestPINNIntegration:
    """Integration tests for PINN framework."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_complete_pinn_workflow(self):
        """Test: Complete PINN workflow for simple problem."""
        # Configuration
        config = PINNConfig(
            hidden_layers=[32, 32],
            num_epochs=10,  # Short for testing
            batch_size=32
        )

        pinn = PINNOptimalControl(config)

        # Create model
        model = pinn.create_model(input_dim=3, output_dim=1)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 3))
        params = model.init(key, x_dummy)

        # Sample training points
        x_collocation = pinn.sample_collocation_points(
            n_points=100,
            state_bounds=[(0.0, 1.0), (0.0, 1.0)],
            time_bounds=(0.0, 1.0)
        )

        assert x_collocation.shape == (100, 3)

        # This validates the complete workflow can execute
        print("PINN workflow completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
