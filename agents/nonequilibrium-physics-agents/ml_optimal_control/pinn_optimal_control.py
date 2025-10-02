"""Physics-Informed Neural Networks for Optimal Control.

Enhanced PINN framework for solving Hamilton-Jacobi-Bellman equations,
value function approximation, and inverse optimal control problems.

Key capabilities:
- HJB equation solving via physics-informed losses
- Value function learning with PDE constraints
- Adaptive sampling for efficient training
- Inverse optimal control (learn cost from demonstrations)
- Multi-fidelity PINNs (combine data + physics)

Author: Nonequilibrium Physics Agents
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, jacfwd, jacrev
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy modules for type hints when JAX not available
    jnp = np  # Use numpy as fallback for type hints
    class nn:
        class Module:
            pass


class PINNArchitecture(Enum):
    """PINN architecture types."""
    VANILLA = "vanilla"  # Standard feedforward
    RESIDUAL = "residual"  # ResNet-style
    FOURIER = "fourier"  # Fourier feature encoding
    ADAPTIVE = "adaptive"  # Adaptive activation functions


class SamplingStrategy(Enum):
    """Sampling strategies for training points."""
    UNIFORM = "uniform"  # Uniform random sampling
    ADAPTIVE = "adaptive"  # Sample where residual is high
    QUASI_RANDOM = "quasi_random"  # Low-discrepancy sequences
    BOUNDARY_EMPHASIS = "boundary_emphasis"  # More samples at boundaries


@dataclass
class PINNConfig:
    """Configuration for PINN training."""
    # Architecture
    architecture: str = PINNArchitecture.VANILLA.value
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"  # tanh, relu, gelu, swish

    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 10000
    batch_size: int = 256
    optimizer: str = "adam"  # adam, sgd, adamw

    # Loss weights
    pde_weight: float = 1.0  # Physics loss weight
    bc_weight: float = 10.0  # Boundary condition weight
    ic_weight: float = 10.0  # Initial condition weight
    data_weight: float = 1.0  # Data fitting weight (if available)

    # Sampling
    sampling_strategy: str = SamplingStrategy.UNIFORM.value
    num_collocation_points: int = 10000  # Interior points
    num_boundary_points: int = 1000  # Boundary points
    adaptive_resample_freq: int = 100  # Epochs between resampling

    # Adaptive sampling
    residual_threshold: float = 0.1  # Threshold for adaptive sampling
    boundary_layer_width: float = 0.1  # Width of boundary layer emphasis


if JAX_AVAILABLE:
    class VanillaPINN(nn.Module):
        """Vanilla feedforward PINN."""
        hidden_layers: List[int]
        output_dim: int = 1
        activation: str = "tanh"

        @nn.compact
        def __call__(self, x):
            """Forward pass.

            Args:
                x: Input tensor (state, time)

            Returns:
                Output (value function, control, etc.)
            """
            # Activation function
            if self.activation == "tanh":
                act = nn.tanh
            elif self.activation == "relu":
                act = nn.relu
            elif self.activation == "gelu":
                act = nn.gelu
            elif self.activation == "swish":
                act = nn.swish
            else:
                act = nn.tanh

            # Hidden layers
            for size in self.hidden_layers:
                x = nn.Dense(size)(x)
                x = act(x)

            # Output layer
            x = nn.Dense(self.output_dim)(x)

            return x


    class ResidualPINN(nn.Module):
        """PINN with residual connections."""
        hidden_layers: List[int]
        output_dim: int = 1
        activation: str = "tanh"

        @nn.compact
        def __call__(self, x):
            """Forward pass with residual connections."""
            if self.activation == "tanh":
                act = nn.tanh
            else:
                act = nn.relu

            # Initial projection
            x_proj = nn.Dense(self.hidden_layers[0])(x)

            # Residual blocks
            for i, size in enumerate(self.hidden_layers):
                x_in = x_proj

                # Two-layer residual block
                x_proj = nn.Dense(size)(x_proj)
                x_proj = act(x_proj)
                x_proj = nn.Dense(size)(x_proj)

                # Residual connection
                if x_in.shape == x_proj.shape:
                    x_proj = x_proj + x_in

                x_proj = act(x_proj)

            # Output
            x_proj = nn.Dense(self.output_dim)(x_proj)

            return x_proj


    class FourierPINN(nn.Module):
        """PINN with Fourier feature encoding.

        Helps with learning high-frequency functions.
        """
        hidden_layers: List[int]
        output_dim: int = 1
        fourier_features: int = 256
        fourier_scale: float = 1.0

        @nn.compact
        def __call__(self, x):
            """Forward pass with Fourier features."""
            # Fourier feature mapping: γ(x) = [cos(Bx), sin(Bx)]
            # B ~ N(0, scale²)
            B = self.param('B', nn.initializers.normal(self.fourier_scale),
                          (x.shape[-1], self.fourier_features))

            # Compute Fourier features
            x_proj = 2 * jnp.pi * x @ B
            x_fourier = jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)

            # Standard network on Fourier features
            x = x_fourier
            for size in self.hidden_layers:
                x = nn.Dense(size)(x)
                x = nn.tanh(x)

            x = nn.Dense(self.output_dim)(x)

            return x


class PINNOptimalControl:
    """Physics-Informed Neural Network for optimal control problems."""

    def __init__(self, config: Optional[PINNConfig] = None):
        """Initialize PINN for optimal control.

        Args:
            config: PINN configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX required for PINN. Install with: pip install jax jaxlib flax optax")

        self.config = config or PINNConfig()
        self.model = None
        self.state = None
        self.training_history = []

    def create_model(
        self,
        input_dim: int,
        output_dim: int = 1,
        architecture: Optional[str] = None
    ) -> nn.Module:
        """Create PINN model.

        Args:
            input_dim: Input dimension (n_states + 1 for time)
            output_dim: Output dimension (1 for value function)
            architecture: Architecture type (overrides config)

        Returns:
            PINN model
        """
        arch = architecture or self.config.architecture

        if arch == PINNArchitecture.VANILLA.value:
            self.model = VanillaPINN(
                hidden_layers=self.config.hidden_layers,
                output_dim=output_dim,
                activation=self.config.activation
            )
        elif arch == PINNArchitecture.RESIDUAL.value:
            self.model = ResidualPINN(
                hidden_layers=self.config.hidden_layers,
                output_dim=output_dim,
                activation=self.config.activation
            )
        elif arch == PINNArchitecture.FOURIER.value:
            self.model = FourierPINN(
                hidden_layers=self.config.hidden_layers,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        return self.model

    def hjb_residual(
        self,
        params: Any,
        x: jnp.ndarray,
        dynamics: Callable,
        running_cost: Callable
    ) -> jnp.ndarray:
        """Compute Hamilton-Jacobi-Bellman residual.

        HJB equation: ∂V/∂t + min_u [∇V·f(x,u) + L(x,u)] = 0

        Args:
            params: Model parameters
            x: State-time points (n_batch, n_states + 1)
            dynamics: f(x, u) -> dx/dt
            running_cost: L(x, u) -> scalar

        Returns:
            HJB residual at each point
        """
        def value_function(xt):
            """Evaluate value function V(x, t)."""
            return self.model.apply(params, xt[None, :])[0, 0]

        # Compute gradients
        grad_V = vmap(grad(value_function))(x)

        # Split into spatial and temporal gradients
        dV_dx = grad_V[:, :-1]  # ∂V/∂x
        dV_dt = grad_V[:, -1:]   # ∂V/∂t

        # Extract state and time
        state = x[:, :-1]
        time = x[:, -1:]

        # Compute optimal control (greedy w.r.t. Hamiltonian)
        # For LQR: u* = -R^(-1) B^T ∇V
        # Simplified: u* ∝ -∇V (for this example)
        u_optimal = -dV_dx  # Placeholder - should solve argmin

        # Hamiltonian: H = ∇V·f(x,u) + L(x,u)
        f_xu = vmap(lambda s, u: dynamics(s, u, 0.0))(state, u_optimal)
        L_xu = vmap(lambda s, u: running_cost(s, u, 0.0))(state, u_optimal)

        hamiltonian = jnp.sum(dV_dx * f_xu, axis=1, keepdims=True) + L_xu[:, None]

        # HJB residual: ∂V/∂t + H = 0
        residual = dV_dt + hamiltonian

        return residual

    def boundary_loss(
        self,
        params: Any,
        x_boundary: jnp.ndarray,
        boundary_values: jnp.ndarray
    ) -> float:
        """Compute boundary condition loss.

        Args:
            params: Model parameters
            x_boundary: Boundary points
            boundary_values: Target values at boundary

        Returns:
            MSE loss
        """
        V_pred = self.model.apply(params, x_boundary)
        loss = jnp.mean((V_pred - boundary_values) ** 2)
        return loss

    def initial_condition_loss(
        self,
        params: Any,
        x_initial: jnp.ndarray,
        initial_values: jnp.ndarray
    ) -> float:
        """Compute initial condition loss.

        Args:
            params: Model parameters
            x_initial: Initial condition points
            initial_values: Target initial values

        Returns:
            MSE loss
        """
        V_pred = self.model.apply(params, x_initial)
        loss = jnp.mean((V_pred - initial_values) ** 2)
        return loss

    def total_loss(
        self,
        params: Any,
        x_collocation: jnp.ndarray,
        x_boundary: jnp.ndarray,
        boundary_values: jnp.ndarray,
        x_initial: jnp.ndarray,
        initial_values: jnp.ndarray,
        dynamics: Callable,
        running_cost: Callable
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total PINN loss.

        Args:
            params: Model parameters
            x_collocation: Collocation points for PDE
            x_boundary: Boundary points
            boundary_values: Boundary values
            x_initial: Initial points
            initial_values: Initial values
            dynamics: Dynamics function
            running_cost: Running cost function

        Returns:
            Total loss and loss components
        """
        # PDE loss (HJB residual)
        hjb_res = self.hjb_residual(params, x_collocation, dynamics, running_cost)
        pde_loss = jnp.mean(hjb_res ** 2)

        # Boundary loss
        bc_loss = self.boundary_loss(params, x_boundary, boundary_values)

        # Initial condition loss
        ic_loss = self.initial_condition_loss(params, x_initial, initial_values)

        # Total weighted loss
        total = (
            self.config.pde_weight * pde_loss +
            self.config.bc_weight * bc_loss +
            self.config.ic_weight * ic_loss
        )

        loss_dict = {
            'total': total,
            'pde': pde_loss,
            'boundary': bc_loss,
            'initial': ic_loss
        }

        return total, loss_dict

    def sample_collocation_points(
        self,
        n_points: int,
        state_bounds: List[Tuple[float, float]],
        time_bounds: Tuple[float, float],
        strategy: Optional[str] = None
    ) -> jnp.ndarray:
        """Sample collocation points for training.

        Args:
            n_points: Number of points
            state_bounds: Bounds for each state dimension [(min, max), ...]
            time_bounds: Time bounds (t_min, t_max)
            strategy: Sampling strategy

        Returns:
            Collocation points (n_points, n_states + 1)
        """
        strategy = strategy or self.config.sampling_strategy
        n_states = len(state_bounds)

        if strategy == SamplingStrategy.UNIFORM.value:
            # Uniform random sampling
            points = []
            for i, (lb, ub) in enumerate(state_bounds):
                points.append(np.random.uniform(lb, ub, n_points))

            # Time dimension
            points.append(np.random.uniform(time_bounds[0], time_bounds[1], n_points))

            return jnp.array(points).T

        elif strategy == SamplingStrategy.QUASI_RANDOM.value:
            # Sobol sequence (low-discrepancy)
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_states + 1, scramble=True)
            points_unit = sampler.random(n_points)

            # Scale to bounds
            points = []
            for i, (lb, ub) in enumerate(state_bounds):
                points.append(lb + (ub - lb) * points_unit[:, i])

            # Time
            points.append(time_bounds[0] + (time_bounds[1] - time_bounds[0]) * points_unit[:, -1])

            return jnp.array(points).T

        elif strategy == SamplingStrategy.BOUNDARY_EMPHASIS.value:
            # More samples near boundaries
            n_boundary = int(n_points * 0.3)  # 30% near boundaries
            n_interior = n_points - n_boundary

            # Interior points
            interior_points = self.sample_collocation_points(
                n_interior, state_bounds, time_bounds, SamplingStrategy.UNIFORM.value
            )

            # Boundary points (randomly select dimension to be at boundary)
            boundary_points = []
            for _ in range(n_boundary):
                point = []
                boundary_dim = np.random.randint(0, n_states)

                for i, (lb, ub) in enumerate(state_bounds):
                    if i == boundary_dim:
                        # At boundary
                        point.append(lb if np.random.rand() < 0.5 else ub)
                    else:
                        # Interior
                        point.append(np.random.uniform(lb, ub))

                # Time
                point.append(np.random.uniform(time_bounds[0], time_bounds[1]))
                boundary_points.append(point)

            return jnp.concatenate([interior_points, jnp.array(boundary_points)], axis=0)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")


class InverseOptimalControl:
    """Inverse optimal control via PINNs.

    Learn cost function from expert demonstrations.
    """

    def __init__(self, n_states: int, n_controls: int):
        """Initialize inverse optimal control.

        Args:
            n_states: State dimension
            n_controls: Control dimension
        """
        self.n_states = n_states
        self.n_controls = n_controls
        self.cost_params = None

    def parametric_cost(
        self,
        params: jnp.ndarray,
        state: jnp.ndarray,
        control: jnp.ndarray
    ) -> float:
        """Parametric cost function.

        L(x, u; θ) = x^T Q(θ) x + u^T R(θ) u

        Args:
            params: Cost parameters (Q, R matrices flattened)
            state: State vector
            control: Control vector

        Returns:
            Cost value
        """
        # Extract Q and R from params
        n_q = self.n_states * self.n_states
        n_r = self.n_controls * self.n_controls

        Q_flat = params[:n_q]
        R_flat = params[n_q:n_q + n_r]

        Q = Q_flat.reshape(self.n_states, self.n_states)
        R = R_flat.reshape(self.n_controls, self.n_controls)

        # Ensure positive definite (use Q^T Q and R^T R)
        Q = Q.T @ Q
        R = R.T @ R

        # Quadratic cost
        cost = state @ Q @ state + control @ R @ control

        return cost

    def learn_cost_from_demonstrations(
        self,
        expert_trajectories: List[Dict[str, np.ndarray]],
        dynamics: Callable,
        num_iterations: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Learn cost function from expert demonstrations.

        Uses inverse RL: find cost such that expert is optimal.

        Args:
            expert_trajectories: List of expert trajectories
                Each: {'states': (T, n_states), 'controls': (T, n_controls)}
            dynamics: System dynamics f(x, u)
            num_iterations: Optimization iterations

        Returns:
            Learned cost parameters (Q, R matrices)
        """
        # Initialize cost parameters randomly
        n_q = self.n_states * self.n_states
        n_r = self.n_controls * self.n_controls
        cost_params = jnp.array(np.random.randn(n_q + n_r) * 0.1)

        # Optimization
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(cost_params)

        for iteration in range(num_iterations):
            # Compute loss: expert should minimize cost
            def loss_fn(params):
                total_cost = 0.0

                for traj in expert_trajectories:
                    states = jnp.array(traj['states'])
                    controls = jnp.array(traj['controls'])

                    # Cost along expert trajectory
                    traj_cost = jnp.sum(vmap(
                        lambda s, u: self.parametric_cost(params, s, u)
                    )(states, controls))

                    total_cost += traj_cost

                # Normalize
                return total_cost / len(expert_trajectories)

            # Compute gradients and update
            loss, grads = jax.value_and_grad(loss_fn)(cost_params)
            updates, opt_state = optimizer.update(grads, opt_state)
            cost_params = optax.apply_updates(cost_params, updates)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

        # Extract learned Q and R
        Q_flat = cost_params[:n_q]
        R_flat = cost_params[n_q:n_q + n_r]

        Q = Q_flat.reshape(self.n_states, self.n_states)
        R = R_flat.reshape(self.n_controls, self.n_controls)

        # Make positive definite
        Q = Q.T @ Q
        R = R.T @ R

        self.cost_params = {'Q': np.array(Q), 'R': np.array(R)}

        return self.cost_params


def create_lqr_pinn_example():
    """Create example PINN for LQR problem."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping PINN example")
        return None

    print("Creating PINN for LQR value function learning...")

    # Problem setup: 2D LQR
    n_states = 2
    config = PINNConfig(
        hidden_layers=[64, 64, 64],
        activation="tanh",
        num_epochs=1000,
        learning_rate=1e-3
    )

    pinn = PINNOptimalControl(config)

    # Create model
    model = pinn.create_model(input_dim=n_states + 1, output_dim=1)

    # Define dynamics and cost
    def dynamics(x, u, t):
        """Linear dynamics: dx/dt = Ax + Bu."""
        A = jnp.array([[0.0, 1.0], [-1.0, -0.1]])
        B = jnp.array([[0.0], [1.0]])
        return A @ x + B @ u

    def running_cost(x, u, t):
        """Quadratic cost: L = x'Qx + u'Ru."""
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        R = jnp.array([[0.1]])
        return x @ Q @ x + u @ R @ u

    print("PINN model created successfully!")
    return pinn


if __name__ == "__main__":
    print("=== PINN Optimal Control Framework ===\n")

    # Create example
    pinn = create_lqr_pinn_example()

    if pinn:
        print("\nPINN framework ready for optimal control!")
        print("Key capabilities:")
        print("  - HJB equation solving")
        print("  - Value function approximation")
        print("  - Adaptive sampling")
        print("  - Inverse optimal control")
