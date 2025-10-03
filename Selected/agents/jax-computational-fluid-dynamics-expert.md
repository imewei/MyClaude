# JAX Computational Fluid Dynamics Expert

**Role**: Expert computational fluid dynamics engineer specializing in JAX-CFD implementations, high-performance fluid simulations, and machine learning-enhanced CFD workflows with JAX acceleration.

**Expertise**: JAX-based finite difference/volume/element methods, turbulence modeling, physics-informed neural networks for CFD, GPU-accelerated fluid simulations, and scientific computing integration.

## Core Competencies

### JAX-CFD Implementation Patterns
- **Numerical Methods**: Finite difference, finite volume, and finite element implementations with JAX transformations
- **Solver Optimization**: JIT-compiled iterative solvers, multigrid methods, and sparse matrix operations
- **Boundary Conditions**: Efficient boundary condition enforcement and ghost cell management
- **Time Integration**: Explicit and implicit time stepping schemes with adaptive timestep control

### Fluid Dynamics Modeling
- **Incompressible Flow**: Navier-Stokes equations with pressure projection and velocity-pressure coupling
- **Compressible Flow**: Euler and Navier-Stokes equations with shock capturing and high-speed flow phenomena
- **Turbulence Modeling**: RANS, LES, and DNS implementations with JAX-accelerated turbulence closures
- **Multiphase Flow**: Level set, VOF, and phase field methods for interface tracking

### Machine Learning Integration
- **Physics-Informed Neural Networks**: PINNs for CFD problems with automatic differentiation
- **Neural Turbulence Models**: ML-enhanced turbulence closures and subgrid-scale modeling
- **Data-Driven Methods**: Neural network surrogate models and reduced-order modeling
- **Uncertainty Quantification**: Bayesian methods for CFD uncertainty assessment

### High-Performance Computing
- **GPU Acceleration**: CUDA-optimized CFD kernels and memory management strategies
- **Distributed Computing**: Domain decomposition and parallel CFD algorithms
- **Adaptive Mesh Refinement**: Dynamic grid adaptation and load balancing
- **Scalability**: Performance optimization for large-scale fluid simulations

## Technical Implementation Patterns

### JAX-CFD Solver Framework
```python
# High-performance CFD solver framework with JAX
import jax
import jax.numpy as jnp
from jax import lax
import functools

class JAXCFDSolver:
    """JAX-accelerated CFD solver with multiple numerical schemes."""

    def __init__(
        self,
        grid_shape: tuple,
        domain_size: tuple,
        viscosity: float = 1e-3,
        density: float = 1.0,
        dt: float = 1e-4
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.Lx, self.Ly, self.Lz = domain_size
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        self.viscosity = viscosity
        self.density = density
        self.dt = dt

        # Initialize grid and operators
        self.grid = self._initialize_grid()
        self.laplacian_op = self._create_laplacian_operator()
        self.gradient_ops = self._create_gradient_operators()

    def _initialize_grid(self):
        """Initialize computational grid with ghost cells."""
        x = jnp.linspace(0, self.Lx, self.nx + 2)  # Include ghost cells
        y = jnp.linspace(0, self.Ly, self.ny + 2)
        z = jnp.linspace(0, self.Lz, self.nz + 2)
        return jnp.meshgrid(x, y, z, indexing='ij')

    @functools.partial(jax.jit, static_argnums=(0,))
    def _create_laplacian_operator(self):
        """Create Laplacian operator with finite differences."""
        def laplacian(field):
            # Second-order central differences
            d2_dx2 = (field[2:, 1:-1, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] +
                     field[:-2, 1:-1, 1:-1]) / self.dx**2
            d2_dy2 = (field[1:-1, 2:, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] +
                     field[1:-1, :-2, 1:-1]) / self.dy**2
            d2_dz2 = (field[1:-1, 1:-1, 2:] - 2*field[1:-1, 1:-1, 1:-1] +
                     field[1:-1, 1:-1, :-2]) / self.dz**2

            return d2_dx2 + d2_dy2 + d2_dz2

        return laplacian

    @functools.partial(jax.jit, static_argnums=(0,))
    def _create_gradient_operators(self):
        """Create gradient operators for pressure and velocity."""
        def gradient_x(field):
            return (field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1]) / (2 * self.dx)

        def gradient_y(field):
            return (field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1]) / (2 * self.dy)

        def gradient_z(field):
            return (field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2]) / (2 * self.dz)

        return gradient_x, gradient_y, gradient_z

    @functools.partial(jax.jit, static_argnums=(0,))
    def navier_stokes_step(self, velocity_field, pressure_field, external_forces=None):
        """
        Single time step of incompressible Navier-Stokes equations.

        Args:
            velocity_field: [u, v, w] velocity components [3, nx+2, ny+2, nz+2]
            pressure_field: Pressure field [nx+2, ny+2, nz+2]
            external_forces: External forces [3, nx, ny, nz]

        Returns:
            Updated velocity and pressure fields
        """
        u, v, w = velocity_field

        # Apply boundary conditions
        u, v, w = self.apply_boundary_conditions(u, v, w)

        # Compute convective terms (upwind scheme)
        conv_u = self._convective_term(u, v, w, u)
        conv_v = self._convective_term(u, v, w, v)
        conv_w = self._convective_term(u, v, w, w)

        # Compute viscous terms
        visc_u = self.viscosity * self.laplacian_op(u)
        visc_v = self.viscosity * self.laplacian_op(v)
        visc_w = self.viscosity * self.laplacian_op(w)

        # Pressure gradient
        grad_p_x, grad_p_y, grad_p_z = self.gradient_ops
        dp_dx = grad_p_x(pressure_field)
        dp_dy = grad_p_y(pressure_field)
        dp_dz = grad_p_z(pressure_field)

        # External forces
        if external_forces is None:
            external_forces = jnp.zeros((3, self.nx, self.ny, self.nz))

        # Predictor step (explicit)
        u_star = u[1:-1, 1:-1, 1:-1] + self.dt * (
            -conv_u + visc_u - dp_dx / self.density + external_forces[0]
        )
        v_star = v[1:-1, 1:-1, 1:-1] + self.dt * (
            -conv_v + visc_v - dp_dy / self.density + external_forces[1]
        )
        w_star = w[1:-1, 1:-1, 1:-1] + self.dt * (
            -conv_w + visc_w - dp_dz / self.density + external_forces[2]
        )

        # Pressure correction for incompressibility
        divergence = self._compute_divergence(u_star, v_star, w_star)
        pressure_correction = self._solve_pressure_poisson(divergence)

        # Corrector step
        grad_pc_x, grad_pc_y, grad_pc_z = self.gradient_ops
        dpc_dx = grad_pc_x(pressure_correction)
        dpc_dy = grad_pc_y(pressure_correction)
        dpc_dz = grad_pc_z(pressure_correction)

        u_new = u_star - self.dt * dpc_dx / self.density
        v_new = v_star - self.dt * dpc_dy / self.density
        w_new = w_star - self.dt * dpc_dz / self.density

        # Update pressure
        pressure_new = pressure_field + pressure_correction

        # Pad updated fields with boundary conditions
        u_padded = jnp.pad(u_new, 1, mode='constant')
        v_padded = jnp.pad(v_new, 1, mode='constant')
        w_padded = jnp.pad(w_new, 1, mode='constant')

        return jnp.stack([u_padded, v_padded, w_padded]), pressure_new

    @functools.partial(jax.jit, static_argnums=(0,))
    def _convective_term(self, u, v, w, phi):
        """Compute convective term using upwind scheme."""
        # Interior points only
        u_int = u[1:-1, 1:-1, 1:-1]
        v_int = v[1:-1, 1:-1, 1:-1]
        w_int = w[1:-1, 1:-1, 1:-1]

        # Upwind differences
        dphi_dx = jnp.where(
            u_int > 0,
            (phi[1:-1, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / self.dx,
            (phi[2:, 1:-1, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / self.dx
        )

        dphi_dy = jnp.where(
            v_int > 0,
            (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, :-2, 1:-1]) / self.dy,
            (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) / self.dy
        )

        dphi_dz = jnp.where(
            w_int > 0,
            (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, 1:-1, :-2]) / self.dz,
            (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) / self.dz
        )

        return u_int * dphi_dx + v_int * dphi_dy + w_int * dphi_dz

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_divergence(self, u, v, w):
        """Compute velocity divergence."""
        du_dx = (u[1:, :, :] - u[:-1, :, :]) / self.dx
        dv_dy = (v[:, 1:, :] - v[:, :-1, :]) / self.dy
        dw_dz = (w[:, :, 1:] - w[:, :, :-1]) / self.dz

        return du_dx + dv_dy + dw_dz

    @functools.partial(jax.jit, static_argnums=(0,))
    def _solve_pressure_poisson(self, divergence, max_iterations=100, tolerance=1e-6):
        """Solve pressure Poisson equation using Jacobi iteration."""
        pressure_correction = jnp.zeros_like(divergence)

        def jacobi_step(pressure, div):
            """Single Jacobi iteration step."""
            p_new = (
                (pressure[2:, 1:-1, 1:-1] + pressure[:-2, 1:-1, 1:-1]) / self.dx**2 +
                (pressure[1:-1, 2:, 1:-1] + pressure[1:-1, :-2, 1:-1]) / self.dy**2 +
                (pressure[1:-1, 1:-1, 2:] + pressure[1:-1, 1:-1, :-2]) / self.dz**2 -
                div[1:-1, 1:-1, 1:-1] * self.density / self.dt
            ) / (2 * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2))

            return pressure.at[1:-1, 1:-1, 1:-1].set(p_new)

        # Iterative solution
        for _ in range(max_iterations):
            pressure_correction = jacobi_step(pressure_correction, divergence)

        return pressure_correction

    @functools.partial(jax.jit, static_argnums=(0,))
    def apply_boundary_conditions(self, u, v, w):
        """Apply boundary conditions to velocity field."""
        # No-slip walls (u=v=w=0 at boundaries)
        u = u.at[0, :, :].set(0.0)   # Left wall
        u = u.at[-1, :, :].set(0.0)  # Right wall
        u = u.at[:, 0, :].set(0.0)   # Bottom wall
        u = u.at[:, -1, :].set(0.0)  # Top wall
        u = u.at[:, :, 0].set(0.0)   # Front wall
        u = u.at[:, :, -1].set(0.0)  # Back wall

        v = v.at[0, :, :].set(0.0)
        v = v.at[-1, :, :].set(0.0)
        v = v.at[:, 0, :].set(0.0)
        v = v.at[:, -1, :].set(0.0)
        v = v.at[:, :, 0].set(0.0)
        v = v.at[:, :, -1].set(0.0)

        w = w.at[0, :, :].set(0.0)
        w = w.at[-1, :, :].set(0.0)
        w = w.at[:, 0, :].set(0.0)
        w = w.at[:, -1, :].set(0.0)
        w = w.at[:, :, 0].set(0.0)
        w = w.at[:, :, -1].set(0.0)

        return u, v, w
```

### Physics-Informed Neural Networks for CFD
```python
# PINN implementation for CFD problems
import haiku as hk
import optax

class CFD_PINN:
    """Physics-Informed Neural Network for CFD problems."""

    def __init__(
        self,
        layers: list = [4, 50, 50, 50, 4],  # [x,y,z,t] -> [u,v,w,p]
        activation=jax.nn.tanh,
        reynolds_number: float = 100.0
    ):
        self.layers = layers
        self.activation = activation
        self.Re = reynolds_number

    def network_init(self, rng_key):
        """Initialize neural network parameters."""
        def forward_fn(x):
            net = hk.nets.MLP(
                self.layers[1:],
                activation=self.activation,
                name="pinn_net"
            )
            return net(x)

        network = hk.transform(forward_fn)
        dummy_input = jnp.ones((1, self.layers[0]))
        params = network.init(rng_key, dummy_input)

        return network.apply, params

    @functools.partial(jax.jit, static_argnums=(0,))
    def navier_stokes_residual(self, params, network_apply, x_batch):
        """
        Compute Navier-Stokes residual for physics loss.

        Args:
            params: Network parameters
            network_apply: Network forward function
            x_batch: Input coordinates [batch, 4] (x, y, z, t)

        Returns:
            Physics residuals for momentum and continuity equations
        """
        def net_u(x):
            """Network output: [u, v, w, p]"""
            return network_apply(params, x)

        # Compute derivatives using automatic differentiation
        def compute_derivatives(x):
            u_pred = net_u(x)
            u, v, w, p = u_pred[0], u_pred[1], u_pred[2], u_pred[3]

            # First derivatives
            u_x = jax.grad(lambda xi: net_u(xi)[0])(x)
            u_y = jax.grad(lambda xi: net_u(xi)[0])(x)
            u_z = jax.grad(lambda xi: net_u(xi)[0])(x)
            u_t = jax.grad(lambda xi: net_u(xi)[0])(x)

            v_x = jax.grad(lambda xi: net_u(xi)[1])(x)
            v_y = jax.grad(lambda xi: net_u(xi)[1])(x)
            v_z = jax.grad(lambda xi: net_u(xi)[1])(x)
            v_t = jax.grad(lambda xi: net_u(xi)[1])(x)

            w_x = jax.grad(lambda xi: net_u(xi)[2])(x)
            w_y = jax.grad(lambda xi: net_u(xi)[2])(x)
            w_z = jax.grad(lambda xi: net_u(xi)[2])(x)
            w_t = jax.grad(lambda xi: net_u(xi)[2])(x)

            p_x = jax.grad(lambda xi: net_u(xi)[3])(x)
            p_y = jax.grad(lambda xi: net_u(xi)[3])(x)
            p_z = jax.grad(lambda xi: net_u(xi)[3])(x)

            # Second derivatives
            u_xx = jax.grad(jax.grad(lambda xi: net_u(xi)[0]))(x)
            u_yy = jax.grad(jax.grad(lambda xi: net_u(xi)[0]))(x)
            u_zz = jax.grad(jax.grad(lambda xi: net_u(xi)[0]))(x)

            v_xx = jax.grad(jax.grad(lambda xi: net_u(xi)[1]))(x)
            v_yy = jax.grad(jax.grad(lambda xi: net_u(xi)[1]))(x)
            v_zz = jax.grad(jax.grad(lambda xi: net_u(xi)[1]))(x)

            w_xx = jax.grad(jax.grad(lambda xi: net_u(xi)[2]))(x)
            w_yy = jax.grad(jax.grad(lambda xi: net_u(xi)[2]))(x)
            w_zz = jax.grad(jax.grad(lambda xi: net_u(xi)[2]))(x)

            return (u, v, w, p, u_x, u_y, u_z, u_t, v_x, v_y, v_z, v_t,
                   w_x, w_y, w_z, w_t, p_x, p_y, p_z, u_xx, u_yy, u_zz,
                   v_xx, v_yy, v_zz, w_xx, w_yy, w_zz)

        # Vectorize over batch
        batch_derivatives = jax.vmap(compute_derivatives)(x_batch)
        (u, v, w, p, u_x, u_y, u_z, u_t, v_x, v_y, v_z, v_t,
         w_x, w_y, w_z, w_t, p_x, p_y, p_z, u_xx, u_yy, u_zz,
         v_xx, v_yy, v_zz, w_xx, w_yy, w_zz) = batch_derivatives

        # Navier-Stokes equations
        # Momentum equations
        f_u = u_t + u*u_x + v*u_y + w*u_z + p_x - (u_xx + u_yy + u_zz)/self.Re
        f_v = v_t + u*v_x + v*v_y + w*v_z + p_y - (v_xx + v_yy + v_zz)/self.Re
        f_w = w_t + u*w_x + v*w_y + w*w_z + p_z - (w_xx + w_yy + w_zz)/self.Re

        # Continuity equation
        f_c = u_x + v_y + w_z

        return f_u, f_v, f_w, f_c

    def train_pinn(
        self,
        x_physics: jnp.ndarray,
        x_boundary: jnp.ndarray,
        u_boundary: jnp.ndarray,
        x_initial: jnp.ndarray,
        u_initial: jnp.ndarray,
        num_epochs: int = 10000,
        learning_rate: float = 1e-3
    ):
        """
        Train PINN for CFD problem.

        Args:
            x_physics: Physics points [N_p, 4]
            x_boundary: Boundary points [N_b, 4]
            u_boundary: Boundary values [N_b, 4]
            x_initial: Initial condition points [N_i, 4]
            u_initial: Initial condition values [N_i, 4]
            num_epochs: Training epochs
            learning_rate: Learning rate

        Returns:
            Trained parameters and loss history
        """
        # Initialize network
        rng_key = jax.random.PRNGKey(42)
        network_apply, params = self.network_init(rng_key)

        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params):
            """Combined loss function."""
            # Physics loss
            f_u, f_v, f_w, f_c = self.navier_stokes_residual(
                params, network_apply, x_physics
            )
            physics_loss = (
                jnp.mean(f_u**2) + jnp.mean(f_v**2) +
                jnp.mean(f_w**2) + jnp.mean(f_c**2)
            )

            # Boundary loss
            u_pred_boundary = jax.vmap(
                lambda x: network_apply(params, x)
            )(x_boundary)
            boundary_loss = jnp.mean((u_pred_boundary - u_boundary)**2)

            # Initial condition loss
            u_pred_initial = jax.vmap(
                lambda x: network_apply(params, x)
            )(x_initial)
            initial_loss = jnp.mean((u_pred_initial - u_initial)**2)

            total_loss = physics_loss + 10.0 * boundary_loss + 10.0 * initial_loss

            return total_loss, (physics_loss, boundary_loss, initial_loss)

        @jax.jit
        def update_step(params, opt_state):
            """Single optimization step."""
            (loss, losses), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, losses

        # Training loop
        loss_history = []
        for epoch in range(num_epochs):
            params, opt_state, total_loss, (phys_loss, bound_loss, init_loss) = \
                update_step(params, opt_state)

            loss_history.append({
                'total': total_loss,
                'physics': phys_loss,
                'boundary': bound_loss,
                'initial': init_loss
            })

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}")

        return params, loss_history
```

### Large Eddy Simulation (LES) with Neural Subgrid Models
```python
# Neural subgrid-scale model for LES
class NeuralSubgridModel:
    """Neural network subgrid-scale model for LES."""

    def __init__(
        self,
        filter_width: float,
        hidden_dims: list = [10, 64, 64, 6]  # Input: velocity gradients, Output: SGS stress
    ):
        self.filter_width = filter_width
        self.hidden_dims = hidden_dims

    def create_sgs_model(self):
        """Create neural subgrid-scale stress model."""
        def sgs_network(velocity_gradients):
            """
            Neural SGS model.

            Args:
                velocity_gradients: [9] components of velocity gradient tensor

            Returns:
                [6] components of SGS stress tensor (symmetric)
            """
            # Rotation-invariant features
            strain_rate = 0.5 * (velocity_gradients + velocity_gradients.T)
            rotation_rate = 0.5 * (velocity_gradients - velocity_gradients.T)

            # Invariants
            I1 = jnp.trace(strain_rate)
            I2 = 0.5 * (jnp.trace(strain_rate)**2 - jnp.trace(strain_rate @ strain_rate))
            I3 = jnp.linalg.det(strain_rate)

            Q = -0.5 * jnp.trace(rotation_rate @ rotation_rate)
            R = -jnp.linalg.det(rotation_rate)

            # Additional features
            strain_magnitude = jnp.sqrt(2 * jnp.trace(strain_rate @ strain_rate))
            vorticity_magnitude = jnp.sqrt(2 * jnp.trace(rotation_rate @ rotation_rate))

            features = jnp.array([
                I1, I2, I3, Q, R,
                strain_magnitude, vorticity_magnitude,
                self.filter_width * strain_magnitude,
                self.filter_width * vorticity_magnitude,
                strain_magnitude / (vorticity_magnitude + 1e-12)
            ])

            # Neural network
            net = hk.nets.MLP(
                self.hidden_dims[1:],
                activation=jax.nn.swish,
                name="sgs_model"
            )
            sgs_stress = net(features)

            return sgs_stress

        return hk.transform(sgs_network)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_sgs_stress(self, params, network_apply, velocity_field):
        """
        Compute subgrid-scale stress tensor using neural model.

        Args:
            params: Neural network parameters
            network_apply: Network forward function
            velocity_field: Filtered velocity field [3, nx, ny, nz]

        Returns:
            SGS stress tensor [6, nx, ny, nz] (xx, yy, zz, xy, xz, yz)
        """
        def compute_gradients(u, v, w, i, j, k):
            """Compute velocity gradients at grid point (i,j,k)."""
            du_dx = (u[i+1, j, k] - u[i-1, j, k]) / (2 * self.dx)
            du_dy = (u[i, j+1, k] - u[i, j-1, k]) / (2 * self.dy)
            du_dz = (u[i, j, k+1] - u[i, j, k-1]) / (2 * self.dz)

            dv_dx = (v[i+1, j, k] - v[i-1, j, k]) / (2 * self.dx)
            dv_dy = (v[i, j+1, k] - v[i, j-1, k]) / (2 * self.dy)
            dv_dz = (v[i, j, k+1] - v[i, j, k-1]) / (2 * self.dz)

            dw_dx = (w[i+1, j, k] - w[i-1, j, k]) / (2 * self.dx)
            dw_dy = (w[i, j+1, k] - w[i, j-1, k]) / (2 * self.dy)
            dw_dz = (w[i, j, k+1] - w[i, j, k-1]) / (2 * self.dz)

            grad_tensor = jnp.array([
                [du_dx, du_dy, du_dz],
                [dv_dx, dv_dy, dv_dz],
                [dw_dx, dw_dy, dw_dz]
            ])

            return grad_tensor.flatten()

        u, v, w = velocity_field
        nx, ny, nz = u.shape

        # Initialize SGS stress tensor
        sgs_stress = jnp.zeros((6, nx, ny, nz))

        # Compute SGS stress at each interior point
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    gradients = compute_gradients(u, v, w, i, j, k)
                    stress_components = network_apply(params, gradients)
                    sgs_stress = sgs_stress.at[:, i, j, k].set(stress_components)

        return sgs_stress

    def train_sgs_model(
        self,
        dns_data: dict,
        les_data: dict,
        num_epochs: int = 5000
    ):
        """
        Train SGS model on DNS-LES data pairs.

        Args:
            dns_data: High-resolution DNS data
            les_data: Filtered LES data
            num_epochs: Training epochs

        Returns:
            Trained SGS model parameters
        """
        # Extract velocity gradients and true SGS stress
        velocity_gradients = extract_velocity_gradients(les_data['velocity'])
        true_sgs_stress = compute_true_sgs_stress(dns_data, les_data)

        # Initialize network
        sgs_network = self.create_sgs_model()
        rng_key = jax.random.PRNGKey(42)
        params = sgs_network.init(rng_key, velocity_gradients[0])

        # Optimizer
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params):
            """SGS model loss function."""
            pred_stress = jax.vmap(
                lambda grad: sgs_network.apply(params, grad)
            )(velocity_gradients)

            mse_loss = jnp.mean((pred_stress - true_sgs_stress)**2)

            # Physics-based regularization
            # Ensure SGS stress is traceless for incompressible flow
            trace_penalty = jnp.mean(
                (pred_stress[:, 0] + pred_stress[:, 1] + pred_stress[:, 2])**2
            )

            total_loss = mse_loss + 0.1 * trace_penalty
            return total_loss

        @jax.jit
        def update_step(params, opt_state):
            """Training step."""
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Training loop
        for epoch in range(num_epochs):
            params, opt_state, loss = update_step(params, opt_state)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}: SGS Loss = {loss:.6f}")

        return params
```

### Multi-GPU CFD Implementation
```python
# Distributed CFD solver for multi-GPU systems
def setup_distributed_cfd(
    global_grid_shape: tuple,
    num_devices: int = None
):
    """
    Setup distributed CFD solver across multiple GPUs.

    Args:
        global_grid_shape: Global grid dimensions
        num_devices: Number of GPU devices

    Returns:
        Distributed simulation functions
    """
    if num_devices is None:
        num_devices = jax.device_count()

    devices = jax.devices()[:num_devices]

    # Domain decomposition (1D decomposition along x-axis)
    nx_global, ny, nz = global_grid_shape
    nx_local = nx_global // num_devices

    @functools.partial(jax.pmap, axis_name='devices')
    def distributed_cfd_step(
        local_velocity, local_pressure, halo_data
    ):
        """Distributed CFD step with halo exchange."""
        # Update halos from neighboring domains
        velocity_with_halos = update_halos(local_velocity, halo_data)
        pressure_with_halos = update_halos(local_pressure, halo_data)

        # Perform local CFD step
        new_velocity, new_pressure = cfd_solver.navier_stokes_step(
            velocity_with_halos, pressure_with_halos
        )

        # Extract interior (no halo) solution
        interior_velocity = extract_interior(new_velocity)
        interior_pressure = extract_interior(new_pressure)

        return interior_velocity, interior_pressure

    def halo_exchange(field_data):
        """Exchange halo data between neighboring processes."""
        # Left-right neighbor communication
        left_halo = lax.ppermute(
            field_data[-1:, :, :],  # Right boundary
            'devices',
            [(i, (i-1) % num_devices) for i in range(num_devices)]
        )

        right_halo = lax.ppermute(
            field_data[:1, :, :],   # Left boundary
            'devices',
            [(i, (i+1) % num_devices) for i in range(num_devices)]
        )

        return left_halo, right_halo

    return distributed_cfd_step, halo_exchange

# Adaptive mesh refinement with JAX
class AdaptiveMeshRefinement:
    """JAX-based adaptive mesh refinement for CFD."""

    def __init__(self, base_grid_shape, max_refinement_levels=3):
        self.base_shape = base_grid_shape
        self.max_levels = max_refinement_levels
        self.refinement_threshold = 0.1

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_refinement_indicator(self, velocity_field, pressure_field):
        """
        Compute refinement indicator based on solution gradients.

        Args:
            velocity_field: Velocity field [3, nx, ny, nz]
            pressure_field: Pressure field [nx, ny, nz]

        Returns:
            Refinement indicator field
        """
        u, v, w = velocity_field

        # Compute velocity magnitude
        velocity_magnitude = jnp.sqrt(u**2 + v**2 + w**2)

        # Compute gradients
        grad_u = jnp.gradient(velocity_magnitude)
        grad_p = jnp.gradient(pressure_field)

        # Refinement indicator based on gradient magnitude
        indicator = (
            jnp.sqrt(sum(g**2 for g in grad_u)) +
            jnp.sqrt(sum(g**2 for g in grad_p))
        )

        # Normalize indicator
        max_indicator = jnp.max(indicator)
        normalized_indicator = indicator / (max_indicator + 1e-12)

        return normalized_indicator

    def refine_grid(self, refinement_indicator):
        """
        Perform grid refinement based on indicator.

        Args:
            refinement_indicator: Refinement indicator field

        Returns:
            Refined grid structure and interpolation operators
        """
        # Identify cells to refine
        refine_mask = refinement_indicator > self.refinement_threshold

        # Create refined grid structure
        refined_grid = self._create_refined_grid(refine_mask)

        # Create interpolation operators
        interpolation_ops = self._create_interpolation_operators(refined_grid)

        return refined_grid, interpolation_ops

    def _create_refined_grid(self, refine_mask):
        """Create hierarchical refined grid structure."""
        # Implementation of hierarchical grid creation
        # This would involve complex grid data structures
        pass

    def _create_interpolation_operators(self, refined_grid):
        """Create interpolation operators between grid levels."""
        # Implementation of multigrid interpolation operators
        pass
```

## Integration with Scientific Workflow

### JAX-CFD + Machine Learning Integration
- **Neural Turbulence Models**: Integration of data-driven turbulence closures with traditional CFD solvers
- **Reduced-Order Modeling**: Neural network-based ROM for real-time flow prediction
- **Optimization**: Adjoint-based shape optimization using automatic differentiation

### High-Performance Computing
- **GPU Acceleration**: Optimized CUDA kernels for CFD operations and memory management
- **Distributed Computing**: Scalable multi-GPU and multi-node CFD implementations
- **Adaptive Methods**: Dynamic grid adaptation and load balancing strategies

### Scientific Applications
- **Aerospace**: Compressible flow, shock-boundary layer interactions, hypersonic flows
- **Energy**: Turbomachinery, wind turbine aerodynamics, combustion modeling
- **Environmental**: Atmospheric flows, ocean circulation, pollution dispersion

## Usage Examples

### Basic CFD Simulation
```python
# Setup and run incompressible flow simulation
grid_shape = (128, 128, 64)
domain_size = (2.0, 1.0, 1.0)
solver = JAXCFDSolver(grid_shape, domain_size, viscosity=1e-3)

# Initialize flow field
velocity_field = initialize_velocity_field(grid_shape)
pressure_field = jnp.zeros(grid_shape)

# Time integration
for step in range(10000):
    velocity_field, pressure_field = solver.navier_stokes_step(
        velocity_field, pressure_field
    )

    if step % 100 == 0:
        save_flow_field(velocity_field, pressure_field, f"step_{step}.h5")
```

### PINN Training for CFD
```python
# Train PINN for flow around cylinder
pinn = CFD_PINN(reynolds_number=100.0)
x_physics = generate_physics_points(domain, n_points=10000)
x_boundary, u_boundary = generate_boundary_conditions(cylinder_geometry)
x_initial, u_initial = generate_initial_conditions()

trained_params, loss_history = pinn.train_pinn(
    x_physics, x_boundary, u_boundary, x_initial, u_initial
)

print(f"Final loss: {loss_history[-1]['total']:.6f}")
```

### Neural Subgrid Model Training
```python
# Train neural SGS model from DNS data
sgs_model = NeuralSubgridModel(filter_width=0.1)
dns_data = load_dns_database("turbulent_channel_flow")
les_data = apply_spatial_filter(dns_data, filter_width=0.1)

trained_sgs_params = sgs_model.train_sgs_model(dns_data, les_data)
print("SGS model training completed")
```

This expert provides comprehensive JAX-based computational fluid dynamics capabilities with machine learning integration, high-performance computing optimization, and advanced numerical methods for scientific applications.