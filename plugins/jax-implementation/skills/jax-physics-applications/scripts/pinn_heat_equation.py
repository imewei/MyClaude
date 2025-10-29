#!/usr/bin/env python3
"""
Complete Physics-Informed Neural Network for Heat Equation

This script demonstrates a full PINN workflow:
1. Define PDE: ∂u/∂t = α∇²u with boundary and initial conditions
2. Create neural network architecture
3. Design physics-informed loss function
4. Train with automatic differentiation
5. Validate against analytical solution

Problem: 1D heat equation on domain [0,1] × [0,1]
  PDE: ∂u/∂t = α∂²u/∂x²
  BC: u(0,t) = 0, u(1,t) = 0
  IC: u(x,0) = sin(πx)

Analytical solution: u(x,t) = exp(-απ²t)sin(πx)

Usage:
    python pinn_heat_equation.py

Requirements:
    pip install jax flax optax matplotlib
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import matplotlib.pyplot as plt


def main():
    # Configuration
    hidden_dim = 64
    n_layers = 3
    learning_rate = 1e-3
    n_epochs = 10000
    n_collocation = 1000
    n_boundary = 100
    n_initial = 100
    alpha = 0.01  # Thermal diffusivity

    print("=" * 60)
    print("Physics-Informed Neural Network for Heat Equation")
    print("=" * 60)
    print(f"PDE: ∂u/∂t = {alpha}∂²u/∂x²")
    print(f"Domain: x ∈ [0,1], t ∈ [0,1]")
    print(f"Hidden dimensions: {hidden_dim}")
    print(f"Number of layers: {n_layers}")
    print(f"Learning rate: {learning_rate}")
    print()

    # 1. Create neural network
    print("Initializing neural network...")
    model = HeatPINN(hidden_dim=hidden_dim, n_layers=n_layers, rngs=nnx.Rngs(0))
    print(f"  Model parameters: {count_parameters(model)}")
    print()

    # 2. Setup optimizer
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    # 3. Training loop
    print("Starting training...")
    losses_history = {'total': [], 'pde': [], 'bc': [], 'ic': []}

    for epoch in range(n_epochs):
        # Sample collocation points
        key = jax.random.PRNGKey(epoch)
        x_pde, t_pde = sample_collocation_points(key, n_collocation)
        x_bc, t_bc = sample_boundary_points(key, n_boundary)
        x_ic, t_ic, u_ic = sample_initial_conditions(key, n_initial)

        # Compute loss and gradients
        loss_grad_fn = nnx.value_and_grad(pinn_loss, has_aux=True)
        (total_loss, losses), grads = loss_grad_fn(
            model, x_pde, t_pde, x_bc, t_bc, x_ic, t_ic, u_ic, alpha
        )

        # Update parameters
        optimizer.update(grads)

        # Record losses
        losses_history['total'].append(total_loss)
        losses_history['pde'].append(losses['pde'])
        losses_history['bc'].append(losses['bc'])
        losses_history['ic'].append(losses['ic'])

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"Loss={total_loss:.6f}, "
                  f"PDE={losses['pde']:.6f}, "
                  f"BC={losses['bc']:.6f}, "
                  f"IC={losses['ic']:.6f}")

    print("✓ Training complete\n")

    # 4. Validation against analytical solution
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Generate test points
    n_test = 100
    x_test = jnp.linspace(0, 1, n_test)
    t_test = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    max_errors = []
    mean_errors = []

    for t_val in t_test:
        # Predict with PINN
        u_pred = jax.vmap(lambda x: model(x, t_val))(x_test)

        # Analytical solution
        u_exact = analytical_solution(x_test, t_val, alpha)

        # Compute error
        error = jnp.abs(u_pred.squeeze() - u_exact)
        max_error = jnp.max(error)
        mean_error = jnp.mean(error)

        max_errors.append(max_error)
        mean_errors.append(mean_error)

        print(f"\nt = {t_val:.2f}:")
        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")

    overall_max_error = jnp.max(jnp.array(max_errors))
    overall_mean_error = jnp.mean(jnp.array(mean_errors))

    print(f"\nOverall Statistics:")
    print(f"  Max error: {overall_max_error:.6f}")
    print(f"  Mean error: {overall_mean_error:.6f}")

    if overall_max_error < 0.01:
        print("  ✓ Solution accuracy: EXCELLENT (error < 1%)")
    elif overall_max_error < 0.05:
        print("  ✓ Solution accuracy: GOOD (error < 5%)")
    else:
        print("  ⚠ Solution accuracy: POOR (consider more training)")

    # Validate PDE residual on test points
    print(f"\nPDE Residual Validation:")
    x_test_grid, t_test_grid = jnp.meshgrid(
        jnp.linspace(0.1, 0.9, 20),
        jnp.linspace(0.1, 0.9, 20)
    )
    x_flat = x_test_grid.flatten()
    t_flat = t_test_grid.flatten()

    residuals = jax.vmap(lambda x, t: compute_pde_residual(model, x, t, alpha))(x_flat, t_flat)
    max_residual = jnp.max(jnp.abs(residuals))
    mean_residual = jnp.mean(jnp.abs(residuals))

    print(f"  Max |residual|: {max_residual:.6f}")
    print(f"  Mean |residual|: {mean_residual:.6f}")

    if max_residual < 1e-3:
        print("  ✓ PDE satisfaction: EXCELLENT (residual < 0.001)")
    elif max_residual < 1e-2:
        print("  ✓ PDE satisfaction: GOOD (residual < 0.01)")
    else:
        print("  ⚠ PDE satisfaction: POOR")

    # 5. Visualization
    print(f"\nGenerating plots...")
    fig = plt.figure(figsize=(16, 10))

    # Loss evolution
    ax1 = plt.subplot(2, 3, 1)
    ax1.semilogy(losses_history['total'], 'b-', label='Total', linewidth=2)
    ax1.semilogy(losses_history['pde'], 'r--', label='PDE', alpha=0.7)
    ax1.semilogy(losses_history['bc'], 'g--', label='BC', alpha=0.7)
    ax1.semilogy(losses_history['ic'], 'm--', label='IC', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Solution at different times
    ax2 = plt.subplot(2, 3, 2)
    x_plot = jnp.linspace(0, 1, 200)

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        u_pred = jax.vmap(lambda x: model(x, t_val))(x_plot)
        u_exact = analytical_solution(x_plot, t_val, alpha)

        ax2.plot(x_plot, u_pred.squeeze(), '-', label=f't={t_val:.2f} (PINN)', alpha=0.7)
        ax2.plot(x_plot, u_exact, '--', label=f't={t_val:.2f} (Exact)', alpha=0.7)

    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x,t)')
    ax2.set_title('Solution Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Error heatmap
    ax3 = plt.subplot(2, 3, 3)
    x_mesh = jnp.linspace(0, 1, 100)
    t_mesh = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x_mesh, t_mesh)

    U_pred = jnp.array([[model(x, t).item() for x in x_mesh] for t in t_mesh])
    U_exact = analytical_solution(X, T, alpha)
    error_field = jnp.abs(U_pred - U_exact)

    im = ax3.contourf(X, T, error_field, levels=20, cmap='hot')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_title('Absolute Error |u_pred - u_exact|')
    plt.colorbar(im, ax=ax3)

    # Solution heatmap (PINN)
    ax4 = plt.subplot(2, 3, 4)
    im = ax4.contourf(X, T, U_pred, levels=20, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_title('PINN Solution u(x,t)')
    plt.colorbar(im, ax=ax4)

    # Solution heatmap (Exact)
    ax5 = plt.subplot(2, 3, 5)
    im = ax5.contourf(X, T, U_exact, levels=20, cmap='viridis')
    ax5.set_xlabel('x')
    ax5.set_ylabel('t')
    ax5.set_title('Analytical Solution u(x,t)')
    plt.colorbar(im, ax=ax5)

    # PDE residual heatmap
    ax6 = plt.subplot(2, 3, 6)
    residual_field = jnp.array([[compute_pde_residual(model, x, t, alpha)
                                 for x in x_mesh[::5]] for t in t_mesh[::5]])
    im = ax6.contourf(x_mesh[::5], t_mesh[::5], jnp.abs(residual_field),
                      levels=20, cmap='hot')
    ax6.set_xlabel('x')
    ax6.set_ylabel('t')
    ax6.set_title('PDE Residual |∂u/∂t - α∂²u/∂x²|')
    plt.colorbar(im, ax=ax6)

    plt.tight_layout()
    plt.savefig('pinn_heat_equation_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: pinn_heat_equation_results.png")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


class HeatPINN(nnx.Module):
    """Physics-Informed Neural Network for Heat Equation"""

    def __init__(self, hidden_dim=64, n_layers=3, *, rngs: nnx.Rngs):
        self.layers = []

        # Input layer
        self.layers.append(nnx.Linear(2, hidden_dim, rngs=rngs))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nnx.Linear(hidden_dim, hidden_dim, rngs=rngs))

        # Output layer
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x, t):
        """Forward pass: (x, t) -> u(x, t)"""
        # Stack inputs
        xt = jnp.array([x, t])

        # Forward through layers with tanh activation
        h = xt
        for layer in self.layers:
            h = jnp.tanh(layer(h))

        # Output
        u = self.output(h)

        return u


def pinn_loss(model, x_pde, t_pde, x_bc, t_bc, x_ic, t_ic, u_ic, alpha):
    """Physics-informed loss = PDE residual + BC + IC"""

    # PDE residual loss
    residuals = jax.vmap(lambda x, t: compute_pde_residual(model, x, t, alpha))(x_pde, t_pde)
    loss_pde = jnp.mean(residuals ** 2)

    # Boundary condition loss: u(0,t) = 0, u(1,t) = 0
    u_bc = jax.vmap(model)(x_bc, t_bc)
    loss_bc = jnp.mean(u_bc ** 2)

    # Initial condition loss
    u_ic_pred = jax.vmap(model)(x_ic, t_ic)
    loss_ic = jnp.mean((u_ic_pred.squeeze() - u_ic) ** 2)

    # Combined loss with weighting
    total_loss = loss_pde + 100 * loss_bc + 100 * loss_ic

    losses = {
        'pde': loss_pde,
        'bc': loss_bc,
        'ic': loss_ic
    }

    return total_loss, losses


def compute_pde_residual(model, x, t, alpha):
    """Compute PDE residual: ∂u/∂t - α∂²u/∂x²"""

    # Automatic differentiation for derivatives
    def u_fn(x_val, t_val):
        return model(x_val, t_val)

    # First derivatives
    u_t = jax.grad(u_fn, argnums=1)(x, t)
    u_x = jax.grad(u_fn, argnums=0)(x, t)

    # Second derivative
    u_xx = jax.grad(lambda x_val: jax.grad(u_fn, argnums=0)(x_val, t))(x)

    # PDE residual
    residual = u_t - alpha * u_xx

    return residual


def sample_collocation_points(key, n_points):
    """Sample random points in domain for PDE residual"""
    key1, key2 = jax.random.split(key)
    x = jax.random.uniform(key1, (n_points,), minval=0.0, maxval=1.0)
    t = jax.random.uniform(key2, (n_points,), minval=0.0, maxval=1.0)
    return x, t


def sample_boundary_points(key, n_points):
    """Sample boundary points: x=0 and x=1"""
    key1, key2 = jax.random.split(key)
    t = jax.random.uniform(key1, (n_points//2,), minval=0.0, maxval=1.0)

    x_left = jnp.zeros(n_points//2)
    x_right = jnp.ones(n_points//2)

    x = jnp.concatenate([x_left, x_right])
    t = jnp.concatenate([t, jax.random.uniform(key2, (n_points//2,), minval=0.0, maxval=1.0)])

    return x, t


def sample_initial_conditions(key, n_points):
    """Sample initial condition: t=0, u(x,0)=sin(πx)"""
    x = jax.random.uniform(key, (n_points,), minval=0.0, maxval=1.0)
    t = jnp.zeros(n_points)
    u = jnp.sin(jnp.pi * x)
    return x, t, u


def analytical_solution(x, t, alpha):
    """Analytical solution: u(x,t) = exp(-απ²t)sin(πx)"""
    return jnp.exp(-alpha * jnp.pi**2 * t) * jnp.sin(jnp.pi * x)


def count_parameters(model):
    """Count total number of parameters in model"""
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


if __name__ == '__main__':
    main()
