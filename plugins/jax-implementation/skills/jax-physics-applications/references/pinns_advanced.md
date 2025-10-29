# Advanced PINN Techniques

## Adaptive Loss Weighting

Automatically balance multiple loss terms during training:

```python
def adaptive_loss_weights(losses_dict, step, tau=1000):
    """
    Adaptive weights using exponential moving average of loss gradients
    """
    # Compute gradient norms for each loss component
    grad_norms = {
        key: jnp.linalg.norm(jax.grad(lambda: loss)(model))
        for key, loss in losses_dict.items()
    }

    # Normalize to sum to 1
    total = sum(grad_norms.values())
    weights = {key: norm / total for key, norm in grad_norms.items()}

    return weights

# Training with adaptive weighting
for epoch in range(10000):
    losses = {
        'pde': compute_pde_loss(model, x_pde, t_pde),
        'bc': compute_bc_loss(model, x_bc, t_bc),
        'ic': compute_ic_loss(model, x_ic, u_ic)
    }

    weights = adaptive_loss_weights(losses, epoch)
    total_loss = sum(w * losses[k] for k, w in weights.items())

    grads = jax.grad(lambda: total_loss)(model)
    optimizer.update(grads)
```

## Stiff PDEs

For stiff problems (e.g., reaction-diffusion), use causal training:

```python
def causal_pinn_training(model, t_intervals, x_domain):
    """Train sequentially in time for stiff problems"""

    for t_start, t_end in t_intervals:
        # Focus on current time window
        t_collocation = jax.random.uniform(key, (N,), minval=t_start, maxval=t_end)

        # Use solution from previous window as IC
        if t_start > 0:
            u_prev = model(x_domain, t_start)
            ic_loss = jnp.mean((model(x_domain, t_start) - u_prev)**2)
        else:
            ic_loss = 0.0

        # Train on this window
        for _ in range(1000):
            pde_loss = compute_pde_residual(model, x_domain, t_collocation)
            loss = pde_loss + ic_loss
            optimizer.update(jax.grad(loss))
```

## Multi-Physics Coupling

Couple multiple PDEs (e.g., Navier-Stokes + Heat Transfer):

```python
class CoupledPhysicsPINN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        # Velocity network
        self.velocity_net = MLP([128, 128, 2], rngs=rngs)  # u, v

        # Temperature network
        self.temp_net = MLP([128, 128, 1], rngs=rngs)  # T

    def __call__(self, x, y, t):
        inputs = jnp.stack([x, y, t], axis=-1)
        u, v = self.velocity_net(inputs)
        T = self.temp_net(inputs)
        return u, v, T

def coupled_loss(model, x, y, t):
    """Navier-Stokes + Heat equation"""
    u, v, T = model(x, y, t)

    # NS residual
    ns_residual = navier_stokes_residual(u, v, x, y, t)

    # Heat equation with advection
    T_t = jax.grad(lambda t: model(x, y, t)[2])(t)
    T_x = jax.grad(lambda x: model(x, y, t)[2])(x)
    T_y = jax.grad(lambda y: model(x, y, t)[2])(y)
    T_xx = jax.grad(jax.grad(lambda x: model(x, y, t)[2]))(x)
    T_yy = jax.grad(jax.grad(lambda y: model(x, y, t)[2]))(y)

    heat_residual = T_t + u*T_x + v*T_y - alpha*(T_xx + T_yy)

    return jnp.mean(ns_residual**2) + jnp.mean(heat_residual**2)
```

## Transfer Learning

Pre-train on similar geometries and fine-tune:

```python
# Train on simple geometry
model_simple = train_pinn(geometry='square', pde='poisson')

# Transfer to complex geometry
model_complex = HeatPINN(rngs=nnx.Rngs(1))

# Copy weights from simple model
model_complex.dense1 = model_simple.dense1  # Freeze early layers
model_complex.dense2 = model_simple.dense2

# Fine-tune on complex geometry
for param in model_complex.dense3.parameters():
    param.requires_grad = True  # Only train final layers

optimizer = nnx.Optimizer(model_complex, optax.adam(1e-4))
train(model_complex, geometry='complex', epochs=1000)
```

## Uncertainty Quantification

Use Monte Carlo dropout for epistemic uncertainty:

```python
class PINNWithDropout(nnx.Module):
    def __init__(self, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(2, 128, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dense2 = nnx.Linear(128, 128, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)
        self.dense3 = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x, t, training=False):
        h = nnx.relu(self.dense1(jnp.stack([x, t], axis=-1)))
        h = self.dropout1(h, deterministic=not training)
        h = nnx.relu(self.dense2(h))
        h = self.dropout2(h, deterministic=not training)
        return self.dense3(h)

# Inference with uncertainty
def predict_with_uncertainty(model, x, t, n_samples=100):
    """MC Dropout for uncertainty estimates"""
    predictions = []

    for _ in range(n_samples):
        u = model(x, t, training=True)  # Dropout enabled
        predictions.append(u)

    predictions = jnp.stack(predictions)
    mean = jnp.mean(predictions, axis=0)
    std = jnp.std(predictions, axis=0)

    return mean, std
```
