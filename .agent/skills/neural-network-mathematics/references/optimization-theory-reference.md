# allow-torch
# Optimization Theory for Neural Networks - Complete Reference

Essential optimization theory for training neural networks, from gradient descent fundamentals to advanced optimization algorithms.

---

## Table of Contents

1. [Optimization Fundamentals](#optimization-fundamentals)
2. [Gradient Descent Variants](#gradient-descent-variants)
3. [Adaptive Learning Rate Methods](#adaptive-learning-rate-methods)
4. [Second-Order Methods](#second-order-methods)
5. [Learning Rate Schedules](#learning-rate-schedules)
6. [Gradient Problems](#gradient-problems)
7. [Loss Landscape Analysis](#loss-landscape-analysis)

---

## Optimization Fundamentals

### The Optimization Problem

**Objective:**
```
θ* = argmin_θ L(θ)

where:
- θ: model parameters (weights and biases)
- L(θ): loss function
- θ*: optimal parameters
```

**Empirical Risk Minimization:**
```
L(θ) = (1/N) Σᵢ₌₁ᴺ ℓ(f(xᵢ; θ), yᵢ)

where:
- N: number of training examples
- ℓ: per-example loss
- f(x; θ): model prediction
- (xᵢ, yᵢ): training data
```

### Gradient Descent

**Basic Algorithm:**
```
θₜ₊₁ = θₜ - η∇L(θₜ)

where:
- η: learning rate (step size)
- ∇L(θₜ): gradient at current parameters
```

**Convergence Conditions:**

For convex L with Lipschitz continuous gradient:
```
If η ≤ 1/L (L is Lipschitz constant):
- Guaranteed convergence to global minimum
- Convergence rate: O(1/t) for convex functions
```

For non-convex (neural networks):
```
- Converges to critical points (∇L(θ) = 0)
- May be local minima, saddle points, or global minima
- No theoretical guarantees, but works empirically
```

**Python Implementation:**

```python
import torch
import torch.nn as nn

def gradient_descent(model: nn.Module,
                     loss_fn,
                     X: torch.Tensor,
                     y: torch.Tensor,
                     learning_rate: float = 0.01,
                     num_iterations: int = 1000):
    """
    Vanilla gradient descent optimizer.
    """
    losses = []

    for iteration in range(num_iterations):
        # Forward pass
        predictions = model(X)
        loss = loss_fn(predictions, y)

        # Backward pass (compute gradients)
        loss.backward()

        # Update parameters manually
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        # Zero gradients for next iteration
        model.zero_grad()

        losses.append(loss.item())

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")

    return losses
```

---

## Gradient Descent Variants

### Batch Gradient Descent

**Algorithm:**
```
Compute gradient using entire dataset:
θₜ₊₁ = θₜ - η∇L(θₜ)

where ∇L(θₜ) = (1/N) Σᵢ₌₁ᴺ ∇ℓᵢ(θₜ)
```

**Properties:**
- Exact gradient computation
- Stable convergence
- Slow for large datasets (compute entire dataset per update)

### Stochastic Gradient Descent (SGD)

**Algorithm:**
```
Sample random example i:
θₜ₊₁ = θₜ - η∇ℓᵢ(θₜ)
```

**Properties:**
- Fast per-iteration updates
- Noisy gradient estimates → exploration of loss landscape
- Can escape shallow local minima
- Requires learning rate decay for convergence

**Python Implementation:**

```python
import torch.optim as optim

# Standard SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()
```

### Mini-Batch Gradient Descent

**Algorithm:**
```
Sample mini-batch B of size m:
θₜ₊₁ = θₜ - η(1/m) Σᵢ∈B ∇ℓᵢ(θₜ)
```

**Batch Size Effects:**

```python
# Common batch sizes and their tradeoffs
batch_size_effects = {
    "small (32-64)": {
        "pros": "Better generalization, escapes sharp minima",
        "cons": "Noisy gradients, slower convergence"
    },
    "medium (128-256)": {
        "pros": "Balanced noise/stability, good GPU utilization",
        "cons": "Moderate memory usage"
    },
    "large (512+)": {
        "pros": "Stable gradients, faster training (fewer updates)",
        "cons": "Worse generalization, requires larger learning rates"
    }
}
```

**Linear Scaling Rule:**
```
When increasing batch size by k:
Scale learning rate by k

Example:
- Batch size 128, lr=0.1 → Batch size 256, lr=0.2
```

### Momentum

**Algorithm:**
```
vₜ = βvₜ₋₁ + ∇L(θₜ)
θₜ₊₁ = θₜ - ηvₜ

where:
- v: velocity (momentum term)
- β: momentum coefficient (typically 0.9)
```

**Intuition:**
- Accumulates past gradients
- Accelerates in consistent directions
- Dampens oscillations in high-curvature directions

**Nesterov Accelerated Gradient (NAG):**
```
vₜ = βvₜ₋₁ + ∇L(θₜ - ηβvₜ₋₁)  # Look-ahead gradient
θₜ₊₁ = θₜ - ηvₜ
```

**Python Implementation:**

```python
# SGD with momentum
optimizer = optim.SGD(model.parameters(),
                     lr=0.01,
                     momentum=0.9)

# SGD with Nesterov momentum
optimizer = optim.SGD(model.parameters(),
                     lr=0.01,
                     momentum=0.9,
                     nesterov=True)
```

---

## Adaptive Learning Rate Methods

### AdaGrad

**Algorithm:**
```
gₜ = ∇L(θₜ)
Gₜ = Gₜ₋₁ + gₜ ⊙ gₜ  # Accumulate squared gradients
θₜ₊₁ = θₜ - η/(√Gₜ + ε) ⊙ gₜ

where:
- Gₜ: sum of squared gradients (element-wise)
- ε: small constant for numerical stability (1e-8)
- ⊙: element-wise multiplication
```

**Properties:**
- Adapts learning rate per parameter
- Larger updates for infrequent features, smaller for frequent
- Learning rate decreases monotonically → may stop too early

### RMSProp

**Algorithm:**
```
gₜ = ∇L(θₜ)
Eₜ = βEₜ₋₁ + (1-β)gₜ ⊙ gₜ  # Exponential moving average
θₜ₊₁ = θₜ - η/(√Eₜ + ε) ⊙ gₜ

where:
- β: decay rate (typically 0.9 or 0.99)
- Eₜ: running average of squared gradients
```

**Properties:**
- Fixes AdaGrad's monotonic decay
- Maintains moving average instead of sum
- Works well for RNNs and non-stationary objectives

**Python Implementation:**

```python
optimizer = optim.RMSprop(model.parameters(),
                         lr=0.01,
                         alpha=0.99,  # decay rate
                         eps=1e-8)
```

### Adam (Adaptive Moment Estimation)

**Algorithm:**
```
gₜ = ∇L(θₜ)

# First moment estimate (mean)
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ

# Second moment estimate (variance)
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²

# Bias correction
m̂ₜ = mₜ/(1-β₁ᵗ)
v̂ₜ = vₜ/(1-β₂ᵗ)

# Update
θₜ₊₁ = θₜ - η·m̂ₜ/(√v̂ₜ + ε)

where:
- β₁: first moment decay (typically 0.9)
- β₂: second moment decay (typically 0.999)
- ε: numerical stability (1e-8)
```

**Why Bias Correction?**
```
At t=0: m₀ = 0, v₀ = 0
Without correction: m₁ ≈ 0, v₁ ≈ 0 (biased towards 0)
With correction: m̂₁ and v̂₁ are unbiased estimates
```

**Python Implementation:**

```python
optimizer = optim.Adam(model.parameters(),
                      lr=0.001,
                      betas=(0.9, 0.999),
                      eps=1e-8,
                      weight_decay=0.0)  # L2 regularization

# Common practice: use default parameters
optimizer = optim.Adam(model.parameters())
```

### AdamW (Adam with Decoupled Weight Decay)

**Algorithm:**
```
Same as Adam, but weight decay applied directly to parameters:

θₜ₊₁ = θₜ - η·m̂ₜ/(√v̂ₜ + ε) - λθₜ

where:
- λ: weight decay coefficient
```

**Why Better than Adam + L2?**
```
Adam + L2: Weight decay affects adaptive learning rates
AdamW: Weight decay independent of gradient-based adaptation
→ Better generalization in practice
```

**Python Implementation:**

```python
optimizer = optim.AdamW(model.parameters(),
                       lr=0.001,
                       betas=(0.9, 0.999),
                       eps=1e-8,
                       weight_decay=0.01)  # Recommended: 0.01-0.1
```

### Comparison Table

| Optimizer | Learning Rate | Momentum | Adaptive | Best For |
|-----------|--------------|----------|----------|----------|
| SGD | Fixed | Optional | No | Simple models, careful tuning |
| SGD+Momentum | Fixed | Yes | No | CNNs, well-tuned systems |
| RMSProp | Adaptive | No | Yes | RNNs, non-stationary problems |
| Adam | Adaptive | Yes | Yes | General purpose, quick prototyping |
| AdamW | Adaptive | Yes | Yes | Transformers, modern architectures |

---

## Second-Order Methods

### Newton's Method

**Algorithm:**
```
θₜ₊₁ = θₜ - H⁻¹∇L(θₜ)

where:
- H: Hessian matrix (second derivatives)
- H = ∇²L(θ) = [∂²L/∂θᵢ∂θⱼ]
```

**Properties:**
- Uses curvature information
- Faster convergence near minima
- Computationally prohibitive for neural networks:
  - O(n²) memory to store Hessian
  - O(n³) computation to invert Hessian

### L-BFGS (Limited-memory BFGS)

**Algorithm:**
```
Approximate H⁻¹ using recent gradients:
θₜ₊₁ = θₜ - ηHₜ⁻¹∇L(θₜ)

Hₜ⁻¹ approximated using last m gradient differences
```

**Properties:**
- Quasi-Newton method (approximates Hessian)
- O(mn) memory (m typically 10-20)
- Works for small models, full-batch training
- Not suitable for mini-batch SGD

**Python Implementation:**

```python
# L-BFGS for small models
optimizer = optim.LBFGS(model.parameters(),
                       lr=1.0,
                       max_iter=20,
                       history_size=10)

# Closure required for line search
def closure():
    optimizer.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, y)
    loss.backward()
    return loss

optimizer.step(closure)
```

### Natural Gradient Descent

**Algorithm:**
```
θₜ₊₁ = θₜ - ηF⁻¹∇L(θₜ)

where:
- F: Fisher information matrix
- F = E[∇log p(y|x,θ) · ∇log p(y|x,θ)ᵀ]
```

**Properties:**
- Invariant to parameter reparametrization
- More stable than Newton's method
- Still computationally expensive

---

## Learning Rate Schedules

### Step Decay

**Schedule:**
```
η(t) = η₀ · γ^⌊t/s⌋

where:
- η₀: initial learning rate
- γ: decay factor (e.g., 0.1)
- s: step size (number of epochs)
```

**Python Implementation:**

```python
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                     step_size=30,
                                     gamma=0.1)

# In training loop
for epoch in range(num_epochs):
    train_epoch()
    scheduler.step()  # Update learning rate
```

### Exponential Decay

**Schedule:**
```
η(t) = η₀ · e^(-λt)

where:
- λ: decay rate
```

**Python Implementation:**

```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                            gamma=0.95)
```

### Cosine Annealing

**Schedule:**
```
η(t) = η_min + (η_max - η_min)/2 · (1 + cos(πt/T))

where:
- T: total number of iterations
- η_min: minimum learning rate
- η_max: maximum learning rate
```

**Properties:**
- Smooth decay
- Popular for transformers and modern architectures
- Can be combined with warm restarts

**Python Implementation:**

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                T_max=num_epochs,
                                                eta_min=1e-6)

# With warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # First restart after 10 epochs
    T_mult=2,  # Double period after each restart
    eta_min=1e-6
)
```

### Warmup + Decay

**Schedule:**
```
Warmup (t ≤ t_warmup):
    η(t) = η_max · (t/t_warmup)

Decay (t > t_warmup):
    η(t) = η_max · decay_schedule(t - t_warmup)
```

**Why Warmup?**
- Adam's bias correction not perfect at start
- Large learning rates can destabilize training initially
- Common in transformer training

**Python Implementation:**

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

scheduler = warmup_cosine_schedule(optimizer,
                                  num_warmup_steps=1000,
                                  num_training_steps=10000)
```

### One Cycle Policy

**Schedule:**
```
Phase 1 (0 to 50%): Linear warmup from η_min to η_max
Phase 2 (50% to 90%): Cosine decay from η_max to η_min
Phase 3 (90% to 100%): Further decay to very small η

Momentum inversely varies with learning rate
```

**Python Implementation:**

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=num_training_steps,
    pct_start=0.3,  # Warmup percentage
    anneal_strategy='cos'
)
```

---

## Gradient Problems

### Vanishing Gradients

**Problem:**
```
For deep networks:
∂L/∂θₗ = ∂L/∂zₙ · ∂zₙ/∂zₙ₋₁ · ... · ∂zₗ₊₁/∂zₗ · ∂zₗ/∂θₗ

If |∂zᵢ/∂zᵢ₋₁| < 1 for all i:
Product → 0 as n-l increases
```

**Causes:**
- Sigmoid/tanh activations (saturate, |derivative| < 0.25)
- Deep networks (many layers to backprop through)
- Poor weight initialization

**Solutions:**

1. **ReLU Activations:**
```python
# ReLU: f(x) = max(0, x)
# Derivative: f'(x) = 1 if x > 0, else 0
# No saturation for positive values
activation = nn.ReLU()
```

2. **Batch Normalization:**
```python
# Normalizes activations, prevents saturation
bn = nn.BatchNorm1d(num_features)
```

3. **Residual Connections:**
```python
# Skip connections allow gradient to flow directly
class ResidualBlock(nn.Module):
    def forward(self, x):
        return x + self.layers(x)  # Identity + transformation
```

4. **Better Initialization:**
```python
# He initialization for ReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### Exploding Gradients

**Problem:**
```
If |∂zᵢ/∂zᵢ₋₁| > 1 for all i:
Product → ∞ as n-l increases
```

**Detection:**

```python
def check_gradients(model):
    """Check for exploding/vanishing gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > 100:
        print(f"⚠️ Exploding gradients detected: norm={total_norm:.2f}")
    elif total_norm < 1e-5:
        print(f"⚠️ Vanishing gradients detected: norm={total_norm:.2e}")

    return total_norm
```

**Solutions:**

1. **Gradient Clipping:**
```python
# Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# In training loop
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

2. **Lower Learning Rate:**
```python
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Smaller lr
```

3. **Better Architecture:**
- Use skip connections (ResNets)
- Use layer normalization
- Use proper weight initialization

---

## Loss Landscape Analysis

### Visualizing Loss Landscapes

**2D Visualization:**

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_landscape_2d(model, loss_fn, X, y,
                          param1_name, param2_name,
                          range_scale=0.5, resolution=50):
    """
    Visualize loss landscape in 2D parameter space.
    """
    # Get original parameters
    param1 = dict(model.named_parameters())[param1_name].data.clone()
    param2 = dict(model.named_parameters())[param2_name].data.clone()

    # Create grid
    p1_range = np.linspace(-range_scale, range_scale, resolution)
    p2_range = np.linspace(-range_scale, range_scale, resolution)

    losses = np.zeros((resolution, resolution))

    for i, d1 in enumerate(p1_range):
        for j, d2 in enumerate(p2_range):
            # Perturb parameters
            dict(model.named_parameters())[param1_name].data = param1 + d1
            dict(model.named_parameters())[param2_name].data = param2 + d2

            # Compute loss
            with torch.no_grad():
                predictions = model(X)
                loss = loss_fn(predictions, y)
            losses[i, j] = loss.item()

    # Restore parameters
    dict(model.named_parameters())[param1_name].data = param1
    dict(model.named_parameters())[param2_name].data = param2

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(p1_range, p2_range, losses.T, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title('Loss Landscape')
    plt.show()
```

### Sharp vs Flat Minima

**Theory:**
```
Sharp minimum: High curvature, sensitive to perturbations
Flat minimum: Low curvature, robust to perturbations

Flat minima → Better generalization
```

**Measuring Sharpness:**

```python
def compute_sharpness(model, loss_fn, X, y, epsilon=0.01):
    """
    Measure sharpness of minimum via eigenvalues of Hessian.
    """
    # Get loss at current parameters
    predictions = model(X)
    loss = loss_fn(predictions, y)

    # Compute Hessian eigenvalues (approximate)
    # In practice, use power iteration or Lanczos method

    # Simple perturbation-based estimate
    original_params = [p.data.clone() for p in model.parameters()]

    max_loss = loss.item()
    for _ in range(10):
        # Random perturbation
        with torch.no_grad():
            for p in model.parameters():
                p.data += epsilon * torch.randn_like(p)

        # Compute perturbed loss
        predictions = model(X)
        perturbed_loss = loss_fn(predictions, y)
        max_loss = max(max_loss, perturbed_loss.item())

        # Restore
        for p, orig in zip(model.parameters(), original_params):
            p.data = orig.clone()

    sharpness = (max_loss - loss.item()) / epsilon
    return sharpness
```

### Mode Connectivity

**Theory:**
```
Different training runs find different minima
Question: Are these minima connected by low-loss paths?

Finding: Neural networks have connected loss basins
→ Multiple good solutions exist
```

---

## Quick Reference

### Optimizer Selection Guide

| Task | Recommended | Learning Rate | Notes |
|------|------------|---------------|-------|
| Simple feedforward | SGD+Momentum | 0.01-0.1 | Needs tuning |
| CNNs (from scratch) | SGD+Momentum | 0.1 | With decay |
| CNNs (fine-tuning) | Adam/AdamW | 1e-4 to 1e-5 | Lower lr |
| RNNs/LSTMs | Adam/RMSProp | 1e-3 | Adaptive helps |
| Transformers | AdamW | 1e-4 | With warmup |
| GANs | Adam | 1e-4 (G), 4e-4 (D) | Different lr for G/D |

### Hyperparameter Defaults

```python
# SGD with momentum
lr=0.1, momentum=0.9, weight_decay=1e-4

# Adam
lr=1e-3, betas=(0.9, 0.999), eps=1e-8

# AdamW (transformers)
lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01

# RMSProp
lr=1e-3, alpha=0.99, eps=1e-8
```

### Common Learning Rate Schedules

```python
# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Reduce on plateau (adaptive)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# One cycle (fast training)
scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=steps)
```

---

## References

1. **"Optimization Methods for Large-Scale Machine Learning"** - Bottou et al., 2018 (SIAM Review)
2. **"Adam: A Method for Stochastic Optimization"** - Kingma & Ba, ICLR 2015
3. **"Decoupled Weight Decay Regularization"** - Loshchilov & Hutter, ICLR 2019
4. **Deep Learning** - Goodfellow et al. (Chapter 8: Optimization)
5. **"Visualizing the Loss Landscape of Neural Nets"** - Li et al., NeurIPS 2018
6. **"Sharp Minima Can Generalize For Deep Nets"** - Dinh et al., ICML 2017
7. **Numerical Optimization** - Nocedal & Wright (for second-order methods)
8. **"Cyclical Learning Rates for Training Neural Networks"** - Smith, 2017
9. **"Adaptive Learning Rates for Neural Networks"** - Zeiler, 2012 (Adadelta paper)

---

*Master optimization algorithms to train neural networks efficiently and achieve better convergence and generalization.*
