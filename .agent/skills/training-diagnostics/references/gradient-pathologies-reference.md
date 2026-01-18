# allow-torch
# Gradient Pathologies - Complete Reference

Comprehensive reference for understanding, diagnosing, and resolving gradient flow issues in neural networks.

---

## Table of Contents

1. [Vanishing Gradients](#vanishing-gradients)
2. [Exploding Gradients](#exploding-gradients)
3. [Dead Neurons](#dead-neurons)
4. [Gradient Saturation](#gradient-saturation)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Diagnostic Procedures](#diagnostic-procedures)
7. [Prevention Strategies](#prevention-strategies)

---

## Vanishing Gradients

### Theory

Gradients diminish exponentially as they propagate backward through layers.

**Mathematical Explanation:**

For an L-layer network:
```
∂L/∂W₁ = ∂L/∂zₗ · ∂zₗ/∂zₗ₋₁ · ... · ∂z₂/∂z₁ · ∂z₁/∂W₁
```

If each term `|∂zᵢ₊₁/∂zᵢ| < 1`, the product shrinks exponentially with depth.

For sigmoid activation: σ(x) = 1/(1 + e⁻ˣ)
- Derivative: σ'(x) = σ(x)(1 - σ(x))
- Maximum: σ'(0) = 0.25
- Chain over 10 layers: 0.25¹⁰ ≈ 9.5 × 10⁻⁷

### Symptoms

1. **Gradient Magnitude**
   - Early layer gradients < 1e-7
   - Exponential decay pattern across depth
   - Later layers update normally

2. **Training Behavior**
   - Loss plateaus early
   - No improvement despite continued training
   - Model acts like shallow network

3. **Weight Updates**
   - Early layer weights barely change
   - Parameter histograms show no movement

### Root Causes

1. **Activation Functions**
   - Sigmoid: max derivative = 0.25
   - Tanh: max derivative = 1.0
   - Both compress large inputs to small ranges

2. **Deep Networks**
   - Depth amplifies multiplicative effect
   - 10+ layers particularly susceptible

3. **Poor Initialization**
   - Weights too small (< 0.01)
   - Activations compressed to saturation regions

4. **No Skip Connections**
   - No gradient highway through network
   - Must flow through all layers sequentially

### Solutions

#### 1. ReLU Activations
```python
# Problem: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

derivative_max = 0.25  # Gradient vanishes!

# Solution: ReLU
def relu(x):
    return np.maximum(0, x)

derivative = 1.0  # For x > 0, no decay!
```

**Why it works:**
- Derivative is 1 for positive inputs
- No multiplicative decay
- Gradients flow unchanged

**Variants:**
- **Leaky ReLU**: Small slope for x < 0
  ```python
  def leaky_relu(x, alpha=0.01):
      return np.where(x > 0, x, alpha * x)
  ```
- **PReLU**: Learnable slope parameter
- **ELU**: Smooth for x < 0

#### 2. Residual Connections
```python
def residual_block(x):
    residual = x
    x = conv_layer1(x)
    x = relu(x)
    x = conv_layer2(x)
    return x + residual  # Identity shortcut
```

**Why it works:**
- Gradient has two paths: through layers AND through shortcut
- Shortcut provides gradient highway
- Even if layers produce small gradients, shortcut preserves flow

**Gradient flow:**
```
∂L/∂x = ∂L/∂(f(x) + x) = ∂L/∂f(x) · ∂f/∂x + ∂L/∂x · 1
                                                      ↑
                                         Direct path with gradient 1
```

#### 3. Proper Initialization

**Xavier/Glorot Initialization:**
```python
import numpy as np

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))
```

- Variance of activations stays constant across layers
- Derived from variance analysis
- Good for tanh/sigmoid

**He Initialization (for ReLU):**
```python
def he_init(fan_in, fan_out):
    std = np.sqrt(2 / fan_in)
    return np.random.randn(fan_in, fan_out) * std
```

- Accounts for ReLU's zero-ing of negative values
- Factor of 2 compensates for half the neurons being off

#### 4. Batch Normalization
```python
import torch.nn as nn

class BatchNormBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Normalize: (x - mean) / std
        x = self.bn(x)
        x = self.relu(x)
        return x
```

**Why it works:**
- Normalizes layer inputs to have mean=0, var=1
- Prevents activation saturation
- Reduces internal covariate shift
- Allows higher learning rates

#### 5. Layer Normalization (for RNNs)
```python
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

- Normalizes across features (not batch)
- Better for recurrent networks
- Used in transformers

---

## Exploding Gradients

### Theory

Gradients grow exponentially during backpropagation.

**Mathematical Explanation:**

If each layer amplifies gradients: `|∂zᵢ₊₁/∂zᵢ| > 1`

Chain rule product grows exponentially:
```
∂L/∂W₁ = ∂L/∂zₗ · (product of derivatives > 1)
```

For 10 layers with amplification factor 1.1 per layer:
1.1¹⁰ ≈ 2.59 (manageable)

But with factor 2.0 per layer:
2.0¹⁰ = 1024 (gradients explode!)

### Symptoms

1. **NaN or Inf in Loss**
   - Loss becomes NaN
   - Parameters become NaN
   - Training crashes

2. **Very Large Gradients**
   - Gradient norms > 100
   - Sudden spikes in gradient magnitude

3. **Unstable Training**
   - Loss oscillates wildly
   - Parameters swing dramatically
   - No convergence

### Root Causes

1. **Large Learning Rate**
   - Updates too large relative to gradient scale
   - Overshoots optimal regions

2. **Large Weight Values**
   - Amplify activations and gradients
   - Positive feedback loop

3. **Recurrent Networks**
   - Gradients flow through time
   - Same weight matrix multiplied many times

4. **Deep Networks Without Regularization**
   - Weights can grow unbounded
   - No constraint on amplification

### Solutions

#### 1. Gradient Clipping (Essential!)
```python
import torch

# Clip by global norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# How it works:
# 1. Compute total gradient norm: ||g|| = sqrt(sum(g_i^2))
# 2. If ||g|| > max_norm:
#    Scale all gradients: g_i = g_i * max_norm / ||g||
```

**Clip by value:**
```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Clamps each gradient element: g_i = clip(g_i, -0.5, 0.5)
```

**Why it works:**
- Prevents any single update from being too large
- Preserves gradient direction
- Essential for RNNs

#### 2. Lower Learning Rate
```python
# Start small
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Use learning rate warmup
def get_lr_warmup(current_step, warmup_steps, base_lr):
    if current_step < warmup_steps:
        return base_lr * current_step / warmup_steps
    return base_lr

# Or use scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)
```

#### 3. Weight Regularization
```python
# L2 regularization (weight decay)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5  # L2 penalty on weights
)

# Or manually:
loss = criterion(output, target) + lambda_l2 * sum(p.pow(2).sum() for p in model.parameters())
```

**Why it works:**
- Penalizes large weights
- Keeps weight values bounded
- Prevents amplification

#### 4. Careful Initialization
```python
# Check initial weight magnitudes
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")

# Should be around 0.01 to 0.1
```

---

## Dead Neurons

### Theory

Neurons that always output zero (for ReLU) or saturate (for sigmoid/tanh).

**ReLU Death:**
```
ReLU(x) = max(0, x)

If x = Wx + b < 0 for all inputs:
- Output always 0
- Gradient always 0
- No updates: dead forever!
```

### Symptoms

1. **Zero Activations**
   - Large percentage of neurons output 0
   - Typically > 20% considered problematic

2. **Zero Gradients**
   - Corresponding gradients are 0
   - Weights don't update

3. **Reduced Capacity**
   - Effective model size smaller than architecture
   - Poor performance despite large model

### Root Causes

1. **Large Learning Rate**
   - Big weight update pushes neuron into always-negative region
   - Single bad update can kill neuron permanently

2. **Large Negative Bias**
   - b << 0 ensures Wx + b < 0
   - Often caused by poor initialization

3. **Inappropriate Activation**
   - Standard ReLU vulnerable
   - No gradient for x < 0

### Solutions

#### 1. Leaky ReLU
```python
import torch.nn as nn

# Standard ReLU (can die)
relu = nn.ReLU()

# Leaky ReLU (small gradient for x < 0)
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# Parametric ReLU (learnable slope)
prelu = nn.PReLU()

# ELU (smooth for x < 0)
elu = nn.ELU(alpha=1.0)
```

**Why it works:**
- Small non-zero gradient for x < 0
- Neurons can recover from negative region
- Still mostly ReLU-like for x > 0

#### 2. Lower Learning Rate
```python
# Reduce by 10x
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

#### 3. Proper Initialization
```python
# He initialization for ReLU
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)  # Zero bias, not negative!
```

#### 4. Monitor During Training
```python
def check_dead_neurons(model, data_loader):
    model.eval()
    activations = defaultdict(list)

    # Register hooks
    def hook(name):
        def fn(module, input, output):
            activations[name].append((output == 0).float().mean().item())
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook(name)))

    # Run data
    with torch.no_grad():
        for inputs, _ in data_loader:
            model(inputs)

    # Check results
    for name, zeros in activations.items():
        dead_pct = 100 * np.mean(zeros)
        if dead_pct > 20:
            print(f"Warning: {name} has {dead_pct:.1f}% dead neurons")

    # Cleanup
    for h in hooks:
        h.remove()
```

---

## Prevention Strategies

### Initialization Checklist

```python
def initialize_model(model):
    """Proper initialization for common architectures."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization for ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

### Training Checklist

- [ ] Use ReLU or Leaky ReLU (not sigmoid/tanh in deep networks)
- [ ] Add residual connections for networks > 10 layers
- [ ] Use batch normalization or layer normalization
- [ ] Initialize with He/Xavier
- [ ] Start with small learning rate (1e-4 to 1e-3)
- [ ] Use gradient clipping (especially for RNNs)
- [ ] Monitor gradient norms during training
- [ ] Check for dead neurons periodically

### Monitoring During Training

```python
# Add to training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()

        # Monitor gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > 100:
            print(f"Warning: Large gradient norm: {total_norm:.2f}")
        if total_norm < 1e-6:
            print(f"Warning: Small gradient norm: {total_norm:.2e}")

        optimizer.step()
```

---

## Quick Reference Table

| Problem | Symptoms | Quick Fix |
|---------|----------|-----------|
| **Vanishing Gradients** | Early layer gradients < 1e-7 | ReLU + residual connections |
| **Exploding Gradients** | Gradients > 100 or NaN | Gradient clipping + lower LR |
| **Dead ReLU** | > 20% zero activations | Leaky ReLU + lower LR |
| **Saturation** | Sigmoid/tanh outputs near 0 or 1 | Switch to ReLU |
| **Unstable Training** | Loss oscillates wildly | Lower LR + gradient clipping |

---

## References

1. "Understanding the difficulty of training deep feedforward neural networks" - Glorot & Bengio, 2010
2. "Deep Residual Learning for Image Recognition" - He et al., 2015
3. "Batch Normalization: Accelerating Deep Network Training" - Ioffe & Szegedy, 2015
4. "On the difficulty of training recurrent neural networks" - Pascanu et al., 2013
