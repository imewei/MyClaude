# allow-torch
# Calculus and Backpropagation - Complete Reference

Comprehensive reference for calculus concepts and backpropagation derivations essential to neural network training.

---

## Table of Contents

1. [Differential Calculus Basics](#differential-calculus-basics)
2. [Multivariable Calculus](#multivariable-calculus)
3. [Chain Rule](#chain-rule)
4. [Backpropagation Algorithm](#backpropagation-algorithm)
5. [Automatic Differentiation](#automatic-differentiation)
6. [Common Layer Gradients](#common-layer-gradients)
7. [Numerical Stability](#numerical-stability)

---

## Differential Calculus Basics

### Derivatives

**Definition:**
```
f'(x) = lim_{h→0} [f(x + h) - f(x)] / h
```

**Interpretation:**
- Rate of change
- Slope of tangent line
- Sensitivity of output to input

### Common Derivatives

```
d/dx (xⁿ) = n·xⁿ⁻¹
d/dx (eˣ) = eˣ
d/dx (ln x) = 1/x
d/dx (sin x) = cos x
d/dx (cos x) = -sin x
```

### Activation Function Derivatives

#### ReLU
```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Alternative:
def relu_derivative_alt(x):
    return np.where(x > 0, 1, 0)
```

#### Sigmoid
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# If you already have sigmoid output
def sigmoid_derivative_from_output(output):
    return output * (1 - output)
```

#### Tanh
```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# From output
def tanh_derivative_from_output(output):
    return 1 - output ** 2
```

#### Leaky ReLU
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

#### Softmax
```python
def softmax(x):
    # Numerically stable version
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    """
    Jacobian of softmax: ∂sᵢ/∂xⱼ
    For a single sample:
    J[i,j] = sᵢ(δᵢⱼ - sⱼ)
    where δᵢⱼ is Kronecker delta
    """
    s = softmax(x)
    # Jacobian: s[i] * (I - s[j])
    return np.diag(s) - np.outer(s, s)
```

---

## Multivariable Calculus

### Partial Derivatives

**Definition:**
```
∂f/∂xᵢ = lim_{h→0} [f(x₁,...,xᵢ+h,...,xₙ) - f(x₁,...,xᵢ,...,xₙ)] / h
```

Hold all other variables constant, differentiate with respect to one variable.

### Gradient

**Definition:**
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Properties:**
- Points in direction of steepest ascent
- Perpendicular to level curves
- Magnitude = rate of steepest increase

**Example:**
```python
# f(x, y) = x²y + y³
# ∇f = [2xy, x² + 3y²]

def f(x, y):
    return x**2 * y + y**3

def gradient_f(x, y):
    df_dx = 2 * x * y
    df_dy = x**2 + 3 * y**2
    return np.array([df_dx, df_dy])
```

### Jacobian Matrix

**For vector function f: ℝⁿ → ℝᵐ:**
```
J = [∂fᵢ/∂xⱼ]  (m×n matrix)

J[i,j] = ∂fᵢ/∂xⱼ
```

**Example: Linear Layer**
```python
# f(x) = Wx + b
# Jacobian ∂f/∂x = W

def linear_jacobian(W):
    return W  # Already the Jacobian!
```

### Hessian Matrix

**For scalar function f: ℝⁿ → ℝ:**
```
H = [∂²f/∂xᵢ∂xⱼ]  (n×n symmetric matrix)

H[i,j] = ∂²f/∂xᵢ∂xⱼ
```

**Properties:**
- Symmetric: H = Hᵀ
- Describes curvature of loss landscape
- Eigenvalues determine convexity

**Application:**
```python
# Second-order optimization (Newton's method)
# x_{t+1} = x_t - H⁻¹∇f

# In practice, approximate Hessian (BFGS, L-BFGS)
```

---

## Chain Rule

### Single Variable

```
If y = f(g(x)), then:
dy/dx = df/dg · dg/dx
```

**Example:**
```python
# y = exp(x²)
# Let u = x²
# y = exp(u)

# dy/dx = dy/du · du/dx = exp(u) · 2x = exp(x²) · 2x

def f(x):
    return np.exp(x**2)

def f_derivative(x):
    return np.exp(x**2) * 2 * x
```

### Multivariable Chain Rule

```
If z = f(y₁, ..., yₘ) and each yᵢ = gᵢ(x₁, ..., xₙ):

∂z/∂xⱼ = Σᵢ (∂z/∂yᵢ · ∂yᵢ/∂xⱼ)
```

**Neural Network Application:**
```
Loss L = f(y)
Output y = g(z)
Pre-activation z = Wx + b

∂L/∂W = ∂L/∂y · ∂y/∂z · ∂z/∂W
```

### Vector Chain Rule

```
If y = f(x) where f: ℝⁿ → ℝᵐ and x = g(t) where g: ℝ → ℝⁿ:

dy/dt = (∂f/∂x) · (dx/dt)
       = Jacobian(f) · (dx/dt)
```

---

## Backpropagation Algorithm

### Forward Pass

**Layer by layer computation:**
```
Input: x⁽⁰⁾
Layer 1: z⁽¹⁾ = W⁽¹⁾x⁽⁰⁾ + b⁽¹⁾
         a⁽¹⁾ = σ(z⁽¹⁾)
Layer 2: z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
         a⁽²⁾ = σ(z⁽²⁾)
...
Loss: L = loss(a⁽ᴸ⁾, y)
```

### Backward Pass

**Start with loss gradient, propagate backward:**

```
∂L/∂a⁽ᴸ⁾ = ∂loss/∂a⁽ᴸ⁾

For layer l (from L to 1):
    ∂L/∂z⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ ⊙ σ'(z⁽ˡ⁾)          (element-wise)
    ∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ        (outer product)
    ∂L/∂b⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾                   (sum over batch)
    ∂L/∂a⁽ˡ⁻¹⁾ = (W⁽ˡ⁾)ᵀ · ∂L/∂z⁽ˡ⁾       (matrix-vector product)
```

### Derivation: Single Layer

**Setup:**
```
Input: x ∈ ℝⁿ
Weight: W ∈ ℝᵐˣⁿ
Bias: b ∈ ℝᵐ
Pre-activation: z = Wx + b ∈ ℝᵐ
Activation: a = σ(z) ∈ ℝᵐ
Loss: L ∈ ℝ
```

**Given:** ∂L/∂a (gradient from next layer)

**Find:** ∂L/∂W, ∂L/∂b, ∂L/∂x

**Step 1: ∂L/∂z**
```
Chain rule: ∂L/∂zᵢ = Σⱼ (∂L/∂aⱼ · ∂aⱼ/∂zᵢ)

Since aⱼ = σ(zⱼ) depends only on zⱼ:
∂L/∂zᵢ = ∂L/∂aᵢ · ∂aᵢ/∂zᵢ = ∂L/∂aᵢ · σ'(zᵢ)

Vectorized: ∂L/∂z = ∂L/∂a ⊙ σ'(z)  (element-wise product)
```

**Step 2: ∂L/∂W**
```
zᵢ = Σⱼ Wᵢⱼxⱼ + bᵢ

∂L/∂Wᵢⱼ = ∂L/∂zᵢ · ∂zᵢ/∂Wᵢⱼ = ∂L/∂zᵢ · xⱼ

Vectorized: ∂L/∂W = (∂L/∂z) · xᵀ  (outer product)
```

**Step 3: ∂L/∂b**
```
∂L/∂bᵢ = ∂L/∂zᵢ · ∂zᵢ/∂bᵢ = ∂L/∂zᵢ · 1 = ∂L/∂zᵢ

Vectorized: ∂L/∂b = ∂L/∂z
```

**Step 4: ∂L/∂x** (for previous layer)
```
zᵢ = Σⱼ Wᵢⱼxⱼ + bᵢ

∂L/∂xⱼ = Σᵢ (∂L/∂zᵢ · ∂zᵢ/∂xⱼ) = Σᵢ (∂L/∂zᵢ · Wᵢⱼ)

Vectorized: ∂L/∂x = Wᵀ(∂L/∂z)  (matrix-vector product)
```

### Complete Implementation

```python
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # He initialization
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((output_dim, 1))

    def forward(self, x):
        """Forward pass: z = Wx + b"""
        self.x = x  # Cache for backward pass
        self.z = self.W @ x + self.b
        return self.z

    def backward(self, grad_output):
        """
        Backward pass.

        Args:
            grad_output: ∂L/∂z (gradient from next layer)

        Returns:
            grad_input: ∂L/∂x (gradient to pass to previous layer)
        """
        batch_size = self.x.shape[1]

        # Gradient w.r.t weights: ∂L/∂W = (∂L/∂z) @ xᵀ
        self.grad_W = grad_output @ self.x.T / batch_size

        # Gradient w.r.t bias: ∂L/∂b = mean(∂L/∂z) over batch
        self.grad_b = np.mean(grad_output, axis=1, keepdims=True)

        # Gradient w.r.t input: ∂L/∂x = Wᵀ @ (∂L/∂z)
        grad_input = self.W.T @ grad_output

        return grad_input


class ReLU:
    def forward(self, x):
        """Forward pass: a = max(0, x)"""
        self.x = x  # Cache for backward pass
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        Backward pass.

        Args:
            grad_output: ∂L/∂a

        Returns:
            grad_input: ∂L/∂x = ∂L/∂a ⊙ (x > 0)
        """
        return grad_output * (self.x > 0)


class MSELoss:
    def forward(self, predictions, targets):
        """Forward pass: L = 1/2 ||pred - target||²"""
        self.diff = predictions - targets
        return 0.5 * np.mean(self.diff ** 2)

    def backward(self):
        """
        Backward pass.

        Returns:
            ∂L/∂pred = (pred - target) / batch_size
        """
        batch_size = self.diff.shape[1]
        return self.diff / batch_size


# Full network
class TwoLayerNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = LinearLayer(input_dim, hidden_dim)
        self.relu = ReLU()
        self.layer2 = LinearLayer(hidden_dim, output_dim)
        self.loss_fn = MSELoss()

    def forward(self, x, targets=None):
        """Forward pass through network."""
        # Layer 1
        z1 = self.layer1.forward(x)
        a1 = self.relu.forward(z1)

        # Layer 2
        z2 = self.layer2.forward(a1)

        if targets is not None:
            loss = self.loss_fn.forward(z2, targets)
            return z2, loss
        return z2

    def backward(self):
        """Backward pass through network."""
        # Loss gradient
        grad_z2 = self.loss_fn.backward()

        # Layer 2 backward
        grad_a1 = self.layer2.backward(grad_z2)

        # ReLU backward
        grad_z1 = self.relu.backward(grad_a1)

        # Layer 1 backward
        grad_x = self.layer1.backward(grad_z1)

        return grad_x

    def get_gradients(self):
        """Return all parameter gradients."""
        return {
            'W1': self.layer1.grad_W,
            'b1': self.layer1.grad_b,
            'W2': self.layer2.grad_W,
            'b2': self.layer2.grad_b
        }


# Usage
network = TwoLayerNetwork(input_dim=784, hidden_dim=128, output_dim=10)

# Forward pass
x = np.random.randn(784, 32)  # Batch of 32
targets = np.random.randn(10, 32)
predictions, loss = network.forward(x, targets)

# Backward pass
network.backward()

# Get gradients
gradients = network.get_gradients()
```

---

## Automatic Differentiation

### Forward Mode vs Reverse Mode

**Forward Mode (builds graph from inputs):**
- Efficient for functions with few inputs, many outputs
- Computes one directional derivative at a time
- ∂y/∂xᵢ for each input xᵢ

**Reverse Mode (backpropagation):**
- Efficient for functions with many inputs, few outputs (like neural networks!)
- Computes ∂L/∂xᵢ for all inputs in one pass
- This is what PyTorch/TensorFlow use

### Computational Graph

**Example: y = (x₁ + x₂) · x₃**

```
Forward pass (build graph):
    v₁ = x₁ + x₂
    v₂ = v₁ · x₃
    y = v₂

Backward pass (compute gradients):
    ∂y/∂v₂ = 1
    ∂y/∂v₁ = ∂y/∂v₂ · ∂v₂/∂v₁ = 1 · x₃ = x₃
    ∂y/∂x₃ = ∂y/∂v₂ · ∂v₂/∂x₃ = 1 · v₁ = (x₁ + x₂)
    ∂y/∂x₂ = ∂y/∂v₁ · ∂v₁/∂x₂ = x₃ · 1 = x₃
    ∂y/∂x₁ = ∂y/∂v₁ · ∂v₁/∂x₁ = x₃ · 1 = x₃
```

### PyTorch Autograd

```python
import torch

# Enable gradient tracking
x = torch.randn(3, requires_grad=True)
W = torch.randn(5, 3, requires_grad=True)
b = torch.randn(5, requires_grad=True)

# Forward pass
z = W @ x + b
a = torch.relu(z)
loss = (a ** 2).sum()

# Backward pass (automatic!)
loss.backward()

# Gradients computed
print(x.grad)  # ∂loss/∂x
print(W.grad)  # ∂loss/∂W
print(b.grad)  # ∂loss/∂b
```

---

## Common Layer Gradients

### Convolutional Layer

**Forward:**
```
output[b, c_out, h_out, w_out] = Σ_{c_in,kh,kw}
    input[b, c_in, h_out+kh, w_out+kw] · kernel[c_out, c_in, kh, kw]
```

**Backward:**
```python
def conv2d_backward(grad_output, input, kernel):
    """
    grad_output: ∂L/∂output
    input: cached from forward pass
    kernel: weight tensor

    Returns:
        grad_input: ∂L/∂input
        grad_kernel: ∂L/∂kernel
    """
    # Gradient w.r.t input: convolve grad_output with flipped kernel
    grad_input = F.conv_transpose2d(grad_output, kernel)

    # Gradient w.r.t kernel: convolve input with grad_output
    grad_kernel = F.conv2d(input.transpose(0, 1), grad_output.transpose(0, 1))

    return grad_input, grad_kernel
```

### Batch Normalization

**Forward:**
```
μ = mean(x)
σ² = var(x)
x_hat = (x - μ) / √(σ² + ε)
y = γ·x_hat + β
```

**Backward:**
```python
def batchnorm_backward(grad_output, x, mean, var, gamma, eps=1e-5):
    """
    Backward pass for batch normalization.

    grad_output: ∂L/∂y
    """
    N = x.shape[0]

    # Normalized input
    x_centered = x - mean
    std = np.sqrt(var + eps)
    x_hat = x_centered / std

    # Gradients
    grad_gamma = np.sum(grad_output * x_hat, axis=0)
    grad_beta = np.sum(grad_output, axis=0)

    # Gradient w.r.t x_hat
    grad_x_hat = grad_output * gamma

    # Gradient w.r.t variance
    grad_var = np.sum(grad_x_hat * x_centered * -0.5 * (var + eps)**(-1.5), axis=0)

    # Gradient w.r.t mean
    grad_mean = np.sum(grad_x_hat * -1.0 / std, axis=0) + \
                grad_var * np.mean(-2.0 * x_centered, axis=0)

    # Gradient w.r.t input
    grad_x = grad_x_hat / std + \
             grad_var * 2.0 * x_centered / N + \
             grad_mean / N

    return grad_x, grad_gamma, grad_beta
```

### Dropout

**Forward:**
```python
def dropout_forward(x, dropout_rate=0.5, training=True):
    if not training:
        return x

    # Binary mask
    mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape)

    # Scale to maintain expected value
    return x * mask / (1 - dropout_rate), mask
```

**Backward:**
```python
def dropout_backward(grad_output, mask, dropout_rate=0.5):
    """
    Gradient is masked and scaled by same factor.
    """
    return grad_output * mask / (1 - dropout_rate)
```

### Attention (Simplified)

**Forward:**
```python
def attention_forward(Q, K, V):
    """
    Q: queries (batch, seq_len_q, d_k)
    K: keys (batch, seq_len_k, d_k)
    V: values (batch, seq_len_v, d_v)
    """
    # Scores
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])

    # Attention weights
    weights = softmax(scores, axis=-1)

    # Output
    output = weights @ V

    return output, weights
```

**Backward (conceptual):**
```python
def attention_backward(grad_output, Q, K, V, weights):
    """
    grad_output: ∂L/∂output

    Returns: grad_Q, grad_K, grad_V
    """
    d_k = K.shape[-1]

    # ∂L/∂V = weightsᵀ @ grad_output
    grad_V = weights.transpose(-2, -1) @ grad_output

    # ∂L/∂weights = grad_output @ Vᵀ
    grad_weights = grad_output @ V.transpose(-2, -1)

    # ∂L/∂scores (softmax backward)
    grad_scores = softmax_backward(grad_weights, weights)

    # ∂L/∂Q = grad_scores @ K / √d_k
    grad_Q = grad_scores @ K / np.sqrt(d_k)

    # ∂L/∂K = grad_scoresᵀ @ Q / √d_k
    grad_K = grad_scores.transpose(-2, -1) @ Q / np.sqrt(d_k)

    return grad_Q, grad_K, grad_V
```

---

## Numerical Stability

### Gradient Checking

```python
def gradient_check(f, x, analytical_grad, eps=1e-7):
    """
    Verify analytical gradient using finite differences.

    f: function to compute
    x: input point
    analytical_grad: computed gradient
    eps: finite difference step size
    """
    numerical_grad = np.zeros_like(x)

    # Compute numerical gradient
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += eps

        x_minus = x.copy()
        x_minus.flat[i] -= eps

        numerical_grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    # Compare
    diff = np.linalg.norm(numerical_grad - analytical_grad)
    norm = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    relative_error = diff / (norm + 1e-8)

    return relative_error, numerical_grad


# Example usage
def test_layer_gradient():
    layer = LinearLayer(10, 5)
    x = np.random.randn(10, 1)

    # Forward
    z = layer.forward(x)
    loss = np.sum(z ** 2)

    # Backward
    grad_z = 2 * z
    layer.backward(grad_z)

    # Gradient check for W
    def f_W(W):
        layer.W = W.reshape(layer.W.shape)
        z = layer.forward(x)
        return np.sum(z ** 2)

    error, _ = gradient_check(f_W, layer.W.flatten(), layer.grad_W.flatten())
    print(f"Gradient check error: {error:.2e}")
    assert error < 1e-5, "Gradient check failed!"
```

### Numerical Stability Tips

1. **Softmax:**
```python
# Bad: overflow risk
def softmax_unstable(x):
    return np.exp(x) / np.sum(np.exp(x))

# Good: subtract max for numerical stability
def softmax_stable(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

2. **Log-Sum-Exp:**
```python
# Compute log(Σ exp(xᵢ)) stably
def logsumexp(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))
```

3. **Cross-Entropy:**
```python
# Bad: compute softmax then log (numerical issues)
def cross_entropy_unstable(logits, targets):
    probs = softmax(logits)
    return -np.sum(targets * np.log(probs + 1e-10))

# Good: log-softmax in one stable operation
def cross_entropy_stable(logits, targets):
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
    return -np.sum(targets * log_probs)
```

---

## Quick Reference

### Derivative Rules
| Function | Derivative |
|----------|------------|
| xⁿ | n·xⁿ⁻¹ |
| eˣ | eˣ |
| ln(x) | 1/x |
| sigmoid(x) | σ(x)·(1 - σ(x)) |
| tanh(x) | 1 - tanh²(x) |
| ReLU(x) | 𝟙(x > 0) |

### Chain Rule
| Context | Formula |
|---------|---------|
| Composition | (f∘g)'(x) = f'(g(x))·g'(x) |
| Neural layer | ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W |
| Backprop | ∂L/∂xₗ = ∂L/∂xₗ₊₁ · ∂xₗ₊₁/∂xₗ |

### Gradient Formulas
| Layer | Gradient |
|-------|----------|
| Linear: z = Wx + b | ∂L/∂W = (∂L/∂z)·xᵀ |
| | ∂L/∂x = Wᵀ·(∂L/∂z) |
| Activation: a = σ(z) | ∂L/∂z = (∂L/∂a) ⊙ σ'(z) |
| Loss: L = ½||y - ŷ||² | ∂L/∂ŷ = ŷ - y |

---

## References

1. "Deep Learning" - Goodfellow, Bengio, Courville (Chapter 6)
2. "Neural Networks and Deep Learning" - Michael Nielsen
3. "Automatic Differentiation in Machine Learning: a Survey" - Baydin et al., 2018
4. "Calculus on Computational Graphs: Backpropagation" - Chris Olah
5. "Yes you should understand backprop" - Andrej Karpathy
