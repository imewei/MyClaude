---
name: neural-network-mathematics
version: "1.0.7"
maturity: "5-Expert"
specialization: NN Mathematical Foundations
description: Apply mathematical foundations of neural networks including linear algebra, calculus, probability theory, optimization, and information theory. Use when deriving backpropagation for custom layers, computing Jacobians/Hessians, implementing automatic differentiation, analyzing gradient flow, proving convergence, working with Bayesian deep learning, deriving loss functions from MLE principles, or understanding PAC learning theory.
---

# Neural Network Mathematics

Mathematical foundations for understanding and implementing neural networks.

---

## Core Mathematical Domains

| Domain | Key Concepts | Applications |
|--------|--------------|--------------|
| Linear Algebra | Eigenvalues, SVD, spectral norm | Gradient flow, initialization, normalization |
| Calculus | Jacobian, Hessian, VJP/JVP | Backpropagation, optimization |
| Probability | MLE, MAP, Bayesian inference | Loss functions, regularization, uncertainty |
| Optimization | SGD, Adam, momentum, natural gradient | Training dynamics, convergence |
| Information Theory | Entropy, KL divergence, mutual information | Compression, VAEs, model selection |

---

## Linear Algebra Essentials

### Forward Pass
```
y = Wx + b
where W ∈ ℝ^(m×n), x ∈ ℝ^n, b ∈ ℝ^m
```

### Gradient Flow Analysis
```python
import numpy as np

def analyze_gradient_flow(W):
    eigenvalues = np.linalg.eigvals(W)
    return {
        'spectral_radius': np.max(np.abs(eigenvalues)),
        'condition_number': np.linalg.cond(W),
    }
```

### Einstein Notation
```python
# Matrix multiplication: C_ij = Σ_k A_ik B_kj
np.einsum('ik,kj->ij', A, B)

# Batch matrix: C_bij = Σ_k A_bik B_bkj
np.einsum('bik,bkj->bij', A, B)

# Attention: attention_ij = Σ_k Q_ik K_jk
np.einsum('ik,jk->ij', Q, K)
```

---

## Calculus for Backpropagation

### Chain Rule
```
df/dx = (df/dg) · (dg/dx)
∂L/∂W_i = ∂L/∂z_{i+1} · ∂z_{i+1}/∂z_i · ∂z_i/∂W_i
```

### VJP vs JVP

| Method | Computation | Use Case |
|--------|-------------|----------|
| VJP (Backprop) | v^T · J_f(x) | Scalar output, vector input (typical ML) |
| JVP (Forward) | J_f(x) · v | Vector output, scalar input |

### Hessian-Vector Product
```python
def hessian_vector_product(loss_fn, params, vector):
    grads = jax.grad(loss_fn)(params)
    hvp = jax.grad(lambda p: jax.vdot(grads(p), vector))(params)
    return hvp  # O(n) not O(n²)
```

---

## Probability and Loss Functions

### Loss as Negative Log-Likelihood

| Loss | Distribution | Formula |
|------|--------------|---------|
| MSE | Gaussian | L = ‖y - ŷ‖² |
| Cross-Entropy | Categorical | L = -Σ_k y_k log ŷ_k |
| Binary CE | Bernoulli | L = -y log ŷ - (1-y) log(1-ŷ) |

### Regularization as Prior
```
θ* = argmin_θ [L(D|θ) + λR(θ)]
```
- L2 regularization = Gaussian prior
- L1 regularization = Laplace prior (sparse)

### Bayesian Approximations

| Method | Description |
|--------|-------------|
| MC Dropout | Approximate Bayesian inference |
| Variational | Approximate p(θ|D) with q(θ) |
| Laplace | Gaussian around MAP estimate |
| Ensemble | Average multiple models |

---

## Optimization Theory

### Momentum Methods
```
# Heavy Ball
v_{t+1} = βv_t - η∇L(θ_t)
θ_{t+1} = θ_t + v_{t+1}

# Nesterov (look-ahead)
v_{t+1} = βv_t - η∇L(θ_t + βv_t)
```

### Adam
```
m_t = β_1 m_{t-1} + (1-β_1)∇L      # Mean
v_t = β_2 v_{t-1} + (1-β_2)(∇L)²   # Variance
θ_{t+1} = θ_t - η m̂_t/(√v̂_t + ε)
```

### Convergence Rates

| Problem Type | Rate |
|--------------|------|
| Strongly convex | O(log(1/ε)) |
| Convex | O(1/ε) |
| Non-convex | Stationary points |

### Loss Landscape

| Critical Point | Hessian Eigenvalues |
|----------------|---------------------|
| Local minimum | All > 0 |
| Local maximum | All < 0 |
| Saddle point | Mixed signs |

---

## Information Theory

### Key Quantities
```
# Shannon Entropy
H(p) = -Σ_x p(x) log p(x)

# Cross-Entropy
H(p, q) = -Σ_x p(x) log q(x)

# KL Divergence
D_KL(p||q) = Σ_x p(x) log(p(x)/q(x))
```

### Information Bottleneck
```
X → Z → Ŷ
L = I(Z; Y) - βI(Z; X)
```
Compress input while retaining predictive information.

---

## Matrix Calculus Reference

```
d(x^T A x)/dx = (A + A^T)x
d(Ax)/dx = A^T
d(||x||²)/dx = 2x
d(log det(A))/dA = A^{-T}

# Matrix multiplication gradient
L = f(WX)
dL/dW = (dL/dY) X^T
dL/dX = W^T (dL/dY)
```

---

## Numerical Gradient Check

```python
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4)
```

---

## Numerical Stability

| Issue | Solution |
|-------|----------|
| exp(large) overflow | Log-sum-exp trick |
| log(small) underflow | Add epsilon |
| Division by ~0 | Epsilon in denominator |
| Precision loss | Double precision for sensitive ops |

---

## Initialization Strategies

| Method | Formula | Use Case |
|--------|---------|----------|
| Xavier/Glorot | Var = 2/(n_in + n_out) | Tanh, sigmoid |
| He | Var = 2/n_in | ReLU |
| Orthogonal | W^T W = I | Prevent gradient issues |

---

## Checklist

- [ ] Chain rule correctly applied in backprop
- [ ] VJP for backward, JVP for forward mode
- [ ] Loss derived from likelihood principle
- [ ] Optimizer hyperparameters tuned
- [ ] Numerical stability ensured (epsilon, log-sum-exp)
- [ ] Gradient checked numerically for custom layers
- [ ] Eigenvalue analysis for initialization
- [ ] Information-theoretic metrics for model analysis

---

**Version**: 1.0.5
