# allow-torch
# Numerical Methods for Neural Networks - Complete Reference

Numerical techniques for stable and efficient computation in deep learning, from floating-point arithmetic to large-scale matrix operations.

---

## Table of Contents

1. [Numerical Stability Fundamentals](#numerical-stability-fundamentals)
2. [Floating-Point Arithmetic](#floating-point-arithmetic)
3. [Stable Implementations](#stable-implementations)
4. [Numerical Linear Algebra](#numerical-linear-algebra)
5. [Sparse Computations](#sparse-computations)
6. [Mixed Precision Training](#mixed-precision-training)

---

## Numerical Stability Fundamentals

### Sources of Numerical Error

**1. Representation Error:**
```
Cannot represent all real numbers exactly in floating-point

Example:
0.1 + 0.2 ≠ 0.3 in floating-point
(0.1 + 0.2 = 0.30000000000000004)
```

**2. Roundoff Error:**
```
Accumulates during arithmetic operations

Example: (a + b) + c ≠ a + (b + c) for large |a| and small |b|, |c|
```

**3. Truncation Error:**
```
From approximating infinite processes with finite steps

Example: Taylor series truncation, iterative methods
```

**4. Catastrophic Cancellation:**
```
Subtracting nearly equal numbers loses precision

Example: x - y when x ≈ y
```

### Condition Number

**Definition:**
```
κ(A) = ||A|| · ||A⁻¹||

Measures sensitivity of output to input perturbations
```

**Interpretation:**
```
κ(A) = 10^k: Lose ~k digits of precision
κ(A) < 10: Well-conditioned
κ(A) > 10^10: Ill-conditioned (numerical instability)
```

**Python Implementation:**

```python
import torch
import numpy as np

def condition_number(A: torch.Tensor, p: float = 2) -> float:
    """
    Compute condition number of matrix A.

    Args:
        A: Matrix (m x n)
        p: Norm type (2 for spectral norm)

    Returns:
        Condition number κ(A)
    """
    if p == 2:
        # Spectral norm: largest singular value
        singular_values = torch.linalg.svdvals(A)
        return (singular_values.max() / singular_values.min()).item()
    else:
        A_inv = torch.linalg.pinv(A)  # Pseudoinverse
        return (torch.linalg.norm(A, ord=p) *
                torch.linalg.norm(A_inv, ord=p)).item()


# Check conditioning of weight matrices
def check_network_conditioning(model):
    """Check condition numbers of all weight matrices."""
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            kappa = condition_number(param.data)
            if kappa > 1e10:
                print(f"⚠️  {name}: κ = {kappa:.2e} (ill-conditioned)")
            else:
                print(f"✓ {name}: κ = {kappa:.2e}")
```

---

## Floating-Point Arithmetic

### IEEE 754 Standard

**32-bit (float32):**
```
Sign: 1 bit
Exponent: 8 bits
Mantissa: 23 bits

Range: ±1.18e-38 to ±3.4e38
Precision: ~7 decimal digits
```

**64-bit (float64/double):**
```
Sign: 1 bit
Exponent: 11 bits
Mantissa: 52 bits

Range: ±2.23e-308 to ±1.80e308
Precision: ~16 decimal digits
```

**16-bit (float16/half):**
```
Sign: 1 bit
Exponent: 5 bits
Mantissa: 10 bits

Range: ±6.1e-5 to ±65504
Precision: ~3 decimal digits
```

### Machine Epsilon

**Definition:**
```
ε_machine: smallest number such that 1 + ε ≠ 1 in floating-point

float16: ε ≈ 9.8e-4
float32: ε ≈ 1.2e-7
float64: ε ≈ 2.2e-16
```

**Python Check:**

```python
def machine_epsilon(dtype=torch.float32):
    """Compute machine epsilon for given dtype."""
    epsilon = torch.tensor(1.0, dtype=dtype)
    while torch.tensor(1.0, dtype=dtype) + epsilon / 2 != torch.tensor(1.0, dtype=dtype):
        epsilon /= 2
    return epsilon.item()

print(f"float16 ε: {machine_epsilon(torch.float16):.2e}")
print(f"float32 ε: {machine_epsilon(torch.float32):.2e}")
print(f"float64 ε: {machine_epsilon(torch.float64):.2e}")
```

### Precision Best Practices

```python
# Avoid catastrophic cancellation
# BAD
def bad_quadratic(a, b, c):
    """Numerically unstable quadratic formula."""
    discriminant = b**2 - 4*a*c
    x1 = (-b + torch.sqrt(discriminant)) / (2*a)
    x2 = (-b - torch.sqrt(discriminant)) / (2*a)
    return x1, x2

# GOOD
def stable_quadratic(a, b, c):
    """Numerically stable quadratic formula."""
    discriminant = b**2 - 4*a*c
    sqrt_discriminant = torch.sqrt(discriminant)

    # Avoid subtracting nearly equal numbers
    if b >= 0:
        x1 = (-b - sqrt_discriminant) / (2*a)
        x2 = (2*c) / (-b - sqrt_discriminant)
    else:
        x1 = (2*c) / (-b + sqrt_discriminant)
        x2 = (-b + sqrt_discriminant) / (2*a)

    return x1, x2
```

---

## Stable Implementations

### Log-Sum-Exp Trick

**Problem:**
```
Computing: log(Σᵢ exp(xᵢ))

Direct computation: exp(xᵢ) can overflow/underflow
```

**Solution:**
```
log(Σᵢ exp(xᵢ)) = a + log(Σᵢ exp(xᵢ - a))

where a = max(x)
```

**Implementation:**

```python
def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable log-sum-exp.

    Args:
        x: Input tensor
        dim: Dimension to sum over

    Returns:
        log(Σ exp(x)) computed stably
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    return x_max.squeeze(dim) + torch.log(
        torch.sum(torch.exp(x - x_max), dim=dim)
    )


# PyTorch built-in (use this!)
result = torch.logsumexp(x, dim=-1)
```

### Softmax Stability

**Problem:**
```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

exp(xᵢ) can overflow for large xᵢ
```

**Solution:**
```
softmax(xᵢ) = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))

Shifting by max(x) prevents overflow while preserving result
```

**Implementation:**

```python
def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax.

    Args:
        x: Logits
        dim: Dimension to apply softmax

    Returns:
        Softmax probabilities
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# PyTorch built-in (use this!)
probs = torch.softmax(x, dim=-1)
```

### Log-Softmax

**Combining log and softmax:**

```python
def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable log-softmax.

    log softmax(xᵢ) = xᵢ - log(Σⱼ exp(xⱼ))
                     = xᵢ - logsumexp(x)

    Args:
        x: Logits
        dim: Dimension to apply

    Returns:
        Log-probabilities
    """
    return x - torch.logsumexp(x, dim=dim, keepdim=True)


# PyTorch built-in (use this!)
log_probs = torch.log_softmax(x, dim=-1)

# NEVER do this (numerically unstable):
# log_probs = torch.log(torch.softmax(x, dim=-1))
```

### Sigmoid Stability

**Problem:**
```
σ(x) = 1 / (1 + exp(-x))

exp(-x) overflows for large negative x
```

**Solution:**
```python
def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid.

    For x ≥ 0: σ(x) = 1 / (1 + exp(-x))
    For x < 0: σ(x) = exp(x) / (1 + exp(x))
    """
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(x))
    )


# PyTorch built-in (use this!)
probs = torch.sigmoid(x)
```

### Binary Cross-Entropy Stability

**Combined sigmoid + BCE:**

```python
def stable_binary_cross_entropy(logits: torch.Tensor,
                                targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable BCE with logits.

    BCE(σ(x), y) = -[y log σ(x) + (1-y) log(1-σ(x))]
                 = -[y log(1/(1+exp(-x))) + (1-y) log(exp(-x)/(1+exp(-x)))]
                 = -[y(-log(1+exp(-x))) + (1-y)(-x - log(1+exp(-x)))]
                 = max(x, 0) - xy + log(1 + exp(-|x|))
    """
    return (torch.clamp(logits, min=0) - logits * targets +
            torch.log(1 + torch.exp(-torch.abs(logits)))).mean()


# PyTorch built-in (use this!)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
```

---

## Numerical Linear Algebra

### Matrix Decompositions

**QR Decomposition:**
```
A = QR

where:
- Q: orthogonal matrix (QᵀQ = I)
- R: upper triangular matrix
```

**Use Cases:**
- Solving least squares
- Computing eigenvalues (QR algorithm)
- Orthogonalization (Gram-Schmidt)

**Implementation:**

```python
def qr_decomposition(A: torch.Tensor) -> tuple:
    """
    QR decomposition of matrix A.

    Args:
        A: Matrix (m x n)

    Returns:
        Q: Orthogonal matrix (m x m)
        R: Upper triangular matrix (m x n)
    """
    Q, R = torch.linalg.qr(A)
    return Q, R


# Example: Solve least squares Ax = b
A = torch.randn(100, 10)
b = torch.randn(100, 1)

Q, R = torch.linalg.qr(A)
x = torch.linalg.solve_triangular(R[:10], Q.T @ b, upper=True)
```

**Cholesky Decomposition:**
```
A = LLᵀ

where:
- L: lower triangular matrix
- A must be positive definite
```

**Use Cases:**
- Solving linear systems with symmetric positive definite matrices
- Sampling from multivariate Gaussians
- Checking positive definiteness

**Implementation:**

```python
def cholesky_decomposition(A: torch.Tensor) -> torch.Tensor:
    """
    Cholesky decomposition of positive definite matrix A.

    Args:
        A: Symmetric positive definite matrix (n x n)

    Returns:
        L: Lower triangular matrix where A = LLᵀ
    """
    try:
        L = torch.linalg.cholesky(A)
        return L
    except RuntimeError:
        # Not positive definite
        print("Matrix is not positive definite")
        # Add small diagonal term for numerical stability
        L = torch.linalg.cholesky(A + 1e-6 * torch.eye(A.size(0)))
        return L


# Example: Sample from multivariate Gaussian N(μ, Σ)
mu = torch.zeros(10)
Sigma = torch.randn(10, 10)
Sigma = Sigma @ Sigma.T + torch.eye(10)  # Make positive definite

L = torch.linalg.cholesky(Sigma)
z = torch.randn(10)
sample = mu + L @ z  # Sample from N(μ, Σ)
```

### Iterative Solvers

**Conjugate Gradient Method:**
```
Solve Ax = b where A is symmetric positive definite

Iterative algorithm:
- Faster than direct methods for large sparse systems
- O(n²) per iteration vs O(n³) for direct solve
```

**Implementation:**

```python
def conjugate_gradient(A: torch.Tensor,
                      b: torch.Tensor,
                      x0: torch.Tensor = None,
                      max_iter: int = None,
                      tol: float = 1e-5) -> torch.Tensor:
    """
    Conjugate gradient method for solving Ax = b.

    Args:
        A: Symmetric positive definite matrix (n x n)
        b: Right-hand side (n,)
        x0: Initial guess (n,)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        x: Solution to Ax = b
    """
    n = b.size(0)
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    if max_iter is None:
        max_iter = n

    r = b - A @ x  # Residual
    p = r.clone()  # Search direction
    rsold = r @ r

    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r

        if torch.sqrt(rsnew) < tol:
            break

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x
```

### Preconditioning

**Idea:** Transform ill-conditioned system into well-conditioned

```
Solve: Ax = b
Precondition: M⁻¹Ax = M⁻¹b

where M ≈ A but easy to invert, and κ(M⁻¹A) << κ(A)
```

**Common Preconditioners:**
- Diagonal (Jacobi): M = diag(A)
- Incomplete Cholesky
- Multigrid methods

---

## Sparse Computations

### Sparse Matrix Formats

**COO (Coordinate Format):**
```
Store (row, col, value) triplets
Good for: Construction, conversion
```

**CSR (Compressed Sparse Row):**
```
Store: values, column indices, row pointers
Good for: Row slicing, matrix-vector products
```

**Implementation:**

```python
# Create sparse matrix
indices = torch.tensor([[0, 1, 1],
                       [2, 0, 2]])
values = torch.tensor([3.0, 4.0, 5.0])
sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 3))

# Convert to dense
dense = sparse_tensor.to_dense()
print(dense)
# tensor([[0., 0., 3.],
#         [4., 0., 5.]])

# Sparse matrix-vector product
x = torch.randn(3)
y = torch.sparse.mm(sparse_tensor, x.unsqueeze(1)).squeeze()
```

### Sparse Neural Networks

**Pruning:**
```python
def magnitude_pruning(tensor: torch.Tensor,
                     sparsity: float) -> torch.Tensor:
    """
    Zero out smallest magnitude weights.

    Args:
        tensor: Weight tensor
        sparsity: Fraction of weights to zero (0 to 1)

    Returns:
        Pruned tensor
    """
    if sparsity == 0:
        return tensor

    # Compute threshold
    k = int(sparsity * tensor.numel())
    threshold = torch.topk(torch.abs(tensor).flatten(), k,
                          largest=False)[0][-1]

    # Create mask
    mask = torch.abs(tensor) > threshold

    return tensor * mask


# Apply to model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.weight.data = magnitude_pruning(module.weight.data,
                                               sparsity=0.5)
```

---

## Mixed Precision Training

### FP16 Training Benefits

**Advantages:**
- 2x faster computation (on modern GPUs)
- 2x less memory usage
- Enables larger batch sizes

**Challenges:**
- Reduced precision can cause:
  - Gradient underflow (very small gradients → 0)
  - Weight update underflow
  - Loss of precision in accumulations

### Automatic Mixed Precision (AMP)

**Strategy:**
```
1. Store weights in FP32 (master weights)
2. Compute forward/backward in FP16
3. Use loss scaling to prevent gradient underflow
4. Update master weights in FP32
```

**Implementation:**

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for gradient scaling
scaler = GradScaler()

for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            predictions = model(x)
            loss = criterion(predictions, y)

        # Backward pass with scaled loss
        scaler.scale(loss).backward()

        # Gradient clipping (optional)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
```

### Loss Scaling

**Dynamic Loss Scaling:**
```
1. Multiply loss by scale factor (e.g., 2¹⁶)
2. Backpropagate scaled loss
3. Unscale gradients before optimizer step
4. Adjust scale factor:
   - If no overflow: increase scale
   - If overflow: decrease scale, skip update
```

**Manual Implementation:**

```python
class ManualMixedPrecision:
    def __init__(self, model, optimizer, initial_scale=2**16):
        self.model = model
        self.optimizer = optimizer
        self.scale = initial_scale
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 2000
        self.steps_since_scale = 0

    def step(self, loss):
        # Scale loss
        scaled_loss = loss * self.scale

        # Backward
        self.optimizer.zero_grad()
        scaled_loss.backward()

        # Check for inf/nan in gradients
        has_overflow = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    has_overflow = True
                    break

        if has_overflow:
            # Skip update, reduce scale
            self.scale *= self.backoff_factor
            self.steps_since_scale = 0
            print(f"Overflow detected, reducing scale to {self.scale}")
        else:
            # Unscale gradients
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data /= self.scale

            # Optimizer step
            self.optimizer.step()

            # Increase scale periodically
            self.steps_since_scale += 1
            if self.steps_since_scale >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_scale = 0
```

---

## Quick Reference

### Numerical Stability Checklist

- [ ] Use `log_softmax` instead of `log(softmax(x))`
- [ ] Use `binary_cross_entropy_with_logits` instead of `BCE(sigmoid(x), y)`
- [ ] Use `logsumexp` for log of sum of exponentials
- [ ] Check condition numbers of weight matrices
- [ ] Clip gradients if using RNNs or very deep networks
- [ ] Use batch normalization or layer normalization
- [ ] Initialize weights properly (Xavier, He initialization)
- [ ] Monitor for NaN/Inf during training
- [ ] Consider mixed precision for large models

### Common Numerical Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Exploding gradients | Loss → NaN, weights → Inf | Gradient clipping, lower LR |
| Vanishing gradients | No learning, gradients → 0 | ReLU, skip connections, better init |
| Underflow in FP16 | Gradients → 0 | Loss scaling, AMP |
| Overflow in softmax | NaN in probabilities | Use log_softmax |
| Ill-conditioned matrices | Unstable inversion | Regularization, preconditioning |

### PyTorch Stability Functions

```python
# Use these instead of manual implementations:
torch.logsumexp(x, dim=-1)
torch.softmax(x, dim=-1)
torch.log_softmax(x, dim=-1)
torch.sigmoid(x)
F.binary_cross_entropy_with_logits(logits, targets)
F.cross_entropy(logits, targets)  # Includes log_softmax
```

---

## References

1. **Numerical Linear Algebra** - Trefethen & Bau (comprehensive textbook)
2. **"What Every Computer Scientist Should Know About Floating-Point Arithmetic"** - Goldberg, 1991
3. **"Mixed Precision Training"** - Micikevicius et al., ICLR 2018
4. **Numerical Recipes** - Press et al. (practical algorithms)
5. **"Numerically Stable Hidden States in RNNs"** - Laurent & von Platen, 2019
6. **PyTorch AMP Documentation** - https://pytorch.org/docs/stable/amp.html
7. **NVIDIA Apex** - Mixed precision training tools
8. **"On Large-Batch Training for Deep Learning"** - Hoffer et al., 2017

---

*Ensure numerical stability and computational efficiency in neural network training and inference.*
