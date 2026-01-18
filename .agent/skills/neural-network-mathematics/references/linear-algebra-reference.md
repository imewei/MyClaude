# allow-torch
# Linear Algebra for Neural Networks - Complete Reference

Comprehensive reference for linear algebra concepts essential to understanding and implementing neural networks.

---

## Table of Contents

1. [Matrix Operations](#matrix-operations)
2. [Vector Spaces and Transformations](#vector-spaces-and-transformations)
3. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
4. [Matrix Decompositions](#matrix-decompositions)
5. [Tensor Operations](#tensor-operations)
6. [Norms and Distances](#norms-and-distances)
7. [Matrix Calculus](#matrix-calculus)
8. [Practical Applications in Neural Networks](#practical-applications)

---

## Matrix Operations

### Basic Operations

**Matrix Multiplication:**
```
C = AB where C[i,j] = Σₖ A[i,k] * B[k,j]

Dimensions: (m×n) × (n×p) → (m×p)
```

**Neural Network Context:**
```python
# Linear layer: y = Wx + b
# W: weight matrix (output_dim × input_dim)
# x: input vector (input_dim × 1)
# b: bias vector (output_dim × 1)

import numpy as np

W = np.random.randn(128, 784)  # 128 neurons, 784 input features
x = np.random.randn(784, 1)    # Input vector
b = np.random.randn(128, 1)    # Bias

y = W @ x + b  # Matrix-vector product
```

**Element-wise Operations (Hadamard Product):**
```python
# Element-wise multiplication: C = A ⊙ B
C = A * B  # In NumPy/PyTorch

# Used in backpropagation
gradient = upstream_gradient * local_gradient  # Element-wise
```

### Transpose

**Definition:**
```
Aᵀ[i,j] = A[j,i]

Properties:
- (Aᵀ)ᵀ = A
- (AB)ᵀ = BᵀAᵀ
- (A + B)ᵀ = Aᵀ + Bᵀ
```

**Neural Network Context:**
```python
# Forward pass: y = Wx
# Backward pass: ∂L/∂x = Wᵀ(∂L/∂y)

grad_x = W.T @ grad_y  # Transpose in backpropagation
```

### Matrix Inverse

**Definition:**
```
A⁻¹A = AA⁻¹ = I

Only exists if A is square and non-singular (det(A) ≠ 0)
```

**Neural Network Context:**
```python
# Solving normal equations: Wᵀ(Wx - y) = 0
# → W = (XᵀX)⁻¹Xᵀy

# Pseudo-inverse for non-square matrices
W = np.linalg.pinv(X.T @ X) @ X.T @ y
```

**Warning:** Direct inversion is numerically unstable. Use:
- QR decomposition
- Cholesky decomposition
- Iterative methods (gradient descent)

---

## Vector Spaces and Transformations

### Linear Transformations

**Definition:**
A function T: ℝⁿ → ℝᵐ is linear if:
1. T(u + v) = T(u) + T(v)
2. T(αu) = αT(u)

**Neural Network Interpretation:**
```python
# Linear layer is a linear transformation
def linear_layer(x, W, b):
    return W @ x + b  # Affine transformation (linear + translation)

# Without bias: purely linear
def linear_only(x, W):
    return W @ x
```

### Rank

**Definition:**
Rank(A) = dimension of column space = number of linearly independent columns

**Properties:**
- Rank(A) ≤ min(m, n) for m×n matrix
- Full rank: Rank(A) = min(m, n)
- Rank deficient: Rank(A) < min(m, n)

**Neural Network Context:**
```python
# Check weight matrix rank
rank = np.linalg.matrix_rank(W)

# Low rank indicates redundancy
if rank < min(W.shape):
    print(f"Weight matrix is rank deficient: {rank}/{min(W.shape)}")
    print("→ Some neurons may be redundant")
```

**Low-Rank Approximation:**
```python
# Compress weight matrix using SVD
U, S, Vt = np.linalg.svd(W, full_matrices=False)

# Keep top k singular values
k = 50
W_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Compression ratio: k(m+n) / (m*n)
```

### Orthogonality

**Definition:**
Vectors u and v are orthogonal if: uᵀv = 0

**Orthogonal Matrix:**
```
Q is orthogonal if: QᵀQ = QQᵀ = I
Equivalently: Qᵀ = Q⁻¹
```

**Properties:**
- Preserves lengths: ||Qx|| = ||x||
- Preserves angles
- Numerically stable

**Neural Network Applications:**

1. **Weight Initialization:**
```python
# Orthogonal initialization for RNNs
def orthogonal_init(n):
    # Generate random matrix
    H = np.random.randn(n, n)
    # QR decomposition gives orthogonal Q
    Q, R = np.linalg.qr(H)
    return Q

W_recurrent = orthogonal_init(256)
```

2. **Batch Normalization:**
```python
# Whitening transformation (orthogonal + scaling)
def whiten(X):
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = X_centered.T @ X_centered / len(X)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Whitening matrix (orthogonal + scaling)
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)) @ eigenvectors.T
    return (X_centered @ W)
```

---

## Eigenvalues and Eigenvectors

### Definitions

**Eigenvector:** Non-zero vector v such that Av = λv
**Eigenvalue:** Scalar λ corresponding to eigenvector v

```
Av = λv
(A - λI)v = 0
```

**Characteristic Equation:**
```
det(A - λI) = 0
```

### Properties

1. **Trace:** tr(A) = Σᵢ λᵢ (sum of eigenvalues)
2. **Determinant:** det(A) = ∏ᵢ λᵢ (product of eigenvalues)
3. **Symmetric matrices:** Real eigenvalues, orthogonal eigenvectors

### Neural Network Applications

#### 1. Gradient Flow Analysis

**Jacobian Eigenvalues:**
```python
# Analyze gradient flow through layer
def analyze_gradient_flow(W):
    eigenvalues = np.linalg.eigvals(W)

    max_eigenvalue = np.max(np.abs(eigenvalues))

    if max_eigenvalue > 1:
        print(f"⚠️ Exploding gradients risk: max |λ| = {max_eigenvalue:.2f}")
    elif max_eigenvalue < 0.1:
        print(f"⚠️ Vanishing gradients risk: max |λ| = {max_eigenvalue:.2f}")

    return eigenvalues

# For RNN weight matrix
W_recurrent = np.random.randn(256, 256) * 0.01
eigenvalues = analyze_gradient_flow(W_recurrent)
```

**Theory:**
- If max |λ| > 1: gradients explode exponentially
- If max |λ| < 1: gradients vanish exponentially
- Target: max |λ| ≈ 1 (edge of chaos)

#### 2. Power Iteration for Weight Spectral Norm

```python
def power_iteration(W, num_iterations=10):
    """Compute largest singular value (spectral norm) of W."""
    u = np.random.randn(W.shape[0], 1)
    u = u / np.linalg.norm(u)

    for _ in range(num_iterations):
        v = W.T @ u
        v = v / np.linalg.norm(v)

        u = W @ v
        u = u / np.linalg.norm(u)

    # Largest singular value
    sigma = (u.T @ W @ v)[0, 0]
    return sigma

# Spectral normalization (Miyato et al., 2018)
def spectral_norm_layer(W):
    sigma = power_iteration(W)
    return W / sigma  # Normalize weights
```

#### 3. Learning Rate Selection

```python
# Hessian eigenvalue approximation
def estimate_max_lr(model, loss_fn, data, tolerance=1e-6):
    """
    Estimate maximum learning rate using Hessian eigenvalue.

    Max LR ≈ 2 / λ_max(Hessian)
    """
    # Compute gradient
    loss = loss_fn(model(data))
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Random vector for power iteration
    v = [torch.randn_like(p) for p in model.parameters()]

    # Hessian-vector product: Hv
    grad_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))
    Hv = torch.autograd.grad(grad_v, model.parameters())

    # Rayleigh quotient: vᵀHv / vᵀv
    numerator = sum((hv * v_i).sum() for hv, v_i in zip(Hv, v))
    denominator = sum((v_i ** 2).sum() for v_i in v)

    lambda_max = (numerator / denominator).item()

    max_lr = 2.0 / (lambda_max + tolerance)
    return max_lr
```

---

## Matrix Decompositions

### Singular Value Decomposition (SVD)

**Definition:**
```
A = UΣVᵀ

Where:
- U: left singular vectors (m×m orthogonal)
- Σ: singular values (m×n diagonal)
- Vᵀ: right singular vectors (n×n orthogonal)
```

**Properties:**
- Singular values σᵢ ≥ 0
- Ordered: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0
- Rank(A) = number of non-zero singular values

**Neural Network Applications:**

#### 1. Weight Matrix Analysis
```python
def analyze_weights(W):
    """Analyze weight matrix using SVD."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    print(f"Singular values: {S[:10]}")
    print(f"Condition number: {S[0] / S[-1]:.2f}")

    # Effective rank (99% energy)
    energy = np.cumsum(S**2) / np.sum(S**2)
    effective_rank = np.argmax(energy > 0.99) + 1
    print(f"Effective rank: {effective_rank}/{len(S)}")

    # Spectral norm (largest singular value)
    spectral_norm = S[0]
    print(f"Spectral norm: {spectral_norm:.4f}")

    return U, S, Vt
```

#### 2. Low-Rank Compression
```python
def compress_layer(W, compression_ratio=0.5):
    """Compress weight matrix using low-rank approximation."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    # Keep top-k singular values
    k = int(len(S) * compression_ratio)

    W_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # Compute compression error
    error = np.linalg.norm(W - W_compressed) / np.linalg.norm(W)

    print(f"Compression: {W.size} → {k * (W.shape[0] + W.shape[1])} params")
    print(f"Relative error: {error:.4%}")

    return W_compressed
```

#### 3. Principal Component Analysis (PCA)
```python
def pca_features(X, n_components=50):
    """Reduce feature dimensionality using PCA."""
    # Center data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD of centered data
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project onto top principal components
    X_reduced = X_centered @ Vt[:n_components].T

    # Variance explained
    var_explained = S[:n_components]**2 / np.sum(S**2)
    print(f"Variance explained: {var_explained.sum():.2%}")

    return X_reduced, Vt[:n_components]
```

### Eigendecomposition

**For symmetric matrices:**
```
A = QΛQᵀ

Where:
- Q: orthogonal matrix of eigenvectors
- Λ: diagonal matrix of eigenvalues
```

**Application: Hessian Analysis**
```python
def analyze_hessian(model, loss_fn, data):
    """Analyze loss landscape curvature."""
    # Compute Hessian eigenvalues
    loss = loss_fn(model(data))

    # ... (Hessian computation)

    eigenvalues = np.linalg.eigvalsh(hessian)

    # Loss landscape analysis
    print(f"Min eigenvalue: {eigenvalues[0]:.4f}")
    print(f"Max eigenvalue: {eigenvalues[-1]:.4f}")
    print(f"Condition number: {eigenvalues[-1] / eigenvalues[0]:.2f}")

    if eigenvalues[0] < 0:
        print("⚠️ Negative eigenvalue → saddle point or local maximum")
    elif eigenvalues[0] > 0:
        print("✅ All positive → local minimum")
```

### QR Decomposition

**Definition:**
```
A = QR

Where:
- Q: orthogonal matrix (m×m)
- R: upper triangular (m×n)
```

**Application: Gram-Schmidt Orthogonalization**
```python
def orthogonalize_weights(W):
    """Orthogonalize weight matrix using QR decomposition."""
    Q, R = np.linalg.qr(W)
    return Q

# Use in RNN initialization
W_recurrent = np.random.randn(256, 256)
W_orthogonal = orthogonalize_weights(W_recurrent)
```

---

## Tensor Operations

### Tensor Basics

**Tensor:** Multi-dimensional array
- 0D: Scalar
- 1D: Vector
- 2D: Matrix
- 3D+: Tensor

**Shape Notation:**
```python
# Image batch
batch = torch.randn(32, 3, 224, 224)
# (batch_size, channels, height, width)

# Sequence batch
sequence = torch.randn(16, 100, 512)
# (batch_size, sequence_length, hidden_size)
```

### Einstein Summation

**Notation:**
```python
import torch

# Matrix multiplication: C_ij = Σₖ A_ik * B_kj
C = torch.einsum('ik,kj->ij', A, B)

# Batch matrix multiplication
# C[b,i,j] = Σₖ A[b,i,k] * B[b,k,j]
C = torch.einsum('bik,bkj->bij', A, B)

# Attention mechanism
# output[b,h,i,d] = Σⱼ attention[b,h,i,j] * V[b,h,j,d]
output = torch.einsum('bhij,bhjd->bhid', attention_weights, V)
```

### Broadcasting

**Rules:**
1. Right-align dimensions
2. Dimensions of size 1 are stretched
3. Missing dimensions are added as size 1

```python
# Example
A = torch.randn(3, 1, 5)  # Shape: (3, 1, 5)
B = torch.randn(4, 5)     # Shape: (4, 5)

# Broadcasting
#   A: (3, 1, 5)
#   B: (1, 4, 5)  <- implicit reshape
# A+B: (3, 4, 5)

C = A + B  # Works!
```

**Neural Network Application:**
```python
# Batch normalization
# X: (batch, features)
# mean: (features,)
# std: (features,)

X_normalized = (X - mean) / std  # Broadcasting over batch dimension
```

### Tensor Reshaping

```python
# Flatten
x = torch.randn(32, 3, 224, 224)
x_flat = x.view(32, -1)  # (32, 3*224*224)

# Transpose dimensions
x = torch.randn(32, 100, 512)
x_transposed = x.transpose(1, 2)  # (32, 512, 100)

# Permute
x = torch.randn(32, 3, 224, 224)  # NCHW
x_nhwc = x.permute(0, 2, 3, 1)   # NHWC: (32, 224, 224, 3)
```

---

## Norms and Distances

### Vector Norms

**L¹ Norm (Manhattan):**
```
||x||₁ = Σᵢ |xᵢ|
```

**L² Norm (Euclidean):**
```
||x||₂ = √(Σᵢ xᵢ²)
```

**L∞ Norm (Maximum):**
```
||x||∞ = maxᵢ |xᵢ|
```

**p-Norm:**
```
||x||ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p)
```

**Implementation:**
```python
def compute_norms(x):
    l1 = torch.norm(x, p=1)
    l2 = torch.norm(x, p=2)
    linf = torch.norm(x, p=float('inf'))
    return l1, l2, linf
```

### Matrix Norms

**Frobenius Norm:**
```
||A||_F = √(Σᵢⱼ Aᵢⱼ²) = √(tr(AᵀA))
```

**Spectral Norm (2-norm):**
```
||A||₂ = max σᵢ (largest singular value)
```

**Nuclear Norm:**
```
||A||_* = Σᵢ σᵢ (sum of singular values)
```

**Neural Network Applications:**
```python
# Weight decay (L² regularization)
l2_penalty = 0.5 * weight_decay * torch.sum(W ** 2)

# Gradient clipping (L² norm)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Spectral normalization
spectral_norm = torch.linalg.matrix_norm(W, ord=2)
```

---

## Matrix Calculus

### Gradient of Scalar Functions

**Function:** f: ℝⁿ → ℝ
**Gradient:** ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

**Examples:**
```
f(x) = xᵀAx  →  ∇f = (A + Aᵀ)x
f(x) = xᵀa   →  ∇f = a
f(x) = ||x||₂²  →  ∇f = 2x
```

### Jacobian Matrix

**Function:** f: ℝⁿ → ℝᵐ
**Jacobian:** J = ∂f/∂x with Jᵢⱼ = ∂fᵢ/∂xⱼ

**Example:**
```python
# Linear layer: f(x) = Wx + b
# Jacobian: ∂f/∂x = W

def linear_jacobian(W, x):
    return W  # Jacobian is just the weight matrix
```

### Matrix Derivatives

**Common Rules:**
```
∂/∂X (tr(X)) = I
∂/∂X (tr(AX)) = Aᵀ
∂/∂X (tr(XAXᵀ)) = X(A + Aᵀ)
∂/∂X (|X|) = |X|(X⁻¹)ᵀ  (determinant)
```

**Backpropagation:**
```
∂L/∂W = ∂L/∂y · ∂y/∂W

Where y = Wx:
∂y/∂W = xᵀ  (as each row of W gets multiplied by x)

Therefore:
∂L/∂W = (∂L/∂y)xᵀ  (outer product)
```

---

## Practical Applications

### 1. Weight Initialization

```python
def initialize_layer(fan_in, fan_out, activation='relu'):
    """Initialize weights based on He/Xavier."""
    if activation == 'relu':
        # He initialization
        std = np.sqrt(2.0 / fan_in)
    else:
        # Xavier initialization
        std = np.sqrt(2.0 / (fan_in + fan_out))

    W = np.random.randn(fan_out, fan_in) * std
    b = np.zeros((fan_out, 1))

    return W, b
```

### 2. Gradient Checking

```python
def numerical_gradient(f, x, eps=1e-7):
    """Compute numerical gradient using finite differences."""
    grad = np.zeros_like(x)

    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += eps

        x_minus = x.copy()
        x_minus.flat[i] -= eps

        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad

def check_gradient(analytical_grad, x, f, tolerance=1e-5):
    """Verify analytical gradient against numerical gradient."""
    numerical_grad = numerical_gradient(f, x)

    diff = np.linalg.norm(analytical_grad - numerical_grad)
    norm = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)

    relative_error = diff / (norm + 1e-8)

    if relative_error < tolerance:
        print(f"✅ Gradient check passed: {relative_error:.2e}")
    else:
        print(f"❌ Gradient check failed: {relative_error:.2e}")

    return relative_error < tolerance
```

### 3. Whitening Transformation

```python
def zca_whitening(X, epsilon=1e-5):
    """Zero-phase Component Analysis whitening."""
    # Center data
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean

    # Covariance matrix
    cov = (X_centered.T @ X_centered) / len(X)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # ZCA whitening matrix
    D = np.diag(1.0 / np.sqrt(eigenvalues + epsilon))
    W = eigenvectors @ D @ eigenvectors.T

    # Whiten
    X_whitened = X_centered @ W

    return X_whitened, W, mean
```

---

## Quick Reference

### Matrix Properties
| Property | Formula |
|----------|---------|
| Transpose | (AB)ᵀ = BᵀAᵀ |
| Inverse | (AB)⁻¹ = B⁻¹A⁻¹ |
| Trace | tr(AB) = tr(BA) |
| Determinant | det(AB) = det(A)det(B) |

### Common Derivatives
| Function | Derivative |
|----------|------------|
| xᵀAx | (A + Aᵀ)x |
| ||x||₂² | 2x |
| Wx | Wᵀ (transpose) |
| xᵀW | W |

### Norms
| Norm | Formula | Use Case |
|------|---------|----------|
| L¹ | Σ\|xᵢ\| | Sparse regularization |
| L² | √(Σxᵢ²) | Weight decay |
| Frobenius | √(ΣΣAᵢⱼ²) | Matrix regularization |
| Spectral | max(σᵢ) | Lipschitz constraint |

---

## References

1. "Deep Learning" - Goodfellow, Bengio, Courville (Chapter 2)
2. "The Matrix Cookbook" - Petersen & Pedersen
3. "Numerical Linear Algebra" - Trefethen & Bau
4. "Matrix Computations" - Golub & Van Loan
5. "Spectral Normalization for GANs" - Miyato et al., 2018
