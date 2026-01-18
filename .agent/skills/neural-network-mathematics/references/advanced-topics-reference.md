# allow-torch
# Advanced Mathematical Topics for Neural Networks - Complete Reference

Advanced mathematical frameworks for understanding deep learning theory, including differential geometry, functional analysis, graph theory, and topological data analysis.

---

## Table of Contents

1. [Differential Geometry](#differential-geometry)
2. [Functional Analysis](#functional-analysis)
3. [Graph Theory](#graph-theory)
4. [Stochastic Processes](#stochastic-processes)
5. [Topological Data Analysis](#topological-data-analysis)
6. [Measure Theory](#measure-theory)

---

## Differential Geometry

### Manifolds and Loss Landscapes

**Manifold:** Locally Euclidean space

**Neural Network Parameter Space:**
```
Parameter space: θ ∈ ℝⁿ
Loss function: L: ℝⁿ → ℝ
Loss landscape: (n+1)-dimensional surface
```

**Riemannian Geometry:**
```
Metric tensor g defines distance on manifold:
ds² = Σᵢⱼ gᵢⱼ dθᵢ dθⱼ

For loss landscape:
gᵢⱼ = ∂²L/∂θᵢ∂θⱼ (Hessian)
```

### Gradient Descent on Manifolds

**Euclidean Gradient:**
```
θₜ₊₁ = θₜ - η∇L(θₜ)
```

**Riemannian Gradient:**
```
θₜ₊₁ = Retractθₜ(-ηG⁻¹∇L(θₜ))

where:
- G: metric tensor
- Retract: projection back to manifold
```

**Natural Gradient Descent:**
```
θₜ₊₁ = θₜ - ηF⁻¹∇L(θₜ)

where F is Fisher Information Matrix:
Fᵢⱼ = E[∂log p(y|x,θ)/∂θᵢ · ∂log p(y|x,θ)/∂θⱼ]
```

**Implementation:**

```python
import torch

def fisher_information_matrix(model, dataloader, num_samples=1000):
    """
    Compute empirical Fisher Information Matrix.

    Args:
        model: Neural network
        dataloader: Data loader
        num_samples: Number of samples for estimation

    Returns:
        F: Fisher Information Matrix (flattened parameter space)
    """
    model.eval()

    # Get total number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    fisher = torch.zeros(n_params, n_params)

    count = 0
    for x, y in dataloader:
        if count >= num_samples:
            break

        # Forward pass
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Sample from predicted distribution
        sampled_y = torch.multinomial(torch.exp(log_probs), 1).squeeze()

        # Compute gradient of log p(y|x, θ)
        model.zero_grad()
        selected_log_probs = log_probs[range(len(sampled_y)), sampled_y]
        selected_log_probs.sum().backward()

        # Flatten gradients
        grads = torch.cat([p.grad.flatten() for p in model.parameters()
                          if p.grad is not None])

        # Accumulate outer product
        fisher += torch.outer(grads, grads)
        count += len(x)

    fisher /= count
    return fisher


def natural_gradient_step(model, loss, fisher, lr=0.01, damping=1e-4):
    """
    Perform natural gradient descent step.

    Args:
        model: Neural network
        loss: Loss value (for backward)
        fisher: Fisher Information Matrix
        lr: Learning rate
        damping: Damping term for numerical stability
    """
    model.zero_grad()
    loss.backward()

    # Flatten gradients
    grads = torch.cat([p.grad.flatten() for p in model.parameters()
                      if p.grad is not None])

    # Solve F·ng = g for natural gradient ng
    # Add damping for numerical stability
    fisher_damped = fisher + damping * torch.eye(fisher.size(0))
    natural_grad = torch.linalg.solve(fisher_damped, grads)

    # Update parameters
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                numel = p.numel()
                p.data -= lr * natural_grad[idx:idx+numel].view_as(p)
                idx += numel
```

### Curvature of Loss Landscape

**Gaussian Curvature:**
```
K = det(H) / (1 + ||∇L||²)^(n+1)

where H is Hessian matrix
```

**Mean Curvature:**
```
H = (1/n) tr(Hessian)
```

**Critical Points:**
```
Minimum: All eigenvalues of H > 0
Maximum: All eigenvalues of H < 0
Saddle: Mixed signs of eigenvalues
```

**Implementation:**

```python
def analyze_critical_point(model, loss_fn, X, y):
    """
    Analyze critical point via Hessian eigenvalues.

    Args:
        model: Neural network
        loss_fn: Loss function
        X, y: Data batch

    Returns:
        Analysis of critical point type
    """
    # Compute Hessian (expensive, use small model/batch)
    params = [p for p in model.parameters() if p.requires_grad]

    # First derivatives
    predictions = model(X)
    loss = loss_fn(predictions, y)

    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vector = torch.cat([g.flatten() for g in grads])

    # Second derivatives (Hessian)
    n_params = grad_vector.size(0)
    hessian = torch.zeros(n_params, n_params)

    for i in range(n_params):
        grad2 = torch.autograd.grad(grad_vector[i], params,
                                   retain_graph=True)
        hessian[i] = torch.cat([g.flatten() for g in grad2])

    # Analyze eigenvalues
    eigenvalues = torch.linalg.eigvalsh(hessian)

    positive = (eigenvalues > 1e-6).sum().item()
    negative = (eigenvalues < -1e-6).sum().item()
    zero = n_params - positive - negative

    if positive == n_params:
        point_type = "Local minimum"
    elif negative == n_params:
        point_type = "Local maximum"
    else:
        point_type = f"Saddle point ({negative} negative, {positive} positive)"

    return {
        'type': point_type,
        'eigenvalues': eigenvalues,
        'min_eigenvalue': eigenvalues.min().item(),
        'max_eigenvalue': eigenvalues.max().item()
    }
```

---

## Functional Analysis

### Universal Approximation Theorem

**Statement (1989):**
```
Any continuous function f: ℝⁿ → ℝᵐ can be approximated
arbitrarily well by a neural network with one hidden layer
and a non-polynomial activation function.

Formally:
∀ε > 0, ∃ network g such that ||f - g||∞ < ε
```

**Width vs Depth Trade-off:**
```
Shallow networks: Require exponentially many neurons
Deep networks: Can represent functions with polynomially many parameters
```

### Function Spaces

**L² Space:**
```
Space of square-integrable functions:
L²(Ω) = {f: ∫_Ω |f(x)|² dx < ∞}

Inner product: ⟨f, g⟩ = ∫ f(x)g(x) dx
Norm: ||f||₂ = √⟨f, f⟩
```

**Reproducing Kernel Hilbert Space (RKHS):**
```
Hilbert space H of functions with kernel k

Reproducing property:
f(x) = ⟨f, k(x, ·)⟩_H

Used in: Kernel methods, GPs, neural tangent kernel
```

### Neural Tangent Kernel (NTK)

**Definition:**
```
For infinitely wide network at initialization:

K(x, x') = E[∂f(x;θ)/∂θ · ∂f(x';θ)/∂θ]

where expectation is over initialization
```

**Properties:**
- Fixed during training (for infinite width)
- Reduces neural network to kernel regression
- Training dynamics become linear

**Implementation:**

```python
def neural_tangent_kernel(model, x1, x2):
    """
    Compute empirical Neural Tangent Kernel.

    Args:
        model: Neural network
        x1, x2: Input samples

    Returns:
        K: NTK matrix
    """
    model.eval()

    # Forward passes
    f1 = model(x1)
    f2 = model(x2)

    n1, n2 = len(x1), len(x2)
    n_out = f1.size(-1)

    # Compute kernel matrix
    K = torch.zeros(n1, n2)

    for i in range(n_out):
        # Jacobians ∂f_i/∂θ
        J1 = []
        for j in range(n1):
            model.zero_grad()
            f1[j, i].backward(retain_graph=True)
            grad = torch.cat([p.grad.flatten() for p in model.parameters()
                            if p.grad is not None])
            J1.append(grad)
        J1 = torch.stack(J1)

        J2 = []
        for j in range(n2):
            model.zero_grad()
            f2[j, i].backward(retain_graph=True)
            grad = torch.cat([p.grad.flatten() for p in model.parameters()
                            if p.grad is not None])
            J2.append(grad)
        J2 = torch.stack(J2)

        # K += J1 @ J2^T
        K += J1 @ J2.T

    return K
```

---

## Graph Theory

### Graph Neural Networks (GNNs)

**Graph Representation:**
```
G = (V, E)
- V: nodes (vertices)
- E: edges
- A: adjacency matrix (n × n)
- X: node features (n × d)
```

**Message Passing:**
```
h_v^(t+1) = UPDATE(h_v^(t), AGGREGATE({h_u^(t) : u ∈ N(v)}))

where:
- h_v: node v's representation
- N(v): neighbors of v
- t: layer/iteration
```

**Graph Convolutional Network (GCN):**
```
H^(t+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(t) W^(t))

where:
- Ã = A + I (adjacency with self-loops)
- D̃: degree matrix of Ã
- W: learnable weight matrix
- σ: activation function
```

**Implementation:**

```python
import torch
import torch.nn as nn

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, X: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Graph convolution operation.

        Args:
            X: Node features (n × d_in)
            adj: Adjacency matrix (n × n)

        Returns:
            Updated node features (n × d_out)
        """
        # Add self-loops
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

        # Degree matrix
        degree = adj_hat.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        norm_adj = degree_inv_sqrt.unsqueeze(1) * adj_hat * degree_inv_sqrt.unsqueeze(0)

        # Graph convolution: σ(D̃^(-1/2) Ã D̃^(-1/2) X W)
        support = X @ self.weight
        output = norm_adj @ support

        return output


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                out_features: int, num_layers: int = 2):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GraphConvLayer(in_features, hidden_features))

        for _ in range(num_layers - 2):
            self.layers.append(GraphConvLayer(hidden_features, hidden_features))

        self.layers.append(GraphConvLayer(hidden_features, out_features))

    def forward(self, X: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN.

        Args:
            X: Node features (n × d)
            adj: Adjacency matrix (n × n)

        Returns:
            Node embeddings (n × out_features)
        """
        for i, layer in enumerate(self.layers[:-1]):
            X = layer(X, adj)
            X = torch.relu(X)

        X = self.layers[-1](X, adj)
        return X
```

### Spectral Graph Theory

**Graph Laplacian:**
```
L = D - A

where:
- D: degree matrix (diagonal)
- A: adjacency matrix

Properties:
- L is positive semi-definite
- Smallest eigenvalue is 0
- Eigenvectors capture graph structure
```

**Normalized Laplacian:**
```
L_norm = I - D^(-1/2) A D^(-1/2)
```

**Spectral Clustering:**

```python
def spectral_clustering(adj: torch.Tensor, num_clusters: int):
    """
    Spectral clustering using graph Laplacian eigenvalues.

    Args:
        adj: Adjacency matrix (n × n)
        num_clusters: Number of clusters

    Returns:
        Cluster assignments
    """
    # Compute degree matrix
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

    # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
    norm_adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
    laplacian = torch.eye(adj.size(0)) - norm_adj

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

    # Use first k eigenvectors (k = num_clusters)
    embedding = eigenvectors[:, :num_clusters]

    # K-means on spectral embedding
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(embedding.numpy())

    return torch.tensor(clusters)
```

---

## Stochastic Processes

### Brownian Motion and Diffusion

**Wiener Process (Brownian Motion):**
```
W(t) ~ N(0, t)

Properties:
- W(0) = 0
- Independent increments
- Continuous paths
- W(t) - W(s) ~ N(0, t-s)
```

**Stochastic Differential Equation (SDE):**
```
dX(t) = μ(X(t), t)dt + σ(X(t), t)dW(t)

where:
- μ: drift
- σ: diffusion coefficient
- dW: Brownian motion increment
```

**Neural SDEs:**
```
Neural network as drift/diffusion:
dh(t) = f_θ(h(t), t)dt + g_θ(h(t), t)dW(t)
```

**Implementation:**

```python
class NeuralSDE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Drift network
        self.drift_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Diffusion network
        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, dt=0.01):
        """
        One step of Euler-Maruyama discretization.

        dX = μ(X,t)dt + σ(X,t)dW
        X(t+dt) ≈ X(t) + μ(X,t)dt + σ(X,t)√dt·ε, ε ~ N(0,1)
        """
        # Drift term
        drift = self.drift_net(x)

        # Diffusion term
        diffusion = self.diffusion_net(x)
        noise = torch.randn_like(x)

        # Euler-Maruyama step
        x_next = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt)) * noise

        return x_next
```

### Diffusion Models

**Forward Process (Add Noise):**
```
q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

xₜ = √(1-βₜ)xₜ₋₁ + √βₜ εₜ, εₜ ~ N(0, I)
```

**Reverse Process (Denoise):**
```
pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))
```

**Training Objective:**
```
L = E_t,x₀,ε[||ε - εθ(√ᾱₜx₀ + √(1-ᾱₜ)ε, t)||²]

where εθ is a neural network predicting noise
```

---

## Topological Data Analysis

### Persistent Homology

**Idea:** Track topological features across scales

**Filtration:**
```
K₀ ⊆ K₁ ⊆ ... ⊆ Kₙ

Sequence of simplicial complexes at increasing scales
```

**Persistence Diagram:**
```
Plot (birth_scale, death_scale) for each topological feature

Long-lived features: signal
Short-lived features: noise
```

**Application to Neural Networks:**
- Analyze decision boundary topology
- Study activation patterns
- Understand representation geometry

### Mapper Algorithm

**Steps:**
1. Cover space with overlapping neighborhoods
2. Cluster points within each neighborhood
3. Build graph where nodes are clusters, edges are overlaps

**Use Case:** Visualize high-dimensional neural network representations

---

## Measure Theory

### Probability Measures

**Measure Space:**
```
(Ω, Σ, μ)

where:
- Ω: sample space
- Σ: σ-algebra (measurable sets)
- μ: measure (assigns "size" to sets)
```

**Probability Measure:**
```
P: Σ → [0, 1]

Properties:
- P(Ω) = 1
- Countable additivity
```

**Radon-Nikodym Theorem:**
```
If P << Q (P absolutely continuous w.r.t. Q):
dP/dQ exists (Radon-Nikodym derivative)

Used in: Importance sampling, variational inference
```

### Change of Variables

**For Normalizing Flows:**
```
x = f(z), z ~ p_Z

p_X(x) = p_Z(f⁻¹(x)) |det(∂f⁻¹/∂x)|

Log-likelihood:
log p_X(x) = log p_Z(z) + log|det(∂f/∂z)|
```

---

## Quick Reference

### Advanced Mathematics → Neural Network Applications

| Math Topic | Neural Network Application |
|-----------|---------------------------|
| Differential Geometry | Loss landscape analysis, natural gradients |
| Functional Analysis | Universal approximation, NTK theory |
| Graph Theory | GNNs, relational learning |
| Stochastic Processes | Neural SDEs, diffusion models |
| Topology | Decision boundary analysis, TDA |
| Measure Theory | Probability theory, normalizing flows |

### Key Theorems

1. **Universal Approximation:** Neural networks can approximate any continuous function
2. **NTK Theory:** Infinite-width networks behave like kernel methods
3. **Lottery Ticket Hypothesis:** Sparse subnetworks exist that train effectively
4. **Double Descent:** Test error can decrease with more parameters (overparameterized regime)

---

## References

1. **Differential Geometry:**
   - "Riemannian Geometry" - do Carmo
   - "Natural Gradient Works Efficiently in Learning" - Amari, 1998

2. **Functional Analysis:**
   - "Functional Analysis" - Rudin
   - "Universal Approximation Bounds for Superpositions" - Barron, 1993

3. **Graph Theory:**
   - "Graph Representation Learning" - William Hamilton
   - "Semi-Supervised Classification with GCNs" - Kipf & Welling, ICLR 2017

4. **Stochastic Processes:**
   - "Introduction to Stochastic Processes" - Lawler
   - "Denoising Diffusion Probabilistic Models" - Ho et al., NeurIPS 2020
   - "Neural SDEs as Infinite-Dimensional GANs" - Kidger et al., ICML 2021

5. **Topology:**
   - "Computational Topology" - Edelsbrunner & Harer
   - "Topology and Data" - Carlsson, 2009

6. **Measure Theory:**
   - "Real Analysis" - Folland
   - "Normalizing Flows" - Papamakarios et al., 2021

---

*Advanced mathematical frameworks provide deep theoretical understanding of neural network behavior and capabilities.*
