# allow-torch
# Information Theory for Neural Networks - Complete Reference

Information-theoretic concepts essential for understanding loss functions, model compression, and learning dynamics in neural networks.

---

## Table of Contents

1. [Information Theory Fundamentals](#information-theory-fundamentals)
2. [Entropy and Cross-Entropy](#entropy-and-cross-entropy)
3. [KL Divergence and Mutual Information](#kl-divergence-and-mutual-information)
4. [Information Bottleneck](#information-bottleneck)
5. [Model Compression](#model-compression)
6. [Variational Inference](#variational-inference)

---

## Information Theory Fundamentals

### Shannon Entropy

**Definition:** Average information content of a random variable

**Discrete:**
```
H(X) = -Σₓ P(X=x) log P(X=x)
     = E[-log P(X)]

Units: bits (log₂) or nats (ln)
```

**Properties:**
- H(X) ≥ 0 (always non-negative)
- H(X) = 0 iff X is deterministic
- H(X) is maximized when X is uniform
- H(X) ≤ log|X| (maximum for uniform distribution)

**Continuous (Differential Entropy):**
```
h(X) = -∫ p(x) log p(x) dx
```

**Python Implementation:**

```python
import torch
import numpy as np

def entropy(probs: torch.Tensor, base: str = 'e') -> torch.Tensor:
    """
    Compute entropy of discrete distribution.

    Args:
        probs: Probability distribution (must sum to 1)
        base: 'e' for nats, '2' for bits

    Returns:
        Entropy value
    """
    # Avoid log(0)
    probs = torch.clamp(probs, min=1e-10)

    if base == 'e':
        log_probs = torch.log(probs)
    elif base == '2':
        log_probs = torch.log2(probs)
    else:
        raise ValueError("base must be 'e' or '2'")

    return -torch.sum(probs * log_probs)


# Example: Uniform distribution has maximum entropy
uniform = torch.ones(10) / 10
print(f"Uniform entropy: {entropy(uniform, base='2'):.4f} bits")  # = log₂(10) ≈ 3.32

# Peaked distribution has lower entropy
peaked = torch.tensor([0.9, 0.05, 0.05])
print(f"Peaked entropy: {entropy(peaked, base='2'):.4f} bits")  # < log₂(3)
```

### Joint, Conditional, and Marginal Entropy

**Joint Entropy:**
```
H(X, Y) = -Σₓ Σᵧ P(X=x, Y=y) log P(X=x, Y=y)
```

**Conditional Entropy:**
```
H(Y|X) = Σₓ P(X=x) H(Y|X=x)
       = -Σₓ Σᵧ P(X=x, Y=y) log P(Y=y|X=x)
       = H(X, Y) - H(X)
```

**Chain Rule:**
```
H(X, Y) = H(X) + H(Y|X)
        = H(Y) + H(X|Y)
```

**Python Implementation:**

```python
def joint_entropy(joint_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute joint entropy H(X, Y).

    Args:
        joint_probs: Joint probability distribution P(X, Y)
                    Shape: (|X|, |Y|)
    """
    joint_probs = torch.clamp(joint_probs, min=1e-10)
    return -torch.sum(joint_probs * torch.log(joint_probs))


def conditional_entropy(joint_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute conditional entropy H(Y|X).

    Args:
        joint_probs: Joint probability distribution P(X, Y)
                    Shape: (|X|, |Y|)
    """
    # P(X) = Σᵧ P(X, Y)
    marginal_x = joint_probs.sum(dim=1, keepdim=True)

    # P(Y|X) = P(X, Y) / P(X)
    conditional_probs = joint_probs / torch.clamp(marginal_x, min=1e-10)

    # H(Y|X) = -Σₓ P(X) Σᵧ P(Y|X) log P(Y|X)
    conditional_probs = torch.clamp(conditional_probs, min=1e-10)
    return -torch.sum(joint_probs * torch.log(conditional_probs))
```

---

## Entropy and Cross-Entropy

### Cross-Entropy

**Definition:** Expected surprise under wrong distribution

**Formula:**
```
H(P, Q) = -Σₓ P(x) log Q(x)
        = E_P[-log Q(X)]

where:
- P: true distribution
- Q: predicted distribution
```

**Relation to Entropy:**
```
H(P, Q) = H(P) + D_KL(P||Q)

where D_KL is KL divergence
```

**Neural Network Loss Function:**

```python
def cross_entropy_loss(predictions: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for multi-class classification.

    Args:
        predictions: Logits (unnormalized scores), shape (N, C)
        targets: True class indices, shape (N,)

    Returns:
        Cross-entropy loss (scalar)
    """
    # Compute log-softmax for numerical stability
    log_probs = torch.log_softmax(predictions, dim=1)

    # Gather log probabilities of true classes
    # -Σᵢ log Q(yᵢ|xᵢ)
    loss = -log_probs[range(len(targets)), targets].mean()

    return loss


# PyTorch built-in (combines log_softmax + nll_loss)
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(predictions, targets)
```

### Binary Cross-Entropy

**Formula:**
```
BCE = -[y log(p) + (1-y) log(1-p)]

where:
- y ∈ {0, 1}: true label
- p ∈ [0, 1]: predicted probability
```

**Connection to Likelihood:**
```
Bernoulli distribution: P(y|p) = p^y (1-p)^(1-y)

Log-likelihood: log P(y|p) = y log p + (1-y) log(1-p)

Maximizing log-likelihood = Minimizing BCE
```

**Python Implementation:**

```python
def binary_cross_entropy(predictions: torch.Tensor,
                        targets: torch.Tensor,
                        from_logits: bool = False) -> torch.Tensor:
    """
    Binary cross-entropy loss.

    Args:
        predictions: Predicted probabilities or logits, shape (N,)
        targets: True labels {0, 1}, shape (N,)
        from_logits: If True, predictions are logits

    Returns:
        BCE loss (scalar)
    """
    if from_logits:
        # Numerically stable version using log-sum-exp trick
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets.float()
        )
    else:
        # Standard BCE
        predictions = torch.clamp(predictions, min=1e-7, max=1-1e-7)
        loss = -(targets * torch.log(predictions) +
                (1 - targets) * torch.log(1 - predictions))
        loss = loss.mean()

    return loss


# PyTorch built-in
criterion = torch.nn.BCEWithLogitsLoss()  # More stable
loss = criterion(logits, targets.float())
```

### Label Smoothing

**Motivation:** Hard targets (one-hot) can cause:
- Overconfidence
- Poor calibration
- Sensitivity to label noise

**Formula:**
```
Smoothed target:
q(k|x) = (1-ε)·y(k) + ε/K

where:
- y(k): one-hot label
- ε: smoothing parameter (e.g., 0.1)
- K: number of classes
```

**Implementation:**

```python
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor,
               targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Logits, shape (N, C)
            targets: True class indices, shape (N,)
        """
        num_classes = predictions.size(-1)
        log_probs = torch.log_softmax(predictions, dim=-1)

        # Create smoothed targets
        # (1-ε) on correct class, ε/(K-1) on others
        # Equivalent to: (1-ε) * one_hot + ε * uniform
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        smoothed_targets = (1 - self.epsilon) * targets_one_hot + \
                          self.epsilon / num_classes

        # Cross-entropy with smoothed targets
        loss = -torch.sum(smoothed_targets * log_probs, dim=-1).mean()

        return loss


# Usage
criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
loss = criterion(logits, targets)
```

---

## KL Divergence and Mutual Information

### Kullback-Leibler (KL) Divergence

**Definition:** Relative entropy between two distributions

**Formula:**
```
D_KL(P||Q) = Σₓ P(x) log[P(x)/Q(x)]
           = E_P[log P(X) - log Q(X)]
           = H(P, Q) - H(P)
```

**Properties:**
- D_KL(P||Q) ≥ 0 (Gibbs' inequality)
- D_KL(P||Q) = 0 iff P = Q
- NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
- NOT a distance metric (violates triangle inequality)

**Forward vs Reverse KL:**

```
Forward KL: D_KL(P||Q) = ∫ P(x) log[P(x)/Q(x)] dx
- Minimizing: Q covers all modes of P (mode-covering)
- Used in: Maximum likelihood estimation

Reverse KL: D_KL(Q||P) = ∫ Q(x) log[Q(x)/P(x)] dx
- Minimizing: Q focuses on single mode of P (mode-seeking)
- Used in: Variational inference
```

**Python Implementation:**

```python
def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence D_KL(P||Q).

    Args:
        p: True distribution P(x)
        q: Approximate distribution Q(x)

    Returns:
        KL divergence (scalar)
    """
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)

    return torch.sum(p * (torch.log(p) - torch.log(q)))


def kl_div_gaussian(mu1: torch.Tensor, logvar1: torch.Tensor,
                   mu2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two Gaussian distributions.

    D_KL(N(μ₁, σ₁²) || N(μ₂, σ₂²))
    = 0.5 * [log(σ₂²/σ₁²) + (σ₁² + (μ₁-μ₂)²)/σ₂² - 1]

    Args:
        mu1, logvar1: Mean and log-variance of first Gaussian
        mu2, logvar2: Mean and log-variance of second Gaussian
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2)**2) / var2 - 1)
    return kl.sum()


# Special case: KL from N(μ, σ²) to N(0, 1) (used in VAE)
def kl_to_standard_normal(mu: torch.Tensor,
                         logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence from N(μ, σ²) to standard normal N(0, 1)."""
    # D_KL(N(μ, σ²) || N(0, 1)) = 0.5 * [σ² + μ² - 1 - log σ²]
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)
```

### Mutual Information

**Definition:** Information shared between two random variables

**Formula:**
```
I(X; Y) = D_KL(P(X,Y) || P(X)P(Y))
        = H(X) - H(X|Y)
        = H(Y) - H(Y|X)
        = H(X) + H(Y) - H(X, Y)
```

**Properties:**
- I(X; Y) ≥ 0
- I(X; Y) = 0 iff X and Y are independent
- Symmetric: I(X; Y) = I(Y; X)
- I(X; X) = H(X) (self-information)

**Neural Network Applications:**
- Feature selection
- Information bottleneck theory
- Disentangled representations

**Python Implementation:**

```python
def mutual_information(joint_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute mutual information I(X; Y).

    Args:
        joint_probs: Joint probability P(X, Y), shape (|X|, |Y|)

    Returns:
        Mutual information (scalar)
    """
    # Marginals
    p_x = joint_probs.sum(dim=1, keepdim=True)  # P(X)
    p_y = joint_probs.sum(dim=0, keepdim=True)  # P(Y)

    # Product of marginals
    p_x_p_y = p_x @ p_y  # P(X)P(Y)

    # MI = Σₓᵧ P(X,Y) log[P(X,Y) / (P(X)P(Y))]
    joint_probs = torch.clamp(joint_probs, min=1e-10)
    p_x_p_y = torch.clamp(p_x_p_y, min=1e-10)

    return torch.sum(joint_probs * torch.log(joint_probs / p_x_p_y))


# Alternative: I(X; Y) = H(X) + H(Y) - H(X, Y)
def mutual_information_from_entropies(joint_probs: torch.Tensor) -> torch.Tensor:
    h_x = entropy(joint_probs.sum(dim=1))
    h_y = entropy(joint_probs.sum(dim=0))
    h_xy = joint_entropy(joint_probs)

    return h_x + h_y - h_xy
```

### Jensen-Shannon Divergence

**Definition:** Symmetric version of KL divergence

**Formula:**
```
D_JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)

where M = 0.5 * (P + Q)
```

**Properties:**
- Symmetric: D_JS(P||Q) = D_JS(Q||P)
- Bounded: 0 ≤ D_JS(P||Q) ≤ log 2
- Square root is a metric: √D_JS satisfies triangle inequality

**Python Implementation:**

```python
def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon divergence between P and Q.

    Args:
        p, q: Probability distributions

    Returns:
        JS divergence (scalar)
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
```

---

## Information Bottleneck

### Theory

**Principle:** Learn representation Z of input X that:
1. Compresses X (minimize I(X; Z))
2. Preserves information about Y (maximize I(Z; Y))

**Objective:**
```
min I(X; Z) - β·I(Z; Y)

where β controls compression-accuracy tradeoff
```

**Connection to Deep Learning:**
- Early layers: compress input (reduce I(X; Z))
- Later layers: extract task-relevant features (increase I(Z; Y))
- Training dynamics: fit → compress phases

**Information Plane:**
```
Plot I(X; Z) vs I(Z; Y) for each layer Z
- Shows information flow through network
- Visualizes compression and relevance
```

### Variational Information Bottleneck

**Objective:**
```
max_θ I(Z; Y) - βI(X; Z)

Approximate with variational bounds:
max_θ E[log p(Y|Z)] - βD_KL(p(Z|X)||r(Z))

where:
- p(Z|X): encoder
- p(Y|Z): decoder
- r(Z): prior (typically N(0, I))
```

**Implementation:**

```python
class VariationalIB(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int,
                output_dim: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta

        # Encoder: X → Z (mean and log-variance)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * latent_dim)  # mu and logvar
        )

        # Decoder: Z → Y
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim)
        )

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ·ε, ε ~ N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_pred = self.decoder(z)
        return y_pred, mu, logvar

    def loss(self, x, y_true):
        """VIB loss: -I(Z; Y) + β·I(X; Z)."""
        y_pred, mu, logvar = self.forward(x)

        # Task loss: -E[log p(Y|Z)]
        task_loss = torch.nn.functional.cross_entropy(y_pred, y_true)

        # Compression loss: I(X; Z) ≈ D_KL(p(Z|X)||N(0,I))
        kl_loss = kl_to_standard_normal(mu, logvar) / len(x)

        # Total loss
        return task_loss + self.beta * kl_loss
```

---

## Model Compression

### Knowledge Distillation

**Idea:** Transfer knowledge from large teacher model to small student

**Distillation Loss:**
```
L = αL_hard + (1-α)L_soft

L_hard = CrossEntropy(student_logits, true_labels)
L_soft = KL(teacher_soft || student_soft)

where soft probabilities use temperature T:
pᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)
```

**Temperature Effect:**
```
T = 1: Standard softmax
T > 1: Softer probabilities (more uniform, reveals dark knowledge)
T → ∞: Uniform distribution
```

**Implementation:**

```python
class DistillationLoss(torch.nn.Module):
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            true_labels: Ground truth labels
        """
        # Hard loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, true_labels)

        # Soft loss (KL divergence with temperature scaling)
        student_soft = torch.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_soft = torch.softmax(
            teacher_logits / self.temperature, dim=-1
        )

        # KL divergence (scaled by T² to match gradients)
        soft_loss = torch.nn.functional.kl_div(
            student_soft, teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# Training loop
teacher_model.eval()
student_model.train()

distillation_criterion = DistillationLoss(temperature=3.0, alpha=0.5)

for x, y in dataloader:
    with torch.no_grad():
        teacher_logits = teacher_model(x)

    student_logits = student_model(x)

    loss = distillation_criterion(student_logits, teacher_logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Entropy-Based Pruning

**Idea:** Prune neurons/channels with low information content

**Neuron Importance:**
```
Importance(zᵢ) = H(output) with zᵢ - H(output) without zᵢ

Prune neurons with low importance
```

---

## Variational Inference

### Variational Autoencoder (VAE)

**Objective:**
```
max_θ,φ E_q[log p(X|Z)] - D_KL(q(Z|X)||p(Z))

where:
- p(X|Z): decoder (likelihood)
- q(Z|X): encoder (variational posterior)
- p(Z): prior (typically N(0, I))
```

**ELBO (Evidence Lower Bound):**
```
log p(X) ≥ E_q[log p(X|Z)] - D_KL(q(Z|X)||p(Z))

Maximizing ELBO ≈ Maximizing log p(X)
```

**Implementation:**

```python
class VAE(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        # Encoder: X → q(Z|X) = N(μ(X), σ²(X))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2 * latent_dim)  # mu and logvar
        )

        # Decoder: Z → p(X|Z)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim),
            torch.nn.Sigmoid()  # For binary data
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(self, x):
        """VAE loss = -ELBO."""
        x_recon, mu, logvar = self.forward(x)

        # Reconstruction loss: -E[log p(X|Z)]
        recon_loss = torch.nn.functional.binary_cross_entropy(
            x_recon, x, reduction='sum'
        )

        # KL divergence: D_KL(q(Z|X)||p(Z))
        kl_loss = kl_to_standard_normal(mu, logvar)

        return recon_loss + kl_loss
```

---

## Quick Reference

### Loss Functions and Information Theory

| Loss Function | Information-Theoretic Interpretation |
|--------------|-------------------------------------|
| Cross-Entropy | H(P, Q) = H(P) + D_KL(P\\|\\|Q) |
| MSE | -log N(y\\|ŷ, σ²) (Gaussian MLE) |
| KL Divergence | Relative entropy D_KL(P\\|\\|Q) |
| VAE Loss | -ELBO = Recon + KL to prior |
| Distillation | αCE + (1-α)KL(teacher\\|\\|student) |

### Key Inequalities

```
1. Non-negativity of KL:
   D_KL(P||Q) ≥ 0, equality iff P = Q

2. Information inequality:
   I(X; Y) ≥ 0, equality iff X ⊥ Y

3. Data processing inequality:
   I(X; Y) ≥ I(X; f(Y)) for any function f

4. Chain rule for entropy:
   H(X, Y) = H(X) + H(Y|X)

5. Conditional reduces entropy:
   H(Y|X) ≤ H(Y), equality iff X ⊥ Y
```

---

## References

1. **Elements of Information Theory** - Cover & Thomas (comprehensive textbook)
2. **"Deep Learning and the Information Bottleneck Principle"** - Tishby & Zaslavsky, 2015
3. **"Distilling the Knowledge in a Neural Network"** - Hinton et al., 2015
4. **"Auto-Encoding Variational Bayes"** - Kingma & Welling, ICLR 2014
5. **"β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"** - Higgins et al., ICLR 2017
6. **"On the Information Bottleneck Theory of Deep Learning"** - Saxe et al., ICLR 2018
7. **Deep Learning** - Goodfellow et al. (Chapter 3: Information Theory)
8. **"Opening the Black Box of Deep Neural Networks via Information"** - Schwartz-Ziv & Tishby, 2017

---

*Understand information flow and compression in neural networks through information-theoretic principles.*
