# allow-torch
# Probability and Statistics for Neural Networks - Complete Reference

Essential probability theory and statistics for understanding neural networks, uncertainty, and Bayesian methods.

---

## Table of Contents

1. [Probability Basics](#probability-basics)
2. [Distributions](#distributions)
3. [Expectation and Variance](#expectation-and-variance)
4. [Bayesian Methods](#bayesian-methods)
5. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
6. [Information Theory Basics](#information-theory-basics)

---

## Probability Basics

### Random Variables

**Discrete:** P(X = x) for countable outcomes
**Continuous:** P(a ≤ X ≤ b) = ∫ₐᵇ p(x)dx

### Key Properties

```
Sum rule: P(X) = Σy P(X, Y)
Product rule: P(X, Y) = P(X|Y)P(Y)
Bayes' theorem: P(Y|X) = P(X|Y)P(Y) / P(X)
Independence: P(X, Y) = P(X)P(Y)
```

### Neural Network Applications

```python
# Classification as probability distribution
import torch.nn.functional as F

logits = model(x)
probs = F.softmax(logits, dim=-1)  # P(y|x)
predicted_class = torch.argmax(probs, dim=-1)

# Sampling from distribution
samples = torch.multinomial(probs, num_samples=1)
```

---

## Distributions

### Gaussian (Normal) Distribution

**PDF:**
```
N(x|μ, σ²) = 1/(√(2πσ²)) exp(-(x-μ)²/(2σ²))
```

**Properties:**
- Mean = μ
- Variance = σ²
- 68-95-99.7 rule

```python
import numpy as np

def gaussian(x, mu, sigma):
    return (1.0 / np.sqrt(2 * np.pi * sigma**2)) * \
           np.exp(-(x - mu)**2 / (2 * sigma**2))

# Sampling
samples = np.random.randn(1000) * sigma + mu
```

**Multivariate Gaussian:**
```
N(x|μ, Σ) = 1/√((2π)^d |Σ|) exp(-1/2(x-μ)ᵀΣ⁻¹(x-μ))

where:
- μ: mean vector (d-dimensional)
- Σ: covariance matrix (d×d)
```

### Bernoulli Distribution

**PMF:**
```
P(X = 1) = p
P(X = 0) = 1 - p
```

**Application: Binary Classification**
```python
# Sigmoid output gives Bernoulli parameter
logit = model(x)
p = torch.sigmoid(logit)  # P(y=1|x)

# Binary cross-entropy loss
loss = -y * torch.log(p) - (1 - y) * torch.log(1 - p)
```

### Categorical Distribution

**PMF:**
```
P(X = k) = pₖ where Σₖ pₖ = 1
```

**Application: Multi-class Classification**
```python
# Softmax gives categorical probabilities
logits = model(x)
probs = F.softmax(logits, dim=-1)  # [p₁, p₂, ..., pₖ]

# Cross-entropy loss
loss = F.cross_entropy(logits, targets)
# = -Σₖ yₖ log(pₖ)
```

---

## Expectation and Variance

### Definitions

**Expectation:**
```
E[X] = Σₓ x·P(X=x)  (discrete)
E[X] = ∫ x·p(x)dx   (continuous)
```

**Variance:**
```
Var[X] = E[(X - E[X])²] = E[X²] - (E[X])²
```

**Covariance:**
```
Cov[X, Y] = E[(X - E[X])(Y - E[Y])]
          = E[XY] - E[X]E[Y]
```

### Neural Network Applications

**1. Batch Normalization:**
```python
# Normalize using batch statistics
mean = x.mean(dim=0)  # E[X]
var = x.var(dim=0)    # Var[X]

x_normalized = (x - mean) / torch.sqrt(var + eps)
```

**2. Weight Initialization:**
```python
# Variance analysis for initialization
# Keep variance constant across layers

# Xavier/Glorot: Var[W] = 2/(fan_in + fan_out)
std = np.sqrt(2.0 / (fan_in + fan_out))
W = np.random.randn(fan_out, fan_in) * std

# He: Var[W] = 2/fan_in (for ReLU)
std = np.sqrt(2.0 / fan_in)
W = np.random.randn(fan_out, fan_in) * std
```

**3. Dropout as Expectation:**
```python
# During training: E[dropout(x)] = x * (1-p) + 0 * p = x * (1-p)
# During inference: scale to match expectation
if training:
    mask = (torch.rand_like(x) > dropout_rate).float()
    return x * mask / (1 - dropout_rate)  # Scale up
else:
    return x  # No scaling needed (expectation already correct)
```

---

## Bayesian Methods

### Bayes' Theorem for Neural Networks

**Posterior:**
```
P(θ|D) = P(D|θ)P(θ) / P(D)

where:
- θ: model parameters (weights)
- D: training data
- P(θ): prior distribution over parameters
- P(D|θ): likelihood
- P(θ|D): posterior distribution
```

### Maximum A Posteriori (MAP) Estimation

```
θ_MAP = argmax_θ P(θ|D)
      = argmax_θ P(D|θ)P(θ)
      = argmax_θ [log P(D|θ) + log P(θ)]
```

**Connection to Regularization:**
```python
# Negative log posterior
loss = -log P(D|θ) - log P(θ)
     = data_loss + regularization

# L2 regularization = Gaussian prior
# L1 regularization = Laplace prior
```

### Bayesian Neural Networks

```python
# Conceptual: distribution over weights
class BayesianLinear:
    def __init__(self, in_features, out_features):
        # Weight distribution parameters
        self.weight_mu = Parameter(torch.randn(out_features, in_features))
        self.weight_sigma = Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Sample weights from distribution
        epsilon = torch.randn_like(self.weight_sigma)
        weight = self.weight_mu + self.weight_sigma * epsilon

        return F.linear(x, weight)

# Prediction: marginalize over weight posterior
def predict_bayesian(model, x, num_samples=100):
    predictions = []
    for _ in range(num_samples):
        pred = model(x)  # Different weights each time
        predictions.append(pred)

    # Average predictions (Monte Carlo estimate)
    return torch.mean(torch.stack(predictions), dim=0)
```

### Uncertainty Quantification

**Aleatoric (Data) Uncertainty:**
```python
# Model outputs mean and variance
def heteroscedastic_loss(pred_mean, pred_log_var, target):
    """
    Gaussian negative log-likelihood with learned variance.
    """
    var = torch.exp(pred_log_var)
    loss = 0.5 * ((pred_mean - target)**2 / var + pred_log_var)
    return loss.mean()
```

**Epistemic (Model) Uncertainty:**
```python
# MC Dropout for uncertainty
def predict_with_uncertainty(model, x, num_samples=100):
    model.train()  # Keep dropout enabled

    predictions = []
    for _ in range(num_samples):
        pred = model(x)
        predictions.append(pred)

    predictions = torch.stack(predictions)

    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)  # Uncertainty estimate

    return mean, std
```

---

## Maximum Likelihood Estimation

### Principle

Find parameters that maximize likelihood of observed data:

```
θ_MLE = argmax_θ P(D|θ)
      = argmax_θ Πᵢ P(xᵢ|θ)
      = argmax_θ Σᵢ log P(xᵢ|θ)  (log-likelihood)
```

### Neural Network Training as MLE

**Regression (Gaussian likelihood):**
```
P(y|x, θ) = N(y|f_θ(x), σ²)

Log-likelihood:
log P(D|θ) = -1/(2σ²) Σᵢ (yᵢ - f_θ(xᵢ))²

Maximizing log-likelihood ≡ Minimizing MSE!
```

**Classification (Categorical likelihood):**
```
P(y|x, θ) = Categorical(y|softmax(f_θ(x)))

Log-likelihood:
log P(D|θ) = Σᵢ log p_θ(yᵢ|xᵢ)

Maximizing log-likelihood ≡ Minimizing cross-entropy!
```

```python
# MSE = Negative log-likelihood (Gaussian assumption)
mse_loss = F.mse_loss(predictions, targets)
# Equivalent to: -log N(targets|predictions, σ²)

# Cross-entropy = Negative log-likelihood (Categorical)
ce_loss = F.cross_entropy(logits, targets)
# Equivalent to: -log Categorical(targets|softmax(logits))
```

---

## Information Theory Basics

### Entropy

**Definition:** Average surprise/information content

```
H(X) = -Σₓ P(X=x) log P(X=x) = E[-log P(X)]
```

**Properties:**
- Always ≥ 0
- Maximum when distribution is uniform
- Measures uncertainty

```python
def entropy(probs):
    """Compute entropy of discrete distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-10))

# Example: uniform distribution has maximum entropy
uniform = torch.ones(10) / 10
H_uniform = entropy(uniform)  # = log(10) ≈ 2.3

peaked = torch.tensor([0.9, 0.1])
H_peaked = entropy(peaked)  # < H_uniform
```

### Cross-Entropy

**Definition:** Expected surprise under wrong distribution

```
H(P, Q) = -Σₓ P(x) log Q(x)
```

**Neural Network Loss:**
```python
# Cross-entropy between true labels (P) and predictions (Q)
def cross_entropy(targets, predictions):
    """
    targets: true distribution P (one-hot or probabilities)
    predictions: predicted distribution Q (probabilities)
    """
    return -torch.sum(targets * torch.log(predictions + 1e-10))

# PyTorch cross-entropy combines softmax + log + nll
loss = F.cross_entropy(logits, target_class_indices)
```

### KL Divergence

**Definition:** Relative entropy between distributions

```
D_KL(P||Q) = Σₓ P(x) log[P(x)/Q(x)]
           = H(P, Q) - H(P)
```

**Properties:**
- Always ≥ 0
- = 0 if and only if P = Q
- NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)

```python
def kl_divergence(p, q):
    """KL divergence D_KL(P||Q)."""
    return torch.sum(p * torch.log((p + 1e-10) / (q + 1e-10)))

# KL divergence in VAE loss
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence D_KL(N(μ,σ²)||N(0,1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
```

### Mutual Information

**Definition:** Information shared between variables

```
I(X; Y) = H(X) - H(X|Y)
        = D_KL(P(X,Y)||P(X)P(Y))
```

**Neural Network Application:**
```python
# Information Bottleneck principle
# min I(X; Z) - β·I(Z; Y)
# Compress input X to representation Z while preserving task info Y
```

---

## Statistical Learning Theory

### Bias-Variance Tradeoff

```
Expected Error = Bias² + Variance + Irreducible Error

Bias = E[f̂(x)] - f(x)     (underfitting)
Variance = E[(f̂(x) - E[f̂(x)])²]  (overfitting)
```

**Neural Network Context:**
```
Small model → High bias, low variance
Large model → Low bias, high variance

Sweet spot: Minimize total error
```

### Generalization

**Training Error vs Test Error:**
```
Test Error = Training Error + Generalization Gap

Generalization gap increases with:
- Model complexity
- Insufficient training data
- Optimization issues
```

### Probably Approximately Correct (PAC) Learning

**Sample Complexity:** How much data needed to learn?

```
With probability ≥ 1-δ:
|Error_test - Error_train| ≤ ε

Sample size n ~ O((d/ε²) log(1/δ))
where d = effective dimension of hypothesis space
```

---

## Quick Reference

### Common Distributions
| Distribution | Parameters | Mean | Variance |
|--------------|------------|------|----------|
| Bernoulli | p | p | p(1-p) |
| Gaussian | μ, σ² | μ | σ² |
| Categorical | p₁,...,pₖ | - | - |

### Information Measures
| Measure | Formula | Interpretation |
|---------|---------|----------------|
| Entropy | -Σ p(x) log p(x) | Uncertainty |
| Cross-Entropy | -Σ p(x) log q(x) | Loss function |
| KL Divergence | Σ p(x) log(p(x)/q(x)) | Distance |

### Loss Functions
| Task | Loss | Probabilistic View |
|------|------|-------------------|
| Regression | MSE | -log N(y\|ŷ, σ²) |
| Binary Classification | BCE | -log Bernoulli(y\|p) |
| Multi-class | CE | -log Categorical(y\|p) |

---

## References

1. "Pattern Recognition and Machine Learning" - Christopher Bishop
2. "Deep Learning" - Goodfellow, Bengio, Courville (Chapter 3)
3. "Information Theory, Inference, and Learning Algorithms" - David MacKay
4. "Bayesian Deep Learning" - Yarin Gal thesis
5. "Weight Uncertainty in Neural Networks" - Blundell et al., 2015
