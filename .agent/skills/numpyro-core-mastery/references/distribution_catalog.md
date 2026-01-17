# NumPyro Distribution Catalog

Quick reference for NumPyro distributions with use cases and parameterizations.

## Continuous Distributions

### Normal (Gaussian)
```python
dist.Normal(loc=0, scale=1)
```
**Support**: (-∞, +∞)
**Use**: General real-valued parameters, measurement errors
**Conjugate**: Normal-Normal

### StudentT
```python
dist.StudentT(df=3, loc=0, scale=1)
```
**Support**: (-∞, +∞)
**Use**: Heavy-tailed alternative to Normal, robust regression
**Note**: df=1 is Cauchy,  df→∞ is Normal

### Cauchy
```python
dist.Cauchy(loc=0, scale=1)
```
**Support**: (-∞, +∞)
**Use**: Very heavy tails, regularizing priors (horseshoe alternative)

### Exponential
```python
dist.Exponential(rate=1)
```
**Support**: [0, +∞)
**Use**: Waiting times, lifetimes, rate parameters
**Conjugate**: Gamma-Exponential

### Gamma
```python
dist.Gamma(concentration=2, rate=1)
```
**Support**: (0, +∞)
**Use**: Positive continuous (precision, variance, rates)
**Conjugate**: Gamma-Poisson, Gamma-Exponential

### HalfNormal
```python
dist.HalfNormal(scale=1)
```
**Support**: [0, +∞)
**Use**: Scale parameters (σ), hierarchical model variances

### HalfCauchy
```python
dist.HalfCauchy(scale=1)
```
**Support**: [0, +∞)
**Use**: Heavy-tailed scale priors, robust to hyperparameter choice

### LogNormal
```python
dist.LogNormal(loc=0, scale=1)
```
**Support**: (0, +∞)
**Use**: Positive with multiplicative effects, income distributions
**Note**: log(X) ~ Normal(loc, scale)

### Beta
```python
dist.Beta(concentration1=2, concentration0=2)
```
**Support**: [0, 1]
**Use**: Probabilities, proportions
**Conjugate**: Beta-Binomial
**Special**: Beta(1, 1) is Uniform(0, 1)

### Uniform
```python
dist.Uniform(low=0, high=1)
```
**Support**: [low, high]
**Use**: Flat priors, bounded parameters

### Dirichlet
```python
dist.Dirichlet(concentration=[1, 1, 1])
```
**Support**: Simplex (sums to 1)
**Use**: Mixture weights, categorical probabilities
**Conjugate**: Dirichlet-Multinomial

## Discrete Distributions

### Bernoulli
```python
dist.Bernoulli(probs=0.5)  # or logits=0
```
**Support**: {0, 1}
**Use**: Binary outcomes, coin flips
**Conjugate**: Beta-Bernoulli

### Binomial
```python
dist.Binomial(total_count=10, probs=0.5)
```
**Support**: {0, 1, ..., total_count}
**Use**: Number of successes in trials
**Conjugate**: Beta-Binomial

### Categorical
```python
dist.Categorical(probs=[0.2, 0.3, 0.5])  # or logits
```
**Support**: {0, 1, ..., K-1}
**Use**: Multiclass classification, discrete choices

### Poisson
```python
dist.Poisson(rate=5)
```
**Support**: {0, 1, 2, ...}
**Use**: Count data, events per interval
**Conjugate**: Gamma-Poisson

### NegativeBinomial
```python
dist.NegativeBinomial(mean=5, concentration=2)
```
**Support**: {0, 1, 2, ...}
**Use**: Overdispersed counts (variance > mean)
**Note**: Poisson-Gamma mixture

### ZeroInflatedPoisson
```python
dist.ZeroInflatedPoisson(rate=5, gate=0.2)
```
**Support**: {0, 1, 2, ...}
**Use**: Count data with excess zeros
**Parameters**: gate = P(extra zero)

## Multivariate Distributions

### MultivariateNormal
```python
# Full covariance
dist.MultivariateNormal(loc=mu, covariance_matrix=Sigma)

# Cholesky factor (more efficient)
dist.MultivariateNormal(loc=mu, scale_tril=L)

# Precision matrix
dist.MultivariateNormal(loc=mu, precision_matrix=Sigma_inv)
```
**Use**: Correlated Gaussians, GP priors

### LKJ (Correlation Matrices)
```python
dist.LKJ(dimension=3, concentration=2)
```
**Support**: Correlation matrices
**Use**: Prior on correlation structure
**Note**: concentration=1 is uniform, >1 concentrates near identity

### Wishart
```python
dist.Wishart(df=10, scale_matrix=S)
```
**Support**: Positive definite matrices
**Use**: Prior on covariance/precision matrices

## Mixture Distributions

### MixtureSameFamily
```python
dist.MixtureSameFamily(
    dist.Categorical(probs=[0.3, 0.7]),       # Mixture weights
    dist.Normal(jnp.array([-2, 2]), jnp.array([1, 1]))  # Components
)
```
**Use**: Gaussian mixtures, clustering models

## Common Use Cases

### Regression Parameters
```python
alpha = numpyro.sample('alpha', dist.Normal(0, 10))  # Intercept
beta = numpyro.sample('beta', dist.Normal(0, 10))    # Slopes
sigma = numpyro.sample('sigma', dist.HalfNormal(5))  # Noise
```

### Hierarchical Hyperpriors
```python
mu_global = numpyro.sample('mu', dist.Normal(0, 10))
sigma_global = numpyro.sample('sigma', dist.HalfCauchy(5))

with numpyro.plate('groups', n_groups):
    theta = numpyro.sample('theta', dist.Normal(mu_global, sigma_global))
```

### Count Data
```python
# Simple counts
count = numpyro.sample('obs', dist.Poisson(rate), obs=y)

# Overdispersed counts
count = numpyro.sample('obs', dist.NegativeBinomial(mean, conc), obs=y)

# Excess zeros
count = numpyro.sample('obs', dist.ZeroInflatedPoisson(rate, gate), obs=y)
```

### Binary Data
```python
# Single trial
y = numpyro.sample('obs', dist.Bernoulli(probs=p), obs=y)

# Multiple trials
y = numpyro.sample('obs', dist.Binomial(n, probs=p), obs=y)

# Logistic regression
logits = alpha + beta * x
y = numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)
```

### Simplex (Mixture Weights)
```python
# Dirichlet prior
weights = numpyro.sample('weights', dist.Dirichlet(jnp.ones(K)))

# Use in mixture
numpyro.sample('obs', dist.MixtureSameFamily(
    dist.Categorical(probs=weights),
    component_dist
), obs=y)
```

## Prior Choice Guidelines

**Weakly informative (default)**:
- Normal(0, 10) for real-valued
- HalfNormal(5) for positive scales
- Beta(2, 2) for probabilities (peaked at 0.5)
- Dirichlet(ones(K)) for mixture weights (uniform)

**Regularizing**:
- Normal(0, 1) for moderate shrinkage
- Cauchy(0, 1) or HalfCauchy(1) for heavy-tailed
- Gamma(2, 2) for positive with mode away from 0

**Flat/Vague (use sparingly)**:
- Uniform(low, high) for bounded
- Normal(0, 100) for very weak information
- Gamma(0.01, 0.01) for scale-invariant

**Informative** (when you have domain knowledge):
- Normal(known_mean, known_sd/2)
- Beta(α, β) tuned to match expert belief
- Truncated distributions for hard constraints

---

**For detailed examples, see SKILL.md workflows and `scripts/` examples.**
