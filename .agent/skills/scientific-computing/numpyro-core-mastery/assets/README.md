# NumPyro Assets - Templates and Starter Files

Production-ready templates for NumPyro Bayesian modeling workflows.

---

## 📁 Contents

### 1. `bayesian_workflow_template.py`

**Complete end-to-end Bayesian workflow template**

**What it includes**:
- Model definition with customizable priors/likelihood
- Data loading placeholder
- Prior predictive checks
- MCMC inference with NUTS
- Comprehensive convergence diagnostics (R-hat, ESS, divergences)
- Trace plot generation
- Posterior predictive checks
- Results extraction and summarization

**Usage**:
```bash
# Copy to your project
cp bayesian_workflow_template.py my_project_analysis.py

# Customize:
# 1. Replace model() function with your model
# 2. Replace load_data() with your data
# 3. Adjust priors based on domain knowledge
# 4. Run complete workflow
python my_project_analysis.py
```

**Output files**:
- `prior_predictive_check.png` - Verify priors are reasonable
- `trace_plots.png` - Visual convergence check
- `posterior_predictive_check.png` - Model fit assessment

**Best for**:
- New NumPyro projects
- Standard regression/classification tasks
- Learning the full Bayesian workflow
- Production inference pipelines

---

### 2. `hierarchical_model_template.py`

**Hierarchical/multilevel modeling template with partial pooling**

**What it includes**:
- Hierarchical model specification (group-level parameters)
- Synthetic data generation with true hierarchical structure
- MCMC inference with higher target_accept_prob (0.9)
- Shrinkage visualization (group estimates → population mean)
- Group-level fit plots
- Population-level parameter extraction

**Model structure**:
```
Population level:  μ_α, σ_α, μ_β, σ_β
     ↓
Group level:      α_j ~ N(μ_α, σ_α)
                  β_j ~ N(μ_β, σ_β)
     ↓
Observation:      y_ij ~ N(α_j + β_j*x_ij, σ)
```

**Usage**:
```bash
# Copy to your project
cp hierarchical_model_template.py my_hierarchical_analysis.py

# Customize:
# 1. Replace hierarchical_model() with your structure
# 2. Replace generate_hierarchical_data() with your data
# 3. Adjust number of groups/observations
# 4. Run workflow
python my_hierarchical_analysis.py
```

**Output files**:
- `hierarchical_shrinkage.png` - Visualize partial pooling effect
- `group_fits.png` - Individual group regression lines

**Best for**:
- Multi-group data (e.g., students in schools, patients in hospitals)
- Repeated measures
- Panel data / longitudinal studies
- When you want to borrow strength across groups

---

## 🚀 Quick Start

### For a New Project

**Standard model**:
```bash
cp bayesian_workflow_template.py my_analysis.py
# Edit model(), load_data(), priors
python my_analysis.py
```

**Hierarchical model**:
```bash
cp hierarchical_model_template.py my_hierarchical_analysis.py
# Edit hierarchical_model(), generate_hierarchical_data()
python my_hierarchical_analysis.py
```

### Customization Checklist

**Both templates**:
- [ ] Define your model equation in `model()` function
- [ ] Specify appropriate priors based on domain knowledge
- [ ] Load your actual data (replace synthetic data generation)
- [ ] Choose appropriate likelihood distribution
- [ ] Adjust MCMC settings if needed (warmup, samples, chains)

**Hierarchical template additionally**:
- [ ] Define your grouping structure (group_idx)
- [ ] Specify population-level hyperpriors
- [ ] Decide on centered vs non-centered parameterization

---

## 📊 Template Comparison

| Feature | Bayesian Workflow | Hierarchical Model |
|---------|-------------------|-------------------|
| **Model Type** | Single-level | Multi-level |
| **Pooling** | No pooling | Partial pooling |
| **Complexity** | Beginner-Intermediate | Intermediate-Advanced |
| **Lines** | ~350 | ~200 |
| **Diagnostics** | Full (R-hat, ESS, divergences) | Standard MCMC |
| **Visualization** | 3 plots | 2 plots |
| **Best For** | Standard inference | Grouped data |
| **Target Accept** | 0.8 | 0.9 (more robust) |

---

## 💡 Usage Tips

### Working with Templates

1. **Copy, don't modify originals** - Keep templates pristine for future use
2. **Rename meaningfully** - `earthquake_magnitude_model.py` not `test.py`
3. **Version control** - Git your modified templates
4. **Document changes** - Add comments explaining your customizations

### Model Customization

**Priors**:
```python
# BEFORE (template default)
alpha = numpyro.sample('alpha', dist.Normal(0, 10))

# AFTER (domain-specific)
# For house prices (always positive, typically 100K-500K)
price = numpyro.sample('price', dist.Normal(300000, 100000))
```

**Likelihood**:
```python
# Count data → Poisson
numpyro.sample('obs', dist.Poisson(rate), obs=y)

# Binary data → Bernoulli
numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

# Overdispersed counts → NegativeBinomial
numpyro.sample('obs', dist.NegativeBinomial(mean, concentration), obs=y)
```

### Performance Tuning

**If divergences occur**:
```python
# Increase target_accept_prob
nuts_kernel = NUTS(model, target_accept_prob=0.95)  # Was 0.8
```

**If slow mixing (low ESS)**:
```python
# Run more samples
mcmc = MCMC(kernel, num_samples=5000)  # Was 2000

# Or use non-centered parameterization (hierarchical models)
```

**For large datasets (N > 100K)**:
```python
# Consider VI instead of MCMC
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

guide = AutoNormal(model)
optimizer = numpyro.optim.Adam(0.001)
svi = SVI(model, guide, optimizer, Trace_ELBO())
svi_result = svi.run(rng_key, 10000, x, y)
```

---

## 🔗 Related Resources

**Within This Skill**:
- `../SKILL.md` - Complete NumPyro workflows
- `../references/mcmc_diagnostics.md` - Troubleshooting guide
- `../references/distribution_catalog.md` - Choose priors/likelihoods
- `../scripts/mcmc_diagnostics.py` - Automated diagnostic checking

**Examples**:
- `../scripts/examples/README.md` - 50+ real-world examples
- NumPyro repository: /Users/b80985/Documents/GitHub/numpyro/examples/

**Documentation**:
- NumPyro: https://num.pyro.ai/
- Tutorial notebooks: https://num.pyro.ai/en/latest/tutorials.html

---

## 📖 Common Workflows

### Workflow 1: Quick Analysis

```bash
# 1. Copy template
cp bayesian_workflow_template.py analysis.py

# 2. Edit model and data (10 min)
# 3. Run
python analysis.py

# 4. Check diagnostics in output
# 5. Examine plots: prior_predictive_check.png, etc.
```

### Workflow 2: Hierarchical Study

```bash
# 1. Copy hierarchical template
cp hierarchical_model_template.py hierarchical_analysis.py

# 2. Define groups and model (20 min)
# 3. Run
python hierarchical_analysis.py

# 4. Examine shrinkage effect in hierarchical_shrinkage.png
# 5. Check group-specific fits in group_fits.png
```

### Workflow 3: Production Pipeline

```bash
# 1. Start with template
cp bayesian_workflow_template.py production_model.py

# 2. Add error handling and logging
# 3. Integrate with data pipeline
# 4. Add monitoring (e.g., divergences → alert)
# 5. Schedule regular runs
```

---

## ⚙️ Template Modification Examples

### Example 1: Change to Logistic Regression

```python
# In model() function:
def model(x, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0, 2))
    beta = numpyro.sample('beta', dist.Normal(0, 2))

    logits = alpha + beta * x

    # Binary likelihood
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)
```

### Example 2: Add More Predictors

```python
# x is now (N, K) matrix
def model(x, y=None):
    N, K = x.shape

    alpha = numpyro.sample('alpha', dist.Normal(0, 10))

    with numpyro.plate('features', K):
        beta = numpyro.sample('beta', dist.Normal(0, 10))

    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    mu = alpha + jnp.dot(x, beta)

    with numpyro.plate('data', N):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)
```

### Example 3: Time Series (AR model)

```python
def model(y=None):
    T = len(y) if y is not None else 100

    # AR(1) coefficient
    rho = numpyro.sample('rho', dist.Uniform(-1, 1))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))

    y_pred = [numpyro.sample('y_0', dist.Normal(0, sigma))]

    for t in range(1, T):
        mu_t = rho * y_pred[t-1]
        y_t = numpyro.sample(f'y_{t}', dist.Normal(mu_t, sigma),
                            obs=y[t] if y is not None else None)
        y_pred.append(y_t)
```

---

**Assets Version**: 1.0.0
**Last Updated**: 2025-10-28
**Templates**: 2 (Bayesian Workflow, Hierarchical Model)
