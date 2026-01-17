# NumPyro Examples Collection

Comprehensive catalog of NumPyro examples for Bayesian modeling and probabilistic programming.

**Total Resources**: 30+ Python scripts + 20+ Jupyter notebooks from official NumPyro repository

**Source**: https://github.com/pyro-ppl/numpyro

---

## 📁 Directory Structure

```
scripts/examples/
├── basic/           # Foundational models (regression, GLM)
├── hierarchical/    # Multilevel models (8 schools, baseball, partial pooling)
├── time_series/     # Sequential models (AR, HMM, state space)
├── advanced/        # Complex models (GP, BNN, VAE, HMM, neural)
└── README.md        # This file

../../assets/        # Interactive notebooks
└── (Jupyter notebooks for tutorials and exploration)
```

---

## 🐍 Python Examples (scripts/)

### Basic Models (Regression, GLM)

**Regression Fundamentals**:
- `bayesian_regression.ipynb` - Linear regression with NumPyro, prior/posterior predictive
- `logistic_regression.ipynb` - Binary classification with uncertainty
- `horseshoe_regression.py` - Sparse regression with horseshoe prior
- `sparse_regression.py` - Lasso-style Bayesian sparse regression
- `ordinal_regression.ipynb` - Ordered categorical outcomes

**Statistical Testing**:
- `proportion_test.py` - Bayesian proportion tests
- `annotation.py` - Inter-rater agreement models

### Hierarchical Models (Multilevel)

**Classic Examples**:
- `baseball.py` - Baseball batting averages (partial pooling)
- `funnel.py` - Neal's funnel (challenging geometry, reparameterization demo)
- `bayesian_hierarchical_linear_regression.ipynb` - Complete hierarchical regression tutorial
- `bayesian_hierarchical_stacking.ipynb` - Model averaging and stacking

**Biological/Medical**:
- `capture_recapture.py` - Ecological population estimation
- `mortality.py` - Mortality modeling with hierarchical structure

### Time Series Models

**Sequential Data**:
- `ar2.py` - AR(2) autoregressive model
- `hmm.py` - Hidden Markov Models
- `hmm_enum.py` - HMM with enumeration for discrete latents
- `holt_winters.py` - Seasonal forecasting
- `time_series_forecasting.ipynb` - Complete time series workflow
- `hierarchical_forecasting.ipynb` - Hierarchical time series

**State Space**:
- `ode.py` - Ordinary differential equations in Bayesian framework
- `lotka_volterra_multiple.ipynb` - Predator-prey dynamics

### Advanced Models

**Gaussian Processes**:
- `gp.py` - GP regression basics
- `hsgp.py` - Hilbert Space GP approximation (scalable)
- `hsgp_example.ipynb` - HSGP tutorial
- `hsgp_nd_example.ipynb` - Multi-dimensional HSGP
- `circulant_gp.ipynb` - Efficient GPs via circulant matrices

**Neural & Deep Models**:
- `bnn.py` - Bayesian Neural Networks
- `stein_bnn.py` - Stein variational BNN
- `cvae.py` - Conditional Variational Autoencoder
- `cvae-flax/` - CVAE with Flax neural network library
- `prodlda.py` - Product of Experts LDA
- `nnx_example.ipynb` - Neural networks with NNX

**Mixture & Clustering**:
- `gmm.ipynb` - Gaussian Mixture Models
- `ssbvm_mixture.py` - Spatial Bayesian variable mixture
- `gaussian_shells.py` - Complex multimodal posteriors

**Specialized Inference**:
- `hmcecs.py` - HMC with Energy Conserving Subsampling (large data)
- `neutra.py` - Neural Transport for reparameterization
- `dais_demo.py` - Differentiable Annealed Importance Sampling
- `stein_dmm.py` - Stein Variational Deep Markov Model

**Other Advanced Topics**:
- `covtype.py` - Large-scale classification
- `tbip.py` - Text-Based Ideal Point model
- `bad_posterior_geometry.ipynb` - Diagnosing posterior issues
- `variationally_inferred_parameterization.ipynb` - VI reparameterization
- `other_samplers.ipynb` - Alternative MCMC kernels

---

## 📓 Interactive Notebooks (assets/)

Jupyter notebooks for tutorials and hands-on learning.

### Regression & Fundamentals
1. **bayesian_regression.ipynb** - Linear regression end-to-end
2. **logistic_regression.ipynb** - Binary classification
3. **ordinal_regression.ipynb** - Ordered outcomes
4. **bayesian_cuped.ipynb** - CUPED for A/B testing

### Hierarchical Modeling
5. **bayesian_hierarchical_linear_regression.ipynb** - Complete hierarchical workflow
6. **bayesian_hierarchical_stacking.ipynb** - Model averaging
7. **censoring.ipynb** - Survival analysis and censored data
8. **bayesian_imputation.ipynb** - Missing data imputation
9. **discrete_imputation.ipynb** - Categorical missing data

### Time Series & Forecasting
10. **time_series_forecasting.ipynb** - Forecasting workflows
11. **hierarchical_forecasting.ipynb** - Multi-level time series
12. **lotka_volterra_multiple.ipynb** - ODEs and dynamics

### Advanced Topics
13. **gmm.ipynb** - Gaussian mixtures
14. **hsgp_example.ipynb** - Scalable Gaussian processes
15. **hsgp_nd_example.ipynb** - Multi-dimensional GP
16. **circulant_gp.ipynb** - Efficient GP via circulant matrices
17. **nnx_example.ipynb** - Neural networks with NNX
18. **variationally_inferred_parameterization.ipynb** - Advanced VI
19. **other_samplers.ipynb** - Alternative MCMC methods

### Diagnostics & Methods
20. **bad_posterior_geometry.ipynb** - Troubleshooting convergence
21. **truncated_distributions.ipynb** - Bounded distributions
22. **model_rendering.ipynb** - Model visualization
23. **tbip.ipynb** - Text modeling

---

## 🚀 Quick Start Guide

### Pattern 1: Learn Basics

```bash
# Start with regression notebooks
jupyter notebook assets/bayesian_regression.ipynb

# Then hierarchical models
jupyter notebook assets/bayesian_hierarchical_linear_regression.ipynb

# Try time series
jupyter notebook assets/time_series_forecasting.ipynb
```

### Pattern 2: Find Domain-Specific Example

**Regression**: bayesian_regression.ipynb, horseshoe_regression.py, sparse_regression.py
**Classification**: logistic_regression.ipynb, covtype.py
**Hierarchical**: baseball.py, bayesian_hierarchical_*.ipynb
**Time Series**: ar2.py, hmm.py, holt_winters.py, time_series_forecasting.ipynb
**Gaussian Processes**: gp.py, hsgp*.ipynb
**Neural**: bnn.py, nnx_example.ipynb, cvae.py
**Mixtures**: gmm.ipynb, ssbvm_mixture.py
**Large Data**: hmcecs.py, hsgp.py
**Survival**: censoring.ipynb

### Pattern 3: Production Deployment

1. Study representative example from your domain
2. Adapt model specification to your data
3. Use diagnostic scripts from `../../scripts/` (mcmc_diagnostics.py, etc.)
4. Implement production patterns from SKILL.md

---

## 📊 Complexity Matrix

| Example | Lines | Level | Domain | Key Concepts |
|---------|-------|-------|--------|--------------|
| bayesian_regression | ~200 | Beginner | Regression | Priors, posterior, predictive |
| baseball | ~260 | Intermediate | Hierarchical | Partial pooling, shrinkage |
| ar2 | ~130 | Intermediate | Time Series | Autoregressive, sequential |
| gp | ~190 | Intermediate | GP | Kernel functions, uncertainty |
| hsgp | ~500 | Advanced | GP | Scalability, approximations |
| bnn | ~160 | Advanced | Neural | Bayesian deep learning |
| hmm | ~280 | Advanced | Time Series | Hidden states, discrete latents |
| hmcecs | ~160 | Advanced | Large Data | Subsampling, efficiency |
| neutra | ~190 | Expert | Inference | Reparameterization, geometry |

---

## 🎓 Learning Paths

### Week 1: Foundations
- bayesian_regression.ipynb
- logistic_regression.ipynb
- proportion_test.py
- Prior/posterior predictive checks

### Week 2: Hierarchical Models
- baseball.py (understand partial pooling)
- funnel.py (learn reparameterization)
- bayesian_hierarchical_linear_regression.ipynb (complete workflow)

### Week 3: Time Series
- ar2.py (autoregressive basics)
- hmm.py (hidden states)
- time_series_forecasting.ipynb (forecasting workflow)

### Week 4: Advanced Topics
- gp.py or hsgp.py (Gaussian processes)
- bnn.py (Bayesian neural networks)
- gmm.ipynb (mixture models)

### Week 5: Production
- hmcecs.py (large-scale inference)
- bad_posterior_geometry.ipynb (diagnostics)
- Implement your own model using patterns

---

## 🔗 Accessing Examples

All examples are available in the NumPyro repository:

**Python scripts**:
```
/Users/b80985/Documents/GitHub/numpyro/examples/
```

**Jupyter notebooks**:
```
/Users/b80985/Documents/GitHub/numpyro/notebooks/source/
```

**Online**:
- GitHub: https://github.com/pyro-ppl/numpyro/tree/master/examples
- Documentation: https://num.pyro.ai/en/latest/examples.html
- Rendered notebooks: https://num.pyro.ai/en/latest/tutorials.html

---

## 💡 Tips for Using Examples

**All examples**:
- Self-contained (import data, define model, run inference)
- Include visualization
- Demonstrate best practices
- Have comprehensive docstrings

**For learning**:
- Start with notebooks (interactive, explanatory)
- Progress through Python scripts (production patterns)
- Experiment with hyperparameters
- Compare MCMC vs VI where applicable

**For production**:
- Adapt model structure to your data
- Add diagnostics from `../../scripts/`
- Implement error handling and monitoring
- Use workflow patterns from SKILL.md

**Common modifications**:
- Replace synthetic data with your data
- Adjust priors based on domain knowledge
- Tune MCMC settings (warmup, samples, chains)
- Add custom derived quantities

---

## 📖 Related Resources

**Within This Skill**:
- `../../SKILL.md` - NumPyro Core Mastery guide
- `../../references/mcmc_diagnostics.md` - Convergence troubleshooting
- `../../references/variational_inference_guide.md` - VI workflows
- `../../references/distribution_catalog.md` - Prior selection
- `../../scripts/*.py` - Diagnostic utilities

**Official NumPyro**:
- Documentation: https://num.pyro.ai/
- GitHub: https://github.com/pyro-ppl/numpyro
- Tutorials: https://num.pyro.ai/en/latest/tutorials.html
- Forum: https://forum.pyro.ai/

**Community**:
- PyMC Discourse (Bayesian community): https://discourse.pymc.io/
- NumPyro Discussions: https://github.com/pyro-ppl/numpyro/discussions

---

**Examples Catalog Version**: 1.0.0
**Source**: NumPyro official repository (github.com/pyro-ppl/numpyro)
**Last Updated**: 2025-10-28
**Total Examples**: 30+ Python scripts, 20+ Jupyter notebooks
