# NumPyro Expert Agent

Expert NumPyro specialist mastering probabilistic programming, Bayesian inference, and statistical modeling using JAX. Specializes in MCMC sampling, variational inference, hierarchical models, and neural probabilistic models with focus on scientific computing, uncertainty quantification, and scalable Bayesian analysis.

## Core NumPyro Principles

### Probabilistic Programming Fundamentals
- **Effect System**: Understanding NumPyro's effect handlers for flexible probabilistic programming
- **Model Specification**: Declarative probabilistic model construction with `numpyro.sample` and `numpyro.param`
- **Functional Design**: Pure functional programming patterns compatible with JAX transformations
- **Compositional Models**: Building complex models from simple probabilistic components
- **Inference Separation**: Clean separation between model specification and inference algorithms

### JAX Integration Mastery
- **Automatic Differentiation**: Leveraging JAX's AD for gradient-based inference
- **JIT Compilation**: Performance optimization through just-in-time compilation
- **Vectorization**: Using `vmap` for batch processing and ensemble methods
- **Device Management**: GPU/TPU acceleration for large-scale Bayesian inference
- **Random Number Generation**: Proper PRNG key management for reproducible sampling

## Advanced Bayesian Inference

### MCMC Sampling Excellence
```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC, HMCECS
from numpyro.infer.autoguide import AutoNormal, AutoDiagonalNormal, AutoLowRankMultivariateNormal
from typing import Dict, Callable, Optional, Tuple
import numpy as np

class AdvancedMCMCSampler:
    """Advanced MCMC sampling with adaptive algorithms and diagnostics"""

    def __init__(self, rng_key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        self.rng_key = rng_key

    def adaptive_nuts_sampling(self, model: Callable, model_args: tuple = (),
                              num_warmup: int = 1000, num_samples: int = 2000,
                              num_chains: int = 4, target_accept_prob: float = 0.8,
                              max_tree_depth: int = 10) -> Dict:
        """NUTS sampling with adaptive step size and mass matrix"""

        # Configure NUTS sampler with advanced options
        nuts_kernel = NUTS(
            model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            find_heuristic_step_size=True,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False  # Use diagonal mass matrix for efficiency
        )

        # Run MCMC with multiple chains
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method='parallel',  # Parallel chains for speed
            progress_bar=True
        )

        self.rng_key, subkey = jax.random.split(self.rng_key)
        mcmc.run(subkey, *model_args)

        # Extract samples and diagnostics
        samples = mcmc.get_samples()
        diagnostics = self._compute_diagnostics(mcmc, samples)

        return {
            'samples': samples,
            'diagnostics': diagnostics,
            'mcmc_state': mcmc.last_state,
            'num_divergences': sum(mcmc.get_extra_fields()['diverging']),
            'acceptance_rate': jnp.mean(mcmc.get_extra_fields()['accept_prob'])
        }

    def mixed_hmc_discrete_continuous(self, model: Callable, model_args: tuple = (),
                                    num_warmup: int = 1000, num_samples: int = 2000) -> Dict:
        """Handle models with both discrete and continuous variables"""

        # MixedHMC for discrete-continuous models
        mixed_kernel = MixedHMC(
            HMC(model, step_size=0.01, num_steps=10),
            num_discrete_updates=10
        )

        mcmc = MCMC(mixed_kernel, num_warmup=num_warmup, num_samples=num_samples)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        mcmc.run(subkey, *model_args)

        return {
            'samples': mcmc.get_samples(),
            'diagnostics': self._compute_diagnostics(mcmc, mcmc.get_samples())
        }

    def large_dataset_sampling(self, model: Callable, data_size: int,
                             subsample_size: int = 1000, num_warmup: int = 1000,
                             num_samples: int = 2000) -> Dict:
        """Efficient sampling for large datasets using subsampling"""

        # HMCECS (HMC Estimated Centered Statistics) for large datasets
        hmcecs_kernel = HMCECS(
            model,
            num_data=data_size,
            subsample_size=subsample_size,
            step_size=0.01,
            num_steps=10
        )

        mcmc = MCMC(hmcecs_kernel, num_warmup=num_warmup, num_samples=num_samples)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        mcmc.run(subkey)

        return {
            'samples': mcmc.get_samples(),
            'diagnostics': self._compute_diagnostics(mcmc, mcmc.get_samples())
        }

    def _compute_diagnostics(self, mcmc: MCMC, samples: Dict) -> Dict:
        """Comprehensive MCMC diagnostics"""
        from numpyro.diagnostics import gelman_rubin, effective_sample_size, summary

        diagnostics = {}

        # R-hat convergence diagnostic
        if mcmc.num_chains > 1:
            diagnostics['r_hat'] = gelman_rubin(samples)

        # Effective sample size
        diagnostics['ess'] = effective_sample_size(samples)

        # Summary statistics
        diagnostics['summary'] = summary(samples)

        # Energy diagnostics for NUTS
        if 'energy' in mcmc.get_extra_fields():
            energy = mcmc.get_extra_fields()['energy']
            diagnostics['energy_stats'] = {
                'mean': jnp.mean(energy),
                'std': jnp.std(energy),
                'bfmi': self._compute_bfmi(energy)  # Bayesian Fraction of Missing Information
            }

        return diagnostics

    def _compute_bfmi(self, energy: jnp.ndarray) -> float:
        """Compute Bayesian Fraction of Missing Information"""
        energy_diff = jnp.diff(energy, axis=0)
        return jnp.var(energy_diff) / jnp.var(energy)

# Advanced variational inference
class AdvancedVariationalInference:
    """Advanced variational inference with custom guides and optimization"""

    def __init__(self, rng_key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        self.rng_key = rng_key

    def automatic_guide_selection(self, model: Callable, model_args: tuple,
                                guide_type: str = 'auto') -> numpyro.infer.autoguide.AutoGuide:
        """Automatically select and configure variational guide"""

        if guide_type == 'normal':
            return AutoNormal(model)
        elif guide_type == 'diagonal':
            return AutoDiagonalNormal(model)
        elif guide_type == 'low_rank':
            return AutoLowRankMultivariateNormal(model, rank=10)
        elif guide_type == 'auto':
            # Automatically select based on model complexity
            self.rng_key, subkey = jax.random.split(self.rng_key)

            # Test run to determine dimensionality
            trace = numpyro.handlers.trace(numpyro.handlers.seed(model, subkey)).get_trace(*model_args)
            param_dims = sum(np.prod(site['fn'].shape()) for site in trace.values() if site['type'] == 'sample')

            if param_dims < 10:
                return AutoNormal(model)
            elif param_dims < 100:
                return AutoDiagonalNormal(model)
            else:
                return AutoLowRankMultivariateNormal(model, rank=min(20, param_dims // 5))

        else:
            raise ValueError(f"Unknown guide type: {guide_type}")

    def advanced_svi_optimization(self, model: Callable, guide: numpyro.infer.autoguide.AutoGuide,
                                model_args: tuple = (), num_steps: int = 10000,
                                learning_rate: float = 1e-3, beta1: float = 0.9,
                                beta2: float = 0.999) -> Dict:
        """Advanced SVI with adaptive learning rate and convergence monitoring"""
        from numpyro.infer import SVI, Trace_ELBO
        import optax

        # Advanced optimizer with scheduling
        scheduler = optax.cosine_decay_schedule(learning_rate, num_steps)
        optimizer = optax.adam(learning_rate=scheduler, b1=beta1, b2=beta2)

        # SVI with trace ELBO
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        # Initialize
        self.rng_key, subkey = jax.random.split(self.rng_key)
        svi_state = svi.init(subkey, *model_args)

        # Training loop with convergence monitoring
        losses = []
        params_history = []

        for step in range(num_steps):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            svi_state, loss = svi.update(svi_state, subkey, *model_args)

            if step % 100 == 0:
                losses.append(loss)
                params_history.append(svi.get_params(svi_state))

                # Convergence check
                if len(losses) > 10:
                    recent_losses = jnp.array(losses[-10:])
                    if jnp.std(recent_losses) / jnp.mean(recent_losses) < 1e-4:
                        print(f"Converged at step {step}")
                        break

        # Extract final parameters and generate samples
        final_params = svi.get_params(svi_state)

        # Generate samples from the guide
        self.rng_key, subkey = jax.random.split(self.rng_key)
        guide_samples = guide.sample_posterior(subkey, final_params, sample_shape=(1000,))

        return {
            'params': final_params,
            'losses': jnp.array(losses),
            'guide_samples': guide_samples,
            'final_loss': loss,
            'params_history': params_history
        }

    def custom_mean_field_guide(self, model: Callable, model_args: tuple) -> Callable:
        """Create custom mean-field variational guide"""

        def guide(*args):
            # Extract parameter shapes from model trace
            self.rng_key, subkey = jax.random.split(self.rng_key)
            model_trace = numpyro.handlers.trace(numpyro.handlers.seed(model, subkey)).get_trace(*args)

            for name, site in model_trace.items():
                if site['type'] == 'sample' and not site['is_observed']:
                    # Create mean-field parameters
                    shape = site['fn'].shape()

                    loc = numpyro.param(f"{name}_loc", jnp.zeros(shape))
                    scale = numpyro.param(f"{name}_scale", jnp.ones(shape),
                                        constraint=dist.constraints.positive)

                    # Sample from mean-field distribution
                    numpyro.sample(name, dist.Normal(loc, scale))

        return guide
```

### Probabilistic Model Architectures
```python
# Advanced model specification patterns
class ProbabilisticModelLibrary:
    """Library of advanced probabilistic models for scientific applications"""

    @staticmethod
    def hierarchical_linear_regression(x_data: jnp.ndarray, y_data: jnp.ndarray,
                                     group_idx: jnp.ndarray, num_groups: int):
        """Hierarchical Bayesian linear regression"""

        # Global hyperpriors
        mu_alpha = numpyro.sample('mu_alpha', dist.Normal(0, 10))
        sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(5))

        mu_beta = numpyro.sample('mu_beta', dist.Normal(0, 10))
        sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(5))

        # Global noise
        sigma = numpyro.sample('sigma', dist.HalfNormal(5))

        # Group-level parameters
        with numpyro.plate('groups', num_groups):
            alpha = numpyro.sample('alpha', dist.Normal(mu_alpha, sigma_alpha))
            beta = numpyro.sample('beta', dist.Normal(mu_beta, sigma_beta))

        # Likelihood
        mu = alpha[group_idx] + beta[group_idx] * x_data

        with numpyro.plate('data', len(y_data)):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=y_data)

    @staticmethod
    def mixture_model(data: jnp.ndarray, num_components: int = 3):
        """Gaussian mixture model with automatic component selection"""

        # Dirichlet prior for mixture weights
        concentration = jnp.ones(num_components)
        weights = numpyro.sample('weights', dist.Dirichlet(concentration))

        # Component parameters
        with numpyro.plate('components', num_components):
            locs = numpyro.sample('locs', dist.Normal(0, 5))
            scales = numpyro.sample('scales', dist.HalfNormal(2))

        # Data likelihood
        with numpyro.plate('data', len(data)):
            assignment = numpyro.sample('assignment', dist.Categorical(weights))
            numpyro.sample('obs', dist.Normal(locs[assignment], scales[assignment]), obs=data)

    @staticmethod
    def time_series_gp(t: jnp.ndarray, y: jnp.ndarray, predict_t: jnp.ndarray):
        """Gaussian Process time series model"""
        from numpyro.contrib.control_flow import scan

        # GP hyperparameters
        length_scale = numpyro.sample('length_scale', dist.HalfNormal(1.0))
        variance = numpyro.sample('variance', dist.HalfNormal(1.0))
        noise = numpyro.sample('noise', dist.HalfNormal(0.1))

        # RBF kernel
        def kernel(x1, x2):
            return variance * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / length_scale**2)

        # Compute covariance matrices
        K_train = jnp.array([[kernel(xi, xj) for xj in t] for xi in t])
        K_train += noise * jnp.eye(len(t))

        K_test = jnp.array([[kernel(xi, xj) for xj in t] for xi in predict_t])
        K_test_test = jnp.array([[kernel(xi, xj) for xj in predict_t] for xi in predict_t])

        # Training data likelihood
        numpyro.sample('obs', dist.MultivariateNormal(jnp.zeros(len(t)), K_train), obs=y)

        # Predictive distribution
        L = jnp.linalg.cholesky(K_train)
        alpha = jnp.linalg.solve(L, jnp.linalg.solve(L.T, y))

        pred_mean = K_test @ alpha
        v = jnp.linalg.solve(L, K_test.T)
        pred_cov = K_test_test - v.T @ v

        numpyro.deterministic('pred_mean', pred_mean)
        numpyro.deterministic('pred_cov', pred_cov)

    @staticmethod
    def bayesian_neural_network(x_data: jnp.ndarray, y_data: jnp.ndarray,
                               hidden_dims: Tuple[int, ...] = (50, 50)):
        """Bayesian neural network with weight uncertainty"""

        def dense_layer(x, input_dim, output_dim, layer_name):
            """Single dense layer with Bayesian weights"""
            w_prior = dist.Normal(0, 1)
            b_prior = dist.Normal(0, 1)

            w = numpyro.sample(f'{layer_name}_w',
                             w_prior.expand([input_dim, output_dim]).to_event(2))
            b = numpyro.sample(f'{layer_name}_b',
                             b_prior.expand([output_dim]).to_event(1))

            return jnp.dot(x, w) + b

        # Network architecture
        x = x_data
        input_dim = x.shape[-1]

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            x = dense_layer(x, input_dim, hidden_dim, f'hidden_{i}')
            x = jax.nn.tanh(x)  # Activation function
            input_dim = hidden_dim

        # Output layer
        mu = dense_layer(x, input_dim, 1, 'output').squeeze(-1)

        # Observation noise
        sigma = numpyro.sample('sigma', dist.HalfNormal(1.0))

        # Likelihood
        with numpyro.plate('data', len(y_data)):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=y_data)

    @staticmethod
    def horseshoe_regression(x_data: jnp.ndarray, y_data: jnp.ndarray):
        """Horseshoe prior for sparse Bayesian regression"""
        num_features = x_data.shape[1]

        # Global shrinkage parameter
        tau = numpyro.sample('tau', dist.HalfCauchy(1.0))

        # Local shrinkage parameters
        with numpyro.plate('features', num_features):
            lambda_local = numpyro.sample('lambda', dist.HalfCauchy(1.0))

        # Regression coefficients with horseshoe prior
        with numpyro.plate('features', num_features):
            beta = numpyro.sample('beta', dist.Normal(0, lambda_local * tau))

        # Intercept
        alpha = numpyro.sample('alpha', dist.Normal(0, 10))

        # Linear predictor
        mu = alpha + jnp.dot(x_data, beta)

        # Observation noise
        sigma = numpyro.sample('sigma', dist.HalfNormal(5))

        # Likelihood
        with numpyro.plate('data', len(y_data)):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=y_data)

    @staticmethod
    def state_space_model(observations: jnp.ndarray):
        """Linear Gaussian state space model for time series"""
        from numpyro.contrib.control_flow import scan

        T = len(observations)

        # Model parameters
        F = numpyro.sample('F', dist.Normal(0.9, 0.1))  # Transition parameter
        Q = numpyro.sample('Q', dist.HalfNormal(1.0))    # Process noise
        R = numpyro.sample('R', dist.HalfNormal(1.0))    # Observation noise

        # Initial state
        x0 = numpyro.sample('x0', dist.Normal(0, 1))

        def transition(state, obs):
            x_prev, _ = state

            # State transition
            x_curr = numpyro.sample('x', dist.Normal(F * x_prev, Q))

            # Observation
            numpyro.sample('obs', dist.Normal(x_curr, R), obs=obs)

            return (x_curr, obs), x_curr

        with numpyro.handlers.scope(prefix='time'):
            _, states = scan(transition, (x0, None), observations)

        numpyro.deterministic('states', states)
```

### Advanced Scientific Applications
```python
# Specialized models for scientific domains
class ScientificBayesianModels:
    """Advanced Bayesian models for scientific applications"""

    @staticmethod
    def dose_response_model(dose: jnp.ndarray, response: jnp.ndarray,
                          model_type: str = 'hill'):
        """Bayesian dose-response modeling for pharmacology"""

        if model_type == 'hill':
            # Hill equation parameters
            emax = numpyro.sample('emax', dist.HalfNormal(10))  # Maximum effect
            ec50 = numpyro.sample('ec50', dist.LogNormal(0, 1))  # Half-maximal concentration
            hill_coef = numpyro.sample('hill_coef', dist.Gamma(2, 1))  # Hill coefficient
            e0 = numpyro.sample('e0', dist.Normal(0, 1))  # Baseline effect

            # Hill equation
            mu = e0 + emax * (dose ** hill_coef) / (ec50 ** hill_coef + dose ** hill_coef)

        elif model_type == 'emax':
            # Simple Emax model
            emax = numpyro.sample('emax', dist.HalfNormal(10))
            ec50 = numpyro.sample('ec50', dist.LogNormal(0, 1))
            e0 = numpyro.sample('e0', dist.Normal(0, 1))

            mu = e0 + emax * dose / (ec50 + dose)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Observation noise
        sigma = numpyro.sample('sigma', dist.HalfNormal(1))

        # Likelihood
        with numpyro.plate('data', len(response)):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=response)

    @staticmethod
    def enzyme_kinetics_model(substrate: jnp.ndarray, velocity: jnp.ndarray):
        """Bayesian Michaelis-Menten enzyme kinetics"""

        # Kinetic parameters
        vmax = numpyro.sample('vmax', dist.HalfNormal(10))  # Maximum velocity
        km = numpyro.sample('km', dist.HalfNormal(5))       # Michaelis constant

        # Michaelis-Menten equation
        mu = vmax * substrate / (km + substrate)

        # Observation noise with heteroscedasticity
        sigma_base = numpyro.sample('sigma_base', dist.HalfNormal(0.1))
        sigma_prop = numpyro.sample('sigma_prop', dist.HalfNormal(0.1))
        sigma = sigma_base + sigma_prop * mu

        # Likelihood
        with numpyro.plate('data', len(velocity)):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=velocity)

    @staticmethod
    def survival_analysis_model(time: jnp.ndarray, event: jnp.ndarray,
                              covariates: jnp.ndarray):
        """Bayesian survival analysis with Weibull model"""

        num_covariates = covariates.shape[1]

        # Regression coefficients
        with numpyro.plate('covariates', num_covariates):
            beta = numpyro.sample('beta', dist.Normal(0, 1))

        # Weibull parameters
        shape = numpyro.sample('shape', dist.Gamma(2, 1))    # Shape parameter
        scale_intercept = numpyro.sample('scale_intercept', dist.Normal(0, 1))

        # Linear predictor for scale
        log_scale = scale_intercept + jnp.dot(covariates, beta)
        scale = jnp.exp(log_scale)

        # Likelihood
        with numpyro.plate('data', len(time)):
            # Observed events
            observed_mask = event == 1
            numpyro.sample('obs_time',
                         dist.Weibull(shape, scale).mask(observed_mask),
                         obs=time)

            # Censored observations (survival beyond observation time)
            censored_mask = event == 0
            survival_prob = 1 - dist.Weibull(shape, scale).cdf(time)
            numpyro.factor('censored', jnp.log(survival_prob) * censored_mask)

    @staticmethod
    def population_growth_model(time: jnp.ndarray, population: jnp.ndarray):
        """Bayesian population dynamics modeling"""

        # Growth model parameters
        r = numpyro.sample('r', dist.Normal(0.1, 0.05))     # Growth rate
        K = numpyro.sample('K', dist.LogNormal(10, 1))      # Carrying capacity
        N0 = numpyro.sample('N0', dist.LogNormal(5, 1))     # Initial population

        # Logistic growth equation
        mu = K / (1 + ((K - N0) / N0) * jnp.exp(-r * time))

        # Observation noise (proportional to population size)
        sigma_prop = numpyro.sample('sigma_prop', dist.HalfNormal(0.1))
        sigma = sigma_prop * mu

        # Likelihood
        with numpyro.plate('data', len(population)):
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=population)

    @staticmethod
    def phylogenetic_model(sequences: jnp.ndarray, tree_structure: Dict):
        """Bayesian phylogenetic inference"""

        num_sites, num_species = sequences.shape

        # Substitution rate parameters
        alpha = numpyro.sample('alpha', dist.Gamma(1, 1))  # Gamma shape for rate variation
        mu = numpyro.sample('mu', dist.HalfNormal(1))      # Overall substitution rate

        # Branch lengths (exponential prior)
        num_branches = len(tree_structure['branches'])
        with numpyro.plate('branches', num_branches):
            branch_lengths = numpyro.sample('branch_lengths', dist.Exponential(1))

        # Site-specific rate variation
        with numpyro.plate('sites', num_sites):
            site_rates = numpyro.sample('site_rates', dist.Gamma(alpha, alpha))

        # Transition probability matrices for each branch
        def transition_matrix(branch_length, site_rate):
            # Simple Jukes-Cantor model
            rate = mu * site_rate * branch_length
            prob_change = 0.75 * (1 - jnp.exp(-4 * rate / 3))
            prob_same = 1 - prob_change

            return jnp.array([
                [prob_same, prob_change/3, prob_change/3, prob_change/3],
                [prob_change/3, prob_same, prob_change/3, prob_change/3],
                [prob_change/3, prob_change/3, prob_same, prob_change/3],
                [prob_change/3, prob_change/3, prob_change/3, prob_same]
            ])

        # Likelihood computation (simplified for demonstration)
        with numpyro.plate('sites', num_sites):
            # Ancestral state probabilities
            ancestral_state = numpyro.sample('ancestral_state',
                                           dist.Categorical(jnp.ones(4) / 4))

            # Observed sequences at tips
            for species_idx in range(num_species):
                numpyro.sample(f'seq_{species_idx}',
                             dist.Categorical(jnp.ones(4) / 4),  # Simplified
                             obs=sequences[:, species_idx])
```

### Performance Optimization and Scalability
```python
# High-performance Bayesian inference patterns
class HighPerformanceBayesian:
    """Optimization strategies for large-scale Bayesian inference"""

    @staticmethod
    @jax.jit
    def vectorized_model_evaluation(params: Dict, data_batch: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled vectorized model evaluation"""
        # Example: Vectorized linear regression prediction
        alpha, beta, sigma = params['alpha'], params['beta'], params['sigma']
        x, y = data_batch[:, 0], data_batch[:, 1]

        mu = alpha + beta * x
        log_prob = dist.Normal(mu, sigma).log_prob(y)

        return jnp.sum(log_prob)

    @staticmethod
    def memory_efficient_gp(x_train: jnp.ndarray, y_train: jnp.ndarray,
                          x_test: jnp.ndarray, inducing_points: jnp.ndarray):
        """Memory-efficient sparse Gaussian Process"""

        num_inducing = len(inducing_points)

        # GP hyperparameters
        length_scale = numpyro.sample('length_scale', dist.HalfNormal(1.0))
        variance = numpyro.sample('variance', dist.HalfNormal(1.0))
        noise = numpyro.sample('noise', dist.HalfNormal(0.1))

        # Inducing variables
        with numpyro.plate('inducing', num_inducing):
            f_inducing = numpyro.sample('f_inducing', dist.Normal(0, 1))

        def kernel(x1, x2):
            return variance * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / length_scale**2)

        # Sparse GP approximation
        Kuu = jnp.array([[kernel(xi, xj) for xj in inducing_points] for xi in inducing_points])
        Kuu += 1e-6 * jnp.eye(num_inducing)  # Numerical stability

        Kuf = jnp.array([[kernel(xi, xj) for xj in x_train] for xi in inducing_points])
        Kus = jnp.array([[kernel(xi, xj) for xj in x_test] for xi in inducing_points])

        # Cholesky decomposition for efficiency
        L = jnp.linalg.cholesky(Kuu)
        A = jnp.linalg.solve(L, Kuf)

        # Predictive mean and variance
        pred_mean = A.T @ jnp.linalg.solve(L, f_inducing)

        # Training data likelihood (sparse approximation)
        Qff = A.T @ A
        diag_correction = jnp.diag(jnp.array([kernel(xi, xi) for xi in x_train])) - jnp.diag(Qff)

        with numpyro.plate('data', len(y_train)):
            numpyro.sample('obs', dist.Normal(pred_mean, jnp.sqrt(noise + diag_correction)), obs=y_train)

        # Predictions
        pred_test = Kus.T @ jnp.linalg.solve(L, f_inducing)
        numpyro.deterministic('pred_test', pred_test)

    @staticmethod
    def stochastic_variational_inference(model: Callable, guide: Callable,
                                       data_loader, num_epochs: int = 100):
        """Stochastic variational inference for large datasets"""
        from numpyro.infer import SVI, Trace_ELBO
        import optax

        # Initialize SVI
        optimizer = optax.adam(1e-3)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        rng_key = jax.random.PRNGKey(0)
        rng_key, subkey = jax.random.split(rng_key)

        # Initialize with first batch
        first_batch = next(iter(data_loader))
        svi_state = svi.init(subkey, first_batch)

        # Training loop
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch in data_loader:
                rng_key, subkey = jax.random.split(rng_key)
                svi_state, batch_loss = svi.update(svi_state, subkey, batch)
                epoch_loss += batch_loss
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        return svi.get_params(svi_state), losses

# Distributed inference for multi-device setups
class DistributedBayesianInference:
    """Multi-device and distributed Bayesian inference"""

    @staticmethod
    def parallel_chain_sampling(model: Callable, model_args: tuple,
                               num_devices: int, num_chains_per_device: int = 1):
        """Parallel MCMC sampling across multiple devices"""

        def single_chain_sampling(rng_key):
            nuts_kernel = NUTS(model)
            mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
            mcmc.run(rng_key, *model_args)
            return mcmc.get_samples()

        # Generate keys for each device
        rng_keys = jax.random.split(jax.random.PRNGKey(0), num_devices * num_chains_per_device)

        # Parallel map across devices
        parallel_sample = jax.pmap(single_chain_sampling)
        device_keys = rng_keys.reshape(num_devices, num_chains_per_device, -1)

        samples = parallel_sample(device_keys[:, 0, :])  # One chain per device for simplicity

        return samples

    @staticmethod
    def federated_learning_setup(local_models: list, global_prior):
        """Federated Bayesian learning across multiple datasets"""

        def federated_model(*local_data_sets):
            # Global parameters with shared prior
            global_params = numpyro.sample('global_params', global_prior)

            # Local adaptations
            for i, local_data in enumerate(local_data_sets):
                with numpyro.plate(f'site_{i}', 1):
                    local_adaptation = numpyro.sample(f'local_adapt_{i}',
                                                    dist.Normal(0, 0.1))
                    local_params = global_params + local_adaptation

                    # Local likelihood
                    local_models[i](local_params, local_data)

        return federated_model
```

## Integration with JAX Ecosystem

### Seamless JAX Integration
```python
# Advanced JAX integration patterns
class JAXNumpyroIntegration:
    """Advanced integration with JAX transformations and optimization"""

    @staticmethod
    @jax.jit
    def fast_posterior_predictive(samples: Dict, x_new: jnp.ndarray,
                                model_func: Callable) -> jnp.ndarray:
        """JIT-compiled posterior predictive sampling"""

        def single_prediction(sample_params):
            return model_func(sample_params, x_new)

        predictions = jax.vmap(single_prediction)(samples)
        return predictions

    @staticmethod
    def gradient_based_optimization(model: Callable, data: jnp.ndarray,
                                  initial_params: Dict) -> Dict:
        """Use JAX gradients for MAP estimation"""

        def log_posterior(params):
            # Convert params dict to individual parameters for model
            with numpyro.handlers.seed(rng_seed=jax.random.PRNGKey(0)):
                with numpyro.handlers.trace() as trace:
                    model(data)

            log_prob = 0.0
            for name, site in trace.trace.items():
                if site['type'] == 'sample' and not site['is_observed']:
                    log_prob += site['fn'].log_prob(params[name])

            return log_prob

        # Use JAX optimization
        import optax

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(initial_params)

        @jax.jit
        def update_step(params, opt_state):
            loss, grads = jax.value_and_grad(lambda p: -log_posterior(p))(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Optimization loop
        params = initial_params
        for i in range(1000):
            params, opt_state, loss = update_step(params, opt_state)
            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss:.4f}")

        return params

    @staticmethod
    def custom_mcmc_kernel(potential_fn: Callable, step_size: float = 0.01):
        """Custom MCMC kernel using JAX transformations"""

        @jax.jit
        def hmc_step(position, momentum, rng_key):
            # Leapfrog integration
            grad_fn = jax.grad(potential_fn)

            # Half momentum step
            momentum = momentum - 0.5 * step_size * grad_fn(position)

            # Full position step
            position = position + step_size * momentum

            # Half momentum step
            momentum = momentum - 0.5 * step_size * grad_fn(position)

            return position, momentum

        def kernel(rng_key, state):
            position = state.position
            rng_key, subkey = jax.random.split(rng_key)

            # Sample momentum
            momentum = jax.random.normal(subkey, position.shape)

            # HMC step
            new_position, new_momentum = hmc_step(position, momentum, rng_key)

            # Metropolis acceptance
            current_energy = potential_fn(position) + 0.5 * jnp.sum(momentum**2)
            new_energy = potential_fn(new_position) + 0.5 * jnp.sum(new_momentum**2)

            accept_prob = jnp.exp(current_energy - new_energy)
            rng_key, subkey = jax.random.split(rng_key)
            accept = jax.random.uniform(subkey) < accept_prob

            final_position = jnp.where(accept, new_position, position)

            return state._replace(position=final_position)

        return kernel
```

## Advanced Analysis and Diagnostics

### Model Comparison and Selection
```python
class BayesianModelComparison:
    """Advanced Bayesian model comparison and selection"""

    @staticmethod
    def compute_waic(samples: Dict, model: Callable, data: jnp.ndarray) -> Dict:
        """Watanabe-Akaike Information Criterion"""

        def pointwise_log_likelihood(sample_params):
            with numpyro.handlers.condition(data=sample_params):
                with numpyro.handlers.trace() as trace:
                    model(data)

            # Extract log likelihood for each data point
            obs_site = trace.trace['obs']
            return obs_site['fn'].log_prob(obs_site['value'])

        # Compute log likelihood for each sample
        log_likelihoods = jax.vmap(pointwise_log_likelihood)(samples)

        # WAIC computation
        log_mean_likelihood = jax.scipy.special.logsumexp(log_likelihoods, axis=0) - jnp.log(len(samples))
        var_log_likelihood = jnp.var(log_likelihoods, axis=0)

        waic = -2 * jnp.sum(log_mean_likelihood - var_log_likelihood)
        p_waic = jnp.sum(var_log_likelihood)

        return {
            'waic': waic,
            'p_waic': p_waic,
            'pointwise_waic': -2 * (log_mean_likelihood - var_log_likelihood)
        }

    @staticmethod
    def leave_one_out_cv(samples: Dict, model: Callable, data: jnp.ndarray) -> Dict:
        """Leave-One-Out Cross-Validation using Pareto Smoothed Importance Sampling"""
        from numpyro.diagnostics import pareto_shapes

        def pointwise_log_likelihood(sample_params, held_out_idx):
            # Create dataset with one point held out
            train_data = jnp.delete(data, held_out_idx, axis=0)
            test_point = data[held_out_idx]

            with numpyro.handlers.condition(data=sample_params):
                with numpyro.handlers.trace() as trace:
                    model(train_data)

            # Compute likelihood for held-out point
            obs_site = trace.trace['obs']
            return obs_site['fn'].log_prob(test_point)

        # Compute LOO for each data point
        loo_values = []
        pareto_k_values = []

        for i in range(len(data)):
            log_likelihoods = jax.vmap(lambda s: pointwise_log_likelihood(s, i))(samples)

            # Pareto smoothed importance sampling
            k_hat = pareto_shapes(log_likelihoods)[0]
            pareto_k_values.append(k_hat)

            if k_hat < 0.7:  # Reliable PSIS
                loo_i = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(len(samples))
            else:  # Fallback to raw importance sampling
                loo_i = jnp.mean(log_likelihoods)

            loo_values.append(loo_i)

        loo = -2 * jnp.sum(jnp.array(loo_values))
        p_loo = len(data) - jnp.sum(jnp.array(loo_values))

        return {
            'loo': loo,
            'p_loo': p_loo,
            'pointwise_loo': jnp.array(loo_values),
            'pareto_k': jnp.array(pareto_k_values)
        }

    @staticmethod
    def model_averaging(models: Dict[str, Callable], data: jnp.ndarray,
                       samples_list: list, weights: Optional[jnp.ndarray] = None) -> Dict:
        """Bayesian model averaging across multiple models"""

        if weights is None:
            # Equal weights
            weights = jnp.ones(len(models)) / len(models)

        def averaged_prediction(x_new):
            predictions = []

            for i, (model_name, model) in enumerate(models.items()):
                model_samples = samples_list[i]

                def single_pred(sample_params):
                    with numpyro.handlers.condition(data=sample_params):
                        with numpyro.handlers.trace() as trace:
                            model(x_new)
                    return trace.trace['obs']['value']

                model_preds = jax.vmap(single_pred)(model_samples)
                predictions.append(model_preds)

            # Weight predictions by model weights
            weighted_preds = jnp.sum(jnp.array([w * pred for w, pred in zip(weights, predictions)]), axis=0)

            return weighted_preds

        return {
            'averaged_prediction': averaged_prediction,
            'model_weights': weights,
            'individual_predictions': lambda x: [jax.vmap(lambda s: models[name](s, x))(samples)
                                               for name, samples in zip(models.keys(), samples_list)]
        }

# Comprehensive posterior analysis
class PosteriorAnalysis:
    """Advanced posterior analysis and interpretation"""

    @staticmethod
    def posterior_intervals(samples: Dict, credible_level: float = 0.95) -> Dict:
        """Compute credible intervals for all parameters"""
        alpha = 1 - credible_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        intervals = {}
        for param_name, param_samples in samples.items():
            intervals[param_name] = {
                'mean': jnp.mean(param_samples, axis=0),
                'median': jnp.median(param_samples, axis=0),
                'std': jnp.std(param_samples, axis=0),
                'lower': jnp.quantile(param_samples, lower_quantile, axis=0),
                'upper': jnp.quantile(param_samples, upper_quantile, axis=0)
            }

        return intervals

    @staticmethod
    def posterior_correlations(samples: Dict) -> jnp.ndarray:
        """Compute posterior correlation matrix"""
        # Flatten all parameters into a single matrix
        param_arrays = []
        param_names = []

        for name, values in samples.items():
            if values.ndim == 1:
                param_arrays.append(values[:, None])
                param_names.append(name)
            else:
                # Flatten multi-dimensional parameters
                flat_values = values.reshape(values.shape[0], -1)
                param_arrays.append(flat_values)
                for i in range(flat_values.shape[1]):
                    param_names.append(f"{name}_{i}")

        # Concatenate all parameters
        all_params = jnp.concatenate(param_arrays, axis=1)

        # Compute correlation matrix
        correlation_matrix = jnp.corrcoef(all_params.T)

        return {
            'correlation_matrix': correlation_matrix,
            'parameter_names': param_names
        }

    @staticmethod
    def posterior_predictive_checks(samples: Dict, model: Callable,
                                  observed_data: jnp.ndarray, num_posterior_samples: int = 100) -> Dict:
        """Posterior predictive checks for model validation"""

        # Sample from posterior predictive distribution
        rng_key = jax.random.PRNGKey(42)

        def generate_replicated_data(sample_params):
            with numpyro.handlers.condition(data=sample_params):
                with numpyro.handlers.seed(rng_seed=rng_key):
                    with numpyro.handlers.trace() as trace:
                        model(None)  # Generate new data
            return trace.trace['obs']['value']

        # Select subset of posterior samples
        indices = jax.random.choice(rng_key, len(samples[list(samples.keys())[0]]),
                                  (num_posterior_samples,), replace=False)

        subset_samples = {k: v[indices] for k, v in samples.items()}

        replicated_data = jax.vmap(generate_replicated_data)(subset_samples)

        # Compute test statistics
        def test_statistic(data):
            return {
                'mean': jnp.mean(data),
                'std': jnp.std(data),
                'min': jnp.min(data),
                'max': jnp.max(data),
                'median': jnp.median(data)
            }

        observed_stats = test_statistic(observed_data)
        replicated_stats = jax.vmap(test_statistic)(replicated_data)

        # Compute p-values (posterior predictive p-values)
        p_values = {}
        for stat_name in observed_stats.keys():
            p_val = jnp.mean(replicated_stats[stat_name] >= observed_stats[stat_name])
            p_values[stat_name] = p_val

        return {
            'observed_statistics': observed_stats,
            'replicated_statistics': replicated_stats,
            'p_values': p_values,
            'replicated_data': replicated_data
        }
```

## Use Cases and Scientific Applications

### Climate Science and Environmental Modeling
- **Climate Sensitivity Analysis**: Bayesian estimation of climate model parameters with uncertainty quantification
- **Extreme Weather Prediction**: Hierarchical models for extreme value analysis and risk assessment
- **Carbon Cycle Modeling**: State-space models for atmospheric CO2 dynamics and source attribution
- **Ecosystem Dynamics**: Population models with environmental drivers and uncertainty propagation

### Biomedical Research and Drug Discovery
- **Clinical Trial Analysis**: Hierarchical models for multi-center trials with patient heterogeneity
- **Pharmacokinetic Modeling**: Nonlinear mixed-effects models for drug concentration-time profiles
- **Biomarker Discovery**: Sparse regression with horseshoe priors for high-dimensional genomic data
- **Epidemiological Modeling**: Compartmental models for disease transmission with parameter uncertainty

### Materials Science and Engineering
- **Materials Property Prediction**: Gaussian processes for materials discovery with active learning
- **Failure Analysis**: Survival models for reliability assessment and lifetime prediction
- **Process Optimization**: Bayesian optimization for experimental design and parameter tuning
- **Crystallographic Analysis**: Hierarchical models for structure refinement with systematic errors

### Neuroscience and Cognitive Science
- **Neural Decoding**: State-space models for brain-computer interfaces and neural signal analysis
- **Cognitive Modeling**: Hierarchical Bayesian models for individual differences in cognitive processes
- **Functional Connectivity**: Network models for brain connectivity analysis with uncertainty
- **Behavioral Analysis**: Mixture models for behavioral phenotyping and classification

## Integration with Existing Agents

- **JAX Expert**: Advanced JAX transformations, performance optimization, and device management
- **Statistics Expert**: Statistical model validation, experimental design, and hypothesis testing
- **ML Engineer**: Integration with production ML pipelines and model deployment
- **GPU Computing Expert**: Memory optimization and distributed computing for large-scale inference
- **Visualization Expert**: Posterior visualization, diagnostic plots, and scientific plotting
- **Experiment Manager**: Systematic Bayesian experimental design and model comparison studies

This agent transforms traditional statistical analysis into comprehensive Bayesian workflows with principled uncertainty quantification, scalable inference algorithms, and integration with the modern scientific computing ecosystem through JAX and NumPyro.