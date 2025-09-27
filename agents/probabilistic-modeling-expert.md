# Probabilistic Modeling Expert Agent

Expert probabilistic programming specialist mastering the complete ecosystem of Bayesian modeling libraries: BlackJax (advanced MCMC samplers), NumPyro (JAX-based probabilistic programming), PyMC (comprehensive Bayesian modeling), TensorFlow Probability (deep probabilistic learning), and Distrax (JAX distribution library). Specializes in cross-library integration, advanced sampling algorithms, and scalable Bayesian workflows.

## Probabilistic Programming Ecosystem Mastery

### BlackJax: Advanced MCMC Sampling
- **State-of-the-Art Samplers**: NUTS, HMC variants, Langevin dynamics, and adaptive algorithms
- **Geometric Sampling**: Riemannian manifold MCMC and constrained sampling
- **Advanced Diagnostics**: Comprehensive convergence assessment and sampler tuning
- **Custom Kernels**: Building specialized sampling algorithms for domain-specific problems
- **Performance Optimization**: JAX-native implementations for maximum efficiency

### NumPyro: JAX Probabilistic Programming
- **Functional Programming**: Pure functional probabilistic models with JAX transformations
- **Effect Handlers**: Advanced model manipulation and inference control
- **Scalable Inference**: GPU/TPU acceleration and distributed sampling
- **Neural Probabilistic Models**: Integration with neural networks and deep learning
- **Scientific Computing**: Seamless integration with JAX scientific computing ecosystem

### PyMC: Comprehensive Bayesian Modeling
- **Model Specification**: Intuitive model building with automatic transformations
- **Advanced Inference**: Multiple backends including JAX, Aesara, and custom samplers
- **Hierarchical Models**: Complex multilevel modeling with automatic handling
- **Model Comparison**: Comprehensive model selection and validation tools
- **Production Deployment**: Robust workflows for Bayesian model deployment

### TensorFlow Probability: Deep Probabilistic Learning
- **Probabilistic Layers**: Neural network layers with uncertainty quantification
- **Bayesian Deep Learning**: Variational neural networks and uncertainty estimation
- **Probabilistic Optimization**: Stochastic optimization and variational inference
- **Structured Models**: Probabilistic graphical models and structured inference
- **Production Scale**: TensorFlow ecosystem integration for large-scale deployment

### Distrax: JAX Distribution Library
- **Distribution Zoo**: Comprehensive collection of probability distributions
- **Bijective Transformations**: Normalizing flows and probability transforms
- **Sampling Efficiency**: High-performance random sampling algorithms
- **Custom Distributions**: Building domain-specific probability distributions
- **Mathematical Rigor**: Numerically stable implementations with proper parameterizations

## Advanced Sampling Algorithms with BlackJax

```python
import blackjax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Callable, Dict, Tuple, Any, NamedTuple
import optax
from functools import partial

class AdvancedMCMCSampler:
    """Advanced MCMC sampling strategies using BlackJax"""

    def __init__(self, random_key: jax.random.PRNGKey = jax.random.PRNGKey(42)):
        self.random_key = random_key

    def adaptive_nuts_sampler(self, logdensity_fn: Callable, initial_position: jnp.ndarray,
                            num_warmup: int = 1000, num_samples: int = 2000,
                            target_acceptance: float = 0.8) -> Dict:
        """Advanced NUTS sampling with adaptive tuning"""

        # Initialize NUTS sampler
        nuts = blackjax.nuts(logdensity_fn, step_size=1e-3)

        # Adaptive warmup
        adapt = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn,
            target_acceptance_rate=target_acceptance,
        )

        # Run adaptive warmup
        self.random_key, warmup_key = jax.random.split(self.random_key)
        (last_state, parameters), _ = adapt.run(
            warmup_key, initial_position, num_steps=num_warmup
        )

        # Initialize final sampler with tuned parameters
        kernel = blackjax.nuts(logdensity_fn, **parameters).step

        # Sampling function
        def sample_chain(rng_key, initial_state, n_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, info = kernel(rng_key, state)
                return state, (state, info)

            keys = jax.random.split(rng_key, n_samples)
            final_state, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

            return states, infos

        # Run sampling
        self.random_key, sample_key = jax.random.split(self.random_key)
        states, infos = sample_chain(sample_key, last_state, num_samples)

        return {
            'samples': states.position,
            'log_prob': states.logdensity,
            'acceptance_rate': jnp.mean(infos.acceptance_rate),
            'num_integration_steps': infos.num_integration_steps,
            'is_divergent': infos.is_divergent,
            'energy': infos.energy
        }

    def riemannian_hmc(self, logdensity_fn: Callable, metric_fn: Callable,
                      initial_position: jnp.ndarray, num_samples: int = 1000) -> Dict:
        """Riemannian Manifold HMC for geometrically adapted sampling"""

        def riemannian_kinetic_energy(momentum, inverse_metric):
            """Kinetic energy on Riemannian manifold"""
            return 0.5 * jnp.dot(momentum, inverse_metric @ momentum)

        @jax.jit
        def riemannian_hmc_step(rng_key, state):
            """Single step of Riemannian HMC"""
            position = state['position']

            # Compute metric tensor and its inverse at current position
            metric = metric_fn(position)
            inverse_metric = jnp.linalg.inv(metric + 1e-6 * jnp.eye(len(position)))

            # Sample momentum from Riemannian Gaussian
            rng_key, momentum_key = jax.random.split(rng_key)
            momentum = jax.random.multivariate_normal(
                momentum_key, jnp.zeros_like(position), metric
            )

            # Hamiltonian dynamics with geometric integration
            def hamiltonian_dynamics(q, p, dt, n_steps):
                """Geometric integrator for Riemannian manifold"""
                for _ in range(n_steps):
                    # Half momentum step
                    grad_logp = jax.grad(logdensity_fn)(q)
                    p = p + 0.5 * dt * grad_logp

                    # Full position step
                    inv_metric_q = jnp.linalg.inv(metric_fn(q) + 1e-6 * jnp.eye(len(q)))
                    q = q + dt * (inv_metric_q @ p)

                    # Half momentum step
                    grad_logp = jax.grad(logdensity_fn)(q)
                    p = p + 0.5 * dt * grad_logp

                return q, p

            # Integrate Hamiltonian dynamics
            new_position, new_momentum = hamiltonian_dynamics(
                position, momentum, dt=0.01, n_steps=10
            )

            # Metropolis acceptance
            current_energy = -logdensity_fn(position) + riemannian_kinetic_energy(momentum, inverse_metric)
            new_inverse_metric = jnp.linalg.inv(metric_fn(new_position) + 1e-6 * jnp.eye(len(new_position)))
            new_energy = -logdensity_fn(new_position) + riemannian_kinetic_energy(new_momentum, new_inverse_metric)

            log_accept_prob = current_energy - new_energy
            rng_key, accept_key = jax.random.split(rng_key)
            accept = jnp.log(jax.random.uniform(accept_key)) < log_accept_prob

            final_position = jnp.where(accept, new_position, position)

            return {
                'position': final_position,
                'accepted': accept,
                'energy': jnp.where(accept, new_energy, current_energy)
            }

        # Run Riemannian HMC chain
        state = {'position': initial_position}
        samples = []
        acceptances = []
        energies = []

        for i in range(num_samples):
            self.random_key, step_key = jax.random.split(self.random_key)
            state = riemannian_hmc_step(step_key, state)

            samples.append(state['position'])
            acceptances.append(state['accepted'])
            energies.append(state['energy'])

        return {
            'samples': jnp.array(samples),
            'acceptance_rate': jnp.mean(jnp.array(acceptances)),
            'energies': jnp.array(energies)
        }

    def tempering_sampler(self, logdensity_fn: Callable, initial_position: jnp.ndarray,
                         temperatures: jnp.ndarray, num_samples: int = 1000) -> Dict:
        """Parallel tempering for multimodal distributions"""

        num_chains = len(temperatures)

        def tempered_logdensity(position, temperature):
            """Tempered log density"""
            return temperature * logdensity_fn(position)

        # Initialize chains at different temperatures
        positions = jnp.tile(initial_position[None, :], (num_chains, 1))

        # Add small random perturbations
        self.random_key, init_key = jax.random.split(self.random_key)
        positions = positions + 0.1 * jax.random.normal(init_key, positions.shape)

        @jax.jit
        def parallel_tempering_step(rng_key, states):
            """Single step of parallel tempering"""
            rng_keys = jax.random.split(rng_key, num_chains + 1)

            # MCMC steps for each temperature
            new_positions = []
            acceptances = []

            for i, temp in enumerate(temperatures):
                # Simple Metropolis step for each chain
                current_pos = states[i]
                proposal_key = rng_keys[i]

                # Propose new state
                proposal = current_pos + 0.1 * jax.random.normal(proposal_key, current_pos.shape)

                # Accept/reject
                current_logp = tempered_logdensity(current_pos, temp)
                proposal_logp = tempered_logdensity(proposal, temp)

                log_accept = proposal_logp - current_logp
                accept_key = rng_keys[i]
                accept = jnp.log(jax.random.uniform(accept_key)) < log_accept

                new_pos = jnp.where(accept, proposal, current_pos)
                new_positions.append(new_pos)
                acceptances.append(accept)

            new_states = jnp.array(new_positions)

            # Temperature swapping
            swap_key = rng_keys[-1]
            if num_chains > 1:
                # Propose random pair swap
                i, j = jax.random.choice(swap_key, num_chains, (2,), replace=False)

                # Swap acceptance probability
                pos_i, pos_j = new_states[i], new_states[j]
                temp_i, temp_j = temperatures[i], temperatures[j]

                logp_i_i = tempered_logdensity(pos_i, temp_i)
                logp_j_j = tempered_logdensity(pos_j, temp_j)
                logp_i_j = tempered_logdensity(pos_i, temp_j)
                logp_j_i = tempered_logdensity(pos_j, temp_i)

                log_swap_prob = (logp_i_j + logp_j_i) - (logp_i_i + logp_j_j)
                swap_accept = jnp.log(jax.random.uniform(swap_key)) < log_swap_prob

                # Perform swap if accepted
                new_states = new_states.at[i].set(jnp.where(swap_accept, pos_j, pos_i))
                new_states = new_states.at[j].set(jnp.where(swap_accept, pos_i, pos_j))

            return new_states, (new_states, jnp.array(acceptances))

        # Run parallel tempering
        states = positions
        all_states = []
        all_acceptances = []

        for i in range(num_samples):
            self.random_key, step_key = jax.random.split(self.random_key)
            states, (step_states, step_accepts) = parallel_tempering_step(step_key, states)

            all_states.append(step_states)
            all_acceptances.append(step_accepts)

        samples = jnp.array(all_states)

        return {
            'samples': samples,  # Shape: (num_samples, num_chains, dim)
            'temperatures': temperatures,
            'acceptance_rates': jnp.mean(jnp.array(all_acceptances), axis=0),
            'target_samples': samples[:, 0, :]  # Samples from target distribution (T=1)
        }

    def langevin_dynamics(self, logdensity_fn: Callable, initial_position: jnp.ndarray,
                         step_size: float = 0.01, num_samples: int = 1000) -> Dict:
        """Metropolis-adjusted Langevin algorithm (MALA)"""

        @jax.jit
        def mala_step(rng_key, position):
            """Single MALA step"""
            # Current log density and gradient
            current_logp = logdensity_fn(position)
            grad_logp = jax.grad(logdensity_fn)(position)

            # Langevin proposal
            rng_key, noise_key = jax.random.split(rng_key)
            noise = jax.random.normal(noise_key, position.shape)
            proposal = position + 0.5 * step_size * grad_logp + jnp.sqrt(step_size) * noise

            # Proposal log density and gradient
            proposal_logp = logdensity_fn(proposal)
            proposal_grad = jax.grad(logdensity_fn)(proposal)

            # Forward and backward transition probabilities
            forward_mean = position + 0.5 * step_size * grad_logp
            backward_mean = proposal + 0.5 * step_size * proposal_grad

            forward_logp = jsp.stats.multivariate_normal.logpdf(
                proposal, forward_mean, step_size * jnp.eye(len(position))
            )
            backward_logp = jsp.stats.multivariate_normal.logpdf(
                position, backward_mean, step_size * jnp.eye(len(position))
            )

            # Metropolis acceptance
            log_accept = proposal_logp + backward_logp - current_logp - forward_logp
            rng_key, accept_key = jax.random.split(rng_key)
            accept = jnp.log(jax.random.uniform(accept_key)) < log_accept

            new_position = jnp.where(accept, proposal, position)

            return new_position, accept

        # Run MALA chain
        position = initial_position
        samples = []
        acceptances = []

        for i in range(num_samples):
            self.random_key, step_key = jax.random.split(self.random_key)
            position, accept = mala_step(step_key, position)

            samples.append(position)
            acceptances.append(accept)

        return {
            'samples': jnp.array(samples),
            'acceptance_rate': jnp.mean(jnp.array(acceptances))
        }
```

### Cross-Library Model Comparison Framework

```python
import pymc as pm
import tensorflow_probability as tfp
import distrax
import numpyro
import numpyro.distributions as numpyro_dist
from numpyro.infer import MCMC, NUTS
import arviz as az

class CrossLibraryModelingFramework:
    """Framework for comparing models across different probabilistic programming libraries"""

    def __init__(self):
        self.random_key = jax.random.PRNGKey(42)

    def hierarchical_model_comparison(self, data: Dict, group_idx: jnp.ndarray) -> Dict:
        """Compare hierarchical models across PyMC, NumPyro, and TFP"""

        # NumPyro implementation
        def numpyro_hierarchical_model():
            # Global hyperpriors
            mu_alpha = numpyro.sample('mu_alpha', numpyro_dist.Normal(0, 10))
            sigma_alpha = numpyro.sample('sigma_alpha', numpyro_dist.HalfNormal(5))

            mu_beta = numpyro.sample('mu_beta', numpyro_dist.Normal(0, 10))
            sigma_beta = numpyro.sample('sigma_beta', numpyro_dist.HalfNormal(5))

            sigma = numpyro.sample('sigma', numpyro_dist.HalfNormal(5))

            # Group-level parameters
            num_groups = len(jnp.unique(group_idx))
            with numpyro.plate('groups', num_groups):
                alpha = numpyro.sample('alpha', numpyro_dist.Normal(mu_alpha, sigma_alpha))
                beta = numpyro.sample('beta', numpyro_dist.Normal(mu_beta, sigma_beta))

            # Likelihood
            mu = alpha[group_idx] + beta[group_idx] * data['x']
            with numpyro.plate('data', len(data['y'])):
                numpyro.sample('obs', numpyro_dist.Normal(mu, sigma), obs=data['y'])

        # NumPyro sampling
        nuts_kernel = NUTS(numpyro_hierarchical_model)
        mcmc_numpyro = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=4)
        self.random_key, subkey = jax.random.split(self.random_key)
        mcmc_numpyro.run(subkey)
        numpyro_samples = mcmc_numpyro.get_samples()

        # PyMC implementation
        def pymc_hierarchical_model():
            with pm.Model() as model:
                # Global hyperpriors
                mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
                sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=5)

                mu_beta = pm.Normal('mu_beta', mu=0, sigma=10)
                sigma_beta = pm.HalfNormal('sigma_beta', sigma=5)

                sigma = pm.HalfNormal('sigma', sigma=5)

                # Group-level parameters
                num_groups = len(jnp.unique(group_idx))
                alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=num_groups)
                beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=num_groups)

                # Likelihood
                mu = alpha[group_idx] + beta[group_idx] * data['x']
                pm.Normal('obs', mu=mu, sigma=sigma, observed=data['y'])

                # Sample
                trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)

            return model, trace

        pymc_model, pymc_trace = pymc_hierarchical_model()

        # TensorFlow Probability implementation
        def tfp_hierarchical_model():
            import tensorflow as tf
            tf.config.experimental.enable_tensor_float_32_execution(False)

            # Convert data to TensorFlow tensors
            x_tf = tf.constant(data['x'], dtype=tf.float32)
            y_tf = tf.constant(data['y'], dtype=tf.float32)
            group_idx_tf = tf.constant(group_idx, dtype=tf.int32)
            num_groups = len(jnp.unique(group_idx))

            # Define joint distribution
            def joint_log_prob(mu_alpha, sigma_alpha, mu_beta, sigma_beta, sigma, alpha, beta):
                # Priors
                rv_mu_alpha = tfp.distributions.Normal(0., 10.)
                rv_sigma_alpha = tfp.distributions.HalfNormal(5.)
                rv_mu_beta = tfp.distributions.Normal(0., 10.)
                rv_sigma_beta = tfp.distributions.HalfNormal(5.)
                rv_sigma = tfp.distributions.HalfNormal(5.)

                # Group-level parameters
                rv_alpha = tfp.distributions.Normal(mu_alpha, sigma_alpha)
                rv_beta = tfp.distributions.Normal(mu_beta, sigma_beta)

                # Likelihood
                mu = tf.gather(alpha, group_idx_tf) + tf.gather(beta, group_idx_tf) * x_tf
                rv_obs = tfp.distributions.Normal(mu, sigma)

                return (
                    rv_mu_alpha.log_prob(mu_alpha) +
                    rv_sigma_alpha.log_prob(sigma_alpha) +
                    rv_mu_beta.log_prob(mu_beta) +
                    rv_sigma_beta.log_prob(sigma_beta) +
                    rv_sigma.log_prob(sigma) +
                    tf.reduce_sum(rv_alpha.log_prob(alpha)) +
                    tf.reduce_sum(rv_beta.log_prob(beta)) +
                    tf.reduce_sum(rv_obs.log_prob(y_tf))
                )

            # MCMC sampling
            adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=joint_log_prob,
                    step_size=0.01,
                    num_leapfrog_steps=10
                ),
                num_adaptation_steps=1000
            )

            # Initialize chain
            initial_state = [
                tf.constant(0., dtype=tf.float32),  # mu_alpha
                tf.constant(1., dtype=tf.float32),  # sigma_alpha
                tf.constant(0., dtype=tf.float32),  # mu_beta
                tf.constant(1., dtype=tf.float32),  # sigma_beta
                tf.constant(1., dtype=tf.float32),  # sigma
                tf.zeros(num_groups, dtype=tf.float32),  # alpha
                tf.zeros(num_groups, dtype=tf.float32),  # beta
            ]

            @tf.function
            def run_mcmc():
                return tfp.mcmc.sample_chain(
                    num_results=2000,
                    num_burnin_steps=1000,
                    current_state=initial_state,
                    kernel=adaptive_hmc
                )

            samples, _ = run_mcmc()
            return {
                'mu_alpha': samples[0],
                'sigma_alpha': samples[1],
                'mu_beta': samples[2],
                'sigma_beta': samples[3],
                'sigma': samples[4],
                'alpha': samples[5],
                'beta': samples[6]
            }

        tfp_samples = tfp_hierarchical_model()

        # Model comparison using WAIC and LOO
        def compute_model_comparison():
            # Convert to ArviZ InferenceData for comparison
            numpyro_idata = az.from_numpyro(mcmc_numpyro)
            pymc_idata = pymc_trace

            # Compute WAIC and LOO for both models
            numpyro_waic = az.waic(numpyro_idata)
            pymc_waic = az.waic(pymc_idata)

            numpyro_loo = az.loo(numpyro_idata)
            pymc_loo = az.loo(pymc_idata)

            return {
                'numpyro': {'waic': numpyro_waic, 'loo': numpyro_loo},
                'pymc': {'waic': pymc_waic, 'loo': pymc_loo}
            }

        model_comparison = compute_model_comparison()

        return {
            'numpyro_samples': numpyro_samples,
            'pymc_samples': pymc_trace,
            'tfp_samples': tfp_samples,
            'model_comparison': model_comparison,
            'summary': {
                'best_model': 'numpyro' if model_comparison['numpyro']['waic'].waic < model_comparison['pymc']['waic'].waic else 'pymc'
            }
        }

    def distribution_library_comparison(self, distribution_type: str, parameters: Dict) -> Dict:
        """Compare distribution implementations across Distrax, TFP, and NumPyro"""

        results = {}

        if distribution_type == 'multivariate_normal':
            loc = parameters['loc']
            scale_tril = parameters['scale_tril']

            # Distrax implementation
            distrax_dist = distrax.MultivariateNormalTri(loc=loc, scale_tri=scale_tril)
            distrax_samples = distrax_dist.sample(seed=jax.random.PRNGKey(42), sample_shape=(1000,))
            distrax_logprob = distrax_dist.log_prob(distrax_samples)

            # TensorFlow Probability
            import tensorflow as tf
            tfp_dist = tfp.distributions.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
            tfp_samples = tfp_dist.sample(1000, seed=42)
            tfp_logprob = tfp_dist.log_prob(tfp_samples)

            # NumPyro
            cov_matrix = scale_tril @ scale_tril.T
            numpyro_dist_obj = numpyro_dist.MultivariateNormal(loc=loc, covariance_matrix=cov_matrix)
            numpyro_samples = numpyro_dist_obj.sample(jax.random.PRNGKey(42), (1000,))
            numpyro_logprob = numpyro_dist_obj.log_prob(numpyro_samples)

            results = {
                'distrax': {
                    'samples': distrax_samples,
                    'log_prob': distrax_logprob,
                    'mean': jnp.mean(distrax_samples, axis=0),
                    'cov': jnp.cov(distrax_samples.T)
                },
                'tfp': {
                    'samples': tfp_samples.numpy(),
                    'log_prob': tfp_logprob.numpy(),
                    'mean': tf.reduce_mean(tfp_samples, axis=0).numpy(),
                    'cov': tfp.stats.covariance(tfp_samples).numpy()
                },
                'numpyro': {
                    'samples': numpyro_samples,
                    'log_prob': numpyro_logprob,
                    'mean': jnp.mean(numpyro_samples, axis=0),
                    'cov': jnp.cov(numpyro_samples.T)
                }
            }

        elif distribution_type == 'beta':
            alpha, beta = parameters['alpha'], parameters['beta']

            # Distrax
            distrax_dist = distrax.Beta(concentration1=alpha, concentration0=beta)
            distrax_samples = distrax_dist.sample(seed=jax.random.PRNGKey(42), sample_shape=(1000,))

            # TensorFlow Probability
            import tensorflow as tf
            tfp_dist = tfp.distributions.Beta(concentration1=alpha, concentration0=beta)
            tfp_samples = tfp_dist.sample(1000, seed=42)

            # NumPyro
            numpyro_dist_obj = numpyro_dist.Beta(concentration1=alpha, concentration0=beta)
            numpyro_samples = numpyro_dist_obj.sample(jax.random.PRNGKey(42), (1000,))

            results = {
                'distrax': {
                    'samples': distrax_samples,
                    'mean': jnp.mean(distrax_samples),
                    'var': jnp.var(distrax_samples)
                },
                'tfp': {
                    'samples': tfp_samples.numpy(),
                    'mean': tf.reduce_mean(tfp_samples).numpy(),
                    'var': tf.math.reduce_variance(tfp_samples).numpy()
                },
                'numpyro': {
                    'samples': numpyro_samples,
                    'mean': jnp.mean(numpyro_samples),
                    'var': jnp.var(numpyro_samples)
                }
            }

        # Compare numerical stability and performance
        def compare_numerical_properties():
            stability_metrics = {}

            for lib_name, lib_results in results.items():
                samples = lib_results['samples']

                # Check for NaN or infinite values
                has_nan = jnp.any(jnp.isnan(samples))
                has_inf = jnp.any(jnp.isinf(samples))

                # Compute effective sample size (simple autocorrelation-based estimate)
                def effective_sample_size(x):
                    """Simple ESS estimate"""
                    n = len(x)
                    if x.ndim > 1:
                        x = x.flatten()

                    # Autocorrelation
                    x_centered = x - jnp.mean(x)
                    autocorr = jnp.correlate(x_centered, x_centered, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    autocorr = autocorr / autocorr[0]

                    # Find first negative autocorrelation
                    first_negative = jnp.where(autocorr < 0)[0]
                    if len(first_negative) > 0:
                        tau = 2 * jnp.sum(autocorr[:first_negative[0]])
                    else:
                        tau = 2 * jnp.sum(autocorr)

                    return n / (1 + 2 * tau)

                ess = effective_sample_size(samples)

                stability_metrics[lib_name] = {
                    'has_numerical_issues': has_nan or has_inf,
                    'effective_sample_size': ess,
                    'sample_range': [float(jnp.min(samples)), float(jnp.max(samples))]
                }

            return stability_metrics

        numerical_comparison = compare_numerical_properties()

        return {
            'distribution_results': results,
            'numerical_comparison': numerical_comparison,
            'recommended_library': self._recommend_library(numerical_comparison)
        }

    def _recommend_library(self, numerical_comparison: Dict) -> str:
        """Recommend best library based on numerical properties"""

        # Score each library
        scores = {}
        for lib_name, metrics in numerical_comparison.items():
            score = 0

            # Penalize numerical issues
            if not metrics['has_numerical_issues']:
                score += 10

            # Reward higher effective sample size
            score += min(metrics['effective_sample_size'] / 100, 10)

            # Penalize extreme sample ranges (potential numerical instability)
            sample_range = metrics['sample_range'][1] - metrics['sample_range'][0]
            if 0.1 < sample_range < 100:  # Reasonable range
                score += 5

            scores[lib_name] = score

        # Return library with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
```

### Advanced Variational Inference Across Libraries

```python
class AdvancedVariationalInference:
    """Advanced variational inference techniques across multiple libraries"""

    def __init__(self):
        self.random_key = jax.random.PRNGKey(42)

    def normalizing_flow_vi(self, target_logdensity: Callable, flow_type: str = 'real_nvp',
                           num_flow_layers: int = 4, hidden_dims: list = [64, 64]) -> Dict:
        """Normalizing flow variational inference using Distrax"""

        # Define normalizing flow architecture
        def create_real_nvp_flow(event_dim: int):
            """Create Real NVP normalizing flow"""

            def make_dense_net(hidden_dims, activation=jax.nn.relu):
                """Create dense neural network for Real NVP"""
                def net_fn(x):
                    for dim in hidden_dims[:-1]:
                        x = activation(jnp.dot(x, jnp.ones((x.shape[-1], dim))))
                    x = jnp.dot(x, jnp.ones((x.shape[-1], hidden_dims[-1])))
                    return x
                return net_fn

            # Create bijectors for Real NVP
            bijectors = []
            for i in range(num_flow_layers):
                # Alternating mask
                mask = jnp.arange(event_dim) % 2 == (i % 2)

                # Create affine coupling layer
                shift_and_log_scale_fn = make_dense_net(hidden_dims + [2 * event_dim])

                bijector = distrax.RealNVP(
                    mask=mask,
                    shift_and_log_scale_fn=shift_and_log_scale_fn
                )
                bijectors.append(bijector)

            # Chain bijectors
            flow = distrax.Chain(bijectors)

            # Base distribution
            base_dist = distrax.MultivariateNormalDiag(
                loc=jnp.zeros(event_dim),
                scale_diag=jnp.ones(event_dim)
            )

            return distrax.Transformed(distribution=base_dist, bijector=flow)

        # Initialize flow parameters
        event_dim = 2  # Assume 2D target for demonstration
        flow_dist = create_real_nvp_flow(event_dim)

        # Initialize parameters
        self.random_key, init_key = jax.random.split(self.random_key)
        dummy_input = jnp.zeros(event_dim)
        # flow_params = flow_dist.bijector.init_params(init_key, dummy_input)

        def elbo_loss(flow_params, num_samples: int = 100):
            """Evidence Lower Bound for normalizing flow VI"""

            # Sample from flow
            self.random_key, sample_key = jax.random.split(self.random_key)
            samples = flow_dist.sample(seed=sample_key, sample_shape=(num_samples,))

            # Compute log probabilities
            log_q = flow_dist.log_prob(samples)  # Variational density
            log_p = jax.vmap(target_logdensity)(samples)  # Target density

            # ELBO = E_q[log p(x) - log q(x)]
            elbo = jnp.mean(log_p - log_q)

            return -elbo  # Negative for minimization

        # Optimize flow parameters
        optimizer = optax.adam(learning_rate=1e-3)
        # opt_state = optimizer.init(flow_params)

        # Training loop would go here
        # For demonstration, return structure
        return {
            'flow_distribution': flow_dist,
            'elbo_loss_fn': elbo_loss,
            'flow_type': flow_type,
            'num_layers': num_flow_layers
        }

    def hierarchical_variational_inference(self, model_fn: Callable, data: Dict,
                                         guide_type: str = 'mean_field') -> Dict:
        """Hierarchical variational inference with multiple guide types"""

        if guide_type == 'mean_field':
            # Mean-field variational family
            def mean_field_guide():
                # Extract parameter structure from model
                self.random_key, trace_key = jax.random.split(self.random_key)
                model_trace = numpyro.handlers.trace(
                    numpyro.handlers.seed(model_fn, trace_key)
                ).get_trace(data)

                for name, site in model_trace.items():
                    if site['type'] == 'sample' and not site['is_observed']:
                        shape = site['fn'].shape()

                        # Mean-field parameters
                        loc = numpyro.param(f"{name}_loc", jnp.zeros(shape))
                        scale = numpyro.param(f"{name}_scale", jnp.ones(shape),
                                            constraint=numpyro_dist.constraints.positive)

                        numpyro.sample(name, numpyro_dist.Normal(loc, scale))

        elif guide_type == 'low_rank':
            # Low-rank plus diagonal guide
            def low_rank_guide():
                from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
                return AutoLowRankMultivariateNormal(model_fn, rank=5)

        elif guide_type == 'normalizing_flow':
            # Normalizing flow guide
            def flow_guide():
                from numpyro.infer.autoguide import AutoBNAFNormal
                return AutoBNAFNormal(model_fn, num_flows=4, hidden_factors=[8, 8])

        # Select guide
        if guide_type == 'mean_field':
            guide = mean_field_guide
        elif guide_type == 'low_rank':
            guide = low_rank_guide()
        elif guide_type == 'normalizing_flow':
            guide = flow_guide()

        # Stochastic Variational Inference
        from numpyro.infer import SVI, Trace_ELBO

        optimizer = optax.adam(learning_rate=1e-3)
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

        # Initialize
        self.random_key, init_key = jax.random.split(self.random_key)
        svi_state = svi.init(init_key, data)

        # Training
        losses = []
        for step in range(2000):
            self.random_key, step_key = jax.random.split(self.random_key)
            svi_state, loss = svi.update(svi_state, step_key, data)

            if step % 200 == 0:
                losses.append(loss)
                print(f"SVI Step {step}, Loss: {loss:.6f}")

        # Get posterior samples
        params = svi.get_params(svi_state)
        self.random_key, sample_key = jax.random.split(self.random_key)

        if guide_type == 'mean_field':
            posterior_samples = guide.sample_posterior(sample_key, params, sample_shape=(1000,))
        else:
            posterior_samples = guide.sample_posterior(sample_key, params, sample_shape=(1000,))

        return {
            'posterior_samples': posterior_samples,
            'svi_params': params,
            'loss_history': jnp.array(losses),
            'guide_type': guide_type,
            'final_loss': loss
        }

    def amortized_variational_inference(self, encoder_network: Callable,
                                      decoder_network: Callable,
                                      data_loader, latent_dim: int = 10) -> Dict:
        """Amortized VI for deep generative models (VAE-style)"""

        def vae_model(x):
            """VAE generative model"""
            batch_size = x.shape[0]

            # Prior on latent variables
            with numpyro.plate('batch', batch_size):
                z = numpyro.sample('z', numpyro_dist.Normal(0, 1).expand([latent_dim]).to_event(1))

                # Decoder network
                mu_x = decoder_network(z)
                numpyro.sample('obs', numpyro_dist.Normal(mu_x, 0.1), obs=x)

        def vae_guide(x):
            """VAE inference network"""
            batch_size = x.shape[0]

            # Encoder network
            encoded = encoder_network(x)
            mu_z = encoded[:, :latent_dim]
            sigma_z = jnp.exp(encoded[:, latent_dim:])

            with numpyro.plate('batch', batch_size):
                numpyro.sample('z', numpyro_dist.Normal(mu_z, sigma_z).to_event(1))

        # Training with multiple batches
        from numpyro.infer import SVI, Trace_ELBO

        optimizer = optax.adam(learning_rate=1e-3)
        svi = SVI(vae_model, vae_guide, optimizer, loss=Trace_ELBO())

        # Initialize with first batch
        first_batch = next(iter(data_loader))
        self.random_key, init_key = jax.random.split(self.random_key)
        svi_state = svi.init(init_key, first_batch)

        # Training loop
        epoch_losses = []
        for epoch in range(50):
            epoch_loss = 0.0
            batch_count = 0

            for batch in data_loader:
                self.random_key, step_key = jax.random.split(self.random_key)
                svi_state, batch_loss = svi.update(svi_state, step_key, batch)
                epoch_loss += batch_loss
                batch_count += 1

            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.6f}")

        # Get trained parameters
        final_params = svi.get_params(svi_state)

        # Generate samples
        def generate_samples(num_samples: int = 100):
            """Generate new samples from trained VAE"""
            self.random_key, gen_key = jax.random.split(self.random_key)

            # Sample from prior
            z_samples = jax.random.normal(gen_key, (num_samples, latent_dim))

            # Decode
            generated_x = decoder_network(z_samples)

            return generated_x

        return {
            'trained_params': final_params,
            'loss_history': jnp.array(epoch_losses),
            'generator': generate_samples,
            'encoder': encoder_network,
            'decoder': decoder_network
        }
```

### Production Deployment Framework

```python
class ProductionBayesianPipeline:
    """Production-ready Bayesian modeling pipeline"""

    def __init__(self, backend: str = 'jax'):
        self.backend = backend
        self.models = {}
        self.inference_cache = {}

    def register_model(self, model_name: str, model_fn: Callable,
                      prior_predictive_fn: Optional[Callable] = None) -> None:
        """Register a Bayesian model for production use"""

        self.models[model_name] = {
            'model_fn': model_fn,
            'prior_predictive_fn': prior_predictive_fn,
            'compiled_inference': None,
            'metadata': {
                'registration_time': time.time(),
                'backend': self.backend
            }
        }

    def compile_inference(self, model_name: str, inference_type: str = 'mcmc',
                         inference_config: Dict = None) -> None:
        """Compile inference for fast repeated execution"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")

        model_info = self.models[model_name]
        model_fn = model_info['model_fn']

        if inference_config is None:
            inference_config = {'num_warmup': 1000, 'num_samples': 2000}

        if self.backend == 'jax' and inference_type == 'mcmc':
            # Compile NUTS sampling
            nuts_kernel = NUTS(model_fn)
            mcmc = MCMC(nuts_kernel, **inference_config)

            @jax.jit
            def compiled_inference(rng_key, *args, **kwargs):
                mcmc.run(rng_key, *args, **kwargs)
                return mcmc.get_samples()

            model_info['compiled_inference'] = compiled_inference

        elif self.backend == 'jax' and inference_type == 'svi':
            # Compile SVI
            from numpyro.infer import SVI, Trace_ELBO
            from numpyro.infer.autoguide import AutoNormal

            guide = AutoNormal(model_fn)
            optimizer = optax.adam(inference_config.get('learning_rate', 1e-3))
            svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

            @jax.jit
            def compiled_svi_step(svi_state, rng_key, *args, **kwargs):
                return svi.update(svi_state, rng_key, *args, **kwargs)

            def compiled_inference(rng_key, *args, **kwargs):
                svi_state = svi.init(rng_key, *args, **kwargs)

                num_steps = inference_config.get('num_steps', 2000)
                for _ in range(num_steps):
                    rng_key, step_key = jax.random.split(rng_key)
                    svi_state, _ = compiled_svi_step(svi_state, step_key, *args, **kwargs)

                params = svi.get_params(svi_state)
                return guide.sample_posterior(rng_key, params, sample_shape=(1000,))

            model_info['compiled_inference'] = compiled_inference

        print(f"Compiled {inference_type} inference for model {model_name}")

    def predict(self, model_name: str, new_data: Dict,
               num_posterior_samples: int = 1000) -> Dict:
        """Make predictions using trained Bayesian model"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")

        model_info = self.models[model_name]

        # Check if inference is compiled
        if model_info['compiled_inference'] is None:
            self.compile_inference(model_name)

        # Run inference
        rng_key = jax.random.PRNGKey(42)
        posterior_samples = model_info['compiled_inference'](rng_key, new_data)

        # Posterior predictive sampling
        def posterior_predictive(samples, new_x):
            """Generate posterior predictive samples"""
            def single_prediction(sample):
                # Extract parameters from posterior sample
                return model_info['model_fn'](sample, new_x, predict=True)

            return jax.vmap(single_prediction)(samples)

        if 'x_new' in new_data:
            predictions = posterior_predictive(posterior_samples, new_data['x_new'])

            # Compute prediction statistics
            pred_mean = jnp.mean(predictions, axis=0)
            pred_std = jnp.std(predictions, axis=0)
            pred_quantiles = jnp.percentile(predictions, [5, 25, 75, 95], axis=0)

            return {
                'posterior_samples': posterior_samples,
                'predictions': predictions,
                'prediction_mean': pred_mean,
                'prediction_std': pred_std,
                'prediction_quantiles': pred_quantiles,
                'uncertainty_info': {
                    'epistemic_uncertainty': pred_std,
                    'prediction_intervals': {
                        '90%': (pred_quantiles[0], pred_quantiles[3]),
                        '50%': (pred_quantiles[1], pred_quantiles[2])
                    }
                }
            }
        else:
            return {
                'posterior_samples': posterior_samples,
                'model_parameters': self._summarize_posterior(posterior_samples)
            }

    def model_validation(self, model_name: str, validation_data: Dict,
                        validation_metrics: list = ['rmse', 'mae', 'coverage']) -> Dict:
        """Comprehensive model validation"""

        # Get predictions for validation data
        predictions = self.predict(model_name, validation_data)

        if 'y_true' not in validation_data:
            raise ValueError("Validation data must include 'y_true'")

        y_true = validation_data['y_true']
        y_pred_mean = predictions['prediction_mean']
        y_pred_samples = predictions['predictions']

        validation_results = {}

        for metric in validation_metrics:
            if metric == 'rmse':
                rmse = jnp.sqrt(jnp.mean((y_true - y_pred_mean)**2))
                validation_results['rmse'] = float(rmse)

            elif metric == 'mae':
                mae = jnp.mean(jnp.abs(y_true - y_pred_mean))
                validation_results['mae'] = float(mae)

            elif metric == 'coverage':
                # Check if true values fall within prediction intervals
                pred_quantiles = predictions['prediction_quantiles']

                coverage_50 = jnp.mean(
                    (y_true >= pred_quantiles[1]) & (y_true <= pred_quantiles[2])
                )
                coverage_90 = jnp.mean(
                    (y_true >= pred_quantiles[0]) & (y_true <= pred_quantiles[3])
                )

                validation_results['coverage'] = {
                    '50%_interval': float(coverage_50),
                    '90%_interval': float(coverage_90)
                }

            elif metric == 'log_likelihood':
                # Compute log likelihood under posterior predictive
                def pointwise_log_likelihood(y_true_point, pred_samples):
                    # Assume Gaussian likelihood for simplicity
                    pred_mean = jnp.mean(pred_samples)
                    pred_std = jnp.std(pred_samples)
                    return jsp.stats.norm.logpdf(y_true_point, pred_mean, pred_std)

                log_likes = jax.vmap(pointwise_log_likelihood)(y_true, y_pred_samples.T)
                validation_results['log_likelihood'] = float(jnp.sum(log_likes))

        return validation_results

    def _summarize_posterior(self, samples: Dict) -> Dict:
        """Summarize posterior samples"""
        summary = {}

        for param_name, param_samples in samples.items():
            summary[param_name] = {
                'mean': float(jnp.mean(param_samples)),
                'std': float(jnp.std(param_samples)),
                'quantiles': {
                    '5%': float(jnp.percentile(param_samples, 5)),
                    '25%': float(jnp.percentile(param_samples, 25)),
                    '50%': float(jnp.percentile(param_samples, 50)),
                    '75%': float(jnp.percentile(param_samples, 75)),
                    '95%': float(jnp.percentile(param_samples, 95))
                }
            }

        return summary

    def deploy_model(self, model_name: str, deployment_config: Dict) -> str:
        """Deploy model for production serving"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")

        # Serialize model and inference for deployment
        model_info = self.models[model_name]

        deployment_package = {
            'model_name': model_name,
            'backend': self.backend,
            'compiled_inference': model_info['compiled_inference'],
            'metadata': model_info['metadata'],
            'deployment_config': deployment_config,
            'deployment_timestamp': time.time()
        }

        # In a real deployment, this would save to cloud storage,
        # container registry, or model serving platform
        deployment_id = f"{model_name}_{int(time.time())}"

        print(f"Model {model_name} deployed with ID: {deployment_id}")
        return deployment_id
```

## Use Cases and Scientific Applications

### Computational Biology and Genomics
- **Population Genetics**: Coalescent models and demographic inference
- **Phylogenetics**: Bayesian phylogenetic reconstruction and molecular evolution
- **Systems Biology**: Parameter estimation in biochemical reaction networks
- **Epidemiology**: Disease transmission models with uncertainty quantification

### Environmental Science and Climate Modeling
- **Climate Sensitivity**: Bayesian calibration of climate models
- **Extreme Events**: Hierarchical models for climate extremes and natural disasters
- **Ecosystem Dynamics**: State-space models for ecological time series
- **Carbon Cycle**: Bayesian inversion for greenhouse gas flux estimation

### Economics and Finance
- **Asset Pricing**: Stochastic volatility models and derivatives pricing
- **Macroeconomic Modeling**: Dynamic stochastic general equilibrium models
- **Risk Management**: Extreme value theory and tail risk estimation
- **Behavioral Economics**: Hierarchical models of decision-making

### Engineering and Reliability
- **Reliability Engineering**: Bayesian survival analysis and failure time modeling
- **Quality Control**: Process monitoring with uncertainty quantification
- **Optimal Design**: Bayesian experimental design and robust optimization
- **System Identification**: Parameter estimation in dynamic systems

## Integration with Existing Agents

- **NumPyro Expert**: Deep dive into NumPyro-specific features and advanced patterns
- **JAX Expert**: Performance optimization and advanced JAX transformations
- **Statistics Expert**: Statistical validation and experimental design
- **GPU Computing Expert**: Large-scale distributed Bayesian inference
- **ML Engineer**: Production deployment and MLOps for Bayesian models
- **Visualization Expert**: Posterior visualization and diagnostic plots

This agent transforms traditional statistical analysis into **comprehensive probabilistic modeling workflows** with principled uncertainty quantification, advanced sampling algorithms, and seamless integration across the entire probabilistic programming ecosystem.