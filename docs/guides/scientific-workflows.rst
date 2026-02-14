Scientific Workflows
====================

Patterns for using the **science-suite** agents and skills in research computing pipelines.

Bayesian Inference Pipeline
---------------------------

A typical Bayesian parameter estimation workflow combines ``@jax-pro`` for
model implementation with ``@research-expert`` for methodology.

1. Define the forward model with JAX (skill: ``jax-core-programming``).
2. Build the probabilistic model in NumPyro (skill: ``numpyro-core-mastery``).
3. Run NUTS sampling and diagnose convergence (skill: ``mcmc-diagnostics``).
4. Visualize posteriors with ArviZ (skill: ``scientific-visualization``).

.. code-block:: python

   import jax
   import numpyro
   from numpyro.infer import MCMC, NUTS

   # 1. Forward model (JIT-compiled)
   @jax.jit
   def model_predict(params, x):
       return params["a"] * jax.numpy.exp(-params["k"] * x)

   # 2. NumPyro model
   def bayesian_model(x, y_obs=None):
       a = numpyro.sample("a", numpyro.distributions.LogNormal(0, 1))
       k = numpyro.sample("k", numpyro.distributions.HalfNormal(1))
       sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(0.1))
       y_pred = model_predict({"a": a, "k": k}, x)
       numpyro.sample("obs", numpyro.distributions.Normal(y_pred, sigma), obs=y_obs)

   # 3. Run MCMC
   kernel = NUTS(bayesian_model)
   mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=4)
   mcmc.run(jax.random.PRNGKey(42), x_data, y_obs=y_data)

**Agent team:** Use :doc:`Team 13 (bayesian-pipeline) </agent-teams-guide>` for
multi-agent Bayesian workflows.

Molecular Dynamics Campaign
---------------------------

For MD simulation campaigns, combine ``@simulation-expert`` with ``@jax-pro``
for differentiable physics.

1. Set up force fields and initial configurations (skill: ``md-simulation-setup``).
2. Run production simulations (skill: ``advanced-simulations``).
3. Analyze trajectories: RDF, MSD, viscosity (skill: ``trajectory-analysis``).
4. Compute correlation functions (skill: ``correlation-computational-methods``).

**Agent team:** Use :doc:`Team 14 (md-campaign) </agent-teams-guide>` for
coordinated MD workflows.

Research Paper Implementation
-----------------------------

Reproducing results from published papers requires systematic methodology.

1. Extract architecture and equations (skill: ``research-paper-implementation``).
2. Implement in JAX or Julia (skills: ``jax-core-programming``, ``core-julia-patterns``).
3. Validate against reported benchmarks (skill: ``research-quality-assessment``).
4. Create publication-quality figures (skill: ``scientific-visualization``).

**Agent team:** Use :doc:`Team 16 (paper-implement) </agent-teams-guide>` for
coordinated reproduction workflows.

Related
-------

- :doc:`/suites/science-suite` — Full science-suite reference
- :doc:`/suites/agent-core` — Orchestration and reasoning agents
