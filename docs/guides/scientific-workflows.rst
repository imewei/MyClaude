Scientific Workflows
====================

Patterns for using the **science-suite** agents and :term:`hub skills <Hub Skill>` in research computing pipelines.

.. note::

   Since v3.1.0, skills use a two-tier :term:`Hub Skill` architecture. The hub
   skills listed below route to specialized sub-skills via their
   :term:`Routing Decision Tree`. You invoke the hub; it dispatches to the
   right sub-skill automatically.

Bayesian Inference Pipeline
---------------------------

A typical Bayesian parameter estimation workflow combines ``@jax-pro`` for
model implementation with ``@research-expert`` for methodology.

1. Define the forward model with JAX (hub: ``jax-computing`` → sub: ``jax-core-programming``).
2. Build the probabilistic model in NumPyro (hub: ``bayesian-inference`` → sub: ``numpyro-core-mastery``).
3. Run NUTS sampling and diagnose convergence (hub: ``bayesian-inference`` → sub: ``mcmc-diagnostics``).
4. Visualize posteriors with ArviZ (hub: ``research-and-domains`` → sub: ``scientific-visualization``).

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

1. Set up force fields and initial configurations (hub: ``simulation-and-hpc`` → sub: ``md-simulation-setup``).
2. Run production simulations (hub: ``simulation-and-hpc`` → sub: ``advanced-simulations``).
3. Analyze trajectories: RDF, MSD, viscosity (hub: ``simulation-and-hpc`` → sub: ``trajectory-analysis``).
4. Compute correlation functions (hub: ``correlation-analysis`` → sub: ``correlation-computational-methods``).

**Agent team:** Use :doc:`Team 14 (md-campaign) </agent-teams-guide>` for
coordinated MD workflows.

Research Paper Implementation
-----------------------------

Reproducing results from published papers requires systematic methodology.

1. Extract architecture and equations (hub: ``research-and-domains`` → sub: ``research-paper-implementation``).
2. Implement in JAX or Julia (hubs: ``jax-computing``, ``julia-language``).
3. Validate against reported benchmarks (hub: ``research-and-domains`` → sub: ``research-quality-assessment``).
4. Create publication-quality figures (hub: ``research-and-domains`` → sub: ``scientific-visualization``).

**Agent team:** Use :doc:`Team 16 (paper-implement) </agent-teams-guide>` for
coordinated reproduction workflows.

Related
-------

- :doc:`/suites/science-suite` — Full science-suite reference (14 hubs → 116 sub-skills)
- :doc:`/suites/agent-core` — Orchestration and reasoning agents
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
