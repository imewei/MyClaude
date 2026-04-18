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
model implementation with ``@research-expert`` (research-suite) for methodology.

1. Define the forward model with JAX (hub: ``jax-computing`` → sub: ``jax-core-programming``).
2. Build the probabilistic model in NumPyro (hub: ``bayesian-inference`` → sub: ``numpyro-core-mastery``).
3. Run NUTS sampling and diagnose convergence (hub: ``bayesian-inference`` → sub: ``mcmc-diagnostics``).
4. Visualize posteriors with ArviZ (hub: ``ml-and-data-science`` → sub: ``scientific-visualization``).

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

Reproducing results from published papers requires systematic methodology. In v3.4.0 the research-methodology skills moved from ``science-suite`` to the new ``research-suite``.

1. Extract architecture and equations (``research-suite`` skill: ``research-paper-implementation``).
2. Implement in JAX or Julia (science-suite hubs: ``jax-computing``, ``julia-language``).
3. Validate against reported benchmarks (``research-suite`` skill: ``research-quality-assessment``).
4. Create publication-quality figures (science-suite hub: ``ml-and-data-science`` → sub: ``scientific-visualization``).

**Agent team:** Use :doc:`Team 16 (paper-implement) </agent-teams-guide>` for
coordinated reproduction workflows.

Peer Review of a Manuscript
---------------------------

Producing a rigorous, journal-ready peer review is a distinct workflow from reproducing papers or assessing quality internally.

1. Trigger ``scientific-review`` skill in ``research-suite`` with the paper (PDF/DOCX/text) and optionally the target journal name.
2. The skill performs six-competency analysis (domain, methodology, critical thinking, communication, integrity, efficiency) and produces a ``.docx`` referee report with Confidential Comments to Editor.
3. For internal scoring without the ``.docx`` deliverable, use ``research-quality-assessment`` instead.

Research-Spark: Idea to Fundable Plan
--------------------------------------

Refining a rough research idea into a scoped, testable, fundable program. Eight-stage artifact-gated pipeline in ``research-suite``:

1. Stage 1 — ``spark-articulator``: rough idea → 3-to-5-sentence articulation.
2. Stage 2 — ``landscape-scanner``: three-layer literature scan + Reviewer 2 pass.
3. Stage 3 — ``falsifiable-claim``: claim + Heilmeier catechism + kill criterion.
4. Stages 4-5 — ``theory-scaffold``: stepwise derivation → LaTeX formalism (delegates to ``nonlinear-dynamics-expert`` or ``statistical-physicist`` in science-suite when applicable).
5. Stage 6 — ``numerical-prototype``: JAX solver + three validation passes (delegates to ``jax-pro`` / ``julia-pro`` / ``simulation-expert`` in science-suite).
6. Stage 7 — ``experiment-designer``: DoE + instrument capability map (3× margin rule).
7. Stage 8 — ``premortem-critique``: failure narratives + simulated reviewers.

The ``research-spark-orchestrator`` agent drives the pipeline, owns ``_state.yaml``, and fans out to parallel sub-agents at natural stage boundaries.

Related
-------

- :doc:`/suites/research-suite` — Full research-suite reference (2 agents, 3 workflow tracks)
- :doc:`/suites/science-suite` — Full science-suite reference (14 hubs → 112 sub-skills)
- :doc:`/suites/agent-core` — Orchestration and reasoning agents
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
