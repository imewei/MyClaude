---
name: ml-and-data-science
description: Meta-orchestrator for machine learning and data science. Routes to classical ML, data analysis, wrangling, statistics, visualization, curve fitting, and experiment tracking skills. Use when training classical ML models, analyzing experimental data, wrangling datasets, running statistical tests, creating scientific visualizations, or storing and converting large N-dimensional array data (Zarr, HDF5 or NetCDF to cloud-native Zarr, chunking and sharding). For tabular classical-model training use machine-learning; for production ML deployment and serving, see the ml-deployment hub.
---

> **Loading a routing target.** `dev-suite:*`, `research-suite:*`, and `science-suite:*` targets below
> are slash-only (`disable-model-invocation: true`) — the Skill tool will not fire them. Load one by
> reading its file with the Read tool at `${CLAUDE_PLUGIN_ROOT}/skills/<name>/SKILL.md` — Claude Code
> substitutes that variable with this plugin's install directory. A target in a sibling suite cannot be
> read by path (installs live in separate directories); reach it by invoking that suite's hub. Targets from other plugins (`superpowers:*`,
> `ecc:*`, …) are unaffected and invoke normally.

# ML and Data Science

Orchestrator for machine learning and data science workflows. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`python-pro`**: Lead for Zarr and HDF5/NetCDF conversion jobs (`zarr-cloud-arrays`), including batches of large XPCS files.
  - *Location*: `${CLAUDE_PLUGIN_ROOT}/agents/python-pro.md`
- **`ml-expert`**: Specialist for classical ML, data pipelines, and production ML systems.
  - *Location*: `${CLAUDE_PLUGIN_ROOT}/agents/ml-expert.md`
  - *Capabilities*: Scikit-learn, feature engineering, statistical modeling, data wrangling, visualization, and experiment tracking.

## Core Skills

### [Machine Learning](../machine-learning/SKILL.md)
Classical ML: tree models, SVMs, ensembles, feature engineering, and model selection.

### [Data Analysis](../data-analysis/SKILL.md)
Exploratory data analysis: pandas, descriptive statistics, correlation, and hypothesis testing.

### [Data Wrangling Communication](../data-wrangling-communication/SKILL.md)
Data cleaning, transformation, merging, and communicating findings to stakeholders.

### [Statistical Analysis Fundamentals](../statistical-analysis-fundamentals/SKILL.md)
Inferential statistics: distributions, hypothesis tests, confidence intervals, and power analysis.

### [Scientific Visualization](../scientific-visualization/SKILL.md)
Publication-quality plots: matplotlib, seaborn, plotly, and domain-specific visualization patterns.

### [NLSQ Core Mastery](../nlsq-core-mastery/SKILL.md)
Non-linear least squares: curve fitting, residual analysis, parameter uncertainty, and JAX-accelerated NLSQ.

### [Experiment Tracking](../experiment-tracking/SKILL.md)
MLflow, Weights & Biases, DVC: logging metrics, artifacts, and reproducible experiment management.

### [Zarr Cloud Arrays](../zarr-cloud-arrays/SKILL.md)
Chunked, compressed N-D arrays on object storage: chunk design, safe parallel writes, NetCDF/HDF5 migration, Python and Julia.

## Routing Decision Tree

```
What is the ML / data science task?
|
+-- Train classical ML models?
|   --> science-suite:machine-learning
|
+-- Deploy / monitor / serve ML in production?
|   --> (see science-suite:ml-deployment hub instead)
|
+-- Explore and understand data?
|   --> science-suite:data-analysis
|
+-- Clean, transform, or merge datasets?
|   --> science-suite:data-wrangling-communication
|
+-- Statistical tests / inference?
|   --> science-suite:statistical-analysis-fundamentals
|
+-- Create plots / figures?
|   --> science-suite:scientific-visualization
|
+-- Fit curves / non-linear models to data?
|   --> science-suite:nlsq-core-mastery
|
+-- Large N-D arrays, data cubes, Zarr, or NetCDF/HDF5-to-cloud conversion?
|   --> science-suite:zarr-cloud-arrays
|
+-- Track experiments and artifacts?
|   --> science-suite:experiment-tracking
|
+-- None of the above / concern is ambiguous?
    --> Delegate to ml-expert for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Checklist

- [ ] Use routing tree to select the most specific sub-skill
- [ ] Perform EDA (`data-analysis`) before training any model
- [ ] Validate train/val/test splits are non-overlapping and stratified if needed
- [ ] Check for data leakage before reporting evaluation metrics
- [ ] Use cross-validation for small datasets; hold-out for large datasets
- [ ] Log all hyperparameters and random seeds via experiment tracking
- [ ] Report confidence intervals, not just point estimates, for fitted parameters
- [ ] Version datasets alongside model artifacts in production pipelines
