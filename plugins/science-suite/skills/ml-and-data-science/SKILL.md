---
name: ml-and-data-science
description: Meta-orchestrator for machine learning and data science. Routes to classical ML, data analysis, wrangling, statistics, visualization, curve fitting, and experiment tracking skills. Use when training classical ML models, analyzing experimental data, wrangling datasets, running statistical tests, or creating scientific visualizations. For production ML deployment and serving, see the ml-deployment hub.
---

# ML and Data Science

Orchestrator for machine learning and data science workflows. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`ml-expert`**: Specialist for classical ML, data pipelines, and production ML systems.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
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

## Routing Decision Tree

```
What is the ML / data science task?
|
+-- Train classical ML models?
|   --> machine-learning
|
+-- Deploy / monitor / serve ML in production?
|   --> (see ml-deployment hub instead)
|
+-- Explore and understand data?
|   --> data-analysis
|
+-- Clean, transform, or merge datasets?
|   --> data-wrangling-communication
|
+-- Statistical tests / inference?
|   --> statistical-analysis-fundamentals
|
+-- Create plots / figures?
|   --> scientific-visualization
|
+-- Fit curves / non-linear models to data?
|   --> nlsq-core-mastery
|
+-- Track experiments and artifacts?
    --> experiment-tracking
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
