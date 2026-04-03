---
name: run-experiment
description: Design and execute a reproducible scientific experiment with proper controls, logging, and analysis
argument-hint: "[experiment description]"
---

# Run Experiment

Design and execute a reproducible scientific experiment.

## Workflow

1. **Hypothesis**: State the hypothesis clearly. Define null and alternative hypotheses.
2. **Controls**: Identify independent, dependent, and control variables. Define baseline.
3. **Setup**: Configure environment with explicit random seeds, version-locked dependencies, and structured logging.
4. **Execution**: Run trials with proper error handling and intermediate checkpointing.
5. **Analysis**: Apply appropriate statistical tests (t-test, ANOVA, bootstrap). Compute effect sizes and confidence intervals.
6. **Report**: Generate figures, tables, and a summary with reproducibility metadata (seeds, versions, hardware).

## Checklist

- [ ] Hypothesis stated with measurable outcome
- [ ] Random seed set and logged
- [ ] Dependencies version-locked (`uv.lock` or `Manifest.toml`)
- [ ] Baseline/control condition defined
- [ ] Sample size justified (power analysis if applicable)
- [ ] Results saved with timestamps and metadata
- [ ] Statistical significance assessed correctly
- [ ] Figures are publication-quality
