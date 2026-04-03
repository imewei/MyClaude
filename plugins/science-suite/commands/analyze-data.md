---
name: analyze-data
description: Comprehensive data analysis workflow with statistical tests, visualization, and reproducible reporting
argument-hint: "[data file or description]"
---

# Analyze Data

Comprehensive data analysis workflow from raw data to publishable results.

## Workflow

1. **Load & Inspect**: Read data, check shape/dtypes, identify missing values and outliers.
2. **Clean**: Handle missing data (imputation vs removal), fix dtypes, validate ranges and monotonicity.
3. **Explore**: Compute summary statistics, distributions, correlations. Generate exploratory plots.
4. **Test**: Apply appropriate statistical tests based on data type and distribution:
   - Continuous: t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis
   - Categorical: Chi-squared, Fisher's exact
   - Correlation: Pearson, Spearman, Kendall
5. **Visualize**: Create publication-quality figures with proper labels, error bars, and legends.
6. **Report**: Summarize findings with effect sizes, confidence intervals, and reproducibility metadata.

## Checklist

- [ ] Data integrity verified (no silent NaN, no truncation)
- [ ] Appropriate statistical test selected for data type
- [ ] Multiple comparison correction applied if needed (Bonferroni, FDR)
- [ ] Effect sizes reported alongside p-values
- [ ] Figures have proper axis labels, units, and legends
- [ ] Analysis is reproducible (seeds, versions logged)
