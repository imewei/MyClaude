# Research Methodology

Research intelligence, methodology design, literature analysis, and evidence-based insights for scientific investigation with systematic research workflows, multi-source synthesis, and rigorous quality assurance.

**Version:** 2.1.0 | **Category:** research | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/research-methodology.html)


## What's New in v2.1.0

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## Agents (1)

### research-intelligence

**Status:** active | **Model:** inherit | **Performance:** comprehensive-research

Research intelligence expert specializing in research methodology and information discovery with systematic 8-step development process, 8 quality assurance principles, and comprehensive examples demonstrating PRISMA-compliant systematic literature reviews with meta-analysis (127 papers analyzed, quantitative performance comparisons with 95% confidence intervals, research gap identification), multi-source competitive intelligence synthesis (3,000+ sources including academic papers, industry reports, patents, financial data for market entry analysis), and trend forecasting with time-series analysis for strategic decision-making.

**Key Capabilities:**
- **Systematic Literature Reviews**: PRISMA-compliant methodology with multi-database search (PubMed, IEEE, ACM, arXiv, Google Scholar), rigorous screening processes, quantitative meta-analysis with 95% confidence intervals
- **Meta-Analysis**: Statistical synthesis of findings across multiple studies with effect size calculations, heterogeneity assessment, publication bias detection
- **Competitive Intelligence**: Multi-source synthesis combining academic papers, industry reports (Gartner, IDC, McKinsey), patent landscape analysis (USPTO, EPO, WIPO), financial data (10-K filings, earnings calls), and market trend forecasting
- **Trend Analysis & Forecasting**: Time-series analysis with ARIMA, exponential smoothing, machine learning forecasting techniques, confidence interval quantification (±20% accuracy), scenario planning
- **Evidence-Based Insights**: Rigorous cross-validation across minimum 3 independent sources, bias mitigation (publication bias, selection bias, confirmation bias), statistical significance verification (p < 0.05), transparent uncertainty quantification
- **Research Quality Assurance**: Source credibility assessment (>95% accuracy requirement), reproducibility documentation (transparent search strategies), methodological transparency, research ethics compliance

**Research Methodologies:**
- Systematic literature reviews and PRISMA-compliant screening
- Meta-analysis with statistical synthesis
- Trend analysis and predictive modeling (ARIMA, ML forecasting)
- Competitive intelligence and patent landscape analysis
- Multi-source synthesis (academic + industry + patents + financial data)
- Hypothesis generation and research gap identification

**Deliverables:**
- Executive summaries (1-2 pages) for strategic decision-making
- Technical reports (15-45 pages) with comprehensive methodology documentation
- Meta-analysis with quantitative performance comparisons and 95% confidence intervals
- Patent landscape visualizations showing technology clusters and vendor positioning
- Research gap analysis for grant proposals and future work
- Market opportunity matrices and technology roadmaps

## Skills (1)

### research-quality-assessment

**Status:** active

Comprehensive evaluation framework for scientific research quality across 6 critical dimensions (methodology soundness, experimental design quality, data quality & sufficiency, statistical analysis rigor, result validity & significance, publication readiness) with systematic assessment workflows, scoring rubrics (0-10 scale with weighted dimensions), and detailed reference guides.

**When to use:** Assessing research projects before manuscript submission to journals (Nature, Science, Cell, PLOS, eLife), evaluating grant proposal methodologies (NSF, NIH, DOE), reviewing experimental designs with statistical power analysis (≥0.80), analyzing data quality and bias detection, checking statistical validity (multiple testing correction, effect size reporting, confidence intervals, sensitivity analysis), preparing manuscripts for publication readiness, conducting pre-submission peer reviews, performing research audits, and generating comprehensive assessment reports.

**Key Assessment Dimensions:**
1. **Methodology Soundness** (20%): Hypothesis clarity, method appropriateness, control adequacy, reproducibility measures, statistical validity
2. **Experimental Design Quality** (20%): Sample size adequacy, statistical power (≥0.80), parameter space coverage, ablation studies, baseline comparisons, replication strategy
3. **Data Quality & Sufficiency** (15%): Completeness, accuracy, consistency, sample size sufficiency, preprocessing appropriateness, bias detection
4. **Statistical Analysis Rigor** (20%): Appropriate test selection, multiple testing correction (Bonferroni, FDR, Holm), effect size reporting (Cohen's d, odds ratio, R²), confidence intervals, sensitivity analysis
5. **Result Validity & Significance** (15%): Statistical significance (p-values, confidence intervals), practical significance (effect magnitude), novelty and contribution, generalizability, limitations
6. **Publication Readiness** (10%): Scientific quality, manuscript completeness, writing clarity, figure quality, reproducibility package (code, data, documentation)

**Assessment Workflows:**
- Quick Assessment Mode: 8 critical factors (must pass) + 8 quality indicators (should pass) for rapid evaluation
- Comprehensive Assessment: 8-step systematic process (define scope, gather materials, methodology assessment, experimental design review, data quality evaluation, statistical rigor review, result validation, publication readiness check, generate assessment report)
- Scoring System: Overall quality score (0-10 scale) with publication target recommendations (9-10: top-tier journals, 7-8: strong journals, 5-6: solid journals)

**Reference Guides:**
- methodology_evaluation.md: Framework for evaluating research methodology
- experimental_design_checklist.md: Checklist for experimental design evaluation
- statistical_rigor_guide.md: Guide to statistical rigor assessment
- publication_readiness.md: Publication readiness evaluation framework

**Assessment Report Templates:**
- research_assessment_template.md: Professional template for comprehensive quality assessment reports with executive summaries, dimension-specific findings, scoring rubrics, and actionable recommendations

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `research-methodology` plugin
3. Activate an agent (e.g., `@research-intelligence`)

## Integration

See the full documentation for integration patterns and compatible plugins.

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/research-methodology.html)

To build documentation locally:

```bash
cd docs/
make -j4 html
```
