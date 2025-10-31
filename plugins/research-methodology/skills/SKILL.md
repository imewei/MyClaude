---
name: research-quality-assessment
description: Comprehensive evaluation framework for scientific research quality across 6 critical dimensions (methodology soundness, experimental design quality, data quality & sufficiency, statistical analysis rigor, result validity & significance, publication readiness) with systematic assessment workflows, scoring rubrics (0-10 scale with weighted dimensions), and detailed reference guides. Use when assessing research projects before submission to journals (Nature, Science, Cell, PLOS, etc.), evaluating grant proposal methodologies (NSF, NIH, DOE proposals), reviewing experimental designs for completeness and statistical power (sample size calculations, power analysis ≥0.80, control adequacy, ablation studies), analyzing data quality and sufficiency (completeness, accuracy, bias detection, preprocessing validation, sample size adequacy), checking statistical validity and rigor (appropriate test selection, multiple testing correction, effect size reporting, confidence intervals, sensitivity analysis), preparing manuscripts for publication readiness (scientific quality, completeness, writing assessment, figure quality, reproducibility package with code/data/documentation), conducting pre-submission peer reviews, performing research audits for quality assurance, evaluating hypothesis clarity and testability, verifying reproducibility measures (code availability, environment specification, data sharing plans), assessing result novelty and contribution to field, identifying critical research quality issues (underpowered studies, missing controls, inappropriate statistics, absent multiple testing corrections), reviewing research documentation (research proposals, manuscripts, methods sections, supplementary materials, analysis code, Jupyter notebooks), preparing assessment reports using structured templates with executive summaries and actionable recommendations, and consulting methodology evaluation frameworks, experimental design checklists, statistical rigor guides, and publication readiness criteria from references/ directory (methodology_evaluation.md, experimental_design_checklist.md, statistical_rigor_guide.md, publication_readiness.md).
---

# Research Quality Assessment

## When to use this skill

- Assessing research projects before manuscript submission to journals (Nature, Science, Cell, PLOS, eLife, etc.) to ensure scientific rigor and publication readiness
- Evaluating grant proposal methodologies for NSF, NIH, DOE, or other funding agencies to verify experimental design quality and feasibility
- Reviewing experimental designs for completeness and statistical power, including sample size calculations, power analysis (target ≥0.80), control adequacy (baselines, negative/positive controls), and ablation study coverage
- Analyzing data quality and sufficiency before publication, checking completeness, accuracy, consistency, sample size adequacy, preprocessing appropriateness, and bias detection (selection bias, measurement bias, sampling bias)
- Checking statistical validity and rigor including appropriate statistical test selection, multiple testing correction (Bonferroni, FDR, Holm), effect size reporting (Cohen's d, odds ratio, R²), confidence interval provision, and sensitivity/robustness analysis
- Preparing manuscripts for publication by evaluating scientific quality, manuscript completeness (all sections present), writing clarity, figure quality, and reproducibility package (code, data, documentation)
- Conducting pre-submission peer reviews to identify critical issues before journal submission, reducing rejection risk
- Performing research audits for quality assurance in academic labs, research institutions, or industry R&D departments
- Evaluating hypothesis clarity and testability to ensure research questions are well-defined and answerable
- Verifying reproducibility measures including code availability, environment specification (dependencies, versions), data sharing plans, and detailed methodology documentation
- Assessing result novelty and contribution to field by comparing to state-of-the-art and evaluating practical significance beyond statistical significance
- Identifying critical research quality issues such as underpowered studies, missing control conditions, inappropriate statistical tests, absent multiple testing corrections, or incomplete reproducibility packages
- Reviewing research documentation files including research proposals (*.docx, *.pdf), manuscript drafts, methods sections, supplementary materials, analysis code (*.py, *.R, *.m, *.jl), Jupyter notebooks (*.ipynb), and data files
- Preparing comprehensive assessment reports using structured templates (assets/research_assessment_template.md) with executive summaries, dimension-specific findings, scoring rubrics, and prioritized actionable recommendations
- Consulting methodology evaluation frameworks (references/methodology_evaluation.md) for systematic assessment of hypothesis clarity, method appropriateness, control adequacy, reproducibility measures, and statistical validity
- Using experimental design checklists (references/experimental_design_checklist.md) for sample size calculations, power analysis, parameter space coverage, ablation study completeness, baseline comparisons, and replication strategies
- Applying statistical rigor guides (references/statistical_rigor_guide.md) for test selection validation, multiple testing correction procedures, effect size reporting standards, uncertainty quantification, and sensitivity analysis protocols
- Evaluating publication readiness criteria (references/publication_readiness.md) covering scientific quality, manuscript completeness, writing assessment, figure standards, reproducibility package requirements, and venue-specific guidance
- Performing rapid quality checks using Quick Assessment Mode with critical factors (must pass: clear hypothesis, appropriate methods, adequate sample size, proper controls, appropriate statistics, multiple testing correction, significant results, stated limitations) and quality indicators (should pass: documented code, data sharing plan, ablation studies, effect sizes, confidence intervals, sensitivity analysis, publication-quality figures, clear writing)
- Scoring research quality across 6 weighted dimensions (Methodology 20%, Experimental Design 20%, Data Quality 15%, Statistical Rigor 20%, Result Validity 15%, Publication Readiness 10%) to generate overall quality scores (0-10 scale) with publication target recommendations
- Identifying and prioritizing issues by severity: critical issues (🔴 must fix: underpowered study, missing controls, inappropriate tests), important issues (⚠️ should fix: limited exploration, missing ablations, unreported effect sizes), and nice-to-have improvements (📋 consider: additional baselines, extended discussion)

## Overview

Systematically evaluate scientific research quality across multiple dimensions: methodology soundness, experimental design appropriateness, data quality and sufficiency, statistical rigor, result validity, and publication readiness. This skill provides comprehensive frameworks, checklists, and templates for research evaluation and improvement.

## Core Assessment Dimensions

### 1. Methodology Soundness

Evaluate research methodology for scientific rigor, reproducibility, and appropriateness.

**Key Criteria:**
- Hypothesis clarity and testability
- Method appropriateness for research question
- Control adequacy (baselines, negative/positive controls)
- Reproducibility (code, data, environment specification)
- Statistical validity (assumptions, power, corrections)

For detailed methodology evaluation framework, read `references/methodology_evaluation.md`.

###  2. Experimental Design Quality

Assess experimental design for completeness, coverage, and statistical power.

**Key Aspects:**
- Sample size adequacy and statistical power
- Parameter space coverage and systematic exploration
- Ablation studies completeness
- Baseline comparisons and control conditions
- Replication strategy

For complete experimental design checklist, read `references/experimental_design_checklist.md`.

### 3. Data Quality & Sufficiency

Evaluate data quality, quantity, preprocessing, and bias assessment.

**Key Factors:**
- Data completeness, accuracy, consistency
- Sample size sufficiency for statistical power
- Preprocessing appropriateness and validation
- Bias detection (selection, measurement, sampling)
- Data availability and sharing plans

### 4. Statistical Analysis Rigor

Assess statistical methods, visualization, error analysis, and sensitivity.

**Key Elements:**
- Appropriate statistical tests and assumptions
- Multiple testing correction
- Effect size reporting
- Uncertainty quantification (confidence intervals, error bars)
- Sensitivity and robustness analysis

For statistical analysis guidelines, read `references/statistical_rigor_guide.md`.

### 5. Result Validity & Significance

Evaluate result validity, novelty, and impact.

**Key Considerations:**
- Statistical significance (p-values, confidence intervals)
- Practical significance (effect magnitude, real-world impact)
- Novelty and contribution to field
- Generalizability and limitations
- Comparison to state-of-the-art

### 6. Publication Readiness

Assess manuscript completeness, writing quality, and reproducibility package.

**Key Components:**
- Scientific quality (rigor, novelty, significance)
- Manuscript completeness (all sections present and complete)
- Writing quality (clarity, organization, grammar)
- Figure quality and completeness
- Reproducibility package (code, data, documentation)

For publication readiness checklist, read `references/publication_readiness.md`.

## Assessment Workflow

### Step 1: Define Scope

Determine assessment focus:
- Full project evaluation
- Specific methodology review
- Experimental design audit
- Data quality check
- Pre-submission manuscript review

### Step 2: Gather Materials

Collect relevant documents:
- Research proposal or manuscript
- Methods documentation
- Data files and metadata
- Analysis code and notebooks
- Figures and supplementary materials

### Step 3: Methodology Assessment

Use methodology evaluation framework (`references/methodology_evaluation.md`):
1. Evaluate hypothesis clarity
2. Assess method appropriateness
3. Check control adequacy
4. Review reproducibility measures
5. Verify statistical validity
6. Generate methodology score and recommendations

### Step 4: Experimental Design Review

Use experimental design checklist (`references/experimental_design_checklist.md`):
1. Perform statistical power analysis
2. Assess parameter space coverage
3. Review ablation study completeness
4. Check baseline comparisons
5. Evaluate replication strategy
6. Identify design gaps and recommendations

### Step 5: Data Quality Evaluation

Assess data across dimensions:
1. Check completeness, accuracy, consistency
2. Verify sample size sufficiency
3. Evaluate preprocessing appropriateness
4. Detect potential biases
5. Review data sharing plans
6. Generate data quality score

### Step 6: Statistical Rigor Review

Use statistical rigor guide (`references/statistical_rigor_guide.md`):
1. Verify appropriate statistical tests
2. Check multiple testing corrections
3. Confirm effect size reporting
4. Assess uncertainty quantification
5. Review sensitivity analysis
6. Evaluate overall statistical rigor

### Step 7: Result Validation

Evaluate result validity and significance:
1. Assess statistical significance
2. Evaluate practical significance
3. Assess novelty and contribution
4. Check generalizability
5. Review stated limitations
6. Identify unstated limitations

### Step 8: Publication Readiness Check

Use publication readiness checklist (`references/publication_readiness.md`):
1. Evaluate scientific quality
2. Check manuscript completeness
3. Assess writing quality
4. Review figure quality
5. Verify reproducibility package
6. Generate overall readiness score

### Step 9: Generate Assessment Report

Use research assessment template (`assets/research_assessment_template.md`):
1. Complete executive summary with scores
2. Document findings for each dimension
3. Identify critical issues and recommendations
4. Prioritize action items
5. Provide publication strategy if applicable

## Quick Assessment Mode

For rapid evaluation, use the **Quick Assessment Checklist**:

### Critical Factors (Must Pass)
- [ ] Clear, testable hypothesis
- [ ] Appropriate methods for question
- [ ] Adequate sample size (statistical power ≥0.80)
- [ ] Proper controls (baselines, negative/positive)
- [ ] Appropriate statistical tests
- [ ] Multiple testing correction applied
- [ ] Results statistically significant
- [ ] Limitations explicitly stated

### Quality Indicators (Should Pass)
- [ ] Code and environment documented
- [ ] Data sharing plan present
- [ ] Ablation studies included
- [ ] Effect sizes reported
- [ ] Confidence intervals provided
- [ ] Sensitivity analysis performed
- [ ] Figures publication-quality
- [ ] Writing clear and well-organized

**Pass Criteria**: All critical factors + ≥6/8 quality indicators

## Assessment Scoring System

### Overall Research Quality Score: X/10

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Methodology | 20% | X/10 | X.XX |
| Experimental Design | 20% | X/10 | X.XX |
| Data Quality | 15% | X/10 | X.XX |
| Statistical Rigor | 20% | X/10 | X.XX |
| Result Validity | 15% | X/10 | X.XX |
| Publication Readiness | 10% | X/10 | X.XX |
| **Overall** | **100%** | - | **X.XX/10** |

### Score Interpretation

| Score | Quality Level | Publication Target | Action Required |
|-------|---------------|-------------------|-----------------|
| 9-10 | Excellent | Top-tier journals | Minor polishing |
| 7-8 | Very Good | Strong journals | Some improvements |
| 5-6 | Good | Solid journals | Moderate revisions |
| 3-4 | Fair | Regional journals | Major revisions |
| 1-2 | Poor | Not publishable | Fundamental changes |

## Resources

### references/

**methodology_evaluation.md**
Comprehensive framework for evaluating research methodology including hypothesis assessment, method appropriateness, control adequacy, reproducibility measures, and statistical validity criteria.

**experimental_design_checklist.md**
Detailed checklist for experimental design evaluation covering sample size calculations, power analysis, parameter space coverage, ablation studies, baseline comparisons, and replication strategies.

**statistical_rigor_guide.md**
Complete guide to statistical rigor assessment including test selection, multiple testing correction, effect size reporting, uncertainty quantification, and sensitivity analysis procedures.

**publication_readiness.md**
Publication readiness evaluation framework covering scientific quality, manuscript completeness, writing assessment, figure quality, reproducibility package requirements, and venue selection guidance.

### assets/

**research_assessment_template.md**
Professional markdown template for comprehensive research quality assessment reports. Includes structured sections for all assessment dimensions, scoring rubrics, and actionable recommendations.

## Example Usage

**Example 1: Pre-submission manuscript review**
```
User: "Review my manuscript for submission to Nature"

Process:
1. Assess methodology soundness (rigor, reproducibility)
2. Evaluate experimental design (sample size, controls, ablations)
3. Check data quality and statistical rigor
4. Validate results (significance, novelty, impact)
5. Review publication readiness (completeness, writing, figures)
6. Generate assessment report with recommendations
7. Provide venue-specific guidance for Nature standards
```

**Example 2: Research proposal evaluation**
```
User: "Evaluate my NSF grant proposal methodology"

Process:
1. Review hypothesis clarity and significance
2. Assess proposed methodology appropriateness
3. Evaluate experimental design and power analysis
4. Check for adequate controls and baselines
5. Review feasibility and timeline
6. Provide improvement recommendations
```

**Example 3: Data quality audit**
```
User: "Assess whether my data is sufficient for publication"

Process:
1. Evaluate sample size and statistical power
2. Check data quality (completeness, accuracy, consistency)
3. Identify potential biases
4. Assess preprocessing appropriateness
5. Review data sharing plans
6. Generate data quality report with recommendations
```

## Best Practices

1. **Be Systematic**: Use checklists to ensure comprehensive coverage
2. **Be Objective**: Base assessments on evidence and established criteria
3. **Be Constructive**: Focus on actionable improvements, not just critique
4. **Quantify When Possible**: Use scores and metrics for clarity
5. **Prioritize Issues**: Flag critical issues (🔴), important (⚠️), and nice-to-have (📋)
6. **Provide Context**: Consider field norms and journal standards
7. **Verify Assumptions**: Check that statistical test assumptions are met
8. **Think Reproducibility**: Assess whether others could replicate the work
9. **Consider Impact**: Evaluate practical significance, not just statistical
10. **Be Thorough**: Don't skip dimensions - weak areas often hide in unchecked sections

## Common Research Quality Issues

### Critical Issues (🔴 Must Fix)
- Insufficient sample size (underpowered study)
- Missing control conditions
- Inappropriate statistical tests
- No multiple testing correction
- Results not statistically significant
- Reproducibility package incomplete/missing
- Major limitations not acknowledged

### Important Issues (⚠️ Should Fix)
- Limited parameter space exploration
- Missing ablation studies
- Effect sizes not reported
- Confidence intervals missing
- Sensitivity analysis absent
- Figures not publication-quality
- Writing clarity issues

### Nice-to-Have (📋 Consider)
- Additional baseline comparisons
- Extended discussion of implications
- Supplementary visualizations
- Code optimization and documentation
- Interactive data exploration tools
