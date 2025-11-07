# Research Reflection Engine

**Version**: 1.0.3
**Purpose**: Scientific methodology validation and publication readiness assessment

---

## Overview

The ResearchReflectionEngine validates research projects across six critical dimensions: methodology soundness, reproducibility, experimental design, data quality, analysis rigor, and publication readiness.

---

## Core Assessment Framework

### 1. Methodology Soundness (Score: /10)

**What We Assess**:
- Hypothesis clarity and testability
- Appropriate method selection for research questions
- Control condition adequacy
- Parameter space coverage
- Statistical assumptions validity

**Scoring Rubric**:
- **9-10**: Exceptional methodology, publishable as-is
- **7-8**: Strong methodology, minor refinements
- **5-6**: Adequate methodology, notable improvements needed
- **3-4**: Weak methodology, major revisions required
- **0-2**: Flawed methodology, requires redesign

**Example Assessment**:
```yaml
methodology_assessment:
  score: 8.5
  strengths:
    - Clear hypothesis formulation
    - Appropriate statistical methods
    - Adequate control conditions
  weaknesses:
    - Some assumptions not explicitly validated
    - Limited exploration of alternative approaches
  recommendations:
    - Add sensitivity analysis for key parameters
    - Validate normality assumption explicitly
    - Compare with 2 alternative methodologies
```

---

### 2. Reproducibility (Score: /10)

**Critical Components**:
- **Code Availability**: Public repository with documentation
- **Environment Specification**: Dependencies with versions
- **Data Availability**: Public data or synthetic alternatives
- **Automation**: Fully automated pipeline
- **Documentation**: Complete README with examples

**Reproducibility Checklist**:
```yaml
reproducibility:
  code_available: true          # GitHub/GitLab repo
  code_documented: true          # README, docstrings
  environment_specified: true    # requirements.txt with versions
  data_available: false          # ⚠️ Proprietary data
  data_synthetic: false          # ❌ No alternative provided
  random_seeds: true             # Specified in code
  pipeline_automated: false      # ⚠️ 3 manual steps
  dependencies_pinned: false     # ❌ Missing version pins
  container_available: false     # No Docker/Singularity

  score: 6.0  # Significant gaps
  critical_issues:
    - Data not publicly available
    - Dependencies not pinned
    - Pipeline has manual steps
```

**Reproducibility Gap Analysis**:
```python
def assess_reproducibility_gaps(project):
    """
    Identify specific reproducibility barriers
    """
    gaps = []

    # Environment specification
    if not project.has_requirements_file():
        gaps.append({
            'type': 'environment',
            'severity': 'high',
            'issue': 'No requirements.txt with pinned versions',
            'fix': 'Create requirements.txt: pip freeze > requirements.txt',
            'effort': '30 minutes'
        })

    # Data availability
    if not project.data_public and not project.data_synthetic:
        gaps.append({
            'type': 'data',
            'severity': 'critical',
            'issue': 'Raw data proprietary, no synthetic alternative',
            'fix': 'Generate synthetic dataset preserving statistical properties',
            'effort': '2-3 days'
        })

    # Pipeline automation
    manual_steps = project.get_manual_steps()
    if manual_steps:
        gaps.append({
            'type': 'automation',
            'severity': 'medium',
            'issue': f'{len(manual_steps)} manual steps in pipeline',
            'fix': 'Automate with Makefile or Snakemake',
            'effort': '1-2 days'
        })

    return gaps
```

---

### 3. Experimental Design (Score: /10)

**Sample Size Analysis**:
```python
def assess_sample_size(n_samples, effect_size, alpha=0.05, power_target=0.80):
    """
    Statistical power analysis for sample size adequacy
    """
    from scipy.stats import power

    # Calculate actual power
    actual_power = power.ttest_power(
        effect_size=effect_size,
        nobs=n_samples,
        alpha=alpha
    )

    # Calculate required sample size
    required_n = power.ttest_sample_size(
        effect_size=effect_size,
        alpha=alpha,
        power=power_target
    )

    assessment = {
        'current_n': n_samples,
        'required_n': required_n,
        'actual_power': actual_power,
        'target_power': power_target,
        'adequate': actual_power >= power_target
    }

    if not assessment['adequate']:
        gap = required_n - n_samples
        assessment['action'] = f"Collect {gap} additional samples"
        assessment['severity'] = 'critical' if gap > n_samples else 'high'

    return assessment
```

**Example Assessment**:
```yaml
sample_size_assessment:
  current_samples:
    training: 30
    validation: 10
    test: 15
    total: 55

  power_analysis:
    effect_size: 0.8  # Cohen's d (large)
    alpha: 0.05
    target_power: 0.80
    actual_power: 0.65  # ⚠️ Underpowered

  required_samples:
    total: 100
    gap: 45  # Need 45 more samples
    recommendation: "CRITICAL: Increase n by 82%"

  score: 6.0  # Penalized for underpowered study
```

**Ablation Studies**:
```yaml
ablation_assessment:
  components_tested:
    - component_a: performance_drop: -15%
    - component_b: performance_drop: -8%

  missing_ablations:
    - component_c: "Not tested (critical component)"
    - combined_effects: "A+B removal not tested"
    - gradual_degradation: "Partial removal not tested"

  completeness: 40%  # Only 2/5 ablations done
  score: 5.0  # Incomplete ablation matrix
  recommendation: "Complete ablation matrix (estimate 1 week)"
```

---

### 4. Data Quality (Score: /10)

**Assessment Dimensions**:

**Measurement Accuracy**:
```python
def assess_measurement_quality(measurements):
    """
    Evaluate measurement accuracy and reliability
    """
    assessment = {
        'error_rate': calculate_error_rate(measurements),
        'precision': calculate_precision(measurements),
        'inter_rater_reliability': None,
        'calibration_verified': False
    }

    # Error rate threshold
    if assessment['error_rate'] < 0.01:
        assessment['error_quality'] = 'excellent'
    elif assessment['error_rate'] < 0.05:
        assessment['error_quality'] = 'good'
    else:
        assessment['error_quality'] = 'concerning'

    return assessment
```

**Bias Detection**:
```yaml
bias_assessment:
  selection_bias:
    detected: true
    severity: medium
    description: "Samples from single institution"
    impact: "May not generalize to other populations"
    mitigation:
      - "Collect validation data from diverse sources"
      - "Explicitly state generalization limits"
      - "Test on external dataset"

  measurement_bias:
    detected: false
    instruments_calibrated: true
    blind_assessment: true
    inter_rater_reliability: 0.92  # Excellent

  survivorship_bias:
    detected: false
    drop_out_rate: 0.03  # Acceptable

  score: 7.5  # Selection bias concerns, otherwise good
```

**Missing Data Analysis**:
```python
def analyze_missing_data(data):
    """
    Assess missing data patterns and severity
    """
    missing_rate = data.isnull().sum() / len(data)

    patterns = {
        'missing_completely_at_random': test_mcar(data),
        'missing_at_random': test_mar(data),
        'missing_not_at_random': test_mnar(data)
    }

    severity = 'high' if missing_rate.max() > 0.10 else 'medium' if missing_rate.max() > 0.05 else 'low'

    recommendations = []
    if patterns['missing_not_at_random']:
        recommendations.append("Consider multiple imputation methods")
        recommendations.append("Assess sensitivity to imputation approach")
    elif missing_rate.max() > 0.05:
        recommendations.append("Document imputation strategy")
        recommendations.append("Report results with/without imputed data")

    return {
        'missing_rate': missing_rate,
        'patterns': patterns,
        'severity': severity,
        'recommendations': recommendations
    }
```

---

### 5. Analysis Rigor (Score: /10)

**Statistical Test Appropriateness**:
```python
def validate_statistical_tests(analysis):
    """
    Verify statistical tests are appropriate for data
    """
    validations = []

    for test in analysis.tests:
        assumptions = check_test_assumptions(test, analysis.data)

        validation = {
            'test': test.name,
            'appropriate': all(assumptions.values()),
            'assumptions': assumptions,
            'alternatives': []
        }

        # Suggest alternatives if assumptions violated
        if not validation['appropriate']:
            if not assumptions['normality'] and test.name == 't-test':
                validation['alternatives'].append('Mann-Whitney U test')
            if not assumptions['homoscedasticity'] and test.name == 'ANOVA':
                validation['alternatives'].append("Welch's ANOVA")

        validations.append(validation)

    return validations
```

**Multiple Testing Correction**:
```yaml
multiple_testing:
  number_of_tests: 15
  correction_method: "Bonferroni"
  adjusted_alpha: 0.003  # 0.05 / 15

  results_after_correction:
    significant_before: 8
    significant_after: 5
    false_discoveries_prevented: ~3

  alternative_methods:
    - "Benjamini-Hochberg (less conservative)"
    - "Holm-Bonferroni (step-down)"

  score: 9.0  # Proper correction applied
```

---

### 6. Publication Readiness (Score: /10)

**Manuscript Completeness**:
```yaml
manuscript_assessment:
  sections:
    abstract: present: true, quality: 8
    introduction: present: true, quality: 7
    methods: present: true, quality: 9
    results: present: true, quality: 8
    discussion: present: true, quality: 6  # ⚠️ Needs expansion
    conclusion: present: true, quality: 5  # ⚠️ Too brief
    limitations: present: false  # ❌ Missing
    future_work: present: false  # ⚠️ Minimal
    broader_impact: present: false  # ❌ Missing

  completeness: 70%
  critical_gaps:
    - "Limitations section missing"
    - "Broader impact statement needed"
    - "Discussion needs expansion"

  score: 7.0  # Good but incomplete
```

**Figure Quality**:
```yaml
figure_assessment:
  publication_ready:
    resolution: 300  # DPI ✅
    format: "vector"  # PDF/SVG ✅
    color_scheme: "colorblind_friendly"  # ✅
    labels: "clear_and_legible"  # ✅
    captions: "informative"  # ✅

  improvements_needed:
    - "Add schematic diagram of method"
    - "Include representative examples in supplement"

  score: 9.0  # Publication-ready
```

---

## Complete Assessment Example

```yaml
Research Project Reflection
Overall Score: 7.5/10 - Strong with Critical Gaps

Dimensional Scores:
  methodology: 8.5/10  # ✅ Strong
  reproducibility: 6.0/10  # ⚠️ Gaps exist
  experimental_design: 6.0/10  # ⚠️ Underpowered
  data_quality: 7.5/10  # ✅ Good
  analysis_rigor: 8.0/10  # ✅ Strong
  publication_readiness: 7.0/10  # ⚠️ Incomplete

Critical Issues:
  🔴 Sample size insufficient (n=55, need n=100)
  🔴 Reproducibility gaps (data, dependencies)
  ⚠️  Ablation studies incomplete
  ⚠️  Limitations section missing

Priority Actions (Next 4 Weeks):
  Week 1-2: Collect 45 additional samples
  Week 3: Complete ablation studies
  Week 4: Fix reproducibility package
  Week 4: Expand discussion and add limitations

Expected Timeline: 4 weeks to publication-ready
Target Venue: Tier 2 journal (high acceptance probability)
Success Probability: 75% (after addressing critical issues)
```

---

## Publication Strategy

### Venue Selection

```python
def recommend_publication_venue(assessment):
    """
    Recommend target venue based on quality assessment
    """
    overall_score = assessment.overall_score

    if overall_score >= 9.0 and assessment.novelty >= 8:
        return {
            'tier': 1,
            'venues': ['Nature', 'Science', 'Top domain journal'],
            'acceptance_rate': '5-10%',
            'probability': 'low' if assessment.has_critical_gaps() else 'medium'
        }
    elif overall_score >= 7.5:
        return {
            'tier': 2,
            'venues': ['Strong domain journal', 'Interdisciplinary journal'],
            'acceptance_rate': '15-25%',
            'probability': 'high'
        }
    else:
        return {
            'tier': 3,
            'venues': ['Solid domain journal', 'Open access venue'],
            'acceptance_rate': '30-50%',
            'probability': 'very_high'
        }
```

---

## Related Documentation

- [Multi-Agent Reflection System](multi-agent-reflection-system.md) - Orchestration patterns
- [Reflection Report Templates](reflection-report-templates.md) - Complete template examples
- [Research Reflection Example](../examples/research-reflection-example.md) - Real project walkthrough

---

*Part of the ai-reasoning plugin documentation*
