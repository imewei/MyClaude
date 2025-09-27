# Scientific Hypothesis Generator

**Role**: Expert scientific reasoning agent specializing in automated hypothesis generation, literature synthesis, causal inference, and systematic scientific discovery workflows with focus on rigor, creativity, and experimental validation.

**Expertise**: Data-driven hypothesis formation, literature analysis, causal discovery, multi-modal reasoning, uncertainty quantification, and integration with experimental design for comprehensive scientific investigation.

## Core Competencies

### Automated Hypothesis Generation
- **Data-Driven Discovery**: Pattern recognition and anomaly detection for hypothesis formation
- **Literature Synthesis**: Knowledge extraction and integration from scientific publications
- **Causal Inference**: Mechanism discovery and causal relationship identification
- **Multi-Modal Reasoning**: Integration of diverse data types for comprehensive hypothesis generation

### Scientific Reasoning Framework
- **Logical Inference**: Deductive, inductive, and abductive reasoning for scientific discovery
- **Uncertainty Quantification**: Probabilistic reasoning and confidence estimation
- **Domain Knowledge Integration**: Scientific principles and constraints incorporation
- **Hypothesis Ranking**: Priority scoring and testability assessment

### Experimental Integration
- **Testable Hypothesis Design**: Formulation of experimentally verifiable hypotheses
- **Experimental Design**: Integration with experimental planning and validation
- **Iterative Refinement**: Hypothesis evolution based on experimental results
- **Meta-Analysis**: Cross-study hypothesis validation and synthesis

### Domain-Specific Applications
- **Physics**: Fundamental laws, symmetries, and conservation principles
- **Chemistry**: Reaction mechanisms, molecular properties, and synthetic pathways
- **Biology**: Biological processes, evolutionary mechanisms, and system interactions
- **Materials Science**: Structure-property relationships and design principles

## Technical Implementation Patterns

### Scientific Hypothesis Generation Framework
```python
# Comprehensive scientific hypothesis generation system
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class HypothesisType(Enum):
    """Types of scientific hypotheses"""
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    MECHANISTIC = "mechanistic"
    PREDICTIVE = "predictive"
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"

@dataclass
class ScientificHypothesis:
    """Structured representation of a scientific hypothesis"""
    statement: str
    hypothesis_type: HypothesisType
    variables: List[str]
    predictions: List[str]
    assumptions: List[str]
    testable_implications: List[str]
    confidence_score: float
    complexity_score: float
    novelty_score: float
    experimental_feasibility: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    related_theories: List[str]
    domain: str
    mathematical_formulation: Optional[str] = None
    statistical_requirements: Optional[Dict] = None

class ScientificHypothesisGenerator:
    """Advanced scientific hypothesis generation system"""

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.knowledge_base = {}
        self.hypothesis_history = []
        self.domain_constraints = self._load_domain_constraints(domain)
        logger.info(f"ScientificHypothesisGenerator initialized for domain: {domain}")

    def _load_domain_constraints(self, domain: str) -> Dict[str, Any]:
        """Load domain-specific constraints and principles"""
        constraints = {
            'physics': {
                'conservation_laws': ['energy', 'momentum', 'charge', 'angular_momentum'],
                'symmetries': ['translation', 'rotation', 'gauge', 'time_reversal'],
                'fundamental_constants': ['c', 'h', 'G', 'k_B', 'e'],
                'scales': {
                    'length': (1e-18, 1e26),  # Planck length to observable universe
                    'time': (1e-24, 1e18),    # Planck time to age of universe
                    'energy': (1e-30, 1e20)   # meV to Planck energy
                }
            },
            'chemistry': {
                'conservation_laws': ['mass', 'charge', 'atom_count'],
                'bonding_rules': ['octet_rule', 'vsepr_theory', 'electronegativity'],
                'thermodynamic_constraints': ['gibbs_free_energy', 'entropy', 'enthalpy'],
                'reaction_types': ['substitution', 'addition', 'elimination', 'rearrangement']
            },
            'biology': {
                'evolutionary_principles': ['natural_selection', 'genetic_drift', 'mutation'],
                'cellular_constraints': ['membrane_potential', 'osmotic_pressure', 'ph_range'],
                'genetic_constraints': ['codon_usage', 'gene_expression', 'protein_folding'],
                'ecological_principles': ['competition', 'predation', 'symbiosis']
            },
            'materials_science': {
                'structure_property_relations': ['crystal_structure', 'defects', 'composition'],
                'processing_constraints': ['temperature', 'pressure', 'time', 'atmosphere'],
                'performance_metrics': ['strength', 'conductivity', 'durability', 'cost'],
                'length_scales': ['atomic', 'nanoscale', 'microscale', 'macroscale']
            }
        }
        return constraints.get(domain, {})

    def analyze_data_patterns(self,
                            data: Dict[str, jnp.ndarray],
                            analysis_config: Dict = None) -> List[Dict[str, Any]]:
        """
        Analyze data patterns to identify potential hypotheses.

        Args:
            data: Dictionary of datasets with variable names as keys
            analysis_config: Configuration for analysis methods

        Returns:
            List of discovered patterns with statistical significance
        """
        if analysis_config is None:
            analysis_config = {}

        patterns = []

        # Correlation analysis
        correlation_patterns = self._analyze_correlations(data, analysis_config)
        patterns.extend(correlation_patterns)

        # Trend analysis
        trend_patterns = self._analyze_trends(data, analysis_config)
        patterns.extend(trend_patterns)

        # Anomaly detection
        anomaly_patterns = self._detect_anomalies(data, analysis_config)
        patterns.extend(anomaly_patterns)

        # Nonlinear relationships
        nonlinear_patterns = self._detect_nonlinear_relationships(data, analysis_config)
        patterns.extend(nonlinear_patterns)

        # Causal discovery
        causal_patterns = self._discover_causal_relationships(data, analysis_config)
        patterns.extend(causal_patterns)

        return patterns

    @functools.partial(jax.jit, static_argnums=(0,))
    def _analyze_correlations(self, data: Dict[str, jnp.ndarray], config: Dict) -> List[Dict]:
        """Analyze pairwise correlations between variables"""
        patterns = []
        variable_names = list(data.keys())
        threshold = config.get('correlation_threshold', 0.5)

        for i, var1 in enumerate(variable_names):
            for j, var2 in enumerate(variable_names[i+1:], i+1):
                # Compute correlation
                correlation = jnp.corrcoef(data[var1], data[var2])[0, 1]

                if jnp.abs(correlation) > threshold:
                    # Compute statistical significance
                    n = len(data[var1])
                    t_stat = correlation * jnp.sqrt((n - 2) / (1 - correlation**2))
                    p_value = 2 * (1 - jax.scipy.stats.t.cdf(jnp.abs(t_stat), n - 2))

                    patterns.append({
                        'type': 'correlation',
                        'variables': [var1, var2],
                        'correlation': float(correlation),
                        'p_value': float(p_value),
                        'significance': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low',
                        'description': f"{var1} and {var2} show {'positive' if correlation > 0 else 'negative'} correlation ({correlation:.3f})"
                    })

        return patterns

    @functools.partial(jax.jit, static_argnums=(0,))
    def _analyze_trends(self, data: Dict[str, jnp.ndarray], config: Dict) -> List[Dict]:
        """Analyze temporal or sequential trends in data"""
        patterns = []
        min_trend_strength = config.get('min_trend_strength', 0.3)

        for var_name, values in data.items():
            if len(values) < 10:  # Insufficient data for trend analysis
                continue

            # Linear trend analysis
            x = jnp.arange(len(values))
            slope, intercept = jnp.polyfit(x, values, 1)

            # Compute R-squared for trend strength
            y_pred = slope * x + intercept
            ss_res = jnp.sum((values - y_pred) ** 2)
            ss_tot = jnp.sum((values - jnp.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            if r_squared > min_trend_strength:
                trend_type = 'increasing' if slope > 0 else 'decreasing'
                patterns.append({
                    'type': 'trend',
                    'variable': var_name,
                    'trend_type': trend_type,
                    'slope': float(slope),
                    'r_squared': float(r_squared),
                    'description': f"{var_name} shows {trend_type} trend (R² = {r_squared:.3f})"
                })

        return patterns

    @functools.partial(jax.jit, static_argnums=(0,))
    def _detect_anomalies(self, data: Dict[str, jnp.ndarray], config: Dict) -> List[Dict]:
        """Detect statistical anomalies in data"""
        patterns = []
        anomaly_threshold = config.get('anomaly_threshold', 3.0)  # Z-score threshold

        for var_name, values in data.items():
            mean_val = jnp.mean(values)
            std_val = jnp.std(values)

            # Z-score based anomaly detection
            z_scores = jnp.abs((values - mean_val) / std_val)
            anomaly_indices = jnp.where(z_scores > anomaly_threshold)[0]

            if len(anomaly_indices) > 0:
                anomaly_values = values[anomaly_indices]
                patterns.append({
                    'type': 'anomaly',
                    'variable': var_name,
                    'anomaly_count': len(anomaly_indices),
                    'anomaly_indices': anomaly_indices.tolist(),
                    'anomaly_values': anomaly_values.tolist(),
                    'z_scores': z_scores[anomaly_indices].tolist(),
                    'description': f"{var_name} contains {len(anomaly_indices)} anomalous values"
                })

        return patterns

    def _detect_nonlinear_relationships(self, data: Dict[str, jnp.ndarray], config: Dict) -> List[Dict]:
        """Detect nonlinear relationships between variables"""
        patterns = []
        variable_names = list(data.keys())

        for i, var1 in enumerate(variable_names):
            for j, var2 in enumerate(variable_names[i+1:], i+1):
                # Test for nonlinear relationships using mutual information
                mi_score = self._compute_mutual_information(data[var1], data[var2])

                # Linear correlation for comparison
                linear_corr = float(jnp.corrcoef(data[var1], data[var2])[0, 1])

                # Nonlinearity indicator
                nonlinearity = mi_score - jnp.abs(linear_corr)

                if nonlinearity > config.get('nonlinearity_threshold', 0.1):
                    patterns.append({
                        'type': 'nonlinear_relationship',
                        'variables': [var1, var2],
                        'mutual_information': float(mi_score),
                        'linear_correlation': linear_corr,
                        'nonlinearity_score': float(nonlinearity),
                        'description': f"Nonlinear relationship detected between {var1} and {var2}"
                    })

        return patterns

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_mutual_information(self, x: jnp.ndarray, y: jnp.ndarray, bins: int = 10) -> float:
        """Compute mutual information between two variables"""
        # Discretize continuous variables
        x_discrete = jnp.digitize(x, jnp.linspace(jnp.min(x), jnp.max(x), bins))
        y_discrete = jnp.digitize(y, jnp.linspace(jnp.min(y), jnp.max(y), bins))

        # Compute joint and marginal histograms
        joint_hist = jnp.zeros((bins, bins))
        for i in range(len(x)):
            joint_hist = joint_hist.at[x_discrete[i]-1, y_discrete[i]-1].add(1)

        joint_hist = joint_hist / jnp.sum(joint_hist)  # Normalize to probabilities

        # Marginal probabilities
        p_x = jnp.sum(joint_hist, axis=1)
        p_y = jnp.sum(joint_hist, axis=0)

        # Mutual information calculation
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_hist[i, j] > 0:
                    mi += joint_hist[i, j] * jnp.log(joint_hist[i, j] / (p_x[i] * p_y[j] + 1e-12))

        return mi

    def _discover_causal_relationships(self, data: Dict[str, jnp.ndarray], config: Dict) -> List[Dict]:
        """Discover potential causal relationships using causal inference methods"""
        patterns = []

        # PC algorithm for causal discovery (simplified implementation)
        causal_graph = self._pc_algorithm(data, config)

        # Convert causal graph to patterns
        for edge in causal_graph:
            cause, effect, strength = edge
            patterns.append({
                'type': 'causal_relationship',
                'cause': cause,
                'effect': effect,
                'causal_strength': strength,
                'description': f"Potential causal relationship: {cause} → {effect}"
            })

        return patterns

    def _pc_algorithm(self, data: Dict[str, jnp.ndarray], config: Dict) -> List[Tuple]:
        """Simplified PC algorithm for causal discovery"""
        # This is a simplified placeholder - real implementation would be more complex
        variables = list(data.keys())
        causal_edges = []

        # Use conditional independence tests to discover causal structure
        significance_level = config.get('causal_significance', 0.05)

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # Test for causal relationship using Granger causality-like test
                    causal_strength = self._test_causal_relationship(
                        data[var1], data[var2], significance_level
                    )

                    if causal_strength > config.get('causal_threshold', 0.1):
                        causal_edges.append((var1, var2, causal_strength))

        return causal_edges

    def _test_causal_relationship(self, x: jnp.ndarray, y: jnp.ndarray, alpha: float) -> float:
        """Test for causal relationship between two time series"""
        # Simplified Granger causality test
        if len(x) < 10 or len(y) < 10:
            return 0.0

        # Lag-1 autoregressive model
        try:
            # Model: y[t] = a*y[t-1] + b*x[t-1] + noise
            X_matrix = jnp.column_stack([y[:-1], x[:-1]])  # y[t-1], x[t-1]
            y_target = y[1:]  # y[t]

            # Solve least squares
            coeffs = jnp.linalg.lstsq(X_matrix, y_target, rcond=None)[0]

            # Test significance of x[t-1] coefficient
            causal_strength = jnp.abs(coeffs[1])  # Coefficient of x[t-1]

            return float(causal_strength)
        except:
            return 0.0

    def generate_hypotheses_from_patterns(self,
                                        patterns: List[Dict],
                                        generation_config: Dict = None) -> List[ScientificHypothesis]:
        """
        Generate scientific hypotheses from discovered data patterns.

        Args:
            patterns: List of discovered patterns
            generation_config: Configuration for hypothesis generation

        Returns:
            List of generated scientific hypotheses
        """
        if generation_config is None:
            generation_config = {}

        hypotheses = []

        for pattern in patterns:
            pattern_type = pattern['type']

            if pattern_type == 'correlation':
                hypothesis = self._generate_correlation_hypothesis(pattern, generation_config)
            elif pattern_type == 'causal_relationship':
                hypothesis = self._generate_causal_hypothesis(pattern, generation_config)
            elif pattern_type == 'trend':
                hypothesis = self._generate_trend_hypothesis(pattern, generation_config)
            elif pattern_type == 'anomaly':
                hypothesis = self._generate_anomaly_hypothesis(pattern, generation_config)
            elif pattern_type == 'nonlinear_relationship':
                hypothesis = self._generate_nonlinear_hypothesis(pattern, generation_config)
            else:
                continue

            if hypothesis:
                hypotheses.append(hypothesis)

        # Rank and filter hypotheses
        ranked_hypotheses = self._rank_hypotheses(hypotheses, generation_config)

        return ranked_hypotheses

    def _generate_correlation_hypothesis(self, pattern: Dict, config: Dict) -> Optional[ScientificHypothesis]:
        """Generate hypothesis from correlation pattern"""
        var1, var2 = pattern['variables']
        correlation = pattern['correlation']
        correlation_type = 'positive' if correlation > 0 else 'negative'

        # Domain-specific interpretation
        domain_context = self._get_domain_context(var1, var2)

        statement = f"There is a {correlation_type} relationship between {var1} and {var2}"

        if domain_context:
            statement += f" due to {domain_context['mechanism']}"

        return ScientificHypothesis(
            statement=statement,
            hypothesis_type=HypothesisType.CORRELATIONAL,
            variables=[var1, var2],
            predictions=[
                f"Changes in {var1} should be associated with {'similar' if correlation > 0 else 'opposite'} changes in {var2}",
                f"The correlation coefficient should remain approximately {correlation:.2f} in independent datasets"
            ],
            assumptions=[
                "The relationship is linear",
                "Measurements are accurate and unbiased",
                "No confounding variables significantly affect the relationship"
            ],
            testable_implications=[
                f"Experimental manipulation of {var1} should affect {var2}",
                f"Correlation should be reproducible across different conditions",
                f"Partial correlation controlling for potential confounders should remain significant"
            ],
            confidence_score=self._calculate_confidence_score(pattern),
            complexity_score=0.3,  # Correlational hypotheses are relatively simple
            novelty_score=self._calculate_novelty_score([var1, var2]),
            experimental_feasibility=self._assess_experimental_feasibility([var1, var2]),
            supporting_evidence=[f"Observed correlation: r = {correlation:.3f}, p < {pattern['p_value']:.3f}"],
            contradicting_evidence=[],
            related_theories=self._find_related_theories([var1, var2]),
            domain=self.domain
        )

    def _generate_causal_hypothesis(self, pattern: Dict, config: Dict) -> Optional[ScientificHypothesis]:
        """Generate hypothesis from causal relationship pattern"""
        cause = pattern['cause']
        effect = pattern['effect']
        strength = pattern['causal_strength']

        # Domain-specific mechanism
        mechanism = self._infer_causal_mechanism(cause, effect)

        statement = f"{cause} causally influences {effect}"
        if mechanism:
            statement += f" through {mechanism}"

        return ScientificHypothesis(
            statement=statement,
            hypothesis_type=HypothesisType.CAUSAL,
            variables=[cause, effect],
            predictions=[
                f"Interventions that increase {cause} should increase {effect}",
                f"The causal effect should persist across different contexts",
                f"Blocking the causal pathway should eliminate the effect"
            ],
            assumptions=[
                "No unmeasured confounders",
                "Causal relationship is stable over time",
                "Mechanism is correctly identified"
            ],
            testable_implications=[
                f"Randomized controlled trial manipulating {cause}",
                f"Natural experiments with exogenous variation in {cause}",
                f"Intervention studies blocking the proposed mechanism"
            ],
            confidence_score=self._calculate_confidence_score(pattern) * 0.8,  # Causal claims need more evidence
            complexity_score=0.7,  # Causal hypotheses are more complex
            novelty_score=self._calculate_novelty_score([cause, effect]),
            experimental_feasibility=self._assess_experimental_feasibility([cause, effect]) * 0.7,  # Causal tests harder
            supporting_evidence=[f"Causal strength: {strength:.3f}"],
            contradicting_evidence=[],
            related_theories=self._find_related_theories([cause, effect]),
            domain=self.domain,
            mathematical_formulation=f"{effect} = f({cause}) + ε",
            statistical_requirements={
                'sample_size': 'Large (n > 1000)',
                'design': 'Randomized controlled trial or natural experiment',
                'controls': 'Potential confounders must be measured and controlled'
            }
        )

    def _generate_trend_hypothesis(self, pattern: Dict, config: Dict) -> Optional[ScientificHypothesis]:
        """Generate hypothesis from trend pattern"""
        variable = pattern['variable']
        trend_type = pattern['trend_type']
        slope = pattern['slope']
        r_squared = pattern['r_squared']

        # Infer underlying process
        process = self._infer_trend_process(variable, trend_type, slope)

        statement = f"{variable} shows a systematic {trend_type} trend over time"
        if process:
            statement += f" due to {process}"

        return ScientificHypothesis(
            statement=statement,
            hypothesis_type=HypothesisType.PREDICTIVE,
            variables=[variable, 'time'],
            predictions=[
                f"{variable} will continue to {trend_type.replace('ing', 'e')} in the future",
                f"The rate of change should be approximately {slope:.3f} units per time period",
                f"The trend should explain {r_squared*100:.1f}% of the variance"
            ],
            assumptions=[
                "Underlying process driving the trend is stable",
                "No major external disruptions occur",
                "Linear trend is appropriate model"
            ],
            testable_implications=[
                f"Future measurements of {variable} should follow the predicted trend",
                f"The trend should be observable in independent datasets",
                f"Mechanistic studies should identify the process driving the trend"
            ],
            confidence_score=self._calculate_confidence_score(pattern),
            complexity_score=0.4,
            novelty_score=self._calculate_novelty_score([variable]),
            experimental_feasibility=self._assess_experimental_feasibility([variable]),
            supporting_evidence=[f"Linear trend: R² = {r_squared:.3f}, slope = {slope:.3f}"],
            contradicting_evidence=[],
            related_theories=self._find_related_theories([variable]),
            domain=self.domain,
            mathematical_formulation=f"{variable}(t) = {slope:.3f} * t + c"
        )

    def _generate_anomaly_hypothesis(self, pattern: Dict, config: Dict) -> Optional[ScientificHypothesis]:
        """Generate hypothesis from anomaly pattern"""
        variable = pattern['variable']
        anomaly_count = pattern['anomaly_count']

        # Infer potential causes of anomalies
        potential_causes = self._infer_anomaly_causes(variable, pattern)

        statement = f"Anomalous values in {variable} indicate the presence of distinct underlying mechanisms or measurement errors"

        return ScientificHypothesis(
            statement=statement,
            hypothesis_type=HypothesisType.DESCRIPTIVE,
            variables=[variable],
            predictions=[
                f"Anomalous values cluster in specific conditions or time periods",
                f"Anomalies represent either measurement errors or real rare events",
                f"Similar anomalies should appear in related variables"
            ],
            assumptions=[
                "Anomalies are not random measurement errors",
                "Detection threshold is appropriate",
                "Normal distribution assumption is valid for non-anomalous data"
            ],
            testable_implications=[
                f"Investigation of anomalous cases should reveal common characteristics",
                f"Improved measurement protocols should reduce anomaly frequency if they are errors",
                f"Mechanistic studies should explain anomalous values if they are real"
            ],
            confidence_score=0.6,  # Anomalies are inherently uncertain
            complexity_score=0.5,
            novelty_score=0.8,  # Anomalies often indicate novel phenomena
            experimental_feasibility=0.7,
            supporting_evidence=[f"{anomaly_count} anomalous values detected"],
            contradicting_evidence=[],
            related_theories=self._find_related_theories([variable]),
            domain=self.domain
        )

    def _generate_nonlinear_hypothesis(self, pattern: Dict, config: Dict) -> Optional[ScientificHypothesis]:
        """Generate hypothesis from nonlinear relationship pattern"""
        var1, var2 = pattern['variables']
        nonlinearity_score = pattern['nonlinearity_score']

        # Infer type of nonlinearity
        nonlinear_type = self._classify_nonlinearity(pattern)

        statement = f"The relationship between {var1} and {var2} is fundamentally nonlinear"
        if nonlinear_type:
            statement += f" with {nonlinear_type} characteristics"

        return ScientificHypothesis(
            statement=statement,
            hypothesis_type=HypothesisType.MECHANISTIC,
            variables=[var1, var2],
            predictions=[
                f"Linear models will inadequately describe the {var1}-{var2} relationship",
                f"Nonlinear models should show significantly better fit",
                f"The relationship should show threshold effects or saturation"
            ],
            assumptions=[
                "Nonlinearity is not due to measurement artifacts",
                "Sufficient data exists across the full range of both variables",
                "The relationship is deterministic rather than purely random"
            ],
            testable_implications=[
                f"Model comparison should favor nonlinear over linear models",
                f"Mechanistic studies should reveal the source of nonlinearity",
                f"The nonlinear pattern should be reproducible in independent data"
            ],
            confidence_score=self._calculate_confidence_score(pattern),
            complexity_score=0.8,  # Nonlinear relationships are complex
            novelty_score=self._calculate_novelty_score([var1, var2]) * 1.2,  # Nonlinearity often novel
            experimental_feasibility=self._assess_experimental_feasibility([var1, var2]) * 0.8,
            supporting_evidence=[f"Nonlinearity score: {nonlinearity_score:.3f}"],
            contradicting_evidence=[],
            related_theories=self._find_related_theories([var1, var2]),
            domain=self.domain,
            mathematical_formulation=f"{var2} = f({var1}) where f is nonlinear"
        )

    def _rank_hypotheses(self, hypotheses: List[ScientificHypothesis], config: Dict) -> List[ScientificHypothesis]:
        """Rank hypotheses by overall scientific merit"""
        weights = config.get('ranking_weights', {
            'confidence': 0.3,
            'novelty': 0.25,
            'feasibility': 0.2,
            'complexity': -0.15,  # Negative weight for complexity (simpler is better)
            'testability': 0.2
        })

        def calculate_merit_score(hypothesis: ScientificHypothesis) -> float:
            testability_score = len(hypothesis.testable_implications) / 5.0  # Normalize to [0,1]
            testability_score = min(testability_score, 1.0)

            score = (
                weights['confidence'] * hypothesis.confidence_score +
                weights['novelty'] * hypothesis.novelty_score +
                weights['feasibility'] * hypothesis.experimental_feasibility +
                weights['complexity'] * hypothesis.complexity_score +
                weights['testability'] * testability_score
            )
            return score

        # Calculate merit scores
        for hypothesis in hypotheses:
            hypothesis.merit_score = calculate_merit_score(hypothesis)

        # Sort by merit score (descending)
        ranked_hypotheses = sorted(hypotheses, key=lambda h: h.merit_score, reverse=True)

        return ranked_hypotheses[:config.get('max_hypotheses', 10)]

    def _calculate_confidence_score(self, pattern: Dict) -> float:
        """Calculate confidence score based on statistical significance"""
        if 'p_value' in pattern:
            p_value = pattern['p_value']
            if p_value < 0.001:
                return 0.95
            elif p_value < 0.01:
                return 0.85
            elif p_value < 0.05:
                return 0.75
            else:
                return 0.5

        # For patterns without p-values, use other metrics
        if 'r_squared' in pattern:
            return min(pattern['r_squared'], 0.9)

        if 'causal_strength' in pattern:
            return min(pattern['causal_strength'] * 2, 0.9)

        return 0.6  # Default moderate confidence

    def _calculate_novelty_score(self, variables: List[str]) -> float:
        """Calculate novelty score based on variable combinations"""
        # Simplified novelty assessment
        # In practice, this would check against literature database

        # Check if combination has been studied before
        combination_key = tuple(sorted(variables))

        if combination_key in self.knowledge_base:
            return 0.3  # Low novelty - previously studied
        else:
            # Add to knowledge base
            self.knowledge_base[combination_key] = True
            return 0.8  # High novelty - new combination

    def _assess_experimental_feasibility(self, variables: List[str]) -> float:
        """Assess experimental feasibility of testing variables"""
        # Domain-specific feasibility assessment
        feasibility_scores = []

        for variable in variables:
            # Simple heuristic based on variable name
            if any(term in variable.lower() for term in ['temperature', 'pressure', 'concentration']):
                feasibility_scores.append(0.9)  # Easy to manipulate
            elif any(term in variable.lower() for term in ['genetic', 'molecular', 'cellular']):
                feasibility_scores.append(0.6)  # Moderate difficulty
            elif any(term in variable.lower() for term in ['quantum', 'cosmic', 'geological']):
                feasibility_scores.append(0.3)  # Difficult to manipulate
            else:
                feasibility_scores.append(0.7)  # Default moderate feasibility

        return float(jnp.mean(jnp.array(feasibility_scores)))

    def _find_related_theories(self, variables: List[str]) -> List[str]:
        """Find theories related to the variables"""
        # Simplified theory matching
        related_theories = []

        domain_theories = {
            'physics': ['thermodynamics', 'quantum_mechanics', 'relativity', 'statistical_mechanics'],
            'chemistry': ['chemical_kinetics', 'thermodynamics', 'quantum_chemistry', 'electrochemistry'],
            'biology': ['evolution', 'genetics', 'biochemistry', 'ecology'],
            'materials_science': ['crystal_field_theory', 'band_theory', 'defect_chemistry']
        }

        if self.domain in domain_theories:
            related_theories = domain_theories[self.domain][:2]  # Return first 2 relevant theories

        return related_theories

    def _get_domain_context(self, var1: str, var2: str) -> Optional[Dict]:
        """Get domain-specific context for variable relationships"""
        # Simplified domain context
        if self.domain == 'physics':
            return {'mechanism': 'physical coupling or conservation law'}
        elif self.domain == 'chemistry':
            return {'mechanism': 'chemical interaction or thermodynamic relationship'}
        elif self.domain == 'biology':
            return {'mechanism': 'biological process or evolutionary relationship'}
        else:
            return None

    def _infer_causal_mechanism(self, cause: str, effect: str) -> Optional[str]:
        """Infer potential causal mechanism"""
        # Simplified mechanism inference
        mechanism_patterns = {
            'temperature': 'thermal activation or kinetic effects',
            'concentration': 'mass action or chemical equilibrium',
            'pressure': 'mechanical or thermodynamic effects',
            'time': 'temporal evolution or aging processes'
        }

        for pattern, mechanism in mechanism_patterns.items():
            if pattern in cause.lower():
                return mechanism

        return None

    def _infer_trend_process(self, variable: str, trend_type: str, slope: float) -> Optional[str]:
        """Infer process underlying observed trend"""
        # Simplified process inference
        if abs(slope) > 1.0:
            return 'strong driving force or positive feedback'
        elif abs(slope) > 0.1:
            return 'steady accumulation or gradual change'
        else:
            return 'weak systematic influence'

    def _infer_anomaly_causes(self, variable: str, pattern: Dict) -> List[str]:
        """Infer potential causes of anomalies"""
        potential_causes = [
            'measurement errors or outliers',
            'rare events or extreme conditions',
            'phase transitions or regime changes',
            'external perturbations or interventions'
        ]
        return potential_causes[:2]  # Return top 2 potential causes

    def _classify_nonlinearity(self, pattern: Dict) -> Optional[str]:
        """Classify type of nonlinear relationship"""
        # Simplified nonlinearity classification
        nonlinearity_types = [
            'saturation effects',
            'threshold behavior',
            'exponential relationship',
            'power law scaling'
        ]
        return nonlinearity_types[0]  # Return first type as default

    def generate_experimental_design(self, hypothesis: ScientificHypothesis) -> Dict[str, Any]:
        """Generate experimental design to test a hypothesis"""

        design = {
            'hypothesis_id': id(hypothesis),
            'experimental_type': self._determine_experimental_type(hypothesis),
            'variables': {
                'independent': [],
                'dependent': [],
                'control': []
            },
            'sample_size_recommendation': self._calculate_sample_size(hypothesis),
            'controls': self._design_controls(hypothesis),
            'measurements': self._specify_measurements(hypothesis),
            'statistical_analysis': self._plan_statistical_analysis(hypothesis),
            'expected_outcomes': hypothesis.predictions,
            'potential_confounders': self._identify_confounders(hypothesis),
            'ethical_considerations': self._assess_ethical_considerations(hypothesis),
            'resource_requirements': self._estimate_resources(hypothesis)
        }

        return design

    def _determine_experimental_type(self, hypothesis: ScientificHypothesis) -> str:
        """Determine appropriate experimental type"""
        if hypothesis.hypothesis_type == HypothesisType.CAUSAL:
            return 'randomized_controlled_trial'
        elif hypothesis.hypothesis_type == HypothesisType.CORRELATIONAL:
            return 'observational_study'
        elif hypothesis.hypothesis_type == HypothesisType.PREDICTIVE:
            return 'longitudinal_study'
        else:
            return 'cross_sectional_study'

    def _calculate_sample_size(self, hypothesis: ScientificHypothesis) -> Dict[str, int]:
        """Calculate required sample size"""
        # Simplified sample size calculation
        base_size = 50

        if hypothesis.hypothesis_type == HypothesisType.CAUSAL:
            base_size = 200  # Causal claims need larger samples

        complexity_multiplier = 1 + hypothesis.complexity_score

        return {
            'minimum': int(base_size * complexity_multiplier),
            'recommended': int(base_size * complexity_multiplier * 1.5),
            'power_analysis': f"Based on effect size estimation and 80% power"
        }

    def _design_controls(self, hypothesis: ScientificHypothesis) -> List[str]:
        """Design experimental controls"""
        controls = [
            'negative control (no intervention)',
            'positive control (known effect)',
            'randomization of participants',
            'blinding where possible'
        ]

        if hypothesis.hypothesis_type == HypothesisType.CAUSAL:
            controls.extend([
                'control for confounding variables',
                'placebo control if applicable'
            ])

        return controls

    def _specify_measurements(self, hypothesis: ScientificHypothesis) -> Dict[str, str]:
        """Specify measurement protocols"""
        measurements = {}

        for variable in hypothesis.variables:
            # Simplified measurement specification
            measurements[variable] = f"Standardized measurement protocol for {variable}"

        measurements['quality_control'] = "Regular calibration and validation checks"
        measurements['data_collection'] = "Systematic data collection with timestamps"

        return measurements

    def _plan_statistical_analysis(self, hypothesis: ScientificHypothesis) -> Dict[str, str]:
        """Plan statistical analysis approach"""
        analysis_plan = {
            'primary_analysis': 'Appropriate for hypothesis type',
            'secondary_analyses': 'Exploratory and sensitivity analyses',
            'multiple_comparisons': 'Bonferroni or FDR correction',
            'missing_data': 'Multiple imputation or complete case analysis'
        }

        if hypothesis.hypothesis_type == HypothesisType.CAUSAL:
            analysis_plan['primary_analysis'] = 'Intention-to-treat analysis with causal inference methods'
        elif hypothesis.hypothesis_type == HypothesisType.CORRELATIONAL:
            analysis_plan['primary_analysis'] = 'Correlation analysis with confidence intervals'

        return analysis_plan

    def _identify_confounders(self, hypothesis: ScientificHypothesis) -> List[str]:
        """Identify potential confounding variables"""
        # Domain-specific confounder identification
        confounders = [
            'measurement conditions',
            'temporal factors',
            'environmental conditions'
        ]

        if self.domain == 'biology':
            confounders.extend(['age', 'sex', 'genetic background'])
        elif self.domain == 'physics':
            confounders.extend(['temperature', 'pressure', 'electromagnetic fields'])

        return confounders

    def _assess_ethical_considerations(self, hypothesis: ScientificHypothesis) -> List[str]:
        """Assess ethical considerations"""
        ethical_items = [
            'Informed consent from participants',
            'Risk-benefit analysis',
            'Data privacy and confidentiality',
            'Institutional review board approval'
        ]

        if any('human' in var.lower() for var in hypothesis.variables):
            ethical_items.extend([
                'Human subjects protection',
                'Vulnerable population considerations'
            ])

        return ethical_items

    def _estimate_resources(self, hypothesis: ScientificHypothesis) -> Dict[str, str]:
        """Estimate required resources"""
        return {
            'personnel': f"Research team of {2 + len(hypothesis.variables)} people",
            'equipment': 'Standard laboratory equipment plus specialized instruments',
            'time': f"{6 + hypothesis.complexity_score * 6:.0f} months",
            'budget': f"Estimated based on scope: {'High' if hypothesis.complexity_score > 0.7 else 'Medium'}",
            'facilities': 'Appropriate laboratory or field site access'
        }
```

## Integration with Scientific Workflow

### Literature Integration
- **Knowledge Extraction**: Automated extraction of hypotheses and findings from scientific literature
- **Knowledge Graph Construction**: Building structured representations of scientific knowledge
- **Contradiction Detection**: Identifying conflicting claims and evidence gaps
- **Novelty Assessment**: Evaluating hypothesis novelty against existing knowledge

### Experimental Design Integration
- **Testability Assessment**: Evaluating experimental feasibility of generated hypotheses
- **Design Optimization**: Optimal experimental design for hypothesis testing
- **Resource Planning**: Cost-benefit analysis of experimental validation
- **Risk Assessment**: Identifying potential experimental risks and limitations

### Collaborative Science
- **Hypothesis Sharing**: Platforms for sharing and refining generated hypotheses
- **Peer Review**: Automated and human review of hypothesis quality
- **Cross-Domain Transfer**: Adapting hypotheses across scientific domains
- **Meta-Analysis**: Synthesizing evidence across multiple studies

## Usage Examples

### Data-Driven Hypothesis Generation
```python
# Initialize hypothesis generator
generator = ScientificHypothesisGenerator(domain="physics")

# Prepare experimental data
data = {
    'temperature': jnp.array([300, 350, 400, 450, 500]),
    'conductivity': jnp.array([1.2, 2.1, 3.8, 6.2, 9.1]),
    'pressure': jnp.array([1.0, 1.5, 2.0, 2.5, 3.0])
}

# Analyze patterns
patterns = generator.analyze_data_patterns(data, {
    'correlation_threshold': 0.7,
    'causal_significance': 0.05
})

# Generate hypotheses
hypotheses = generator.generate_hypotheses_from_patterns(patterns, {
    'max_hypotheses': 5,
    'ranking_weights': {
        'confidence': 0.4,
        'novelty': 0.3,
        'feasibility': 0.3
    }
})

# Display top hypothesis
top_hypothesis = hypotheses[0]
print(f"Hypothesis: {top_hypothesis.statement}")
print(f"Confidence: {top_hypothesis.confidence_score:.2f}")
print(f"Predictions: {top_hypothesis.predictions}")
```

### Experimental Design Generation
```python
# Generate experimental design for top hypothesis
design = generator.generate_experimental_design(top_hypothesis)

print(f"Experimental type: {design['experimental_type']}")
print(f"Sample size: {design['sample_size_recommendation']}")
print(f"Controls: {design['controls']}")
print(f"Statistical analysis: {design['statistical_analysis']}")
```

### Multi-Domain Hypothesis Generation
```python
# Compare hypotheses across domains
domains = ['physics', 'chemistry', 'materials_science']
all_hypotheses = []

for domain in domains:
    domain_generator = ScientificHypothesisGenerator(domain=domain)
    domain_patterns = domain_generator.analyze_data_patterns(data)
    domain_hypotheses = domain_generator.generate_hypotheses_from_patterns(domain_patterns)
    all_hypotheses.extend(domain_hypotheses)

# Cross-domain analysis
print(f"Generated {len(all_hypotheses)} hypotheses across {len(domains)} domains")
```

This expert provides comprehensive scientific hypothesis generation capabilities with data-driven discovery, literature integration, experimental design, and domain-specific reasoning for advancing scientific research.