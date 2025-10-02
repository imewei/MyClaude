"""Uncertainty Quantification Agent - Comprehensive UQ Methods.

Capabilities:
- Monte Carlo sampling and estimation
- Latin Hypercube Sampling (LHS)
- Variance-based sensitivity analysis (Sobol indices)
- Confidence intervals and prediction bands
- Risk assessment and rare event estimation
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from scipy import stats
from scipy.stats import qmc

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_computational_method_agent import ComputationalMethodAgent
from base_agent import (
    AgentResult, AgentStatus, ValidationResult, ResourceRequirement,
    AgentMetadata, Capability, ExecutionEnvironment
)
from computational_models import (
    ComputationalResult, ConvergenceReport
)


class UncertaintyQuantificationAgent(ComputationalMethodAgent):
    """Agent for uncertainty quantification and sensitivity analysis."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_n_samples = config.get('n_samples', 1000) if config else 1000
        self.default_confidence_level = config.get('confidence_level', 0.95) if config else 0.95

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="UncertaintyQuantificationAgent",
            version=self.VERSION,
            description="Uncertainty quantification and sensitivity analysis",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'numpy.ndarray', 'callable']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="monte_carlo_sampling",
                description="Monte Carlo uncertainty propagation",
                input_types=["model", "input_distributions", "n_samples"],
                output_types=["samples", "statistics", "confidence_intervals"],
                typical_use_cases=["Uncertainty propagation", "Risk assessment"]
            ),
            Capability(
                name="latin_hypercube_sampling",
                description="Efficient stratified sampling",
                input_types=["bounds", "n_samples", "dimensions"],
                output_types=["lhs_samples"],
                typical_use_cases=["Design of experiments", "Efficient sampling"]
            ),
            Capability(
                name="sensitivity_analysis",
                description="Variance-based sensitivity (Sobol indices)",
                input_types=["model", "input_ranges", "method"],
                output_types=["first_order_indices", "total_order_indices"],
                typical_use_cases=["Input importance ranking", "Model reduction"]
            ),
            Capability(
                name="confidence_intervals",
                description="Statistical confidence and prediction intervals",
                input_types=["samples", "confidence_level"],
                output_types=["mean", "std", "intervals"],
                typical_use_cases=["Uncertainty bounds", "Statistical inference"]
            ),
            Capability(
                name="rare_event_estimation",
                description="Estimate probability of rare events",
                input_types=["model", "threshold", "sampling_method"],
                output_types=["failure_probability", "confidence_bounds"],
                typical_use_cases=["Reliability analysis", "Risk quantification"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing 'problem_type'")
        elif problem_type not in ['monte_carlo', 'lhs', 'sensitivity', 'confidence_interval', 'rare_event']:
            errors.append(f"Invalid problem_type: {problem_type}")

        if problem_type == 'monte_carlo':
            if 'model' not in data:
                errors.append("Missing 'model' function")
            if 'input_distributions' not in data:
                errors.append("Missing 'input_distributions'")

        elif problem_type == 'lhs':
            if 'bounds' not in data and 'n_dimensions' not in data:
                errors.append("Missing 'bounds' or 'n_dimensions'")

        elif problem_type == 'sensitivity':
            if 'model' not in data:
                errors.append("Missing 'model' function")
            if 'input_ranges' not in data:
                warnings.append("No input_ranges specified, using default [-1, 1]")

        elif problem_type == 'confidence_interval':
            if 'samples' not in data and 'data' not in data:
                errors.append("Missing 'samples' or 'data'")

        elif problem_type == 'rare_event':
            if 'model' not in data:
                errors.append("Missing 'model' function")
            if 'threshold' not in data:
                warnings.append("No threshold specified for rare event")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        problem_type = data.get('problem_type', 'monte_carlo')

        cpu_cores = 1
        memory_gb = 1.0
        estimated_time_sec = 5.0
        environment = ExecutionEnvironment.LOCAL

        n_samples = data.get('n_samples', self.default_n_samples)

        if problem_type == 'monte_carlo':
            if n_samples > 10000:
                cpu_cores = 4
                memory_gb = 4.0
                estimated_time_sec = 30.0
                environment = ExecutionEnvironment.HPC

        elif problem_type == 'sensitivity':
            # Sensitivity analysis requires many model evaluations
            n_params = len(data.get('input_ranges', [[0, 1]]))
            total_evals = n_samples * (2 * n_params + 2)  # Saltelli sampling

            if total_evals > 50000:
                cpu_cores = 8
                memory_gb = 8.0
                estimated_time_sec = 60.0
                environment = ExecutionEnvironment.HPC

        elif problem_type == 'rare_event':
            # Rare event estimation may require many samples
            estimated_time_sec = 20.0
            memory_gb = 2.0

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            estimated_time_sec=estimated_time_sec,
            execution_environment=environment
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = datetime.now()

        try:
            validation = self.validate_input(input_data)
            if not validation.valid:
                return AgentResult(
                    agent_name=self.metadata.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=validation.errors,
                    warnings=validation.warnings
                )

            problem_type = input_data['problem_type']

            if problem_type == 'monte_carlo':
                comp_result = self._monte_carlo_sampling(input_data)
            elif problem_type == 'lhs':
                comp_result = self._latin_hypercube_sampling(input_data)
            elif problem_type == 'sensitivity':
                comp_result = self._sensitivity_analysis(input_data)
            elif problem_type == 'confidence_interval':
                comp_result = self._confidence_intervals(input_data)
            elif problem_type == 'rare_event':
                comp_result = self._rare_event_estimation(input_data)
            else:
                raise ValueError(f"Unsupported problem_type: {problem_type}")

            return self.wrap_result_in_agent_result(
                comp_result, input_data, start_time, warnings=validation.warnings
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                metadata={'execution_time_sec': execution_time},
                errors=[f"Execution failed: {str(e)}"]
            )

    def _monte_carlo_sampling(self, data: Dict[str, Any]) -> ComputationalResult:
        """Monte Carlo uncertainty propagation."""
        model = data['model']
        input_distributions = data['input_distributions']
        n_samples = data.get('n_samples', self.default_n_samples)

        # Generate samples from input distributions
        np.random.seed(data.get('seed', None))

        samples = []
        for dist_spec in input_distributions:
            dist_type = dist_spec.get('type', 'normal')

            if dist_type == 'normal':
                mean = dist_spec.get('mean', 0)
                std = dist_spec.get('std', 1)
                samples.append(np.random.normal(mean, std, n_samples))

            elif dist_type == 'uniform':
                low = dist_spec.get('low', 0)
                high = dist_spec.get('high', 1)
                samples.append(np.random.uniform(low, high, n_samples))

            elif dist_type == 'lognormal':
                mean = dist_spec.get('mean', 0)
                sigma = dist_spec.get('sigma', 1)
                samples.append(np.random.lognormal(mean, sigma, n_samples))

            else:
                samples.append(np.random.randn(n_samples))

        input_samples = np.column_stack(samples)

        # Evaluate model
        output_samples = np.array([model(x) for x in input_samples])

        # Compute statistics
        mean = np.mean(output_samples)
        std = np.std(output_samples, ddof=1)
        variance = std**2

        # Confidence intervals
        confidence_level = data.get('confidence_level', self.default_confidence_level)
        alpha = 1 - confidence_level

        # Bootstrap for confidence intervals
        ci_lower = np.percentile(output_samples, 100 * alpha / 2)
        ci_upper = np.percentile(output_samples, 100 * (1 - alpha / 2))

        # Additional statistics
        median = np.median(output_samples)
        skewness = stats.skew(output_samples)
        kurtosis = stats.kurtosis(output_samples)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=n_samples,
            final_residual=std / np.sqrt(n_samples),  # Standard error
            tolerance=self.tolerance
        )

        metadata = {
            'n_samples': n_samples,
            'n_inputs': len(input_distributions),
            'confidence_level': confidence_level,
            'method': 'Monte_Carlo_Sampling'
        }

        solution = {
            'input_samples': input_samples,
            'output_samples': output_samples,
            'mean': float(mean),
            'std': float(std),
            'variance': float(variance),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'median': float(median),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'percentiles': {
                '5': float(np.percentile(output_samples, 5)),
                '25': float(np.percentile(output_samples, 25)),
                '75': float(np.percentile(output_samples, 75)),
                '95': float(np.percentile(output_samples, 95))
            }
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _latin_hypercube_sampling(self, data: Dict[str, Any]) -> ComputationalResult:
        """Latin Hypercube Sampling for efficient space-filling design."""
        n_samples = data.get('n_samples', self.default_n_samples)

        # Get bounds
        if 'bounds' in data:
            bounds = np.asarray(data['bounds'])
            n_dims = len(bounds)
        else:
            n_dims = data.get('n_dimensions', 2)
            bounds = np.array([[0, 1]] * n_dims)

        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=n_dims, seed=data.get('seed'))
        samples_unit = sampler.random(n=n_samples)

        # Scale to bounds
        samples = qmc.scale(samples_unit, bounds[:, 0], bounds[:, 1])

        # Compute space-filling quality (discrepancy)
        discrepancy = qmc.discrepancy(samples_unit)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=n_samples,
            final_residual=float(discrepancy),
            tolerance=self.tolerance
        )

        metadata = {
            'n_samples': n_samples,
            'n_dimensions': n_dims,
            'discrepancy': float(discrepancy),
            'method': 'Latin_Hypercube_Sampling'
        }

        solution = {
            'samples': samples,
            'samples_unit': samples_unit,
            'bounds': bounds,
            'discrepancy': float(discrepancy)
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _sensitivity_analysis(self, data: Dict[str, Any]) -> ComputationalResult:
        """Variance-based sensitivity analysis (Sobol indices)."""
        model = data['model']
        input_ranges = data.get('input_ranges', [[-1, 1], [-1, 1]])
        n_samples = data.get('n_samples', self.default_n_samples)

        input_ranges = np.asarray(input_ranges)
        n_params = len(input_ranges)

        # Saltelli sampling for Sobol indices
        # Generate base samples
        np.random.seed(data.get('seed', None))

        # Sample matrices A and B
        A = np.random.uniform(
            input_ranges[:, 0],
            input_ranges[:, 1],
            size=(n_samples, n_params)
        )
        B = np.random.uniform(
            input_ranges[:, 0],
            input_ranges[:, 1],
            size=(n_samples, n_params)
        )

        # Evaluate model on A and B
        Y_A = np.array([model(x) for x in A])
        Y_B = np.array([model(x) for x in B])

        # Compute C matrices and evaluate
        Y_C = {}
        for i in range(n_params):
            C_i = A.copy()
            C_i[:, i] = B[:, i]
            Y_C[i] = np.array([model(x) for x in C_i])

        # Compute Sobol indices
        f0 = np.mean(np.concatenate([Y_A, Y_B]))
        var_y = np.var(np.concatenate([Y_A, Y_B]), ddof=1)

        first_order = {}
        total_order = {}

        for i in range(n_params):
            # First-order index
            Si = np.mean(Y_B * (Y_C[i] - Y_A)) / var_y if var_y > 1e-10 else 0
            first_order[f'S{i+1}'] = float(Si)

            # Total-order index
            STi = 1 - np.mean(Y_A * (Y_C[i] - Y_B)) / var_y if var_y > 1e-10 else 0
            total_order[f'ST{i+1}'] = float(STi)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=n_samples,
            final_residual=0.0,
            tolerance=self.tolerance
        )

        metadata = {
            'n_samples': n_samples,
            'n_parameters': n_params,
            'total_evaluations': n_samples * (n_params + 2),
            'output_variance': float(var_y),
            'method': 'Sobol_Sensitivity_Analysis'
        }

        solution = {
            'first_order_indices': first_order,
            'total_order_indices': total_order,
            'output_variance': float(var_y),
            'output_mean': float(f0),
            'samples_A': A,
            'samples_B': B,
            'outputs_A': Y_A,
            'outputs_B': Y_B
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _confidence_intervals(self, data: Dict[str, Any]) -> ComputationalResult:
        """Compute confidence intervals and statistical summaries."""
        samples = data.get('samples', data.get('data'))
        samples = np.asarray(samples).flatten()

        confidence_level = data.get('confidence_level', self.default_confidence_level)
        alpha = 1 - confidence_level

        # Basic statistics
        n = len(samples)
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        sem = std / np.sqrt(n)  # Standard error of mean

        # Confidence interval for mean (t-distribution)
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        ci_mean_lower = mean - t_crit * sem
        ci_mean_upper = mean + t_crit * sem

        # Prediction interval (wider, for new observations)
        pi_lower = mean - t_crit * std * np.sqrt(1 + 1/n)
        pi_upper = mean + t_crit * std * np.sqrt(1 + 1/n)

        # Percentile-based intervals
        ci_percentile_lower = np.percentile(samples, 100 * alpha / 2)
        ci_percentile_upper = np.percentile(samples, 100 * (1 - alpha / 2))

        # Additional statistics
        median = np.median(samples)
        iqr = np.percentile(samples, 75) - np.percentile(samples, 25)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=n,
            final_residual=float(sem),
            tolerance=self.tolerance
        )

        metadata = {
            'n_samples': n,
            'confidence_level': confidence_level,
            'method': 'Confidence_Intervals'
        }

        solution = {
            'mean': float(mean),
            'std': float(std),
            'sem': float(sem),
            'median': float(median),
            'iqr': float(iqr),
            'confidence_interval_mean': (float(ci_mean_lower), float(ci_mean_upper)),
            'prediction_interval': (float(pi_lower), float(pi_upper)),
            'percentile_interval': (float(ci_percentile_lower), float(ci_percentile_upper)),
            'samples': samples
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _rare_event_estimation(self, data: Dict[str, Any]) -> ComputationalResult:
        """Estimate probability of rare events."""
        model = data['model']
        threshold = data.get('threshold', 0)
        n_samples = data.get('n_samples', self.default_n_samples)

        # Input distribution (default: standard normal)
        input_dist = data.get('input_distribution', 'normal')
        n_dims = data.get('n_dimensions', 1)

        np.random.seed(data.get('seed', None))

        # Generate samples
        if input_dist == 'normal':
            samples = np.random.randn(n_samples, n_dims)
        elif input_dist == 'uniform':
            samples = np.random.uniform(-1, 1, size=(n_samples, n_dims))
        else:
            samples = np.random.randn(n_samples, n_dims)

        # Evaluate model
        if n_dims == 1:
            outputs = np.array([model(x[0]) for x in samples])
        else:
            outputs = np.array([model(x) for x in samples])

        # Check for rare event (exceeding threshold)
        failures = outputs > threshold
        n_failures = np.sum(failures)

        # Failure probability
        p_failure = n_failures / n_samples

        # Confidence bounds (binomial distribution)
        confidence_level = data.get('confidence_level', self.default_confidence_level)

        if n_failures > 0:
            # Wilson score interval
            z = stats.norm.ppf((1 + confidence_level) / 2)
            denominator = 1 + z**2 / n_samples
            center = (p_failure + z**2 / (2 * n_samples)) / denominator
            margin = z * np.sqrt(p_failure * (1 - p_failure) / n_samples + z**2 / (4 * n_samples**2)) / denominator

            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
        else:
            # No failures observed
            ci_lower = 0.0
            ci_upper = 1 - (1 - confidence_level)**(1 / n_samples)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=n_samples,
            final_residual=float(p_failure),
            tolerance=self.tolerance
        )

        metadata = {
            'n_samples': n_samples,
            'n_failures': int(n_failures),
            'threshold': float(threshold),
            'confidence_level': confidence_level,
            'method': 'Rare_Event_Estimation'
        }

        solution = {
            'failure_probability': float(p_failure),
            'confidence_bounds': (float(ci_lower), float(ci_upper)),
            'n_failures': int(n_failures),
            'failure_samples': samples[failures] if n_failures > 0 else np.array([]),
            'outputs': outputs
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        import uuid
        job_id = f"uq_{uuid.uuid4().hex[:8]}"
        if not hasattr(self, '_jobs'):
            self._jobs = {}
        self._jobs[job_id] = {'input': input_data, 'status': 'submitted'}
        return job_id

    def check_status(self, job_id: str) -> AgentStatus:
        if hasattr(self, '_jobs') and job_id in self._jobs:
            return AgentStatus.PENDING
        return AgentStatus.FAILED

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        if hasattr(self, '_jobs') and job_id in self._jobs:
            return self._jobs[job_id]
        return {}
