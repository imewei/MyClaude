"""Base class for computational method agents.

This module extends ComputationalAgent for numerical method specialization.
All computational method agents (ODE/PDE solver, optimization, etc.) inherit from this.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from base_agent import (
    ComputationalAgent,
    AgentResult,
    AgentStatus,
    ValidationResult,
    ResourceRequirement,
    Provenance
)
from computational_models import (
    ProblemSpecification,
    AlgorithmRecommendation,
    ComputationalResult,
    ConvergenceReport,
    PerformanceMetrics,
    ValidationReport,
    NumericalKernel
)


class ComputationalMethodAgent(ComputationalAgent):
    """Base class for all computational method agents.

    Extends ComputationalAgent with numerical computing specific features:
    - Numerical validation (NaN/Inf checks, convergence)
    - Performance profiling
    - Result validation against analytical solutions
    - Automatic visualization
    """

    VERSION = "1.0.0"
    DOMAIN = "computational_methods"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize computational method agent.

        Args:
            config: Configuration including:
                - backend: 'local', 'hpc', 'cloud'
                - tolerance: numerical tolerance
                - max_iterations: maximum iterations
                - enable_profiling: performance profiling
        """
        super().__init__(config)
        self.tolerance = config.get('tolerance', 1e-6) if config else 1e-6
        self.max_iterations = config.get('max_iterations', 10000) if config else 10000
        self.enable_profiling = config.get('enable_profiling', True) if config else True
        self.numerical_kernels = {}  # Registry of numerical kernels

    def validate_numerical_output(self, output: Any) -> ValidationReport:
        """Validate numerical output for common issues.

        Args:
            output: Numerical output to validate (scalar, array, or dict)

        Returns:
            ValidationReport with validation results
        """
        checks_performed = []
        checks_passed = []
        checks_failed = []
        warnings = []

        # Check for NaN
        checks_performed.append('nan_check')
        if isinstance(output, np.ndarray):
            if np.any(np.isnan(output)):
                checks_failed.append('nan_check')
                warnings.append(f"Output contains {np.sum(np.isnan(output))} NaN values")
            else:
                checks_passed.append('nan_check')
        elif isinstance(output, (int, float)):
            if np.isnan(output):
                checks_failed.append('nan_check')
                warnings.append("Output is NaN")
            else:
                checks_passed.append('nan_check')
        elif isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, np.ndarray) and np.any(np.isnan(value)):
                    checks_failed.append(f'nan_check_{key}')
                    warnings.append(f"Output['{key}'] contains NaN values")

        # Check for Inf
        checks_performed.append('inf_check')
        if isinstance(output, np.ndarray):
            if np.any(np.isinf(output)):
                checks_failed.append('inf_check')
                warnings.append(f"Output contains {np.sum(np.isinf(output))} Inf values")
            else:
                checks_passed.append('inf_check')
        elif isinstance(output, (int, float)):
            if np.isinf(output):
                checks_failed.append('inf_check')
                warnings.append("Output is Inf")
            else:
                checks_passed.append('inf_check')

        # Check for abnormally large values
        checks_performed.append('magnitude_check')
        if isinstance(output, np.ndarray):
            max_val = np.max(np.abs(output[np.isfinite(output)])) if np.any(np.isfinite(output)) else 0
            if max_val > 1e10:
                warnings.append(f"Output contains very large values (max: {max_val:.2e})")
            checks_passed.append('magnitude_check')

        valid = len(checks_failed) == 0

        return ValidationReport(
            valid=valid,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings
        )

    def check_convergence(
        self,
        residual: float,
        tolerance: Optional[float] = None,
        iteration: Optional[int] = None,
        residual_history: Optional[List[float]] = None
    ) -> ConvergenceReport:
        """Check convergence of iterative method.

        Args:
            residual: Current residual value
            tolerance: Convergence tolerance (default: self.tolerance)
            iteration: Current iteration number
            residual_history: History of residuals for rate calculation

        Returns:
            ConvergenceReport with convergence status
        """
        tol = tolerance if tolerance is not None else self.tolerance
        converged = residual < tol

        # Estimate convergence rate if history available
        convergence_rate = None
        if residual_history and len(residual_history) >= 3:
            # Estimate rate from last 3 iterations
            r = residual_history[-3:]
            if r[1] > 0 and r[0] > 0:
                rate1 = np.log(r[2] / r[1]) / np.log(r[1] / r[0])
                convergence_rate = rate1

        failure_reason = None
        if not converged and iteration and iteration >= self.max_iterations:
            failure_reason = f"Maximum iterations ({self.max_iterations}) reached"
        elif not converged and np.isnan(residual):
            failure_reason = "Residual became NaN"
        elif not converged and np.isinf(residual):
            failure_reason = "Residual became Inf (diverged)"

        return ConvergenceReport(
            converged=converged,
            iterations=iteration if iteration is not None else 0,
            final_residual=residual,
            tolerance=tol,
            convergence_rate=convergence_rate,
            failure_reason=failure_reason,
            iteration_history=residual_history if residual_history else []
        )

    def profile_performance(
        self,
        start_time: datetime,
        end_time: datetime,
        memory_mb: float,
        flops: Optional[float] = None
    ) -> PerformanceMetrics:
        """Profile computational performance.

        Args:
            start_time: Computation start time
            end_time: Computation end time
            memory_mb: Peak memory usage in MB
            flops: Floating-point operations (if available)

        Returns:
            PerformanceMetrics with timing and memory data
        """
        wall_time = (end_time - start_time).total_seconds()

        return PerformanceMetrics(
            wall_time_sec=wall_time,
            cpu_time_sec=wall_time,  # TODO: distinguish wall vs CPU time
            memory_peak_mb=memory_mb,
            flops=flops,
            efficiency=None
        )

    def register_kernel(self, kernel: NumericalKernel):
        """Register a numerical kernel for reuse.

        Args:
            kernel: NumericalKernel to register
        """
        self.numerical_kernels[kernel.name] = kernel

    def get_kernel(self, name: str) -> Optional[NumericalKernel]:
        """Retrieve a registered numerical kernel.

        Args:
            name: Kernel name

        Returns:
            NumericalKernel if found, None otherwise
        """
        return self.numerical_kernels.get(name)

    def create_computational_result(
        self,
        solution: Any,
        metadata: Dict[str, Any],
        convergence_info: Optional[ConvergenceReport] = None,
        performance: Optional[PerformanceMetrics] = None,
        validate_output: bool = True
    ) -> ComputationalResult:
        """Create standardized computational result.

        Args:
            solution: Primary solution (array, scalar, or dict)
            metadata: Metadata (grid points, time steps, etc.)
            convergence_info: Convergence diagnostics
            performance: Performance metrics
            validate_output: Whether to validate numerical output

        Returns:
            ComputationalResult with all data and diagnostics
        """
        # Validate output if requested
        validation = None
        if validate_output:
            validation = self.validate_numerical_output(solution)

        # Compute diagnostics
        diagnostics = {}
        if isinstance(solution, np.ndarray):
            diagnostics['solution_norm'] = float(np.linalg.norm(solution.flatten()))
            diagnostics['solution_min'] = float(np.min(solution))
            diagnostics['solution_max'] = float(np.max(solution))
        elif isinstance(solution, (int, float)):
            diagnostics['solution_value'] = float(solution)

        return ComputationalResult(
            solution=solution,
            metadata=metadata,
            diagnostics=diagnostics,
            convergence_info=convergence_info,
            performance=performance,
            validation=validation
        )

    def wrap_result_in_agent_result(
        self,
        computational_result: ComputationalResult,
        input_data: Dict[str, Any],
        start_time: datetime,
        warnings: Optional[List[str]] = None
    ) -> AgentResult:
        """Wrap ComputationalResult in AgentResult for standardization.

        Args:
            computational_result: ComputationalResult to wrap
            input_data: Original input data
            start_time: Computation start time
            warnings: Any warnings to include

        Returns:
            AgentResult with provenance
        """
        execution_time = (datetime.now() - start_time).total_seconds()

        # Create provenance
        provenance = Provenance(
            agent_name=self.metadata.name,
            agent_version=self.VERSION,
            timestamp=start_time,
            input_hash=self._compute_cache_key(input_data),
            parameters=input_data.get('parameters', {}),
            execution_time_sec=execution_time,
            environment={
                'backend': self.compute_backend,
                'tolerance': self.tolerance,
                'max_iterations': self.max_iterations
            }
        )

        # Determine status
        status = AgentStatus.SUCCESS
        if computational_result.validation and not computational_result.validation.valid:
            status = AgentStatus.FAILED
        elif computational_result.convergence_info and not computational_result.convergence_info.converged:
            status = AgentStatus.FAILED

        # Create data dict with full solution (not just metadata)
        data = {
            'solution': computational_result.solution,
            'metadata': computational_result.metadata,
            'diagnostics': computational_result.diagnostics,
            'convergence_info': computational_result.convergence_info,
            'performance': computational_result.performance,
            'validation': computational_result.validation
        }

        return AgentResult(
            agent_name=self.metadata.name,
            status=status,
            data=data,
            metadata={
                'execution_time_sec': execution_time,
                'backend': self.compute_backend
            },
            warnings=warnings if warnings else [],
            provenance=provenance
        )
