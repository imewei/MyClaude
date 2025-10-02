"""Integration Agent - Numerical Integration Expert.

Capabilities:
- 1D integration: Adaptive quadrature, Simpson, Romberg
- Multi-dimensional integration: Monte Carlo, cubature
- Special integrals: Improper, oscillatory, singular
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from scipy.integrate import quad, dblquad, nquad

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
from numerical_kernels.integration import (
    adaptive_quadrature, simpson_rule, monte_carlo_integrate
)


class IntegrationAgent(ComputationalMethodAgent):
    """Agent for numerical integration problems."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_method = config.get('default_method', 'quad') if config else 'quad'

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="IntegrationAgent",
            version=self.VERSION,
            description="Numerical integration and quadrature",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'callable']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="integrate_1d",
                description="1D numerical integration",
                input_types=["function", "bounds"],
                output_types=["integral_value", "error_estimate"],
                typical_use_cases=["Area calculation", "Probability", "Physics"]
            ),
            Capability(
                name="integrate_multidim",
                description="Multi-dimensional integration",
                input_types=["function", "bounds"],
                output_types=["integral_value", "error_estimate"],
                typical_use_cases=["Volume", "Multi-variate probability"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing 'problem_type'")
        elif problem_type not in ['integrate_1d', 'integrate_2d', 'integrate_nd', 'monte_carlo']:
            errors.append(f"Invalid problem_type: {problem_type}")

        if 'function' not in data:
            errors.append("Missing 'function'")
        elif not callable(data['function']):
            errors.append("Function must be callable")

        if 'bounds' not in data:
            errors.append("Missing 'bounds'")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        problem_type = data.get('problem_type', 'integrate_1d')
        
        cpu_cores = 1
        memory_gb = 0.5
        estimated_time_sec = 1.0
        environment = ExecutionEnvironment.LOCAL

        if problem_type in ['integrate_nd', 'monte_carlo']:
            estimated_time_sec = 5.0
            memory_gb = 1.0

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

            if problem_type == 'integrate_1d':
                comp_result = self._integrate_1d(input_data)
            elif problem_type == 'integrate_2d':
                comp_result = self._integrate_2d(input_data)
            elif problem_type in ['integrate_nd', 'monte_carlo']:
                comp_result = self._integrate_nd(input_data)
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

    def _integrate_1d(self, data: Dict[str, Any]) -> ComputationalResult:
        func = data['function']
        bounds = data['bounds']
        tol = data.get('tolerance', self.tolerance)
        
        a, b = bounds
        result, error = quad(func, a, b, epsabs=tol)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=float(error),
            tolerance=tol
        )

        metadata = {
            'bounds': bounds,
            'error_estimate': float(error),
            'method': 'quad'
        }

        return self.create_computational_result(
            solution={'value': float(result), 'error': float(error)},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _integrate_2d(self, data: Dict[str, Any]) -> ComputationalResult:
        func = data['function']
        bounds = data['bounds']
        tol = data.get('tolerance', self.tolerance)
        
        result, error = dblquad(func, bounds[0][0], bounds[0][1], 
                                lambda x: bounds[1][0], lambda x: bounds[1][1],
                                epsabs=tol)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=float(error),
            tolerance=tol
        )

        metadata = {
            'bounds': bounds,
            'error_estimate': float(error),
            'method': 'dblquad'
        }

        return self.create_computational_result(
            solution={'value': float(result), 'error': float(error)},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _integrate_nd(self, data: Dict[str, Any]) -> ComputationalResult:
        func = data['function']
        bounds = data['bounds']
        method = data.get('method', 'monte_carlo')

        if method == 'monte_carlo':
            n_samples = data.get('n_samples', 10000)
            result, error = monte_carlo_integrate(func, bounds, n_samples)
        else:
            result, error = nquad(func, bounds)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=float(error),
            tolerance=self.tolerance
        )

        metadata = {
            'bounds': bounds,
            'error_estimate': float(error),
            'method': method
        }

        return self.create_computational_result(
            solution={'value': float(result), 'error': float(error)},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        import uuid
        job_id = f"integ_{uuid.uuid4().hex[:8]}"
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
