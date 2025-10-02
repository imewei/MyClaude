"""Special Functions Agent - Special Mathematical Functions Expert.

Capabilities:
- Special functions: Bessel, Legendre, Hermite, Laguerre
- Orthogonal polynomials
- Error functions, Gamma, Beta
- Transforms: FFT, Discrete Cosine/Sine
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from scipy import special
from scipy.fft import fft, ifft, fft2, dct, dst

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


class SpecialFunctionsAgent(ComputationalMethodAgent):
    """Agent for special mathematical functions and transforms."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="SpecialFunctionsAgent",
            version=self.VERSION,
            description="Special functions and transforms",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'numpy.ndarray']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="compute_special_function",
                description="Compute special mathematical functions",
                input_types=["function_type", "arguments"],
                output_types=["function_values"],
                typical_use_cases=["Physics", "Engineering", "Statistics"]
            ),
            Capability(
                name="compute_transform",
                description="Compute mathematical transforms (FFT, DCT)",
                input_types=["transform_type", "data"],
                output_types=["transformed_data"],
                typical_use_cases=["Signal processing", "Image processing"]
            ),
            Capability(
                name="orthogonal_polynomials",
                description="Compute orthogonal polynomials",
                input_types=["polynomial_type", "degree", "points"],
                output_types=["polynomial_values"],
                typical_use_cases=["Numerical methods", "Approximation"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing 'problem_type'")
        elif problem_type not in ['special_function', 'transform', 'orthogonal_polynomial']:
            errors.append(f"Invalid problem_type: {problem_type}")

        if problem_type == 'special_function':
            if 'function_type' not in data:
                errors.append("Missing 'function_type'")
            if 'x' not in data:
                errors.append("Missing 'x' (function argument)")

        elif problem_type == 'transform':
            if 'transform_type' not in data:
                errors.append("Missing 'transform_type'")
            if 'data' not in data:
                errors.append("Missing 'data' for transform")

        elif problem_type == 'orthogonal_polynomial':
            if 'polynomial_type' not in data:
                errors.append("Missing 'polynomial_type'")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        problem_type = data.get('problem_type', 'special_function')
        
        cpu_cores = 1
        memory_gb = 0.5
        estimated_time_sec = 0.1
        environment = ExecutionEnvironment.LOCAL

        # Transforms can be more expensive
        if problem_type == 'transform':
            data_array = data.get('data')
            if data_array is not None and hasattr(data_array, 'size'):
                if data_array.size > 1000000:
                    estimated_time_sec = 5.0
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

            if problem_type == 'special_function':
                comp_result = self._compute_special_function(input_data)
            elif problem_type == 'transform':
                comp_result = self._compute_transform(input_data)
            elif problem_type == 'orthogonal_polynomial':
                comp_result = self._compute_orthogonal_polynomial(input_data)
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

    def _compute_special_function(self, data: Dict[str, Any]) -> ComputationalResult:
        func_type = data['function_type']
        x = np.atleast_1d(data['x'])
        
        # Map function types to scipy.special functions
        func_map = {
            'bessel_j0': special.j0,
            'bessel_j1': special.j1,
            'bessel_y0': special.y0,
            'bessel_y1': special.y1,
            'erf': special.erf,
            'erfc': special.erfc,
            'gamma': special.gamma,
            'beta': lambda x: special.beta(x[0], x[1]) if len(x) == 2 else None,
            'legendre': lambda x: special.legendre(int(x[0]))(x[1:]) if len(x) > 1 else None,
        }

        if func_type not in func_map:
            raise ValueError(f"Unknown function type: {func_type}")

        result = func_map[func_type](x)

        metadata = {
            'function_type': func_type,
            'input_shape': x.shape,
            'output_shape': result.shape if hasattr(result, 'shape') else ()
        }

        return self.create_computational_result(
            solution={'values': result, 'x': x},
            metadata=metadata
        )

    def _compute_transform(self, data: Dict[str, Any]) -> ComputationalResult:
        transform_type = data['transform_type']
        input_data = np.asarray(data['data'])
        
        # Compute transform
        if transform_type == 'fft':
            result = fft(input_data)
        elif transform_type == 'ifft':
            result = ifft(input_data)
        elif transform_type == 'fft2':
            result = fft2(input_data)
        elif transform_type == 'dct':
            result = dct(input_data)
        elif transform_type == 'dst':
            result = dst(input_data)
        else:
            raise ValueError(f"Unknown transform: {transform_type}")

        metadata = {
            'transform_type': transform_type,
            'input_shape': input_data.shape,
            'output_shape': result.shape
        }

        return self.create_computational_result(
            solution={'transformed': result, 'original': input_data},
            metadata=metadata
        )

    def _compute_orthogonal_polynomial(self, data: Dict[str, Any]) -> ComputationalResult:
        poly_type = data['polynomial_type']
        n = data.get('degree', 5)
        x = data.get('x', np.linspace(-1, 1, 100))
        x = np.atleast_1d(x)

        # Compute orthogonal polynomials
        if poly_type == 'legendre':
            poly = special.legendre(n)
            values = poly(x)
        elif poly_type == 'chebyshev':
            poly = special.chebyt(n)
            values = poly(x)
        elif poly_type == 'hermite':
            poly = special.hermite(n)
            values = poly(x)
        elif poly_type == 'laguerre':
            poly = special.laguerre(n)
            values = poly(x)
        else:
            raise ValueError(f"Unknown polynomial: {poly_type}")

        metadata = {
            'polynomial_type': poly_type,
            'degree': n,
            'num_points': len(x)
        }

        return self.create_computational_result(
            solution={'values': values, 'x': x},
            metadata=metadata
        )

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        import uuid
        job_id = f"specfunc_{uuid.uuid4().hex[:8]}"
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
