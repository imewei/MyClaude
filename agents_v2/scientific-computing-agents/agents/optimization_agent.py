"""Optimization Agent - Optimization and Root-Finding Expert.

This agent specializes in solving optimization and root-finding problems
using gradient-based and derivative-free methods.

Capabilities:
- Unconstrained optimization: BFGS, Nelder-Mead, CG, Newton
- Constrained optimization: SLSQP, trust-constr, barrier methods
- Root finding: Newton, bisection, Brent, secant
- Global optimization: Differential evolution, basin hopping
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from scipy.optimize import minimize, root_scalar, differential_evolution

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_computational_method_agent import ComputationalMethodAgent
from base_agent import (
    AgentResult, AgentStatus, ValidationResult, ResourceRequirement,
    AgentMetadata, Capability, ExecutionEnvironment
)
from computational_models import (
    ProblemSpecification, ProblemType, AlgorithmRecommendation,
    ComputationalResult, ConvergenceReport, PerformanceMetrics,
    MethodCategory, NumericalKernel
)
from numerical_kernels.optimization import (
    minimize_bfgs, find_root_newton, line_search_backtracking,
    golden_section_search
)


class OptimizationAgent(ComputationalMethodAgent):
    """Agent for solving optimization and root-finding problems.

    Supports multiple problem types and solution methods:
    - Unconstrained: BFGS, L-BFGS-B, Nelder-Mead, CG, Newton-CG
    - Constrained: SLSQP, trust-constr, COBYLA
    - Root finding: Newton, bisection, Brent, secant
    - Global: Differential evolution, basin hopping, dual annealing
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Optimization agent.

        Args:
            config: Configuration including:
                - backend: 'local', 'hpc', 'cloud'
                - tolerance: relative tolerance (default: 1e-6)
                - max_iterations: max iterations (default: 10000)
                - default_method: default optimization method (default: 'BFGS')
        """
        super().__init__(config)
        self.default_method = config.get('default_method', 'BFGS') if config else 'BFGS'
        self.supported_unconstrained = ['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'CG', 'Newton-CG']
        self.supported_constrained = ['SLSQP', 'trust-constr', 'COBYLA']
        self.supported_global = ['differential_evolution', 'basin_hopping', 'dual_annealing']
        self.supported_root_finding = ['newton', 'bisect', 'brentq', 'secant']

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="OptimizationAgent",
            version=self.VERSION,
            description="Solve optimization and root-finding problems",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'callable']
        )

    def get_capabilities(self) -> List[Capability]:
        """Return list of agent capabilities."""
        return [
            Capability(
                name="minimize_unconstrained",
                description="Minimize unconstrained objective function",
                input_types=["objective_function", "initial_guess", "gradient"],
                output_types=["optimal_point", "optimal_value", "convergence_info"],
                typical_use_cases=[
                    "Parameter estimation",
                    "Curve fitting",
                    "Machine learning training",
                    "Energy minimization"
                ]
            ),
            Capability(
                name="minimize_constrained",
                description="Minimize objective with constraints",
                input_types=["objective", "constraints", "bounds", "initial_guess"],
                output_types=["optimal_point", "optimal_value", "constraint_violations"],
                typical_use_cases=[
                    "Resource allocation",
                    "Portfolio optimization",
                    "Engineering design",
                    "Process optimization"
                ]
            ),
            Capability(
                name="find_root",
                description="Find roots of scalar or vector functions",
                input_types=["function", "initial_guess", "derivative"],
                output_types=["root", "function_value", "iterations"],
                typical_use_cases=[
                    "Equation solving",
                    "Fixed-point problems",
                    "Equilibrium finding",
                    "Inverse problems"
                ]
            ),
            Capability(
                name="global_optimization",
                description="Find global minimum using stochastic methods",
                input_types=["objective", "bounds"],
                output_types=["global_minimum", "function_value"],
                typical_use_cases=[
                    "Non-convex optimization",
                    "Multimodal problems",
                    "Hyperparameter tuning",
                    "Combinatorial optimization"
                ]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data before execution.

        Args:
            data: Input data containing:
                - problem_type: 'optimization_unconstrained', 'optimization_constrained',
                               'root_finding', 'global_optimization'
                - objective: objective function (for optimization)
                - function: function to find root of (for root finding)
                - initial_guess: starting point
                - bounds: variable bounds (optional)
                - constraints: constraint functions (optional)

        Returns:
            ValidationResult with status and errors/warnings
        """
        errors = []
        warnings = []

        # Check problem type
        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing required field: 'problem_type'")
        elif problem_type not in ['optimization_unconstrained', 'optimization_constrained',
                                   'root_finding', 'global_optimization']:
            errors.append(f"Invalid problem_type: {problem_type}")

        # Check function
        if problem_type in ['optimization_unconstrained', 'optimization_constrained', 'global_optimization']:
            if 'objective' not in data and 'objective_function' not in data:
                errors.append("Optimization requires 'objective' or 'objective_function'")
            else:
                obj = data.get('objective') if 'objective' in data else data.get('objective_function')
                if not callable(obj):
                    errors.append("Objective must be callable")

        elif problem_type == 'root_finding':
            if 'function' not in data:
                errors.append("Root finding requires 'function'")
            elif not callable(data['function']):
                errors.append("Function must be callable")

        # Check initial guess (except for global optimization)
        if problem_type not in ['global_optimization']:
            if 'initial_guess' not in data and 'x0' not in data:
                errors.append("Requires 'initial_guess' or 'x0'")

        # Check bounds for global optimization
        if problem_type == 'global_optimization':
            if 'bounds' not in data:
                errors.append("Global optimization requires 'bounds'")

        # Check constraints for constrained optimization
        if problem_type == 'optimization_constrained':
            if 'constraints' not in data:
                warnings.append("Constrained optimization without constraints - will use unconstrained method")

        # Check method if specified
        method = data.get('method')
        if method and problem_type == 'optimization_unconstrained':
            if method not in self.supported_unconstrained:
                warnings.append(f"Method '{method}' not in recommended list: {self.supported_unconstrained}")
        elif method and problem_type == 'optimization_constrained':
            if method not in self.supported_constrained:
                warnings.append(f"Method '{method}' not in recommended list: {self.supported_constrained}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed.

        Args:
            data: Input data

        Returns:
            ResourceRequirement with estimated resources
        """
        problem_type = data.get('problem_type', 'optimization_unconstrained')
        method = data.get('method', self.default_method)

        # Default estimates
        cpu_cores = 1
        memory_gb = 0.5
        estimated_time_sec = 5.0
        environment = ExecutionEnvironment.LOCAL

        # Estimate based on problem dimension
        x0 = data.get('initial_guess') if 'initial_guess' in data else data.get('x0')
        if x0 is not None:
            if hasattr(x0, '__len__'):
                n = len(x0)
            else:
                n = 1

            # Higher dimensions need more time
            if n > 10:
                estimated_time_sec = max(10.0, n * 0.5)
            if n > 100:
                estimated_time_sec = max(30.0, n * 1.0)
                memory_gb = 2.0

        # Global optimization is more expensive
        if problem_type == 'global_optimization':
            estimated_time_sec *= 10
            memory_gb = max(1.0, memory_gb)
            if estimated_time_sec > 60:
                environment = ExecutionEnvironment.HPC

        # Constrained optimization is moderately expensive
        elif problem_type == 'optimization_constrained':
            estimated_time_sec *= 2

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            estimated_time_sec=estimated_time_sec,
            execution_environment=environment
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute optimization or root finding.

        Args:
            input_data: Input data containing:
                - problem_type: type of problem
                - objective/function: objective or constraint function
                - initial_guess: starting point
                - method: solution method (optional)
                - tolerance: numerical tolerance (optional)

        Returns:
            AgentResult with solution and diagnostics

        Example:
            >>> agent = OptimizationAgent()
            >>> result = agent.execute({
            ...     'problem_type': 'optimization_unconstrained',
            ...     'objective': lambda x: (x[0] - 2)**2 + (x[1] - 3)**2,
            ...     'initial_guess': [0, 0],
            ...     'method': 'BFGS'
            ... })
        """
        start_time = datetime.now()

        try:
            # Validate input
            validation = self.validate_input(input_data)
            if not validation.valid:
                return AgentResult(
                    agent_name=self.metadata.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=validation.errors,
                    warnings=validation.warnings
                )

            # Route to appropriate solver
            problem_type = input_data['problem_type']

            if problem_type == 'optimization_unconstrained':
                comp_result = self._minimize_unconstrained(input_data)
            elif problem_type == 'optimization_constrained':
                comp_result = self._minimize_constrained(input_data)
            elif problem_type == 'root_finding':
                comp_result = self._find_root(input_data)
            elif problem_type == 'global_optimization':
                comp_result = self._global_optimization(input_data)
            else:
                raise ValueError(f"Unsupported problem_type: {problem_type}")

            # Wrap in agent result
            return self.wrap_result_in_agent_result(
                comp_result,
                input_data,
                start_time,
                warnings=validation.warnings
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

    def _minimize_unconstrained(self, data: Dict[str, Any]) -> ComputationalResult:
        """Minimize unconstrained objective function.

        Args:
            data: Input data with objective, initial_guess, method

        Returns:
            ComputationalResult with solution
        """
        obj = data.get('objective') if 'objective' in data else data.get('objective_function')
        x0 = data.get('initial_guess') if 'initial_guess' in data else data.get('x0')
        method = data.get('method', self.default_method)
        tol = data.get('tolerance', self.tolerance)
        grad = data.get('gradient')

        # Convert to numpy array
        x0 = np.atleast_1d(x0)

        # Minimize using scipy
        result = minimize(
            obj,
            x0,
            method=method,
            jac=grad,
            tol=tol,
            options={'maxiter': self.max_iterations}
        )

        # Create convergence report
        converged = result.success
        convergence_info = ConvergenceReport(
            converged=converged,
            iterations=result.nit,
            final_residual=float(np.linalg.norm(result.fun)) if hasattr(result.fun, '__len__') else float(result.fun),
            tolerance=tol,
            failure_reason=result.message if not converged else None
        )

        # Create metadata
        metadata = {
            'method': method,
            'initial_value': float(obj(x0)),
            'final_value': float(result.fun),
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'success': result.success,
            'message': result.message
        }

        return self.create_computational_result(
            solution={'x': result.x, 'fun': result.fun},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _minimize_constrained(self, data: Dict[str, Any]) -> ComputationalResult:
        """Minimize constrained objective function.

        Args:
            data: Input data with objective, constraints, bounds, initial_guess

        Returns:
            ComputationalResult with solution
        """
        obj = data.get('objective') if 'objective' in data else data.get('objective_function')
        x0 = data.get('initial_guess') if 'initial_guess' in data else data.get('x0')
        method = data.get('method', 'SLSQP')
        tol = data.get('tolerance', self.tolerance)
        constraints = data.get('constraints', [])
        bounds = data.get('bounds')

        # Convert to numpy array
        x0 = np.atleast_1d(x0)

        # Minimize with constraints
        result = minimize(
            obj,
            x0,
            method=method,
            constraints=constraints,
            bounds=bounds,
            tol=tol,
            options={'maxiter': self.max_iterations}
        )

        # Create convergence report
        converged = result.success
        convergence_info = ConvergenceReport(
            converged=converged,
            iterations=result.nit,
            final_residual=float(result.fun),
            tolerance=tol,
            failure_reason=result.message if not converged else None
        )

        # Create metadata
        metadata = {
            'method': method,
            'final_value': float(result.fun),
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'success': result.success,
            'message': result.message,
            'has_constraints': len(constraints) > 0,
            'has_bounds': bounds is not None
        }

        return self.create_computational_result(
            solution={'x': result.x, 'fun': result.fun},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _find_root(self, data: Dict[str, Any]) -> ComputationalResult:
        """Find root of function.

        Args:
            data: Input data with function, initial_guess

        Returns:
            ComputationalResult with root
        """
        func = data['function']
        method = data.get('method', 'newton')
        tol = data.get('tolerance', self.tolerance)

        # Check if scalar or vector problem
        x0 = data.get('initial_guess') if 'initial_guess' in data else data.get('x0')

        if np.isscalar(x0) or (hasattr(x0, '__len__') and len(x0) == 1):
            # Scalar root finding
            x0_scalar = float(x0) if hasattr(x0, '__len__') else x0

            if method in ['bisect', 'brentq']:
                # Need bracket
                bracket = data.get('bracket', [x0_scalar - 1, x0_scalar + 1])
                result = root_scalar(func, method=method, bracket=bracket, xtol=tol)
            else:
                # Newton-like methods
                fprime = data.get('derivative')
                result = root_scalar(func, method=method, x0=x0_scalar, fprime=fprime, xtol=tol)

            x_root = result.root
            f_root = result.function_calls

        else:
            # Vector root finding (use scipy.optimize.root)
            from scipy.optimize import root
            result = root(func, x0, method='hybr', tol=tol)
            x_root = result.x
            f_root = result.nfev

        # Create convergence report
        converged = result.converged
        convergence_info = ConvergenceReport(
            converged=converged,
            iterations=result.iterations if hasattr(result, 'iterations') else result.function_calls,
            final_residual=float(abs(result.root - x0_scalar)) if np.isscalar(x0) else float(np.linalg.norm(result.x - x0)),
            tolerance=tol,
            failure_reason=None if converged else "Did not converge"
        )

        # Create metadata
        metadata = {
            'method': method,
            'iterations': result.iterations if hasattr(result, 'iterations') else result.function_calls,
            'function_evaluations': f_root,
            'success': result.converged
        }

        return self.create_computational_result(
            solution={'root': x_root},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _global_optimization(self, data: Dict[str, Any]) -> ComputationalResult:
        """Global optimization using stochastic methods.

        Args:
            data: Input data with objective, bounds

        Returns:
            ComputationalResult with global minimum
        """
        obj = data.get('objective') if 'objective' in data else data.get('objective_function')
        bounds = data['bounds']
        method = data.get('method', 'differential_evolution')
        tol = data.get('tolerance', self.tolerance)

        # Global optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                obj,
                bounds,
                tol=tol,
                maxiter=self.max_iterations // 10,  # Typically fewer iterations needed
                workers=1
            )
        else:
            # Fallback to differential evolution
            result = differential_evolution(obj, bounds, tol=tol)

        # Create convergence report
        converged = result.success
        convergence_info = ConvergenceReport(
            converged=converged,
            iterations=result.nit,
            final_residual=float(result.fun),
            tolerance=tol,
            failure_reason=result.message if not converged else None
        )

        # Create metadata
        metadata = {
            'method': method,
            'final_value': float(result.fun),
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'success': result.success,
            'message': result.message
        }

        return self.create_computational_result(
            solution={'x': result.x, 'fun': result.fun},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit calculation to compute backend.

        Args:
            input_data: Calculation input

        Returns:
            Job ID for tracking
        """
        import uuid
        job_id = f"optim_{uuid.uuid4().hex[:8]}"
        if not hasattr(self, '_jobs'):
            self._jobs = {}
        self._jobs[job_id] = {'input': input_data, 'status': 'submitted'}
        return job_id

    def check_status(self, job_id: str) -> AgentStatus:
        """Check calculation status.

        Args:
            job_id: Job identifier

        Returns:
            AgentStatus
        """
        if hasattr(self, '_jobs') and job_id in self._jobs:
            return AgentStatus.PENDING
        return AgentStatus.FAILED

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve calculation results.

        Args:
            job_id: Job identifier

        Returns:
            Calculation results
        """
        if hasattr(self, '_jobs') and job_id in self._jobs:
            return self._jobs[job_id]
        return {}
