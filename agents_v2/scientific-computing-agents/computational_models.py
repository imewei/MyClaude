"""Data models for scientific computing agents.

This module defines specialized data models for computational methods:
- Problem specifications
- Algorithm recommendations
- Computational results
- Convergence diagnostics
- Numerical kernels
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np


# ============================================================================
# Problem Type Taxonomy
# ============================================================================

class ProblemType(Enum):
    """Classification of computational problems."""
    # Differential Equations
    ODE_IVP = "ode_ivp"  # Initial value problem
    ODE_BVP = "ode_bvp"  # Boundary value problem
    ODE_DAE = "ode_dae"  # Differential-algebraic equations
    PDE_ELLIPTIC = "pde_elliptic"  # Laplace, Poisson
    PDE_PARABOLIC = "pde_parabolic"  # Heat equation
    PDE_HYPERBOLIC = "pde_hyperbolic"  # Wave equation
    PDE_MIXED = "pde_mixed"  # Navier-Stokes, etc.

    # Linear Algebra
    LINEAR_SYSTEM_DENSE = "linear_system_dense"
    LINEAR_SYSTEM_SPARSE = "linear_system_sparse"
    EIGENVALUE_PROBLEM = "eigenvalue_problem"
    LEAST_SQUARES = "least_squares"
    MATRIX_FACTORIZATION = "matrix_factorization"

    # Optimization
    OPTIMIZATION_UNCONSTRAINED = "optimization_unconstrained"
    OPTIMIZATION_CONSTRAINED = "optimization_constrained"
    OPTIMIZATION_GLOBAL = "optimization_global"
    ROOT_FINDING = "root_finding"

    # Integration
    INTEGRATION_1D = "integration_1d"
    INTEGRATION_ND = "integration_nd"
    INTEGRATION_PATH = "integration_path"

    # Data-Driven
    REGRESSION = "regression"
    SURROGATE_MODELING = "surrogate_modeling"
    INVERSE_PROBLEM = "inverse_problem"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    PHYSICS_INFORMED_ML = "physics_informed_ml"


class MethodCategory(Enum):
    """Categories of numerical methods."""
    # ODE methods
    EXPLICIT_RK = "explicit_rk"  # Runge-Kutta
    IMPLICIT_BDF = "implicit_bdf"  # Backward differentiation
    MULTISTEP = "multistep"  # Adams-Bashforth, etc.

    # PDE methods
    FINITE_DIFFERENCE = "finite_difference"
    FINITE_ELEMENT = "finite_element"
    FINITE_VOLUME = "finite_volume"
    SPECTRAL = "spectral"

    # Linear solvers
    DIRECT_SOLVER = "direct_solver"  # LU, Cholesky
    ITERATIVE_SOLVER = "iterative_solver"  # GMRES, CG

    # Optimization
    GRADIENT_BASED = "gradient_based"
    DERIVATIVE_FREE = "derivative_free"
    GLOBAL_SEARCH = "global_search"

    # ML methods
    NEURAL_NETWORK = "neural_network"
    GAUSSIAN_PROCESS = "gaussian_process"
    POLYNOMIAL_CHAOS = "polynomial_chaos"


# ============================================================================
# Problem Specification
# ============================================================================

@dataclass
class Constraint:
    """Mathematical constraint."""
    type: str  # 'equality', 'inequality', 'bound'
    expression: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Domain:
    """Spatial or temporal domain."""
    dimensions: int
    bounds: List[tuple]  # [(x_min, x_max), (y_min, y_max), ...]
    discretization: Optional[np.ndarray] = None  # Grid points


@dataclass
class ProblemSpecification:
    """Complete problem description."""
    type: ProblemType
    description: str
    equations: List[str]  # Mathematical expressions
    variables: List[str]
    parameters: Dict[str, float] = field(default_factory=dict)
    boundary_conditions: Optional[Dict[str, Any]] = None
    initial_conditions: Optional[Dict[str, Any]] = None
    constraints: List[Constraint] = field(default_factory=list)
    domain: Optional[Domain] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': self.type.value,
            'description': self.description,
            'equations': self.equations,
            'variables': self.variables,
            'parameters': self.parameters,
            'boundary_conditions': self.boundary_conditions,
            'initial_conditions': self.initial_conditions,
            'constraints': [{'type': c.type, 'expression': c.expression} for c in self.constraints],
            'domain': {
                'dimensions': self.domain.dimensions,
                'bounds': self.domain.bounds
            } if self.domain else None
        }


# ============================================================================
# Algorithm Recommendation
# ============================================================================

@dataclass
class AlgorithmRecommendation:
    """Algorithm selection with metadata."""
    algorithm_name: str
    method_category: MethodCategory
    confidence_score: float  # 0-1
    expected_accuracy: float
    expected_runtime_sec: float
    memory_requirement_mb: float
    stability_properties: Dict[str, bool] = field(default_factory=dict)
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    selection_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'algorithm_name': self.algorithm_name,
            'method_category': self.method_category.value,
            'confidence_score': self.confidence_score,
            'expected_accuracy': self.expected_accuracy,
            'expected_runtime_sec': self.expected_runtime_sec,
            'memory_requirement_mb': self.memory_requirement_mb,
            'stability_properties': self.stability_properties,
            'recommended_parameters': self.recommended_parameters,
            'alternatives': self.alternatives,
            'references': self.references,
            'selection_rationale': self.selection_rationale
        }


# ============================================================================
# Computational Results
# ============================================================================

@dataclass
class ConvergenceReport:
    """Convergence diagnostics."""
    converged: bool
    iterations: int
    final_residual: float
    tolerance: float
    convergence_rate: Optional[float] = None  # Order of convergence
    failure_reason: Optional[str] = None
    iteration_history: List[float] = field(default_factory=list)  # Residual history

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'converged': self.converged,
            'iterations': self.iterations,
            'final_residual': self.final_residual,
            'tolerance': self.tolerance,
            'convergence_rate': self.convergence_rate,
            'failure_reason': self.failure_reason,
            'iteration_count': len(self.iteration_history)
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for computation."""
    wall_time_sec: float
    cpu_time_sec: float
    memory_peak_mb: float
    flops: Optional[float] = None  # Floating-point operations
    efficiency: Optional[float] = None  # Parallel efficiency if applicable

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'wall_time_sec': self.wall_time_sec,
            'cpu_time_sec': self.cpu_time_sec,
            'memory_peak_mb': self.memory_peak_mb,
            'flops': self.flops,
            'efficiency': self.efficiency
        }


@dataclass
class ValidationReport:
    """Validation of computational results."""
    valid: bool
    checks_performed: List[str]
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str] = field(default_factory=list)
    numerical_error: Optional[float] = None  # If analytical solution available

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'valid': self.valid,
            'checks_performed': self.checks_performed,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'warnings': self.warnings,
            'numerical_error': self.numerical_error
        }


@dataclass
class ComputationalResult:
    """Standardized computational output."""
    solution: Union[np.ndarray, float, Dict[str, np.ndarray]]  # Primary solution
    metadata: Dict[str, Any] = field(default_factory=dict)  # Grid points, time steps, etc.
    diagnostics: Dict[str, float] = field(default_factory=dict)  # Residual, error estimate
    convergence_info: Optional[ConvergenceReport] = None
    performance: Optional[PerformanceMetrics] = None
    validation: Optional[ValidationReport] = None
    visualization_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excluding large arrays)."""
        return {
            'solution_shape': self.solution.shape if isinstance(self.solution, np.ndarray) else 'scalar',
            'metadata': self.metadata,
            'diagnostics': self.diagnostics,
            'convergence_info': self.convergence_info.to_dict() if self.convergence_info else None,
            'performance': self.performance.to_dict() if self.performance else None,
            'validation': self.validation.to_dict() if self.validation else None
        }


# ============================================================================
# Numerical Kernels
# ============================================================================

@dataclass
class NumericalKernel:
    """Reusable numerical implementation."""
    name: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    complexity: str = "O(n)"  # Computational complexity
    stability: str = "stable"  # 'A-stable', 'conditionally stable', etc.
    accuracy_order: int = 1  # Order of accuracy
    description: str = ""
    reference: str = ""

    def __call__(self, *args, **kwargs):
        """Call the underlying function."""
        return self.function(*args, **kwargs)


@dataclass
class AlgorithmMetadata:
    """Metadata for numerical algorithms."""
    name: str
    category: MethodCategory
    complexity: str
    stability: str
    accuracy_order: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommended_for: List[str] = field(default_factory=list)
    not_recommended_for: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


# ============================================================================
# Generated Code
# ============================================================================

@dataclass
class GeneratedCode:
    """Code generated by CodeGeneratorAgent."""
    source_code: str
    imports: List[str]
    function_name: str
    docstring: str
    validation_code: str = ""
    test_code: str = ""

    def get_full_code(self) -> str:
        """Get complete executable code."""
        import_str = '\n'.join(self.imports)
        return f"{import_str}\n\n{self.source_code}\n\n{self.validation_code}"


# ============================================================================
# Execution Results
# ============================================================================

@dataclass
class ExecutionResult:
    """Result from ExecutorAgent."""
    success: bool
    output: Any
    runtime_sec: float
    memory_mb: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


# ============================================================================
# Uncertainty Quantification
# ============================================================================

@dataclass
class UncertaintyMetrics:
    """Uncertainty quantification metrics."""
    mean: Union[float, np.ndarray]
    std: Union[float, np.ndarray]
    variance: Union[float, np.ndarray]
    confidence_interval: tuple  # (lower, upper)
    percentiles: Dict[int, float] = field(default_factory=dict)  # {5: val_5, 95: val_95}
    sobol_indices: Optional[Dict[str, float]] = None  # Sensitivity analysis

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'mean': float(self.mean) if isinstance(self.mean, (int, float, np.number)) else 'array',
            'std': float(self.std) if isinstance(self.std, (int, float, np.number)) else 'array',
            'confidence_interval': self.confidence_interval,
            'percentiles': self.percentiles,
            'sobol_indices': self.sobol_indices
        }
