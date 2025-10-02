"""Problem Analyzer Agent - Intelligent Problem Analysis and Classification.

Capabilities:
- Problem type identification (ODE, PDE, optimization, etc.)
- Complexity estimation (problem size, computational cost)
- Required capability identification
- Resource requirement estimation
- Algorithm recommendation
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_agent import (
    BaseAgent, AgentResult, AgentStatus, ValidationResult, ResourceRequirement,
    AgentMetadata, Capability, ExecutionEnvironment
)


class ProblemType(Enum):
    """Problem type classification."""
    ODE_IVP = "ode_ivp"
    ODE_BVP = "ode_bvp"
    PDE = "pde"
    LINEAR_SYSTEM = "linear_system"
    EIGENVALUE = "eigenvalue"
    OPTIMIZATION = "optimization"
    INTEGRATION = "integration"
    INVERSE_PROBLEM = "inverse_problem"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    SURROGATE_MODELING = "surrogate_modeling"
    UNKNOWN = "unknown"


class ProblemComplexity(Enum):
    """Problem complexity classification."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ProblemAnalyzerAgent(BaseAgent):
    """Agent for analyzing and classifying computational problems."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="ProblemAnalyzerAgent",
            version=self.VERSION,
            description="Intelligent problem analysis and classification",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy'],
            supported_formats=['dict']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="classify_problem",
                description="Identify problem type and characteristics",
                input_types=["problem_description", "data"],
                output_types=["problem_type", "complexity", "characteristics"],
                typical_use_cases=["Automatic problem routing", "Algorithm selection"]
            ),
            Capability(
                name="estimate_complexity",
                description="Estimate computational complexity",
                input_types=["problem_data"],
                output_types=["complexity_class", "estimated_cost"],
                typical_use_cases=["Resource planning", "Feasibility analysis"]
            ),
            Capability(
                name="identify_requirements",
                description="Identify required capabilities and agents",
                input_types=["problem_type", "constraints"],
                output_types=["required_agents", "optional_agents"],
                typical_use_cases=["Workflow planning", "Agent orchestration"]
            ),
            Capability(
                name="recommend_approach",
                description="Recommend computational approach",
                input_types=["problem_analysis"],
                output_types=["recommended_algorithms", "execution_plan"],
                typical_use_cases=["Automatic solver selection", "Workflow design"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        if 'analysis_type' not in data:
            errors.append("Missing 'analysis_type'")

        analysis_type = data.get('analysis_type')
        if analysis_type not in ['classify', 'complexity', 'requirements', 'recommend']:
            errors.append(f"Invalid analysis_type: {analysis_type}")

        if analysis_type == 'classify':
            if 'problem_description' not in data and 'problem_data' not in data:
                warnings.append("No problem description or data provided")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources for problem analysis."""
        # Problem analysis is lightweight - always runs locally
        return ResourceRequirement(
            cpu_cores=1,
            memory_gb=1,
            estimated_runtime_seconds=1,
            recommended_environment=ExecutionEnvironment.LOCAL
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

            analysis_type = input_data['analysis_type']

            if analysis_type == 'classify':
                result = self._classify_problem(input_data)
            elif analysis_type == 'complexity':
                result = self._estimate_complexity(input_data)
            elif analysis_type == 'requirements':
                result = self._identify_requirements(input_data)
            elif analysis_type == 'recommend':
                result = self._recommend_approach(input_data)
            else:
                raise ValueError(f"Unsupported analysis_type: {analysis_type}")

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result,
                metadata={'execution_time_sec': execution_time},
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

    def _classify_problem(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify problem type based on description and data."""
        description = data.get('problem_description', '').lower()
        problem_data = data.get('problem_data', {})

        # Keyword-based classification
        problem_type = ProblemType.UNKNOWN
        confidence = 0.0

        # ODE classification
        if any(kw in description for kw in ['ode', 'differential equation', 'ivp', 'initial value']):
            problem_type = ProblemType.ODE_IVP
            confidence = 0.9
        elif 'bvp' in description or 'boundary value' in description:
            problem_type = ProblemType.ODE_BVP
            confidence = 0.9

        # PDE classification
        elif any(kw in description for kw in ['pde', 'partial differential', 'heat equation', 'wave equation']):
            problem_type = ProblemType.PDE
            confidence = 0.85

        # Linear algebra
        elif any(kw in description for kw in ['linear system', 'solve ax=b', 'matrix solve']):
            problem_type = ProblemType.LINEAR_SYSTEM
            confidence = 0.95
        elif any(kw in description for kw in ['eigenvalue', 'eigenvector', 'spectrum']):
            problem_type = ProblemType.EIGENVALUE
            confidence = 0.9

        # Optimization
        elif any(kw in description for kw in ['optimize', 'minimize', 'maximize', 'objective']):
            problem_type = ProblemType.OPTIMIZATION
            confidence = 0.9

        # Integration
        elif any(kw in description for kw in ['integrate', 'quadrature', 'area', 'volume']):
            problem_type = ProblemType.INTEGRATION
            confidence = 0.85

        # Inverse problems
        elif any(kw in description for kw in ['inverse', 'parameter estimation', 'data assimilation']):
            problem_type = ProblemType.INVERSE_PROBLEM
            confidence = 0.8

        # UQ
        elif any(kw in description for kw in ['uncertainty', 'monte carlo', 'sensitivity', 'stochastic']):
            problem_type = ProblemType.UNCERTAINTY_QUANTIFICATION
            confidence = 0.85

        # Surrogate
        elif any(kw in description for kw in ['surrogate', 'gaussian process', 'emulator', 'reduced order']):
            problem_type = ProblemType.SURROGATE_MODELING
            confidence = 0.8

        # Data-based classification
        if 'matrix_A' in problem_data or 'A' in problem_data:
            if problem_type == ProblemType.UNKNOWN:
                problem_type = ProblemType.LINEAR_SYSTEM
                confidence = 0.7

        if 'objective_function' in problem_data or 'objective' in problem_data:
            if problem_type == ProblemType.UNKNOWN:
                problem_type = ProblemType.OPTIMIZATION
                confidence = 0.7

        # Determine problem characteristics
        characteristics = self._extract_characteristics(problem_data, problem_type)

        return {
            'problem_type': problem_type.value,
            'confidence': float(confidence),
            'characteristics': characteristics,
            'classification_method': 'keyword_and_data_based'
        }

    def _estimate_complexity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate computational complexity."""
        problem_data = data.get('problem_data', {})
        problem_type = data.get('problem_type', ProblemType.UNKNOWN.value)

        complexity = ProblemComplexity.SIMPLE
        estimated_cost = 1.0  # Arbitrary units
        memory_requirement = 'LOW'
        time_requirement = 'FAST'

        # Analyze dimensions
        n_dof = problem_data.get('n_dof', 0)
        n_variables = problem_data.get('n_variables', 0)

        if 'matrix_A' in problem_data:
            A = problem_data['matrix_A']
            if hasattr(A, 'shape'):
                n_dof = A.shape[0]

        # Complexity estimation based on problem type and size
        if problem_type == ProblemType.LINEAR_SYSTEM.value:
            if n_dof == 0:
                complexity = ProblemComplexity.SIMPLE
                estimated_cost = 1.0
            elif n_dof < 100:
                complexity = ProblemComplexity.SIMPLE
                estimated_cost = n_dof**2
            elif n_dof < 1000:
                complexity = ProblemComplexity.MODERATE
                estimated_cost = n_dof**2.5
                memory_requirement = 'MEDIUM'
                time_requirement = 'MODERATE'
            elif n_dof < 10000:
                complexity = ProblemComplexity.COMPLEX
                estimated_cost = n_dof**2.5
                memory_requirement = 'HIGH'
                time_requirement = 'SLOW'
            else:
                complexity = ProblemComplexity.VERY_COMPLEX
                estimated_cost = n_dof**3
                memory_requirement = 'VERY_HIGH'
                time_requirement = 'VERY_SLOW'

        elif problem_type == ProblemType.OPTIMIZATION.value:
            if n_variables < 10:
                complexity = ProblemComplexity.SIMPLE
                estimated_cost = 100
            elif n_variables < 100:
                complexity = ProblemComplexity.MODERATE
                estimated_cost = 1000
                time_requirement = 'MODERATE'
            else:
                complexity = ProblemComplexity.COMPLEX
                estimated_cost = 10000
                time_requirement = 'SLOW'

        elif problem_type == ProblemType.UNCERTAINTY_QUANTIFICATION.value:
            n_samples = problem_data.get('n_samples', 1000)
            if n_samples < 1000:
                complexity = ProblemComplexity.SIMPLE
                estimated_cost = n_samples
            elif n_samples < 10000:
                complexity = ProblemComplexity.MODERATE
                estimated_cost = n_samples * 2
                time_requirement = 'MODERATE'
            else:
                complexity = ProblemComplexity.COMPLEX
                estimated_cost = n_samples * 5
                time_requirement = 'SLOW'

        return {
            'complexity_class': complexity.value,
            'estimated_cost': float(estimated_cost),
            'memory_requirement': memory_requirement,
            'time_requirement': time_requirement,
            'scalability': 'LINEAR' if estimated_cost < 1e6 else 'SUPERLINEAR'
        }

    def _identify_requirements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify required agents and capabilities."""
        problem_type = data.get('problem_type', ProblemType.UNKNOWN.value)

        # Map problem types to required agents
        agent_mapping = {
            ProblemType.ODE_IVP.value: ['ODEPDESolverAgent'],
            ProblemType.ODE_BVP.value: ['ODEPDESolverAgent'],
            ProblemType.PDE.value: ['ODEPDESolverAgent'],
            ProblemType.LINEAR_SYSTEM.value: ['LinearAlgebraAgent'],
            ProblemType.EIGENVALUE.value: ['LinearAlgebraAgent'],
            ProblemType.OPTIMIZATION.value: ['OptimizationAgent'],
            ProblemType.INTEGRATION.value: ['IntegrationAgent'],
            ProblemType.INVERSE_PROBLEM.value: ['InverseProblemsAgent'],
            ProblemType.UNCERTAINTY_QUANTIFICATION.value: ['UncertaintyQuantificationAgent'],
            ProblemType.SURROGATE_MODELING.value: ['SurrogateModelingAgent']
        }

        required_agents = agent_mapping.get(problem_type, [])

        # Optional agents based on problem characteristics
        optional_agents = []

        if data.get('requires_uncertainty', False):
            optional_agents.append('UncertaintyQuantificationAgent')

        if data.get('requires_sensitivity', False):
            optional_agents.append('UncertaintyQuantificationAgent')

        if data.get('expensive_model', False):
            optional_agents.append('SurrogateModelingAgent')

        # Required capabilities
        required_capabilities = []
        if problem_type == ProblemType.LINEAR_SYSTEM.value:
            required_capabilities = ['solve_linear_system']
        elif problem_type == ProblemType.OPTIMIZATION.value:
            required_capabilities = ['minimize', 'optimize']
        elif problem_type == ProblemType.ODE_IVP.value:
            required_capabilities = ['solve_ode_ivp']

        return {
            'required_agents': required_agents,
            'optional_agents': optional_agents,
            'required_capabilities': required_capabilities,
            'estimated_agents_needed': len(required_agents)
        }

    def _recommend_approach(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend computational approach."""
        problem_type = data.get('problem_type', ProblemType.UNKNOWN.value)
        complexity = data.get('complexity_class', ProblemComplexity.SIMPLE.value)

        recommendations = []
        execution_plan = []

        if problem_type == ProblemType.LINEAR_SYSTEM.value:
            if complexity in [ProblemComplexity.SIMPLE.value, ProblemComplexity.MODERATE.value]:
                recommendations.append({
                    'method': 'Direct solver (LU, QR, Cholesky)',
                    'agent': 'LinearAlgebraAgent',
                    'priority': 1,
                    'rationale': 'Problem size suitable for direct methods'
                })
            else:
                recommendations.append({
                    'method': 'Iterative solver (CG, GMRES)',
                    'agent': 'LinearAlgebraAgent',
                    'priority': 1,
                    'rationale': 'Large system benefits from iterative methods'
                })

            execution_plan = [
                {'step': 1, 'action': 'Validate input matrix and vector'},
                {'step': 2, 'action': 'Select appropriate method based on matrix properties'},
                {'step': 3, 'action': 'Solve linear system'},
                {'step': 4, 'action': 'Validate solution accuracy'}
            ]

        elif problem_type == ProblemType.OPTIMIZATION.value:
            recommendations.append({
                'method': 'Gradient-based optimization (BFGS, L-BFGS)',
                'agent': 'OptimizationAgent',
                'priority': 1,
                'rationale': 'Efficient for smooth problems'
            })

            if complexity == ProblemComplexity.COMPLEX.value:
                recommendations.append({
                    'method': 'Global optimization (Differential Evolution)',
                    'agent': 'OptimizationAgent',
                    'priority': 2,
                    'rationale': 'May have multiple local minima'
                })

            execution_plan = [
                {'step': 1, 'action': 'Define objective function'},
                {'step': 2, 'action': 'Choose optimization algorithm'},
                {'step': 3, 'action': 'Run optimization'},
                {'step': 4, 'action': 'Validate optimality conditions'}
            ]

        elif problem_type == ProblemType.UNCERTAINTY_QUANTIFICATION.value:
            recommendations.append({
                'method': 'Monte Carlo sampling',
                'agent': 'UncertaintyQuantificationAgent',
                'priority': 1,
                'rationale': 'Robust and widely applicable'
            })

            recommendations.append({
                'method': 'Latin Hypercube Sampling',
                'agent': 'UncertaintyQuantificationAgent',
                'priority': 2,
                'rationale': 'More efficient than simple MC'
            })

            execution_plan = [
                {'step': 1, 'action': 'Define input distributions'},
                {'step': 2, 'action': 'Generate samples (LHS or MC)'},
                {'step': 3, 'action': 'Evaluate model'},
                {'step': 4, 'action': 'Compute statistics and confidence intervals'}
            ]

        return {
            'recommendations': recommendations,
            'execution_plan': execution_plan,
            'estimated_steps': len(execution_plan)
        }

    def _extract_characteristics(self, problem_data: Dict[str, Any],
                                 problem_type: ProblemType) -> Dict[str, Any]:
        """Extract problem characteristics from data."""
        characteristics = {}

        # Dimension analysis
        if 'matrix_A' in problem_data:
            A = problem_data['matrix_A']
            if hasattr(A, 'shape'):
                characteristics['dimensions'] = A.shape
                characteristics['is_square'] = A.shape[0] == A.shape[1]

        if 'n_variables' in problem_data:
            characteristics['n_variables'] = problem_data['n_variables']

        if 'n_constraints' in problem_data:
            characteristics['n_constraints'] = problem_data['n_constraints']

        # Problem-specific characteristics
        if problem_type == ProblemType.LINEAR_SYSTEM:
            characteristics['problem_class'] = 'linear_algebra'
        elif problem_type == ProblemType.OPTIMIZATION:
            characteristics['problem_class'] = 'optimization'
            characteristics['has_constraints'] = 'n_constraints' in problem_data
        elif problem_type in [ProblemType.ODE_IVP, ProblemType.ODE_BVP]:
            characteristics['problem_class'] = 'differential_equations'

        return characteristics
