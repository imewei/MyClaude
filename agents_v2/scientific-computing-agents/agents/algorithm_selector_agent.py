"""Algorithm Selector Agent - Intelligent Algorithm and Agent Selection.

Capabilities:
- Algorithm selection based on problem type and characteristics
- Agent selection and configuration
- Parameter tuning and optimization
- Multi-agent workflow design
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
from agents.problem_analyzer_agent import ProblemType, ProblemComplexity


class AlgorithmCategory(Enum):
    """Algorithm category classification."""
    DIRECT = "direct"
    ITERATIVE = "iterative"
    GRADIENT_BASED = "gradient_based"
    GRADIENT_FREE = "gradient_free"
    STOCHASTIC = "stochastic"
    DETERMINISTIC = "deterministic"
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"


class AlgorithmSelectorAgent(BaseAgent):
    """Agent for intelligent algorithm and agent selection."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Initialize algorithm database
        self._algorithm_db = self._build_algorithm_database()

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="AlgorithmSelectorAgent",
            version=self.VERSION,
            description="Intelligent algorithm and agent selection for computational problems",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy'],
            supported_formats=['dict']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="select_algorithm",
                description="Select optimal algorithm for problem",
                input_types=["problem_type", "problem_characteristics"],
                output_types=["algorithm", "parameters"],
                typical_use_cases=["Solver selection", "Method optimization"]
            ),
            Capability(
                name="select_agents",
                description="Select required agents for problem",
                input_types=["problem_type", "requirements"],
                output_types=["agent_list", "execution_order"],
                typical_use_cases=["Workflow planning", "Multi-agent coordination"]
            ),
            Capability(
                name="tune_parameters",
                description="Suggest optimal algorithm parameters",
                input_types=["algorithm", "problem_size"],
                output_types=["parameters", "tolerances"],
                typical_use_cases=["Performance optimization", "Convergence tuning"]
            ),
            Capability(
                name="design_workflow",
                description="Design multi-agent computational workflow",
                input_types=["problem_analysis", "resource_constraints"],
                output_types=["workflow", "dependencies"],
                typical_use_cases=["Complex problem solving", "Pipeline automation"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        if 'selection_type' not in data:
            errors.append("Missing 'selection_type'")

        selection_type = data.get('selection_type')
        if selection_type not in ['algorithm', 'agents', 'parameters', 'workflow']:
            errors.append(f"Invalid selection_type: {selection_type}")

        if selection_type in ['algorithm', 'agents', 'workflow']:
            if 'problem_type' not in data:
                warnings.append("No problem_type provided")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources for algorithm selection."""
        # Selection is lightweight - always runs locally
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

            selection_type = input_data['selection_type']

            if selection_type == 'algorithm':
                result = self._select_algorithm(input_data)
            elif selection_type == 'agents':
                result = self._select_agents(input_data)
            elif selection_type == 'parameters':
                result = self._tune_parameters(input_data)
            elif selection_type == 'workflow':
                result = self._design_workflow(input_data)
            else:
                raise ValueError(f"Unsupported selection_type: {selection_type}")

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

    def _select_algorithm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal algorithm based on problem characteristics."""
        problem_type = data.get('problem_type', ProblemType.UNKNOWN.value)
        complexity = data.get('complexity_class', ProblemComplexity.SIMPLE.value)
        characteristics = data.get('characteristics', {})

        # Get candidate algorithms
        candidates = self._algorithm_db.get(problem_type, [])

        if not candidates:
            return {
                'selected_algorithm': None,
                'confidence': 0.0,
                'reason': 'No algorithms available for this problem type'
            }

        # Score algorithms based on problem characteristics
        scored_algorithms = []
        for algo in candidates:
            score = self._score_algorithm(algo, complexity, characteristics)
            scored_algorithms.append((algo, score))

        # Sort by score
        scored_algorithms.sort(key=lambda x: x[1], reverse=True)

        # Select best algorithm
        best_algo, best_score = scored_algorithms[0]

        # Get alternatives
        alternatives = [
            {'algorithm': algo['name'], 'score': score}
            for algo, score in scored_algorithms[1:3]
        ]

        return {
            'selected_algorithm': best_algo['name'],
            'algorithm_type': best_algo['type'],
            'confidence': min(best_score / 100.0, 1.0),
            'rationale': best_algo['rationale'],
            'alternatives': alternatives,
            'expected_performance': self._estimate_performance(best_algo, complexity)
        }

    def _select_agents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Select required agents and determine execution order."""
        problem_type = data.get('problem_type', ProblemType.UNKNOWN.value)
        required_capabilities = data.get('required_capabilities', [])
        optional_requirements = data.get('optional_requirements', [])

        # Agent mapping
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

        primary_agents = agent_mapping.get(problem_type, [])

        # Add supporting agents
        supporting_agents = []
        if 'uncertainty' in optional_requirements:
            supporting_agents.append('UncertaintyQuantificationAgent')
        if 'sensitivity' in optional_requirements:
            supporting_agents.append('UncertaintyQuantificationAgent')
        if 'surrogate' in optional_requirements:
            supporting_agents.append('SurrogateModelingAgent')

        # Remove duplicates
        supporting_agents = list(set(supporting_agents) - set(primary_agents))

        # Determine execution order
        execution_order = self._determine_execution_order(
            primary_agents, supporting_agents, problem_type
        )

        return {
            'primary_agents': primary_agents,
            'supporting_agents': supporting_agents,
            'execution_order': execution_order,
            'total_agents': len(primary_agents) + len(supporting_agents),
            'parallel_capable': len(primary_agents) == 1
        }

    def _tune_parameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal parameters for algorithm."""
        algorithm = data.get('algorithm', '')
        problem_size = data.get('problem_size', 100)
        tolerance = data.get('desired_tolerance', 1e-6)

        parameters = {}

        # Linear algebra parameters
        if 'conjugate gradient' in algorithm.lower() or 'cg' in algorithm.lower():
            parameters = {
                'max_iterations': max(problem_size, 1000),
                'tolerance': tolerance,
                'preconditioner': 'jacobi' if problem_size < 1000 else 'ilu',
                'restart': None
            }
        elif 'gmres' in algorithm.lower():
            parameters = {
                'max_iterations': min(problem_size, 100),
                'tolerance': tolerance,
                'restart': min(30, problem_size // 10),
                'preconditioner': 'ilu'
            }

        # Optimization parameters
        elif 'bfgs' in algorithm.lower():
            parameters = {
                'max_iterations': 1000,
                'tolerance': tolerance,
                'line_search': 'strong_wolfe',
                'history_size': 10 if 'l-bfgs' in algorithm.lower() else None
            }
        elif 'nelder-mead' in algorithm.lower():
            parameters = {
                'max_iterations': problem_size * 200,
                'tolerance': tolerance,
                'adaptive': True
            }

        # ODE solver parameters
        elif 'rk45' in algorithm.lower() or 'runge-kutta' in algorithm.lower():
            parameters = {
                'atol': tolerance,
                'rtol': tolerance * 10,
                'max_step': np.inf,
                'first_step': None
            }

        # Monte Carlo parameters
        elif 'monte carlo' in algorithm.lower():
            parameters = {
                'n_samples': max(1000, int(1.0 / tolerance**2)),
                'seed': None,
                'batch_size': 100
            }

        # Default parameters
        else:
            parameters = {
                'tolerance': tolerance,
                'max_iterations': 1000
            }

        return {
            'algorithm': algorithm,
            'recommended_parameters': parameters,
            'tuning_rationale': self._get_tuning_rationale(algorithm, problem_size),
            'sensitivity': self._get_parameter_sensitivity(algorithm)
        }

    def _design_workflow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Design multi-agent computational workflow."""
        problem_type = data.get('problem_type', ProblemType.UNKNOWN.value)
        complexity = data.get('complexity_class', ProblemComplexity.SIMPLE.value)
        requirements = data.get('requirements', {})

        workflow_steps = []
        dependencies = []

        # Step 1: Problem analysis (always first)
        workflow_steps.append({
            'step': 1,
            'agent': 'ProblemAnalyzerAgent',
            'action': 'classify_and_analyze',
            'inputs': ['problem_description', 'problem_data'],
            'outputs': ['problem_type', 'complexity', 'characteristics']
        })

        # Step 2: Algorithm selection
        workflow_steps.append({
            'step': 2,
            'agent': 'AlgorithmSelectorAgent',
            'action': 'select_algorithm_and_agents',
            'inputs': ['problem_type', 'complexity'],
            'outputs': ['algorithm', 'agents', 'parameters']
        })
        dependencies.append({'step': 2, 'depends_on': [1]})

        # Step 3: Primary computation
        if problem_type == ProblemType.LINEAR_SYSTEM.value:
            workflow_steps.append({
                'step': 3,
                'agent': 'LinearAlgebraAgent',
                'action': 'solve_linear_system',
                'inputs': ['matrix_A', 'vector_b', 'parameters'],
                'outputs': ['solution', 'residual']
            })
        elif problem_type == ProblemType.OPTIMIZATION.value:
            workflow_steps.append({
                'step': 3,
                'agent': 'OptimizationAgent',
                'action': 'minimize',
                'inputs': ['objective', 'x0', 'parameters'],
                'outputs': ['optimal_x', 'optimal_value']
            })
        elif problem_type == ProblemType.UNCERTAINTY_QUANTIFICATION.value:
            workflow_steps.append({
                'step': 3,
                'agent': 'UncertaintyQuantificationAgent',
                'action': 'monte_carlo_sampling',
                'inputs': ['model', 'distributions', 'n_samples'],
                'outputs': ['statistics', 'confidence_intervals']
            })
        else:
            workflow_steps.append({
                'step': 3,
                'agent': 'TBD',
                'action': 'solve',
                'inputs': ['problem_data'],
                'outputs': ['solution']
            })

        dependencies.append({'step': 3, 'depends_on': [2]})

        # Step 4: Validation (always last)
        workflow_steps.append({
            'step': 4,
            'agent': 'ExecutorValidatorAgent',
            'action': 'validate_results',
            'inputs': ['solution', 'problem_data'],
            'outputs': ['validation_report', 'quality_metrics']
        })
        dependencies.append({'step': 4, 'depends_on': [3]})

        # Add optional UQ step if needed
        if requirements.get('uncertainty_quantification', False):
            workflow_steps.insert(-1, {
                'step': len(workflow_steps),
                'agent': 'UncertaintyQuantificationAgent',
                'action': 'sensitivity_analysis',
                'inputs': ['model', 'solution'],
                'outputs': ['sensitivity_indices']
            })
            dependencies.insert(-1, {'step': len(workflow_steps) - 1, 'depends_on': [3]})
            dependencies[-1]['depends_on'] = [len(workflow_steps) - 1]

        return {
            'workflow_steps': workflow_steps,
            'dependencies': dependencies,
            'total_steps': len(workflow_steps),
            'estimated_runtime': self._estimate_workflow_runtime(workflow_steps, complexity),
            'parallel_opportunities': self._identify_parallel_steps(workflow_steps, dependencies)
        }

    def _build_algorithm_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build database of algorithms for different problem types."""
        return {
            ProblemType.LINEAR_SYSTEM.value: [
                {
                    'name': 'LU Decomposition',
                    'type': AlgorithmCategory.DIRECT.value,
                    'complexity': 'O(n^3)',
                    'best_for': ['small_to_medium', 'dense'],
                    'rationale': 'Direct method, accurate for well-conditioned systems',
                    'score_base': 80
                },
                {
                    'name': 'Conjugate Gradient',
                    'type': AlgorithmCategory.ITERATIVE.value,
                    'complexity': 'O(n^2) per iteration',
                    'best_for': ['large', 'sparse', 'spd'],
                    'rationale': 'Efficient iterative method for symmetric positive definite matrices',
                    'score_base': 75
                },
                {
                    'name': 'GMRES',
                    'type': AlgorithmCategory.ITERATIVE.value,
                    'complexity': 'O(n^2) per iteration',
                    'best_for': ['large', 'sparse', 'nonsymmetric'],
                    'rationale': 'General iterative method for nonsymmetric systems',
                    'score_base': 70
                }
            ],
            ProblemType.OPTIMIZATION.value: [
                {
                    'name': 'L-BFGS',
                    'type': AlgorithmCategory.GRADIENT_BASED.value,
                    'complexity': 'O(n) per iteration',
                    'best_for': ['smooth', 'large_scale', 'unconstrained'],
                    'rationale': 'Memory-efficient quasi-Newton method',
                    'score_base': 85
                },
                {
                    'name': 'Nelder-Mead',
                    'type': AlgorithmCategory.GRADIENT_FREE.value,
                    'complexity': 'O(n) evaluations per iteration',
                    'best_for': ['small', 'noisy', 'non_smooth'],
                    'rationale': 'Derivative-free simplex method',
                    'score_base': 65
                },
                {
                    'name': 'Differential Evolution',
                    'type': AlgorithmCategory.STOCHASTIC.value,
                    'complexity': 'O(n*pop_size) per generation',
                    'best_for': ['global', 'multimodal', 'constrained'],
                    'rationale': 'Global optimization with good exploration',
                    'score_base': 70
                }
            ],
            ProblemType.ODE_IVP.value: [
                {
                    'name': 'RK45',
                    'type': AlgorithmCategory.NUMERICAL.value,
                    'complexity': 'Adaptive step size',
                    'best_for': ['smooth', 'moderate_accuracy'],
                    'rationale': 'Adaptive Runge-Kutta, good balance of speed and accuracy',
                    'score_base': 85
                },
                {
                    'name': 'BDF',
                    'type': AlgorithmCategory.NUMERICAL.value,
                    'complexity': 'Adaptive step size',
                    'best_for': ['stiff', 'implicit'],
                    'rationale': 'Backward differentiation for stiff problems',
                    'score_base': 75
                }
            ],
            ProblemType.UNCERTAINTY_QUANTIFICATION.value: [
                {
                    'name': 'Monte Carlo',
                    'type': AlgorithmCategory.STOCHASTIC.value,
                    'complexity': 'O(n_samples)',
                    'best_for': ['general', 'high_dimensional'],
                    'rationale': 'Robust and widely applicable',
                    'score_base': 80
                },
                {
                    'name': 'Latin Hypercube Sampling',
                    'type': AlgorithmCategory.STOCHASTIC.value,
                    'complexity': 'O(n_samples)',
                    'best_for': ['efficient_sampling', 'space_filling'],
                    'rationale': 'More efficient than simple Monte Carlo',
                    'score_base': 85
                },
                {
                    'name': 'Polynomial Chaos Expansion',
                    'type': AlgorithmCategory.ANALYTICAL.value,
                    'complexity': 'O(P) where P is basis size',
                    'best_for': ['smooth_response', 'low_to_medium_dimensional'],
                    'rationale': 'Spectral method with analytical sensitivity',
                    'score_base': 75
                }
            ]
        }

    def _score_algorithm(self, algo: Dict[str, Any], complexity: str,
                         characteristics: Dict[str, Any]) -> float:
        """Score algorithm suitability for problem."""
        score = algo['score_base']

        # Adjust for complexity
        if complexity == ProblemComplexity.SIMPLE.value:
            if algo['type'] == AlgorithmCategory.DIRECT.value:
                score += 10
        elif complexity in [ProblemComplexity.COMPLEX.value, ProblemComplexity.VERY_COMPLEX.value]:
            if algo['type'] == AlgorithmCategory.ITERATIVE.value:
                score += 15

        # Adjust for characteristics
        best_for = algo.get('best_for', [])
        if 'sparse' in characteristics and 'sparse' in best_for:
            score += 20
        if 'symmetric' in characteristics and 'spd' in best_for:
            score += 15

        return score

    def _estimate_performance(self, algo: Dict[str, Any], complexity: str) -> Dict[str, Any]:
        """Estimate algorithm performance."""
        if complexity == ProblemComplexity.SIMPLE.value:
            runtime = 'fast'
            memory = 'low'
        elif complexity == ProblemComplexity.MODERATE.value:
            runtime = 'moderate'
            memory = 'medium'
        else:
            runtime = 'slow'
            memory = 'high'

        return {
            'expected_runtime': runtime,
            'memory_usage': memory,
            'accuracy': 'high' if algo['type'] == AlgorithmCategory.DIRECT.value else 'good',
            'scalability': 'good' if algo['type'] == AlgorithmCategory.ITERATIVE.value else 'limited'
        }

    def _determine_execution_order(self, primary: List[str], supporting: List[str],
                                   problem_type: str) -> List[Dict[str, Any]]:
        """Determine optimal execution order for agents."""
        order = []

        # Primary agents first
        for i, agent in enumerate(primary):
            order.append({
                'order': i + 1,
                'agent': agent,
                'role': 'primary',
                'parallel': False
            })

        # Supporting agents after primary
        for i, agent in enumerate(supporting):
            order.append({
                'order': len(primary) + i + 1,
                'agent': agent,
                'role': 'supporting',
                'parallel': True  # Supporting agents can often run in parallel
            })

        return order

    def _get_tuning_rationale(self, algorithm: str, problem_size: int) -> str:
        """Get rationale for parameter tuning."""
        if 'iterative' in algorithm.lower():
            return f"Parameters tuned for problem size {problem_size} with iterative convergence"
        elif 'monte carlo' in algorithm.lower():
            return "Sample size chosen to achieve desired tolerance with 95% confidence"
        else:
            return "Standard parameters for this algorithm type"

    def _get_parameter_sensitivity(self, algorithm: str) -> Dict[str, str]:
        """Get parameter sensitivity information."""
        return {
            'tolerance': 'high',
            'max_iterations': 'medium',
            'other': 'low'
        }

    def _estimate_workflow_runtime(self, steps: List[Dict[str, Any]], complexity: str) -> str:
        """Estimate total workflow runtime."""
        n_steps = len(steps)

        if complexity == ProblemComplexity.SIMPLE.value:
            return f"{n_steps * 2}-{n_steps * 5} seconds"
        elif complexity == ProblemComplexity.MODERATE.value:
            return f"{n_steps * 5}-{n_steps * 30} seconds"
        else:
            return f"{n_steps * 30} seconds - several minutes"

    def _identify_parallel_steps(self, steps: List[Dict[str, Any]],
                                 dependencies: List[Dict[str, Any]]) -> List[int]:
        """Identify steps that can run in parallel."""
        # Simple heuristic: steps with no dependencies or same dependencies can be parallel
        parallel_candidates = []

        for i, step in enumerate(steps):
            step_num = step['step']
            # Find dependencies for this step
            deps = [d for d in dependencies if d['step'] == step_num]

            if not deps or (deps and len(deps[0]['depends_on']) == 1):
                # Steps with 0 or 1 dependency might be parallelizable
                parallel_candidates.append(step_num)

        return parallel_candidates
