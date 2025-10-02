"""Executor Validator Agent - Multi-Agent Workflow Execution and Validation.

Capabilities:
- Execute multi-agent workflows
- Validate computational results
- Monitor convergence and quality
- Generate comprehensive reports
- Handle agent orchestration
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


class ValidationLevel(Enum):
    """Validation rigor level."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class QualityMetric(Enum):
    """Quality metrics for results."""
    ACCURACY = "accuracy"
    CONVERGENCE = "convergence"
    STABILITY = "stability"
    CONSISTENCY = "consistency"


class ExecutorValidatorAgent(BaseAgent):
    """Agent for executing workflows and validating results."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="ExecutorValidatorAgent",
            version=self.VERSION,
            description="Multi-agent workflow execution and result validation",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy'],
            supported_formats=['dict']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="execute_workflow",
                description="Execute multi-agent computational workflow",
                input_types=["workflow", "problem_data"],
                output_types=["results", "execution_log"],
                typical_use_cases=["Workflow orchestration", "Multi-step computation"]
            ),
            Capability(
                name="validate_solution",
                description="Validate computational solution",
                input_types=["solution", "problem_data", "validation_level"],
                output_types=["validation_report", "quality_metrics"],
                typical_use_cases=["Result verification", "Quality assurance"]
            ),
            Capability(
                name="check_convergence",
                description="Check convergence of iterative methods",
                input_types=["residuals", "tolerance"],
                output_types=["converged", "convergence_rate"],
                typical_use_cases=["Iterative solver monitoring", "Convergence analysis"]
            ),
            Capability(
                name="generate_report",
                description="Generate comprehensive computational report",
                input_types=["results", "validation", "metadata"],
                output_types=["report"],
                typical_use_cases=["Documentation", "Result summarization"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        if 'task_type' not in data:
            errors.append("Missing 'task_type'")

        task_type = data.get('task_type')
        if task_type not in ['execute_workflow', 'validate', 'check_convergence', 'generate_report']:
            errors.append(f"Invalid task_type: {task_type}")

        if task_type == 'validate' and 'solution' not in data:
            warnings.append("No solution provided for validation")

        if task_type == 'check_convergence' and 'residuals' not in data:
            errors.append("Missing 'residuals' for convergence check")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources for execution/validation."""
        task_type = data.get('task_type', 'validate')

        if task_type == 'execute_workflow':
            # Workflow execution might be expensive
            return ResourceRequirement(
                cpu_cores=4,
                memory_gb=8,
                estimated_runtime_seconds=60,
                recommended_environment=ExecutionEnvironment.HPC
            )
        else:
            # Validation is lightweight
            return ResourceRequirement(
                cpu_cores=1,
                memory_gb=2,
                estimated_runtime_seconds=5,
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

            task_type = input_data['task_type']

            if task_type == 'execute_workflow':
                result = self._execute_workflow(input_data)
            elif task_type == 'validate':
                result = self._validate_solution(input_data)
            elif task_type == 'check_convergence':
                result = self._check_convergence(input_data)
            elif task_type == 'generate_report':
                result = self._generate_report(input_data)
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

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

    def _execute_workflow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-agent workflow (mock implementation)."""
        workflow = data.get('workflow', [])
        problem_data = data.get('problem_data', {})

        execution_log = []
        results = {}
        total_runtime = 0.0

        # Simulate workflow execution
        for step in workflow:
            step_num = step.get('step', 0)
            agent = step.get('agent', 'Unknown')
            action = step.get('action', 'unknown')

            # Mock execution
            step_start = datetime.now()
            step_result = {
                'status': 'success',
                'output': f"Result from {agent}",
                'duration': 0.1
            }
            step_duration = (datetime.now() - step_start).total_seconds()

            execution_log.append({
                'step': step_num,
                'agent': agent,
                'action': action,
                'status': step_result['status'],
                'duration_sec': step_duration
            })

            results[f"step_{step_num}"] = step_result
            total_runtime += step_duration

        return {
            'execution_log': execution_log,
            'results': results,
            'total_steps': len(workflow),
            'successful_steps': len([log for log in execution_log if log['status'] == 'success']),
            'total_runtime_sec': total_runtime,
            'workflow_status': 'completed'
        }

    def _validate_solution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate computational solution."""
        solution = data.get('solution')
        problem_data = data.get('problem_data', {})
        validation_level = data.get('validation_level', ValidationLevel.STANDARD.value)

        validation_checks = []

        # Check 1: Solution exists and is valid
        if solution is None:
            validation_checks.append({
                'check': 'solution_exists',
                'passed': False,
                'message': 'No solution provided'
            })
        else:
            validation_checks.append({
                'check': 'solution_exists',
                'passed': True,
                'message': 'Solution provided'
            })

        # Check 2: Solution has correct shape
        if solution is not None and hasattr(solution, 'shape'):
            expected_shape = problem_data.get('expected_shape')
            if expected_shape:
                shape_correct = solution.shape == expected_shape
                validation_checks.append({
                    'check': 'correct_shape',
                    'passed': shape_correct,
                    'message': f"Shape: {solution.shape}, Expected: {expected_shape}"
                })
            else:
                validation_checks.append({
                    'check': 'has_shape',
                    'passed': True,
                    'message': f"Solution shape: {solution.shape}"
                })

        # Check 3: No NaN or Inf values
        if solution is not None:
            try:
                solution_array = np.asarray(solution)
                no_nan = not np.any(np.isnan(solution_array))
                no_inf = not np.any(np.isinf(solution_array))

                validation_checks.append({
                    'check': 'no_nan',
                    'passed': no_nan,
                    'message': 'No NaN values' if no_nan else 'Contains NaN values'
                })
                validation_checks.append({
                    'check': 'no_inf',
                    'passed': no_inf,
                    'message': 'No Inf values' if no_inf else 'Contains Inf values'
                })
            except:
                validation_checks.append({
                    'check': 'numeric_validation',
                    'passed': False,
                    'message': 'Could not validate numeric properties'
                })

        # Check 4: Residual check (if applicable)
        if 'matrix_A' in problem_data and 'vector_b' in problem_data and solution is not None:
            try:
                A = np.asarray(problem_data['matrix_A'])
                b = np.asarray(problem_data['vector_b']).flatten()
                x = np.asarray(solution).flatten()

                residual = np.linalg.norm(A @ x - b)
                relative_residual = residual / np.linalg.norm(b)

                residual_ok = relative_residual < 1e-3

                validation_checks.append({
                    'check': 'residual',
                    'passed': residual_ok,
                    'message': f"Relative residual: {relative_residual:.2e}",
                    'value': float(relative_residual)
                })
            except:
                pass

        # Compute overall quality metrics
        quality_metrics = self._compute_quality_metrics(solution, problem_data, validation_checks)

        # Determine overall validation result
        all_passed = all(check['passed'] for check in validation_checks)

        return {
            'validation_level': validation_level,
            'validation_checks': validation_checks,
            'all_checks_passed': all_passed,
            'quality_metrics': quality_metrics,
            'overall_quality': self._assess_overall_quality(quality_metrics)
        }

    def _check_convergence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check convergence of iterative method."""
        residuals = np.asarray(data.get('residuals', []))
        tolerance = data.get('tolerance', 1e-6)
        max_iterations = data.get('max_iterations', 1000)

        if len(residuals) == 0:
            return {
                'converged': False,
                'reason': 'No residuals provided',
                'final_residual': None
            }

        final_residual = residuals[-1] if len(residuals) > 0 else np.inf
        converged = final_residual < tolerance

        # Estimate convergence rate
        convergence_rate = None
        if len(residuals) > 2:
            # Linear convergence: ||r_{k+1}|| / ||r_k|| ~ constant
            ratios = residuals[1:] / (residuals[:-1] + 1e-16)
            convergence_rate = float(np.mean(ratios[-5:]))  # Average last 5 ratios

        # Determine convergence quality
        if converged:
            iterations_used = len(residuals)
            if iterations_used < max_iterations * 0.1:
                quality = "excellent"
            elif iterations_used < max_iterations * 0.5:
                quality = "good"
            else:
                quality = "acceptable"
        else:
            quality = "failed"

        return {
            'converged': bool(converged),
            'final_residual': float(final_residual),
            'tolerance': float(tolerance),
            'iterations': len(residuals),
            'convergence_rate': convergence_rate,
            'convergence_quality': quality,
            'residual_history': residuals.tolist() if len(residuals) < 100 else None
        }

    def _generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive computational report."""
        results = data.get('results', {})
        validation = data.get('validation', {})
        metadata = data.get('metadata', {})

        report = {
            'report_generated': datetime.now().isoformat(),
            'summary': self._generate_summary(results, validation),
            'results_overview': self._format_results_overview(results),
            'validation_summary': self._format_validation_summary(validation),
            'performance_metrics': self._extract_performance_metrics(metadata),
            'recommendations': self._generate_recommendations(results, validation)
        }

        return report

    def _compute_quality_metrics(self, solution: Any, problem_data: Dict[str, Any],
                                 validation_checks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute quality metrics for solution."""
        metrics = {}

        # Accuracy metric (based on residual if available)
        residual_check = next((c for c in validation_checks if c['check'] == 'residual'), None)
        if residual_check and 'value' in residual_check:
            # Map residual to accuracy score (0-100)
            residual = residual_check['value']
            accuracy = max(0, min(100, 100 * (1 - np.log10(residual + 1e-16) / 10)))
            metrics['accuracy'] = float(accuracy)
        else:
            metrics['accuracy'] = 100.0 if all(c['passed'] for c in validation_checks) else 50.0

        # Consistency metric (all checks passed)
        consistency = 100.0 * sum(c['passed'] for c in validation_checks) / max(len(validation_checks), 1)
        metrics['consistency'] = float(consistency)

        # Stability metric (no NaN/Inf)
        no_nan = any(c['check'] == 'no_nan' and c['passed'] for c in validation_checks)
        no_inf = any(c['check'] == 'no_inf' and c['passed'] for c in validation_checks)
        stability = 100.0 if (no_nan and no_inf) else 0.0
        metrics['stability'] = float(stability)

        return metrics

    def _assess_overall_quality(self, quality_metrics: Dict[str, float]) -> str:
        """Assess overall quality from metrics."""
        if not quality_metrics:
            return "unknown"

        avg_quality = np.mean(list(quality_metrics.values()))

        if avg_quality >= 95:
            return "excellent"
        elif avg_quality >= 80:
            return "good"
        elif avg_quality >= 60:
            return "acceptable"
        else:
            return "poor"

    def _generate_summary(self, results: Dict[str, Any], validation: Dict[str, Any]) -> str:
        """Generate executive summary."""
        if not results:
            return "No results to summarize."

        validation_status = "passed" if validation.get('all_checks_passed', False) else "failed"
        quality = validation.get('overall_quality', 'unknown')

        return f"Computation completed. Validation {validation_status}. Overall quality: {quality}."

    def _format_results_overview(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results overview."""
        return {
            'total_results': len(results),
            'result_keys': list(results.keys())[:10]  # First 10 keys
        }

    def _format_validation_summary(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Format validation summary."""
        if not validation:
            return {'status': 'no_validation_performed'}

        checks = validation.get('validation_checks', [])
        return {
            'total_checks': len(checks),
            'passed_checks': sum(c['passed'] for c in checks),
            'overall_status': 'passed' if validation.get('all_checks_passed', False) else 'failed'
        }

    def _extract_performance_metrics(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics."""
        return {
            'execution_time_sec': metadata.get('execution_time_sec', 0.0),
            'memory_usage_mb': metadata.get('memory_usage_mb', 'N/A')
        }

    def _generate_recommendations(self, results: Dict[str, Any],
                                  validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results and validation."""
        recommendations = []

        if not validation.get('all_checks_passed', True):
            recommendations.append("Review failed validation checks")

        quality = validation.get('overall_quality', 'unknown')
        if quality in ['poor', 'acceptable']:
            recommendations.append("Consider refining parameters or using higher accuracy method")

        quality_metrics = validation.get('quality_metrics', {})
        if quality_metrics.get('accuracy', 100) < 80:
            recommendations.append("Accuracy below 80% - verify problem formulation")

        if not recommendations:
            recommendations.append("Results look good - no major issues detected")

        return recommendations
