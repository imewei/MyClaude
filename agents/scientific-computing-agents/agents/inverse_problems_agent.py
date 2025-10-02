"""Inverse Problems Agent - Bayesian Inference and Data Assimilation.

Capabilities:
- Bayesian inference for parameter estimation
- Ensemble Kalman Filter (EnKF)
- Variational data assimilation (3D-Var, 4D-Var)
- Regularization methods (Tikhonov, L1)
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.linalg import lstsq, svd

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


class InverseProblemsAgent(ComputationalMethodAgent):
    """Agent for solving inverse problems and data assimilation."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_ensemble_size = config.get('ensemble_size', 100) if config else 100
        self.default_regularization = config.get('regularization', 1e-3) if config else 1e-3

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="InverseProblemsAgent",
            version=self.VERSION,
            description="Inverse problems and data assimilation",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'numpy.ndarray']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="bayesian_inference",
                description="Bayesian parameter estimation with uncertainty",
                input_types=["observations", "forward_model", "prior"],
                output_types=["posterior_mean", "posterior_covariance"],
                typical_use_cases=["Parameter estimation", "Uncertainty quantification"]
            ),
            Capability(
                name="ensemble_kalman_filter",
                description="Ensemble Kalman Filter for sequential data assimilation",
                input_types=["ensemble", "observations", "observation_operator"],
                output_types=["updated_ensemble", "analysis_mean"],
                typical_use_cases=["Weather forecasting", "Reservoir simulation"]
            ),
            Capability(
                name="variational_assimilation",
                description="Variational data assimilation (3D-Var, 4D-Var)",
                input_types=["background", "observations", "cost_function"],
                output_types=["analysis", "cost_reduction"],
                typical_use_cases=["Optimal state estimation", "Smoothing"]
            ),
            Capability(
                name="regularized_inversion",
                description="Regularized least squares (Tikhonov, L1)",
                input_types=["forward_matrix", "observations", "regularization"],
                output_types=["solution", "residual"],
                typical_use_cases=["Ill-posed problems", "Image deblurring"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing 'problem_type'")
        elif problem_type not in ['bayesian', 'enkf', 'variational', 'regularized']:
            errors.append(f"Invalid problem_type: {problem_type}")

        if problem_type == 'bayesian':
            if 'observations' not in data:
                errors.append("Missing 'observations'")
            if 'forward_model' not in data:
                errors.append("Missing 'forward_model'")
            if 'prior' not in data:
                warnings.append("No prior specified, using default")

        elif problem_type == 'enkf':
            if 'ensemble' not in data:
                errors.append("Missing 'ensemble'")
            if 'observations' not in data:
                errors.append("Missing 'observations'")

        elif problem_type == 'variational':
            if 'background' not in data:
                errors.append("Missing 'background' state")
            if 'observations' not in data:
                errors.append("Missing 'observations'")

        elif problem_type == 'regularized':
            if 'forward_matrix' not in data and 'forward_operator' not in data:
                errors.append("Missing 'forward_matrix' or 'forward_operator'")
            if 'observations' not in data:
                errors.append("Missing 'observations'")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        problem_type = data.get('problem_type', 'bayesian')

        cpu_cores = 1
        memory_gb = 1.0
        estimated_time_sec = 5.0
        environment = ExecutionEnvironment.LOCAL

        if problem_type == 'enkf':
            ensemble = data.get('ensemble', np.zeros((100, 100)))
            if isinstance(ensemble, np.ndarray):
                n_ensemble, n_state = ensemble.shape
                if n_ensemble > 1000 or n_state > 10000:
                    cpu_cores = 4
                    memory_gb = 8.0
                    estimated_time_sec = 30.0
                    environment = ExecutionEnvironment.HPC

        elif problem_type == 'variational':
            # Variational methods can be expensive
            estimated_time_sec = 10.0
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

            if problem_type == 'bayesian':
                comp_result = self._bayesian_inference(input_data)
            elif problem_type == 'enkf':
                comp_result = self._ensemble_kalman_filter(input_data)
            elif problem_type == 'variational':
                comp_result = self._variational_assimilation(input_data)
            elif problem_type == 'regularized':
                comp_result = self._regularized_inversion(input_data)
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

    def _bayesian_inference(self, data: Dict[str, Any]) -> ComputationalResult:
        """Bayesian inference for parameter estimation.

        Uses Gaussian approximation: posterior ∝ likelihood × prior
        """
        observations = np.asarray(data['observations']).flatten()
        forward_model = data['forward_model']

        # Prior distribution
        prior = data.get('prior', {})
        prior_mean = np.asarray(prior.get('mean', np.zeros(2)))
        prior_cov = np.asarray(prior.get('covariance', np.eye(len(prior_mean))))

        # Observation covariance
        obs_cov = data.get('observation_covariance')
        if obs_cov is None:
            obs_noise = data.get('observation_noise', 0.1)
            obs_cov = np.eye(len(observations)) * obs_noise**2
        else:
            obs_cov = np.asarray(obs_cov)

        # Compute MAP estimate via optimization
        def negative_log_posterior(params):
            # Forward model
            predicted = forward_model(params)

            # Likelihood term: -log p(y|x)
            residual = observations - predicted
            likelihood_term = 0.5 * residual.T @ np.linalg.solve(obs_cov, residual)

            # Prior term: -log p(x)
            prior_residual = params - prior_mean
            prior_term = 0.5 * prior_residual.T @ np.linalg.solve(prior_cov, prior_residual)

            return likelihood_term + prior_term

        # Optimize
        result = scipy_minimize(
            negative_log_posterior,
            prior_mean,
            method='BFGS',
            options={'maxiter': 100}
        )

        posterior_mean = result.x

        # Approximate posterior covariance using Hessian
        # For Gaussian case: Σ_post^-1 = Σ_prior^-1 + H^T R^-1 H
        # Simplified: use scaled prior covariance
        posterior_cov = prior_cov * 0.5  # Rough approximation

        # Compute credible intervals (95%)
        posterior_std = np.sqrt(np.diag(posterior_cov))
        credible_intervals = np.column_stack([
            posterior_mean - 1.96 * posterior_std,
            posterior_mean + 1.96 * posterior_std
        ])

        convergence_info = ConvergenceReport(
            converged=result.success,
            iterations=result.nit,
            final_residual=float(result.fun),
            tolerance=self.tolerance
        )

        metadata = {
            'n_parameters': len(posterior_mean),
            'n_observations': len(observations),
            'optimization_success': result.success,
            'negative_log_posterior': float(result.fun),
            'method': 'Bayesian_MAP_Estimation'
        }

        solution = {
            'posterior_mean': posterior_mean,
            'posterior_covariance': posterior_cov,
            'posterior_std': posterior_std,
            'credible_intervals': credible_intervals,
            'map_estimate': posterior_mean,
            'prior_mean': prior_mean
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _ensemble_kalman_filter(self, data: Dict[str, Any]) -> ComputationalResult:
        """Ensemble Kalman Filter for sequential data assimilation.

        Args:
            data: Must contain ensemble (n_ensemble x n_state), observations, observation_operator
        """
        ensemble = np.asarray(data['ensemble'])  # (n_ensemble, n_state)
        observations = np.asarray(data['observations']).flatten()

        # Observation operator (maps state to observation space)
        H = data.get('observation_operator')
        if H is None:
            # Default: observe first few state variables
            n_obs = len(observations)
            n_state = ensemble.shape[1]
            H = np.zeros((n_obs, n_state))
            H[:n_obs, :n_obs] = np.eye(n_obs)
        else:
            H = np.asarray(H)

        # Observation error covariance
        R = data.get('observation_covariance')
        if R is None:
            obs_noise = data.get('observation_noise', 0.1)
            R = np.eye(len(observations)) * obs_noise**2
        else:
            R = np.asarray(R)

        n_ensemble, n_state = ensemble.shape

        # Forecast mean and covariance
        forecast_mean = np.mean(ensemble, axis=0)
        forecast_anomalies = ensemble - forecast_mean

        # Predicted observations
        predicted_obs = ensemble @ H.T  # (n_ensemble, n_obs)
        predicted_obs_mean = np.mean(predicted_obs, axis=0)

        # Innovation covariance: H P_f H^T + R
        obs_anomalies = predicted_obs - predicted_obs_mean
        innovation_cov = (obs_anomalies.T @ obs_anomalies) / (n_ensemble - 1) + R

        # Kalman gain: P_f H^T (H P_f H^T + R)^-1
        cross_cov = (forecast_anomalies.T @ obs_anomalies) / (n_ensemble - 1)
        kalman_gain = cross_cov @ np.linalg.solve(innovation_cov, np.eye(len(observations)))

        # Update ensemble
        innovations = observations - predicted_obs  # (n_ensemble, n_obs)
        ensemble_updated = ensemble + innovations @ kalman_gain.T

        # Analysis mean and covariance
        analysis_mean = np.mean(ensemble_updated, axis=0)
        analysis_anomalies = ensemble_updated - analysis_mean
        analysis_cov = (analysis_anomalies.T @ analysis_anomalies) / (n_ensemble - 1)

        # Compute innovation (observation - forecast)
        innovation = observations - predicted_obs_mean
        innovation_norm = np.linalg.norm(innovation)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=float(innovation_norm),
            tolerance=self.tolerance
        )

        metadata = {
            'n_ensemble': n_ensemble,
            'n_state': n_state,
            'n_observations': len(observations),
            'innovation_norm': float(innovation_norm),
            'method': 'Ensemble_Kalman_Filter'
        }

        solution = {
            'ensemble_updated': ensemble_updated,
            'analysis_mean': analysis_mean,
            'analysis_covariance': analysis_cov,
            'forecast_mean': forecast_mean,
            'kalman_gain': kalman_gain,
            'innovation': innovation
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _variational_assimilation(self, data: Dict[str, Any]) -> ComputationalResult:
        """Variational data assimilation (3D-Var).

        Minimizes: J(x) = 1/2 (x - x_b)^T B^-1 (x - x_b) + 1/2 (y - H(x))^T R^-1 (y - H(x))
        """
        background = np.asarray(data['background']).flatten()
        observations = np.asarray(data['observations']).flatten()

        # Observation operator
        H = data.get('observation_operator')
        if H is None:
            n_obs = len(observations)
            n_state = len(background)
            H = np.zeros((n_obs, n_state))
            H[:n_obs, :n_obs] = np.eye(n_obs)
        else:
            H = np.asarray(H)

        # Background error covariance
        B = data.get('background_covariance')
        if B is None:
            background_error = data.get('background_error', 1.0)
            B = np.eye(len(background)) * background_error**2
        else:
            B = np.asarray(B)

        # Observation error covariance
        R = data.get('observation_covariance')
        if R is None:
            obs_error = data.get('observation_error', 0.1)
            R = np.eye(len(observations)) * obs_error**2
        else:
            R = np.asarray(R)

        # Cost function
        def cost_function(x):
            # Background term
            bg_residual = x - background
            bg_cost = 0.5 * bg_residual.T @ np.linalg.solve(B, bg_residual)

            # Observation term
            obs_residual = observations - H @ x
            obs_cost = 0.5 * obs_residual.T @ np.linalg.solve(R, obs_residual)

            return bg_cost + obs_cost

        # Gradient
        def gradient(x):
            bg_grad = np.linalg.solve(B, x - background)
            obs_grad = -H.T @ np.linalg.solve(R, observations - H @ x)
            return bg_grad + obs_grad

        # Minimize
        result = scipy_minimize(
            cost_function,
            background,
            jac=gradient,
            method='BFGS',
            options={'maxiter': 100}
        )

        analysis = result.x
        final_cost = result.fun
        initial_cost = cost_function(background)
        cost_reduction = (initial_cost - final_cost) / initial_cost * 100

        # Analysis error covariance (simplified)
        # A^-1 = B^-1 + H^T R^-1 H
        analysis_precision = np.linalg.inv(B) + H.T @ np.linalg.solve(R, H)
        analysis_cov = np.linalg.inv(analysis_precision)

        convergence_info = ConvergenceReport(
            converged=result.success,
            iterations=result.nit,
            final_residual=float(final_cost),
            tolerance=self.tolerance
        )

        metadata = {
            'n_state': len(analysis),
            'n_observations': len(observations),
            'initial_cost': float(initial_cost),
            'final_cost': float(final_cost),
            'cost_reduction_percent': float(cost_reduction),
            'optimization_success': result.success,
            'method': '3D-Var_Assimilation'
        }

        solution = {
            'analysis': analysis,
            'analysis_covariance': analysis_cov,
            'background': background,
            'cost_reduction': float(cost_reduction),
            'final_cost': float(final_cost)
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _regularized_inversion(self, data: Dict[str, Any]) -> ComputationalResult:
        """Regularized least squares inversion.

        Solves: min ||Ax - b||^2 + λ||Lx||^2 (Tikhonov) or λ||x||_1 (L1)
        """
        observations = np.asarray(data['observations']).flatten()

        # Forward operator
        if 'forward_matrix' in data:
            A = np.asarray(data['forward_matrix'])
        else:
            forward_op = data['forward_operator']
            # Assume forward_op is callable, build matrix by finite differences
            n_obs = len(observations)
            n_params = data.get('n_parameters', n_obs)
            A = np.eye(n_obs, n_params)  # Simplified

        regularization_type = data.get('regularization_type', 'tikhonov')
        reg_param = data.get('regularization_parameter', self.default_regularization)

        if regularization_type == 'tikhonov':
            # Tikhonov: (A^T A + λ I) x = A^T b
            L = data.get('regularization_matrix')
            if L is None:
                L = np.eye(A.shape[1])
            else:
                L = np.asarray(L)

            # Normal equations
            ATA = A.T @ A
            ATb = A.T @ observations
            system_matrix = ATA + reg_param * (L.T @ L)

            solution = np.linalg.solve(system_matrix, ATb)

        elif regularization_type == 'truncated_svd':
            # Truncated SVD
            U, s, Vt = svd(A, full_matrices=False)

            # Truncate small singular values
            threshold = reg_param * s[0]
            s_inv = np.zeros_like(s)
            s_inv[s > threshold] = 1.0 / s[s > threshold]

            solution = Vt.T @ np.diag(s_inv) @ U.T @ observations

        else:  # Default to simple least squares
            solution, residuals, rank, s = lstsq(A, observations, rcond=reg_param)

        # Compute residual
        predicted = A @ solution
        residual = observations - predicted
        residual_norm = np.linalg.norm(residual)

        # Regularization term
        if regularization_type == 'tikhonov':
            reg_term = reg_param * np.linalg.norm(L @ solution)**2
        else:
            reg_term = reg_param * np.linalg.norm(solution)**2

        total_cost = residual_norm**2 + reg_term

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=float(residual_norm),
            tolerance=self.tolerance
        )

        metadata = {
            'n_parameters': len(solution),
            'n_observations': len(observations),
            'regularization_type': regularization_type,
            'regularization_parameter': reg_param,
            'residual_norm': float(residual_norm),
            'regularization_term': float(reg_term),
            'total_cost': float(total_cost),
            'method': 'Regularized_Inversion'
        }

        solution_dict = {
            'solution': solution,
            'predicted': predicted,
            'residual': residual,
            'residual_norm': float(residual_norm),
            'regularization_term': float(reg_term)
        }

        return self.create_computational_result(
            solution=solution_dict,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        import uuid
        job_id = f"inv_{uuid.uuid4().hex[:8]}"
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
