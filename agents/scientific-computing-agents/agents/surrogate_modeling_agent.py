"""Surrogate Modeling Agent - Fast Approximations of Expensive Models.

Capabilities:
- Gaussian Process Regression (GPR)
- Polynomial Chaos Expansion (PCE)
- Kriging interpolation
- Reduced-Order Models (ROM)
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular

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


class SurrogateModelingAgent(ComputationalMethodAgent):
    """Agent for building surrogate models of expensive computational models."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_kernel = config.get('kernel', 'rbf') if config else 'rbf'
        self.default_noise = config.get('noise_level', 1e-6) if config else 1e-6

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="SurrogateModelingAgent",
            version=self.VERSION,
            description="Surrogate modeling for expensive simulations",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'numpy.ndarray']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="gaussian_process",
                description="Gaussian Process Regression for surrogate modeling",
                input_types=["training_x", "training_y", "kernel"],
                output_types=["surrogate_model", "predictions", "uncertainty"],
                typical_use_cases=["Expensive simulations", "Uncertainty quantification"]
            ),
            Capability(
                name="polynomial_chaos",
                description="Polynomial Chaos Expansion for uncertainty",
                input_types=["samples", "polynomial_order"],
                output_types=["pce_coefficients", "sensitivity_indices"],
                typical_use_cases=["Uncertainty propagation", "Sensitivity analysis"]
            ),
            Capability(
                name="kriging",
                description="Kriging interpolation for spatial data",
                input_types=["locations", "values", "variogram"],
                output_types=["interpolated_values", "prediction_variance"],
                typical_use_cases=["Geostatistics", "Spatial prediction"]
            ),
            Capability(
                name="reduced_order_model",
                description="Reduced-Order Models via POD/SVD",
                input_types=["snapshot_matrix", "n_modes"],
                output_types=["reduced_basis", "reconstruction"],
                typical_use_cases=["Large-scale simulations", "Real-time prediction"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing 'problem_type'")
        elif problem_type not in ['gp_regression', 'polynomial_chaos', 'kriging', 'rom']:
            errors.append(f"Invalid problem_type: {problem_type}")

        if problem_type == 'gp_regression':
            if 'training_x' not in data:
                errors.append("Missing 'training_x'")
            if 'training_y' not in data:
                errors.append("Missing 'training_y'")

            if 'training_x' in data and 'training_y' in data:
                x = np.asarray(data['training_x'])
                y = np.asarray(data['training_y'])
                if len(x) != len(y):
                    errors.append("training_x and training_y must have same length")

        elif problem_type == 'polynomial_chaos':
            if 'samples' not in data:
                errors.append("Missing 'samples' data")
            if 'polynomial_order' not in data:
                warnings.append("No polynomial_order specified, using default")

        elif problem_type == 'kriging':
            if 'locations' not in data:
                errors.append("Missing 'locations'")
            if 'values' not in data:
                errors.append("Missing 'values'")

        elif problem_type == 'rom':
            if 'snapshots' not in data:
                errors.append("Missing 'snapshots' matrix")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        problem_type = data.get('problem_type', 'gp_regression')

        cpu_cores = 1
        memory_gb = 1.0
        estimated_time_sec = 1.0
        environment = ExecutionEnvironment.LOCAL

        if problem_type == 'gp_regression':
            n_train = len(data.get('training_x', [])) if 'training_x' in data else 100

            if n_train > 1000:
                cpu_cores = 4
                memory_gb = 4.0
                estimated_time_sec = 30.0
                environment = ExecutionEnvironment.HPC
            elif n_train > 5000:
                cpu_cores = 8
                memory_gb = 16.0
                estimated_time_sec = 120.0
                environment = ExecutionEnvironment.HPC

        elif problem_type == 'rom':
            snapshots = data.get('snapshots', np.zeros((100, 10)))
            if isinstance(snapshots, np.ndarray):
                n_dof, n_snapshots = snapshots.shape
                if n_dof > 10000:
                    memory_gb = 8.0
                    estimated_time_sec = 10.0
                    environment = ExecutionEnvironment.HPC

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

            if problem_type == 'gp_regression':
                comp_result = self._gaussian_process_regression(input_data)
            elif problem_type == 'polynomial_chaos':
                comp_result = self._polynomial_chaos_expansion(input_data)
            elif problem_type == 'kriging':
                comp_result = self._kriging_interpolation(input_data)
            elif problem_type == 'rom':
                comp_result = self._reduced_order_model(input_data)
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

    def _gaussian_process_regression(self, data: Dict[str, Any]) -> ComputationalResult:
        """Gaussian Process Regression for surrogate modeling.

        Args:
            data: Must contain training_x, training_y, optionally test_x, kernel, noise_level
        """
        X_train = np.asarray(data['training_x'])
        y_train = np.asarray(data['training_y']).flatten()

        # Ensure 2D
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        kernel_type = data.get('kernel', self.default_kernel)
        noise_level = data.get('noise_level', self.default_noise)
        length_scale = data.get('length_scale', 1.0)

        # Build kernel matrix K
        K = self._compute_kernel(X_train, X_train, kernel_type, length_scale)
        K += noise_level * np.eye(len(X_train))  # Add noise

        # Cholesky decomposition for numerical stability
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more noise
            K += 1e-3 * np.eye(len(X_train))
            L = cholesky(K, lower=True)

        alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True))

        # Make predictions if test points provided
        predictions = None
        uncertainties = None
        X_test = data.get('test_x')

        if X_test is not None:
            X_test = np.asarray(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)

            # Predict
            K_star = self._compute_kernel(X_test, X_train, kernel_type, length_scale)
            predictions = K_star @ alpha

            # Predictive variance
            v = solve_triangular(L, K_star.T, lower=True)
            K_star_star = self._compute_kernel(X_test, X_test, kernel_type, length_scale)
            uncertainties = np.sqrt(np.diag(K_star_star) - np.sum(v**2, axis=0))

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=0.0,
            tolerance=self.tolerance
        )

        metadata = {
            'kernel': kernel_type,
            'n_training': len(X_train),
            'length_scale': length_scale,
            'noise_level': noise_level,
            'method': 'Gaussian_Process_Regression'
        }

        solution = {
            'K': K,
            'alpha': alpha,
            'L': L,
            'X_train': X_train,
            'y_train': y_train
        }

        if predictions is not None:
            solution['predictions'] = predictions
            solution['uncertainties'] = uncertainties
            solution['X_test'] = X_test

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _polynomial_chaos_expansion(self, data: Dict[str, Any]) -> ComputationalResult:
        """Polynomial Chaos Expansion for uncertainty quantification.

        Args:
            data: Must contain samples (N x d), polynomial_order
        """
        samples = np.asarray(data['samples'])
        polynomial_order = data.get('polynomial_order', 2)

        n_samples, n_dims = samples.shape

        # Generate polynomial basis (using Legendre-like orthogonal polynomials)
        # For simplicity, use tensorized univariate polynomials
        basis = self._generate_polynomial_basis(samples, polynomial_order, n_dims)

        # If function values provided, compute PCE coefficients
        function_values = data.get('function_values')
        coefficients = None

        if function_values is not None:
            function_values = np.asarray(function_values).flatten()
            # Least squares fit
            coefficients = np.linalg.lstsq(basis, function_values, rcond=None)[0]

        # Compute sensitivity indices (Sobol indices)
        sensitivity_indices = None
        if coefficients is not None:
            sensitivity_indices = self._compute_sobol_indices(coefficients, polynomial_order, n_dims)

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=0.0,
            tolerance=self.tolerance
        )

        metadata = {
            'polynomial_order': polynomial_order,
            'n_samples': n_samples,
            'n_dimensions': n_dims,
            'n_coefficients': len(coefficients) if coefficients is not None else 0,
            'method': 'Polynomial_Chaos_Expansion'
        }

        solution = {
            'basis': basis,
            'coefficients': coefficients,
            'sensitivity_indices': sensitivity_indices,
            'samples': samples
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _kriging_interpolation(self, data: Dict[str, Any]) -> ComputationalResult:
        """Kriging (Gaussian Process) interpolation for spatial data.

        Args:
            data: Must contain locations, values, optionally prediction_locations
        """
        locations = np.asarray(data['locations'])
        values = np.asarray(data['values']).flatten()

        if locations.ndim == 1:
            locations = locations.reshape(-1, 1)

        # Use RBF interpolation as a proxy for kriging
        # In practice, kriging uses variogram fitting
        kernel = data.get('kernel', 'thin_plate_spline')

        # Simplified kriging using RBF
        from scipy.interpolate import RBFInterpolator
        interpolator = RBFInterpolator(locations, values, kernel=kernel)

        # Predict at new locations if provided
        prediction_locations = data.get('prediction_locations')
        predictions = None
        prediction_variance = None

        if prediction_locations is not None:
            prediction_locations = np.asarray(prediction_locations)
            if prediction_locations.ndim == 1:
                prediction_locations = prediction_locations.reshape(-1, 1)

            predictions = interpolator(prediction_locations)

            # Estimate prediction variance (simplified)
            distances = cdist(prediction_locations, locations)
            min_distances = np.min(distances, axis=1)
            prediction_variance = min_distances * 0.1  # Heuristic

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=0.0,
            tolerance=self.tolerance
        )

        metadata = {
            'kernel': kernel,
            'n_training_points': len(locations),
            'spatial_dimensions': locations.shape[1],
            'method': 'Kriging_Interpolation'
        }

        solution = {
            'interpolator': interpolator,
            'locations': locations,
            'values': values
        }

        if predictions is not None:
            solution['predictions'] = predictions
            solution['prediction_variance'] = prediction_variance
            solution['prediction_locations'] = prediction_locations

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _reduced_order_model(self, data: Dict[str, Any]) -> ComputationalResult:
        """Reduced-Order Model via Proper Orthogonal Decomposition (POD).

        Args:
            data: Must contain snapshots (n_dof x n_snapshots), optionally n_modes
        """
        snapshots = np.asarray(data['snapshots'])
        n_modes = data.get('n_modes', min(10, min(snapshots.shape)))

        # Center the snapshots
        mean_snapshot = np.mean(snapshots, axis=1, keepdims=True)
        snapshots_centered = snapshots - mean_snapshot

        # SVD for POD
        U, s, Vt = np.linalg.svd(snapshots_centered, full_matrices=False)

        # Keep top n_modes
        U_reduced = U[:, :n_modes]
        s_reduced = s[:n_modes]

        # Reconstruction error
        energy_captured = np.sum(s_reduced**2) / np.sum(s**2) if np.sum(s**2) > 0 else 1.0

        # Project snapshots onto reduced basis
        reduced_coordinates = U_reduced.T @ snapshots_centered

        # Reconstruct
        reconstructed = U_reduced @ reduced_coordinates + mean_snapshot

        # Compute reconstruction error
        reconstruction_error = np.linalg.norm(snapshots - reconstructed, 'fro') / np.linalg.norm(snapshots, 'fro')

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=float(reconstruction_error),
            tolerance=self.tolerance
        )

        metadata = {
            'n_dof': snapshots.shape[0],
            'n_snapshots': snapshots.shape[1],
            'n_modes': n_modes,
            'energy_captured': float(energy_captured),
            'reconstruction_error': float(reconstruction_error),
            'method': 'POD_Reduced_Order_Model'
        }

        solution = {
            'reduced_basis': U_reduced,
            'singular_values': s_reduced,
            'reduced_coordinates': reduced_coordinates,
            'mean_snapshot': mean_snapshot,
            'reconstructed': reconstructed,
            'energy_captured': float(energy_captured)
        }

        return self.create_computational_result(
            solution=solution,
            metadata=metadata,
            convergence_info=convergence_info
        )

    # Helper methods

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray,
                       kernel_type: str, length_scale: float) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if kernel_type == 'rbf' or kernel_type == 'squared_exponential':
            # Squared exponential kernel
            dists = cdist(X1, X2, metric='sqeuclidean')
            K = np.exp(-0.5 * dists / length_scale**2)

        elif kernel_type == 'matern':
            # Matern 5/2 kernel
            dists = cdist(X1, X2, metric='euclidean')
            scaled_dists = np.sqrt(5) * dists / length_scale
            K = (1 + scaled_dists + scaled_dists**2 / 3) * np.exp(-scaled_dists)

        elif kernel_type == 'linear':
            K = X1.astype(float) @ X2.T.astype(float)

        else:
            # Default to RBF
            dists = cdist(X1, X2, metric='sqeuclidean')
            K = np.exp(-0.5 * dists / length_scale**2)

        return K

    def _generate_polynomial_basis(self, samples: np.ndarray,
                                   order: int, n_dims: int) -> np.ndarray:
        """Generate polynomial basis functions.

        For simplicity, uses monomial basis. In practice, use orthogonal polynomials.
        """
        n_samples = samples.shape[0]

        # Generate all multi-indices up to total degree 'order'
        multi_indices = []

        def generate_indices(current, remaining_order, dim):
            if dim == n_dims:
                if remaining_order >= 0:
                    multi_indices.append(current[:])
                return
            for i in range(remaining_order + 1):
                current.append(i)
                generate_indices(current, remaining_order - i, dim + 1)
                current.pop()

        generate_indices([], order, 0)

        # Evaluate basis functions
        n_basis = len(multi_indices)
        basis = np.ones((n_samples, n_basis))

        for i, multi_index in enumerate(multi_indices):
            for d, power in enumerate(multi_index):
                if power > 0:
                    basis[:, i] *= samples[:, d] ** power

        return basis

    def _compute_sobol_indices(self, coefficients: np.ndarray,
                               order: int, n_dims: int) -> Dict[str, float]:
        """Compute Sobol sensitivity indices from PCE coefficients.

        Simplified version: returns first-order indices only.
        """
        # Total variance
        total_variance = np.sum(coefficients[1:]**2)  # Exclude constant term

        if total_variance < 1e-12:
            return {f'S{i+1}': 0.0 for i in range(n_dims)}

        # First-order indices (simplified)
        # In full PCE, need to identify which coefficient corresponds to which variable
        first_order_indices = {}
        for i in range(n_dims):
            # Approximate: use first few coefficients
            if i + 1 < len(coefficients):
                first_order_indices[f'S{i+1}'] = coefficients[i+1]**2 / total_variance
            else:
                first_order_indices[f'S{i+1}'] = 0.0

        return first_order_indices

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        import uuid
        job_id = f"surr_{uuid.uuid4().hex[:8]}"
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
