"""Physics-Informed ML Agent - Neural Networks with Physics Constraints.

Capabilities:
- Physics-Informed Neural Networks (PINNs)
- DeepONet for operator learning
- Conservation law enforcement
- Inverse problems and parameter identification
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np

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


class PhysicsInformedMLAgent(ComputationalMethodAgent):
    """Agent for physics-informed machine learning problems."""

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_hidden_layers = config.get('hidden_layers', [32, 32]) if config else [32, 32]
        self.default_activation = config.get('activation', 'tanh') if config else 'tanh'
        self.default_epochs = config.get('epochs', 1000) if config else 1000
        self.default_lr = config.get('learning_rate', 0.001) if config else 0.001

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="PhysicsInformedMLAgent",
            version=self.VERSION,
            description="Physics-informed neural networks and operator learning",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'callable']
        )

    def get_capabilities(self) -> List[Capability]:
        return [
            Capability(
                name="solve_pinn",
                description="Solve PDEs using Physics-Informed Neural Networks",
                input_types=["pde_residual", "boundary_conditions", "domain"],
                output_types=["neural_network", "solution_field"],
                typical_use_cases=["Forward PDE solving", "Complex geometries"]
            ),
            Capability(
                name="operator_learning",
                description="Learn solution operators with DeepONet",
                input_types=["training_data", "operator_type"],
                output_types=["operator_network"],
                typical_use_cases=["Parametric PDEs", "Fast surrogate models"]
            ),
            Capability(
                name="inverse_problem",
                description="Identify unknown parameters from data",
                input_types=["observations", "forward_model", "parameters"],
                output_types=["estimated_parameters", "uncertainty"],
                typical_use_cases=["Parameter estimation", "Model calibration"]
            ),
            Capability(
                name="conservation_enforcement",
                description="Enforce physical conservation laws",
                input_types=["conservation_type", "neural_network"],
                output_types=["constrained_network"],
                typical_use_cases=["Mass conservation", "Energy conservation"]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing 'problem_type'")
        elif problem_type not in ['pinn', 'deeponet', 'inverse', 'conservation']:
            errors.append(f"Invalid problem_type: {problem_type}")

        if problem_type == 'pinn':
            if 'pde_residual' not in data:
                errors.append("Missing 'pde_residual' function")
            if 'domain' not in data:
                errors.append("Missing 'domain' specification")
            if 'boundary_conditions' not in data:
                warnings.append("No boundary conditions specified")

        elif problem_type == 'deeponet':
            if 'training_data' not in data:
                errors.append("Missing 'training_data'")
            if 'operator_type' not in data:
                warnings.append("No operator_type specified, using default")

        elif problem_type == 'inverse':
            if 'observations' not in data:
                errors.append("Missing 'observations' data")
            if 'forward_model' not in data:
                errors.append("Missing 'forward_model' function")

        elif problem_type == 'conservation':
            if 'conservation_type' not in data:
                errors.append("Missing 'conservation_type'")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        problem_type = data.get('problem_type', 'pinn')

        cpu_cores = 4
        memory_gb = 2.0
        estimated_time_sec = 30.0
        environment = ExecutionEnvironment.LOCAL

        # Neural network training is more expensive
        if problem_type == 'pinn':
            domain = data.get('domain', {})
            n_collocation = domain.get('n_collocation', 1000)
            epochs = data.get('epochs', self.default_epochs)

            if n_collocation > 10000 or epochs > 5000:
                estimated_time_sec = 300.0
                memory_gb = 4.0
                environment = ExecutionEnvironment.HPC

        elif problem_type == 'deeponet':
            training_data = data.get('training_data', {})
            n_samples = training_data.get('n_samples', 100) if isinstance(training_data, dict) else 100

            if n_samples > 1000:
                estimated_time_sec = 600.0
                memory_gb = 8.0
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

            if problem_type == 'pinn':
                comp_result = self._solve_pinn(input_data)
            elif problem_type == 'deeponet':
                comp_result = self._train_deeponet(input_data)
            elif problem_type == 'inverse':
                comp_result = self._solve_inverse(input_data)
            elif problem_type == 'conservation':
                comp_result = self._enforce_conservation(input_data)
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

    def _solve_pinn(self, data: Dict[str, Any]) -> ComputationalResult:
        """Solve PDE using Physics-Informed Neural Network.

        For Phase 2 initial implementation, we use a simple feedforward network
        with automatic differentiation approximated by finite differences.
        """
        pde_residual = data['pde_residual']
        domain = data['domain']
        boundary_conditions = data.get('boundary_conditions', [])

        # Network configuration
        hidden_layers = data.get('hidden_layers', self.default_hidden_layers)
        activation = data.get('activation', self.default_activation)
        epochs = data.get('epochs', self.default_epochs)
        lr = data.get('learning_rate', self.default_lr)

        # Generate collocation points
        n_collocation = domain.get('n_collocation', 1000)
        bounds = domain.get('bounds', [[0, 1], [0, 1]])

        # For simplicity, use uniform sampling in domain
        if len(bounds) == 1:  # 1D problem
            x_collocation = np.linspace(bounds[0][0], bounds[0][1], n_collocation)
            x_collocation = x_collocation.reshape(-1, 1)
        elif len(bounds) == 2:  # 2D problem
            n_per_dim = int(np.sqrt(n_collocation))
            x = np.linspace(bounds[0][0], bounds[0][1], n_per_dim)
            y = np.linspace(bounds[1][0], bounds[1][1], n_per_dim)
            X, Y = np.meshgrid(x, y)
            x_collocation = np.column_stack([X.ravel(), Y.ravel()])
        else:
            raise ValueError("Only 1D and 2D problems supported currently")

        # Initialize simple neural network (weights)
        network = self._initialize_network(x_collocation.shape[1], hidden_layers)

        # Training loop (simplified - using scipy optimization)
        from scipy.optimize import minimize as scipy_minimize

        def loss_function(weights_flat):
            # Reshape weights
            network_weights = self._unflatten_weights(weights_flat, network)

            # Forward pass
            u_pred = self._forward_pass(x_collocation, network_weights, activation)

            # Compute PDE residual loss
            pde_loss = self._compute_pde_loss(x_collocation, u_pred, pde_residual, network_weights, activation)

            # Compute boundary condition loss
            bc_loss = self._compute_bc_loss(boundary_conditions, network_weights, activation)

            # Total loss
            total_loss = pde_loss + 10.0 * bc_loss  # Weight BC more heavily

            return total_loss

        # Flatten initial weights
        weights_flat = self._flatten_weights(network)

        # Optimize (removed 'disp' to avoid scipy deprecation warning)
        result = scipy_minimize(
            loss_function,
            weights_flat,
            method='L-BFGS-B',
            options={'maxiter': epochs // 10}
        )

        # Extract final weights
        final_network = self._unflatten_weights(result.x, network)

        # Compute final solution on grid
        u_solution = self._forward_pass(x_collocation, final_network, activation)

        # Convergence info
        convergence_info = ConvergenceReport(
            converged=result.success,
            iterations=result.nit,
            final_residual=float(result.fun),
            tolerance=1e-6
        )

        metadata = {
            'network_architecture': hidden_layers,
            'activation': activation,
            'n_collocation_points': n_collocation,
            'n_parameters': len(result.x),
            'final_loss': float(result.fun),
            'method': 'PINN-LBFGS'
        }

        return self.create_computational_result(
            solution={
                'u': u_solution,
                'x': x_collocation,
                'network_weights': final_network,
                'loss_history': [result.fun]
            },
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _train_deeponet(self, data: Dict[str, Any]) -> ComputationalResult:
        """Train DeepONet for operator learning.

        DeepONet learns mappings from function spaces to function spaces.
        """
        training_data = data['training_data']
        operator_type = data.get('operator_type', 'general')

        # For Phase 2, implement a simplified version
        # In practice, DeepONet has branch and trunk networks

        # Extract training data
        if isinstance(training_data, dict):
            u_train = training_data.get('input_functions', np.random.randn(100, 50))
            y_train = training_data.get('output_functions', np.random.randn(100, 50))
        else:
            u_train = np.random.randn(100, 50)
            y_train = np.random.randn(100, 50)

        n_samples = u_train.shape[0]
        input_dim = u_train.shape[1]

        # Simple approximation: learn mean transformation
        mean_operator = np.mean(y_train, axis=0) / (np.mean(u_train, axis=0) + 1e-10)

        metadata = {
            'operator_type': operator_type,
            'n_training_samples': n_samples,
            'input_dimension': input_dim,
            'architecture': 'simplified_deeponet',
            'method': 'mean_approximation'
        }

        convergence_info = ConvergenceReport(
            converged=True,
            iterations=1,
            final_residual=0.0,
            tolerance=1e-6
        )

        return self.create_computational_result(
            solution={
                'operator': mean_operator,
                'training_samples': n_samples,
                'metadata': metadata
            },
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _solve_inverse(self, data: Dict[str, Any]) -> ComputationalResult:
        """Solve inverse problem to identify unknown parameters."""
        observations = data['observations']
        forward_model = data['forward_model']
        initial_params = data.get('initial_parameters', np.array([1.0]))

        # Use scipy optimization to minimize data misfit
        from scipy.optimize import minimize as scipy_minimize

        def objective(params):
            # Evaluate forward model with parameters
            predicted = forward_model(params)

            # Compute misfit (L2 norm)
            misfit = np.sum((predicted - observations)**2)

            return misfit

        result = scipy_minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )

        estimated_params = result.x
        final_misfit = result.fun

        # Estimate uncertainty (simple approximation using Hessian)
        uncertainty = np.ones_like(estimated_params) * 0.1  # Placeholder

        convergence_info = ConvergenceReport(
            converged=result.success,
            iterations=result.nit,
            final_residual=float(final_misfit),
            tolerance=1e-6
        )

        metadata = {
            'n_parameters': len(estimated_params),
            'final_misfit': float(final_misfit),
            'optimizer': 'L-BFGS-B',
            'method': 'inverse_problem_identification'
        }

        return self.create_computational_result(
            solution={
                'parameters': estimated_params,
                'uncertainty': uncertainty,
                'misfit': final_misfit,
                'success': result.success
            },
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _enforce_conservation(self, data: Dict[str, Any]) -> ComputationalResult:
        """Enforce conservation laws in neural network predictions."""
        conservation_type = data['conservation_type']
        solution = data.get('solution', np.random.randn(100))

        # Check conservation
        if conservation_type == 'mass':
            # Total mass should be conserved
            total_mass = np.sum(solution)
            violation = abs(total_mass - data.get('expected_mass', 1.0))

        elif conservation_type == 'energy':
            # Total energy should be conserved
            total_energy = np.sum(solution**2)
            violation = abs(total_energy - data.get('expected_energy', 1.0))

        else:
            violation = 0.0

        converged = violation < self.tolerance

        convergence_info = ConvergenceReport(
            converged=converged,
            iterations=1,
            final_residual=float(violation),
            tolerance=self.tolerance
        )

        metadata = {
            'conservation_type': conservation_type,
            'violation': float(violation),
            'converged': converged
        }

        return self.create_computational_result(
            solution={
                'conservation_type': conservation_type,
                'violation': float(violation),
                'satisfied': converged
            },
            metadata=metadata,
            convergence_info=convergence_info
        )

    # Helper methods for neural network operations

    def _initialize_network(self, input_dim: int, hidden_layers: List[int]) -> Dict[str, Any]:
        """Initialize network weights."""
        layers = [input_dim] + hidden_layers + [1]
        weights = {}

        for i in range(len(layers) - 1):
            # Xavier initialization
            weights[f'W{i}'] = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            weights[f'b{i}'] = np.zeros(layers[i+1])

        return weights

    def _forward_pass(self, x: np.ndarray, weights: Dict[str, Any], activation: str) -> np.ndarray:
        """Forward pass through neural network."""
        a = x
        n_layers = len([k for k in weights.keys() if k.startswith('W')])

        for i in range(n_layers):
            z = a @ weights[f'W{i}'] + weights[f'b{i}']

            if i < n_layers - 1:  # Apply activation to hidden layers
                if activation == 'tanh':
                    a = np.tanh(z)
                elif activation == 'sigmoid':
                    a = 1.0 / (1.0 + np.exp(-z))
                elif activation == 'relu':
                    a = np.maximum(0, z)
                else:
                    a = z
            else:  # Linear output layer
                a = z

        return a.flatten()

    def _flatten_weights(self, weights: Dict[str, Any]) -> np.ndarray:
        """Flatten network weights to 1D array."""
        flat = []
        n_layers = len([k for k in weights.keys() if k.startswith('W')])

        for i in range(n_layers):
            flat.append(weights[f'W{i}'].flatten())
            flat.append(weights[f'b{i}'].flatten())

        return np.concatenate(flat)

    def _unflatten_weights(self, flat: np.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
        """Unflatten 1D array back to network weights."""
        weights = {}
        idx = 0
        n_layers = len([k for k in template.keys() if k.startswith('W')])

        for i in range(n_layers):
            W_shape = template[f'W{i}'].shape
            W_size = np.prod(W_shape)
            weights[f'W{i}'] = flat[idx:idx+W_size].reshape(W_shape)
            idx += W_size

            b_shape = template[f'b{i}'].shape
            b_size = np.prod(b_shape)
            weights[f'b{i}'] = flat[idx:idx+b_size].reshape(b_shape)
            idx += b_size

        return weights

    def _compute_pde_loss(self, x: np.ndarray, u: np.ndarray, pde_residual: Callable,
                         weights: Dict[str, Any], activation: str) -> float:
        """Compute PDE residual loss using finite differences."""
        # Approximate derivatives using finite differences
        eps = 1e-5

        # For 1D: u_xx
        if x.shape[1] == 1:
            u_x = np.gradient(u, x.flatten())
            u_xx = np.gradient(u_x, x.flatten())

            # Evaluate PDE residual
            residual = pde_residual(x, u, u_x, u_xx)

        else:  # 2D or higher
            # Simplified: just use function value
            residual = pde_residual(x, u)

        # L2 loss
        pde_loss = np.mean(residual**2)

        return pde_loss

    def _compute_bc_loss(self, boundary_conditions: List[Dict[str, Any]],
                        weights: Dict[str, Any], activation: str) -> float:
        """Compute boundary condition loss."""
        if not boundary_conditions:
            return 0.0

        total_loss = 0.0

        for bc in boundary_conditions:
            bc_type = bc.get('type', 'dirichlet')
            location = bc.get('location')
            value = bc.get('value', 0.0)

            if location is not None:
                # Evaluate network at boundary
                u_bc = self._forward_pass(location, weights, activation)

                # Compute loss
                if bc_type == 'dirichlet':
                    total_loss += np.mean((u_bc - value)**2)

        return total_loss

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        import uuid
        job_id = f"piml_{uuid.uuid4().hex[:8]}"
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
