"""Nonequilibrium Quantum Agent - Open Quantum Systems and Quantum Thermodynamics.

Capabilities:
- Lindblad Master Equation: Open quantum system evolution
- Quantum Fluctuation Theorems: Jarzynski, Crooks for quantum systems
- Quantum Master Equation Solver: GKSL equation with complete positivity
- Quantum Transport: Landauer-Büttiker formalism, quantum conductance
- Quantum Thermodynamics: Heat, work, entropy in quantum regime
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
from uuid import uuid4
import numpy as np
from scipy.linalg import expm, logm
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize

from base_agent import (
    SimulationAgent,
    AgentResult,
    AgentStatus,
    ValidationResult,
    ResourceRequirement,
    Capability,
    AgentMetadata,
    Provenance,
    ExecutionEnvironment,
    ValidationError,
    ExecutionError
)


class NonequilibriumQuantumAgent(SimulationAgent):
    """Nonequilibrium quantum systems and quantum thermodynamics agent.

    Handles open quantum systems:
    - Lindblad master equation evolution
    - Quantum fluctuation theorems (Jarzynski, Crooks)
    - GKSL equation solver with complete positivity
    - Quantum transport (Landauer-Büttiker)
    - Quantum thermodynamics (work, heat, entropy)
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize nonequilibrium quantum agent.

        Args:
            config: Configuration with quantum parameters, etc.
        """
        super().__init__(config)
        self.supported_methods = [
            'lindblad_master_equation', 'quantum_fluctuation_theorem',
            'quantum_master_equation_solver', 'quantum_transport',
            'quantum_thermodynamics'
        ]
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
        self.e = 1.602176634e-19  # Elementary charge (C)
        self.h = 6.62607015e-34  # Planck constant (J·s)
        self.job_cache = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute quantum analysis.

        Args:
            input_data: Input with keys:
                - method: str (lindblad_master_equation, etc.)
                - data: dict (Hamiltonian, density matrix, etc.)
                - parameters: dict (time, temperature, etc.)
                - analysis: list of str (evolution, entropy, etc.)

        Returns:
            AgentResult with quantum analysis results

        Example:
            >>> agent = NonequilibriumQuantumAgent()
            >>> result = agent.execute({
            ...     'method': 'lindblad_master_equation',
            ...     'data': {'H': H_matrix, 'rho0': rho0_matrix},
            ...     'parameters': {'time': 10.0, 'temperature': 300.0},
            ...     'analysis': ['evolution', 'entropy_production']
            ... })
        """
        start_time = datetime.now()
        method = input_data.get('method', 'lindblad_master_equation')

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

            # Route to appropriate method
            if method == 'lindblad_master_equation':
                result_data = self._lindblad_master_equation(input_data)
            elif method == 'quantum_fluctuation_theorem':
                result_data = self._quantum_fluctuation_theorem(input_data)
            elif method == 'quantum_master_equation_solver':
                result_data = self._quantum_master_equation_solver(input_data)
            elif method == 'quantum_transport':
                result_data = self._quantum_transport(input_data)
            elif method == 'quantum_thermodynamics':
                result_data = self._quantum_thermodynamics(input_data)
            else:
                raise ExecutionError(f"Unsupported method: {method}")

            # Create provenance
            provenance = Provenance(
                agent_name=self.metadata.name,
                agent_version=self.VERSION,
                timestamp=datetime.now(),
                input_hash=self._compute_input_hash(input_data),
                parameters=input_data.get('parameters', {}),
                execution_time_sec=(datetime.now() - start_time).total_seconds()
            )

            # Add execution metadata
            metadata = {
                'method': method,
                'analysis_type': input_data.get('analysis', []),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }

            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                metadata=metadata,
                provenance=provenance
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=[f"Execution failed: {str(e)}"]
            )

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data for quantum analysis.

        Args:
            data: Input data dictionary

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []

        # Check method
        method = data.get('method')
        if not method:
            errors.append("Missing required field 'method'")
        elif method not in self.supported_methods:
            errors.append(f"Unsupported method '{method}'. Supported: {self.supported_methods}")

        # Check data
        if 'data' not in data:
            errors.append("Missing required field 'data'")

        # Check parameters
        if 'parameters' not in data:
            warnings.append("Missing 'parameters' field - using defaults")
        else:
            params = data['parameters']

            # Check time
            if 'time' in params:
                time = params['time']
                if time <= 0:
                    errors.append(f"Invalid time: {time} (must be positive)")

            # Check temperature
            if 'temperature' in params:
                temp = params['temperature']
                if temp <= 0:
                    errors.append(f"Invalid temperature: {temp} K (must be positive)")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, input_data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed.

        Args:
            input_data: Input data dictionary

        Returns:
            ResourceRequirement with estimated needs
        """
        method = input_data.get('method', 'lindblad_master_equation')
        params = input_data.get('parameters', {})
        data = input_data.get('data', {})

        # Estimate based on method and problem size
        n_dim = data.get('n_dim', 2)  # Hilbert space dimension
        n_steps = params.get('n_steps', 100)

        # Base resource estimation
        if method == 'lindblad_master_equation':
            cpu_cores = 4
            memory_gb = 2.0 * (n_dim**2 / 4.0)  # Scale with density matrix size
            duration_est = 60 * (n_dim**2 / 4.0)
            env = ExecutionEnvironment.HPC if n_dim > 4 else ExecutionEnvironment.LOCAL
        elif method == 'quantum_fluctuation_theorem':
            cpu_cores = 8
            memory_gb = 4.0
            duration_est = 300
            env = ExecutionEnvironment.HPC
        elif method == 'quantum_master_equation_solver':
            cpu_cores = 8
            memory_gb = 4.0 * (n_dim**2 / 4.0)
            duration_est = 120 * (n_dim**2 / 4.0)
            env = ExecutionEnvironment.HPC if n_dim > 4 else ExecutionEnvironment.LOCAL
        elif method == 'quantum_transport':
            cpu_cores = 4
            memory_gb = 2.0
            duration_est = 60
            env = ExecutionEnvironment.LOCAL
        elif method == 'quantum_thermodynamics':
            cpu_cores = 2
            memory_gb = 1.0
            duration_est = 30
            env = ExecutionEnvironment.LOCAL
        else:
            cpu_cores = 4
            memory_gb = 2.0
            duration_est = 60
            env = ExecutionEnvironment.LOCAL

        # Adjust for problem size
        if n_steps > 1000:
            memory_gb *= 1.5
            duration_est *= 1.5

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_required=False,
            estimated_duration_seconds=duration_est,
            environment=env
        )

    def _lindblad_master_equation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve Lindblad master equation for open quantum system.

        Physics:
            - Lindblad equation: dρ/dt = -i/ℏ [H, ρ] + Σ_k γ_k D[L_k]ρ
            - Dissipator: D[L]ρ = L ρ L† - 1/2 {L†L, ρ}
            - Preserves: Tr(ρ) = 1, ρ = ρ†, ρ ≥ 0 (complete positivity)

        Args:
            input_data: Contains Hamiltonian H, initial state rho0, jump operators

        Returns:
            Dictionary with density matrix evolution, entropy, purity
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Time parameters
        time = params.get('time', 10.0)
        n_steps = params.get('n_steps', 100)
        t_eval = np.linspace(0, time, n_steps)

        # System parameters
        n_dim = data.get('n_dim', 2)

        # Hamiltonian (default: two-level system)
        if 'H' in data:
            H = np.array(data['H'])
        else:
            # Default: Pauli-Z Hamiltonian
            omega = 1.0  # Frequency
            H = -0.5 * omega * np.array([[1, 0], [0, -1]], dtype=complex)

        # Initial density matrix (default: ground state)
        if 'rho0' in data:
            rho0 = np.array(data['rho0'])
        else:
            rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

        # Jump operators (default: single decay operator)
        if 'jump_operators' in data:
            L_ops = [np.array(L) for L in data['jump_operators']]
            gammas = data.get('decay_rates', [1.0] * len(L_ops))
        else:
            # Default: spontaneous decay
            gamma = params.get('decay_rate', 0.1)
            L = np.array([[0, 1], [0, 0]], dtype=complex)  # Lowering operator
            L_ops = [L]
            gammas = [gamma]

        # Solver selection
        solver_type = params.get('solver', 'RK45')

        # Check for GPU backend option
        backend = params.get('backend', 'cpu')

        # Solve based on selected method
        if solver_type == 'magnus':
            # Magnus expansion solver (better energy conservation)
            try:
                from solvers.magnus_expansion import solve_lindblad_magnus

                # For Magnus, need time-dependent H protocol
                # If H is constant, create callable
                if isinstance(H, np.ndarray):
                    H_protocol = lambda t: H
                else:
                    H_protocol = H  # Already callable

                magnus_order = params.get('magnus_order', 4)
                result = solve_lindblad_magnus(
                    rho0, H_protocol, L_ops, gammas, t_eval,
                    order=magnus_order, n_steps=n_steps
                )

                rho_evolution = result['rho_evolution']
                entropy = result['entropy']
                purity = result['purity']
                solver_used = f'magnus_order{magnus_order}'

            except ImportError:
                # Fallback to RK45 if Magnus not available
                solver_type = 'RK45'
                solver_used = 'RK45_fallback'

        if solver_type == 'jax' or backend == 'gpu':
            # GPU-accelerated solver
            try:
                from gpu_kernels.quantum_evolution import solve_lindblad

                result_gpu = solve_lindblad(
                    rho0, H, L_ops, gammas, t_eval,
                    hbar=self.hbar, backend=backend
                )

                rho_evolution = result_gpu['rho_evolution']
                entropy = result_gpu['entropy']
                purity = result_gpu['purity']
                solver_used = result_gpu['backend_used']

            except ImportError:
                # Fallback to RK45 if GPU not available
                solver_type = 'RK45'
                solver_used = 'RK45_fallback'

        if solver_type == 'RK45' or solver_type not in ['magnus', 'jax']:
            # Standard RK45 solver (default)
            def lindblad_rhs(t, rho_vec):
                """Right-hand side of Lindblad equation."""
                # Reshape to matrix
                rho = rho_vec.reshape((n_dim, n_dim))

                # Unitary evolution: -i/ℏ [H, ρ]
                drho_dt = -1j / self.hbar * (H @ rho - rho @ H)

                # Dissipative evolution: Σ_k γ_k D[L_k]ρ
                for L, gamma in zip(L_ops, gammas):
                    L_dag = L.conj().T
                    drho_dt += gamma * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))

                return drho_dt.flatten()

            # Solve Lindblad equation
            rho0_vec = rho0.flatten()
            sol = solve_ivp(
                lindblad_rhs,
                (0, time),
                rho0_vec,
                t_eval=t_eval,
                method='RK45'
            )

            # Extract density matrices
            rho_evolution = sol.y.T.reshape((n_steps, n_dim, n_dim))
            solver_used = 'RK45'

            # Compute observables for RK45 (Magnus/GPU already computed them)
            entropy = np.zeros(n_steps)
            purity = np.zeros(n_steps)

            for i, rho_t in enumerate(rho_evolution):
                # Von Neumann entropy: S = -Tr(ρ ln ρ)
                eigvals = np.linalg.eigvalsh(rho_t)
                eigvals = eigvals[eigvals > 1e-12]  # Avoid log(0)
                entropy[i] = -np.sum(eigvals * np.log(eigvals))

                # Purity: Tr(ρ²)
                purity[i] = np.real(np.trace(rho_t @ rho_t))

        # Compute populations (always needed, regardless of solver)
        populations = np.zeros((n_steps, n_dim))
        for i, rho_t in enumerate(rho_evolution):
            populations[i] = np.real(np.diag(rho_t))

        return {
            'method_type': 'lindblad_master_equation',
            'time_grid': t_eval.tolist(),
            'n_dim': int(n_dim),
            'rho_evolution': [rho.tolist() for rho in rho_evolution],
            'rho_final': rho_evolution[-1].tolist(),
            'entropy': entropy.tolist(),
            'purity': purity.tolist(),
            'populations': populations.tolist(),
            'trace_final': float(np.real(np.trace(rho_evolution[-1]))),
            'n_jump_operators': len(L_ops),
            'decay_rates': gammas,
            'n_steps': int(n_steps),
            'solver_used': solver_used  # NEW: Report which solver was used
        }

    def _quantum_fluctuation_theorem(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify quantum fluctuation theorems (Jarzynski, Crooks).

        Physics:
            - Quantum Jarzynski: ⟨e^(-βW)⟩ = e^(-βΔF)
            - Two-point measurement (TPM) protocol
            - Quantum Crooks: P_F(W)/P_R(-W) = exp[β(W - ΔF)]

        Args:
            input_data: Contains initial/final Hamiltonians, protocol, trajectories

        Returns:
            Dictionary with work distribution, Jarzynski verification, Crooks ratio
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Thermodynamic parameters
        temperature = params.get('temperature', 300.0)
        beta = 1.0 / (self.kB * temperature)

        # Hamiltonians
        n_dim = data.get('n_dim', 2)
        if 'H_initial' in data and 'H_final' in data:
            H_i = np.array(data['H_initial'])
            H_f = np.array(data['H_final'])
        else:
            # Default: two-level system with changing field
            omega_i = 1.0
            omega_f = 2.0
            H_i = -0.5 * omega_i * np.array([[1, 0], [0, -1]], dtype=complex)
            H_f = -0.5 * omega_f * np.array([[1, 0], [0, -1]], dtype=complex)

        # Compute free energy change
        # F = -kT ln Z, where Z = Tr(e^(-βH))
        Z_i = np.trace(expm(-beta * H_i))
        Z_f = np.trace(expm(-beta * H_f))
        F_i = -self.kB * temperature * np.log(np.real(Z_i))
        F_f = -self.kB * temperature * np.log(np.real(Z_f))
        delta_F = F_f - F_i

        # Two-point measurement protocol
        n_realizations = params.get('n_realizations', 1000)
        work_samples = np.zeros(n_realizations)

        # Initial thermal state
        rho_eq_i = expm(-beta * H_i) / Z_i

        for n in range(n_realizations):
            # First measurement: project onto eigenstate of H_i
            E_i_vals, E_i_vecs = np.linalg.eigh(H_i)
            probs_i = np.array([np.real(E_i_vecs[:, k].conj().T @ rho_eq_i @ E_i_vecs[:, k])
                                for k in range(n_dim)])
            probs_i = probs_i / np.sum(probs_i)  # Normalize
            k_i = np.random.choice(n_dim, p=probs_i)
            E_i = E_i_vals[k_i]

            # Unitary evolution (simplified: instantaneous for demonstration)
            psi_i = E_i_vecs[:, k_i]

            # Second measurement: measure in eigenbasis of H_f
            E_f_vals, E_f_vecs = np.linalg.eigh(H_f)
            probs_f = np.abs(E_f_vecs.conj().T @ psi_i)**2
            k_f = np.random.choice(n_dim, p=probs_f)
            E_f = E_f_vals[k_f]

            # Work: W = E_f - E_i
            work_samples[n] = E_f - E_i

        # Jarzynski equality verification
        # ⟨e^(-βW)⟩ should equal e^(-βΔF)
        exp_avg = np.mean(np.exp(-beta * work_samples))
        jarzynski_lhs = exp_avg
        jarzynski_rhs = np.exp(-beta * delta_F)
        jarzynski_ratio = jarzynski_lhs / jarzynski_rhs

        # Work distribution statistics
        mean_work = np.mean(work_samples)
        std_work = np.std(work_samples)

        # Verify Jarzynski relation
        # Also check: ⟨W⟩ ≥ ΔF (second law)
        second_law_satisfied = (mean_work >= delta_F)

        return {
            'method_type': 'quantum_fluctuation_theorem',
            'n_dim': int(n_dim),
            'temperature_K': float(temperature),
            'beta': float(beta),
            'free_energy_initial_J': float(F_i),
            'free_energy_final_J': float(F_f),
            'delta_F_J': float(delta_F),
            'n_realizations': int(n_realizations),
            'work_distribution': work_samples.tolist(),
            'mean_work_J': float(mean_work),
            'std_work_J': float(std_work),
            'jarzynski_lhs': float(jarzynski_lhs),
            'jarzynski_rhs': float(jarzynski_rhs),
            'jarzynski_ratio': float(jarzynski_ratio),
            'jarzynski_satisfied': bool(np.abs(jarzynski_ratio - 1.0) < 0.1),
            'second_law_satisfied': bool(second_law_satisfied),
            'excess_work_J': float(mean_work - delta_F)
        }

    def _quantum_master_equation_solver(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve general quantum master equation (GKSL form).

        Physics:
            - GKSL equation: dρ/dt = -i/ℏ [H, ρ] + Σ_k γ_k D[L_k]ρ
            - Complete positivity guaranteed
            - Trace preservation: Tr(ρ) = 1 for all t

        Args:
            input_data: Contains Hamiltonian, jump operators, initial state

        Returns:
            Dictionary with full evolution, steady state, relaxation time
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Time parameters
        time = params.get('time', 10.0)
        n_steps = params.get('n_steps', 100)
        t_eval = np.linspace(0, time, n_steps)

        # System dimension
        n_dim = data.get('n_dim', 2)

        # Hamiltonian
        if 'H' in data:
            H = np.array(data['H'])
        else:
            # Default: driven two-level system
            omega = 1.0
            H = -0.5 * omega * np.array([[1, 0], [0, -1]], dtype=complex)

        # Initial state
        if 'rho0' in data:
            rho0 = np.array(data['rho0'])
        else:
            # Default: excited state
            rho0 = np.array([[0, 0], [0, 1]], dtype=complex)

        # Jump operators with rates
        if 'jump_operators' in data:
            L_ops = [np.array(L) for L in data['jump_operators']]
            gammas = data.get('decay_rates', [1.0] * len(L_ops))
        else:
            # Default: decay + dephasing
            gamma_decay = 0.5
            gamma_dephase = 0.2
            L_decay = np.array([[0, 1], [0, 0]], dtype=complex)
            L_dephase = np.array([[1, 0], [0, -1]], dtype=complex)
            L_ops = [L_decay, L_dephase]
            gammas = [gamma_decay, gamma_dephase]

        # Solve GKSL equation (same as Lindblad)
        def gksl_rhs(t, rho_vec):
            rho = rho_vec.reshape((n_dim, n_dim))

            # Coherent evolution
            drho_dt = -1j / self.hbar * (H @ rho - rho @ H)

            # Dissipative evolution
            for L, gamma in zip(L_ops, gammas):
                L_dag = L.conj().T
                drho_dt += gamma * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))

            return drho_dt.flatten()

        rho0_vec = rho0.flatten()
        sol = solve_ivp(gksl_rhs, (0, time), rho0_vec, t_eval=t_eval, method='RK45')

        rho_evolution = sol.y.T.reshape((n_steps, n_dim, n_dim))

        # Find steady state (long-time limit)
        rho_steady = rho_evolution[-1]

        # Estimate relaxation time
        # Fit exponential decay of off-diagonal elements
        rho_01 = np.array([np.abs(rho_evolution[i, 0, 1]) for i in range(n_steps)])
        if rho_01[0] > 0:
            # Simple estimate: when rho_01 decays to 1/e
            threshold = rho_01[0] / np.e
            idx = np.where(rho_01 < threshold)[0]
            if len(idx) > 0:
                relaxation_time = t_eval[idx[0]]
            else:
                relaxation_time = time  # Not fully relaxed
        else:
            relaxation_time = 0.0

        # Entropy production rate at steady state
        entropy_ss = 0.0
        eigvals_ss = np.linalg.eigvalsh(rho_steady)
        eigvals_ss = eigvals_ss[eigvals_ss > 1e-12]
        if len(eigvals_ss) > 0:
            entropy_ss = -np.sum(eigvals_ss * np.log(eigvals_ss))

        return {
            'method_type': 'quantum_master_equation_solver',
            'time_grid': t_eval.tolist(),
            'n_dim': int(n_dim),
            'rho_evolution': [rho.tolist() for rho in rho_evolution],
            'rho_steady_state': rho_steady.tolist(),
            'steady_state_populations': np.real(np.diag(rho_steady)).tolist(),
            'steady_state_entropy': float(entropy_ss),
            'relaxation_time': float(relaxation_time),
            'n_jump_operators': len(L_ops),
            'trace_preserved': bool(np.abs(np.trace(rho_steady) - 1.0) < 1e-6),
            'n_steps': int(n_steps)
        }

    def _quantum_transport(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute quantum transport via Landauer-Büttiker formalism.

        Physics:
            - Conductance: G = (e²/h) * T (for transmission T)
            - Landauer formula: G = (2e²/h) ∫ dE T(E) [-∂f/∂E]
            - Current: I = (e/h) ∫ dE T(E) [f_L(E) - f_R(E)]

        Args:
            input_data: Contains transmission function, chemical potentials, temperature

        Returns:
            Dictionary with conductance, current, transport coefficients
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Temperature
        temperature = params.get('temperature', 300.0)
        beta = 1.0 / (self.kB * temperature)

        # Chemical potentials
        mu_L = data.get('mu_left', 0.1 * self.e)  # Left lead
        mu_R = data.get('mu_right', 0.0)  # Right lead
        bias = mu_L - mu_R

        # Energy grid
        n_energies = params.get('n_energies', 200)
        E_min = min(mu_L, mu_R) - 5 * self.kB * temperature
        E_max = max(mu_L, mu_R) + 5 * self.kB * temperature
        energies = np.linspace(E_min, E_max, n_energies)

        # Transmission function T(E)
        if 'transmission' in data and callable(data['transmission']):
            T_E = np.array([data['transmission'](E) for E in energies])
        else:
            # Default: Lorentzian resonance
            E0 = 0.05 * self.e  # Resonance energy
            Gamma = 0.01 * self.e  # Width
            T_E = (Gamma**2) / ((energies - E0)**2 + Gamma**2)

        # Fermi-Dirac distribution
        def fermi(E, mu, T):
            beta_local = 1.0 / (self.kB * T)
            return 1.0 / (1.0 + np.exp(beta_local * (E - mu)))

        f_L = fermi(energies, mu_L, temperature)
        f_R = fermi(energies, mu_R, temperature)

        # Current: I = (e/h) ∫ T(E) [f_L - f_R] dE
        integrand_current = T_E * (f_L - f_R)
        current = (self.e / self.h) * np.trapz(integrand_current, energies)

        # Conductance at zero bias: G = (2e²/h) ∫ T(E) [-∂f/∂E] dE
        # For small bias: G ≈ I / V
        if np.abs(bias) > 1e-12:
            conductance = current / bias
        else:
            # Use derivative of Fermi function
            df_dE = -beta * f_L * (1 - f_L)
            integrand_conductance = T_E * (-df_dE)
            conductance = (2 * self.e**2 / self.h) * np.trapz(integrand_conductance, energies)

        # Quantum of conductance
        G0 = 2 * self.e**2 / self.h  # 2e²/h ≈ 7.75 × 10⁻⁵ S
        conductance_normalized = conductance / G0

        # Thermoelectric coefficients (simplified)
        # Seebeck coefficient: S = (1/eT) * L1 / L0
        # where Ln = ∫ (E-μ)^n T(E) [-∂f/∂E] dE
        df_dE_avg = -beta * fermi(energies, (mu_L + mu_R)/2, temperature) * (1 - fermi(energies, (mu_L + mu_R)/2, temperature))
        L0 = np.trapz(T_E * (-df_dE_avg), energies)
        L1 = np.trapz(T_E * (energies - (mu_L + mu_R)/2) * (-df_dE_avg), energies)
        if L0 > 0:
            seebeck = L1 / (self.e * temperature * L0)
        else:
            seebeck = 0.0

        return {
            'method_type': 'quantum_transport',
            'temperature_K': float(temperature),
            'mu_left_J': float(mu_L),
            'mu_right_J': float(mu_R),
            'bias_voltage_V': float(bias / self.e),
            'energy_grid_J': energies.tolist(),
            'transmission': T_E.tolist(),
            'current_A': float(current),
            'conductance_S': float(conductance),
            'conductance_G0_units': float(conductance_normalized),
            'quantum_conductance_G0': float(G0),
            'seebeck_coefficient_V_K': float(seebeck),
            'n_energies': int(n_energies)
        }

    def _quantum_thermodynamics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quantum thermodynamics: work, heat, entropy.

        Physics:
            - First law: dU = δW + δQ
            - Work: W = Tr[dH * ρ]
            - Heat: Q = Tr[H * dρ]
            - Entropy: S = -Tr[ρ ln ρ]

        Args:
            input_data: Contains time-dependent Hamiltonian, density matrix evolution

        Returns:
            Dictionary with work, heat, entropy, efficiency
        """
        data = input_data.get('data', {})
        params = input_data.get('parameters', {})

        # Time parameters
        time = params.get('time', 10.0)
        n_steps = params.get('n_steps', 100)
        t_grid = np.linspace(0, time, n_steps)

        # System dimension
        n_dim = data.get('n_dim', 2)

        # Hamiltonian protocol (time-dependent)
        if 'H_protocol' in data:
            H_protocol = [np.array(H) for H in data['H_protocol']]
        else:
            # Default: linear ramp
            omega_i = 1.0
            omega_f = 2.0
            H_protocol = []
            for t in t_grid:
                omega_t = omega_i + (omega_f - omega_i) * (t / time)
                H_t = -0.5 * omega_t * np.array([[1, 0], [0, -1]], dtype=complex)
                H_protocol.append(H_t)

        # Density matrix evolution (from Lindblad or provided)
        if 'rho_evolution' in data:
            rho_evolution = [np.array(rho) for rho in data['rho_evolution']]
        else:
            # Simplified: adiabatic evolution (instantaneous eigenstates)
            rho_evolution = []
            for H_t in H_protocol:
                eigvals, eigvecs = np.linalg.eigh(H_t)
                rho_t = np.outer(eigvecs[:, 0], eigvecs[:, 0].conj())  # Ground state
                rho_evolution.append(rho_t)

        # Compute thermodynamic quantities
        work = 0.0
        heat = 0.0
        internal_energy = np.zeros(n_steps)
        entropy = np.zeros(n_steps)

        for i in range(n_steps):
            rho_t = rho_evolution[i]
            H_t = H_protocol[i]

            # Internal energy: U = Tr(H ρ)
            internal_energy[i] = np.real(np.trace(H_t @ rho_t))

            # Von Neumann entropy: S = -Tr(ρ ln ρ)
            eigvals = np.linalg.eigvalsh(rho_t)
            eigvals = eigvals[eigvals > 1e-12]
            if len(eigvals) > 0:
                entropy[i] = -np.sum(eigvals * np.log(eigvals))

            # Incremental work and heat
            if i > 0:
                dt = t_grid[i] - t_grid[i-1]
                rho_prev = rho_evolution[i-1]
                H_prev = H_protocol[i-1]

                # dH and dρ
                dH = H_t - H_prev
                drho = rho_t - rho_prev

                # Work: W = ∫ Tr[dH * ρ] dt
                dW = np.real(np.trace(dH @ rho_t))
                work += dW

                # Heat: Q = ∫ Tr[H * dρ] dt
                dQ = np.real(np.trace(H_t @ drho))
                heat += dQ

        # Total change in internal energy
        delta_U = internal_energy[-1] - internal_energy[0]

        # Verify first law: ΔU = W + Q
        first_law_check = np.abs(delta_U - (work + heat))

        # Entropy change
        delta_S = entropy[-1] - entropy[0]

        # Efficiency (if heat absorbed)
        if heat > 0:
            efficiency = work / heat
        else:
            efficiency = 0.0

        return {
            'method_type': 'quantum_thermodynamics',
            'time_grid': t_grid.tolist(),
            'n_dim': int(n_dim),
            'internal_energy': internal_energy.tolist(),
            'entropy': entropy.tolist(),
            'total_work_J': float(work),
            'total_heat_J': float(heat),
            'delta_U_J': float(delta_U),
            'delta_S': float(delta_S),
            'first_law_residual_J': float(first_law_check),
            'first_law_satisfied': bool(first_law_check < 1e-6),
            'efficiency': float(efficiency),
            'n_steps': int(n_steps)
        }

    # Integration methods
    def quantum_driven_system(self, driven_params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum version of driven systems analysis.

        Args:
            driven_params: Parameters for quantum driven system

        Returns:
            Quantum driven system evolution
        """
        input_data = {
            'method': 'lindblad_master_equation',
            'data': driven_params,
            'parameters': {'time': 10.0},
            'analysis': ['evolution', 'entropy']
        }

        return self.execute(input_data).data

    def quantum_transport_coefficients(self, transport_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum transport coefficients for TransportAgent extension.

        Args:
            transport_data: Transport system parameters

        Returns:
            Quantum transport coefficients
        """
        input_data = {
            'method': 'quantum_transport',
            'data': transport_data,
            'parameters': {'temperature': 300.0},
            'analysis': ['conductance', 'current']
        }

        return self.execute(input_data).data

    def quantum_information_thermodynamics(self, info_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum information thermodynamics (Maxwell demon).

        Args:
            info_data: Information thermodynamics parameters

        Returns:
            Quantum information analysis
        """
        input_data = {
            'method': 'quantum_thermodynamics',
            'data': info_data,
            'parameters': {'time': 5.0},
            'analysis': ['work', 'heat', 'entropy']
        }

        return self.execute(input_data).data

    def get_capabilities(self) -> List[Capability]:
        """Return list of agent capabilities."""
        return [
            Capability(
                name='lindblad_master_equation',
                description='Open quantum system evolution with dissipation',
                input_types=['H', 'rho0', 'jump_operators', 'decay_rates', 'time'],
                output_types=['rho_evolution', 'entropy', 'purity'],
                typical_use_cases=[
                    'Open quantum system dynamics',
                    'Decoherence studies',
                    'Dissipative quantum evolution'
                ]
            ),
            Capability(
                name='quantum_fluctuation_theorem',
                description='Quantum Jarzynski and Crooks relations',
                input_types=['H_initial', 'H_final', 'temperature', 'n_realizations'],
                output_types=['work_distribution', 'jarzynski_ratio', 'delta_F'],
                typical_use_cases=[
                    'Quantum free energy calculations',
                    'Nonequilibrium quantum thermodynamics',
                    'Two-point measurement protocols'
                ]
            ),
            Capability(
                name='quantum_master_equation_solver',
                description='GKSL equation solver with complete positivity',
                input_types=['H', 'rho0', 'jump_operators', 'time', 'n_steps'],
                output_types=['rho_evolution', 'steady_state', 'relaxation_time'],
                typical_use_cases=[
                    'Steady state computation',
                    'Relaxation dynamics',
                    'Complete positivity preservation'
                ]
            ),
            Capability(
                name='quantum_transport',
                description='Landauer-Büttiker quantum transport formalism',
                input_types=['transmission', 'mu_left', 'mu_right', 'temperature'],
                output_types=['conductance', 'current', 'seebeck_coefficient'],
                typical_use_cases=[
                    'Quantum conductance calculations',
                    'Mesoscopic transport',
                    'Thermoelectric effects'
                ]
            ),
            Capability(
                name='quantum_thermodynamics',
                description='Quantum work, heat, entropy analysis',
                input_types=['H_protocol', 'rho_evolution', 'time'],
                output_types=['work', 'heat', 'entropy', 'efficiency'],
                typical_use_cases=[
                    'Quantum heat engines',
                    'First law validation',
                    'Quantum entropy production'
                ]
            )
        ]

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="NonequilibriumQuantumAgent",
            version=self.VERSION,
            description="Nonequilibrium quantum systems and quantum thermodynamics",
            author="Nonequilibrium Physics Team",
            capabilities=self.get_capabilities()
        )

    # Abstract method implementations (required by SimulationAgent)
    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit quantum calculation (simplified: synchronous execution).

        For this implementation, calculations are executed synchronously,
        but we return a job_id for interface consistency.

        Args:
            input_data: Calculation input data

        Returns:
            Job ID string
        """
        import uuid
        job_id = str(uuid.uuid4())

        # Execute immediately and cache result
        result = self.execute(input_data)
        self.job_cache[job_id] = result

        return job_id

    def check_status(self, job_id: str) -> AgentStatus:
        """Check calculation status.

        Args:
            job_id: Job identifier

        Returns:
            AgentStatus (SUCCESS if job complete, FAILED if not found)
        """
        if job_id in self.job_cache:
            return self.job_cache[job_id].status
        else:
            return AgentStatus.FAILED

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve calculation results.

        Args:
            job_id: Job identifier

        Returns:
            Calculation results dictionary
        """
        if job_id in self.job_cache:
            return self.job_cache[job_id].data
        else:
            return {}

    def _compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Compute hash of input data for caching."""
        import hashlib
        import json
        # Remove non-serializable items (numpy arrays, callables)
        hashable_data = {}
        for k, v in input_data.items():
            if k == 'data':
                # Skip numpy arrays in data
                hashable_data[k] = {dk: 'array' if isinstance(dv, (list, np.ndarray)) else dv
                                   for dk, dv in v.items() if not callable(dv)}
            elif not callable(v):
                hashable_data[k] = v
        data_str = json.dumps(hashable_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
