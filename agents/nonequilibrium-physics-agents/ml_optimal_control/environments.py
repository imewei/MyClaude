"""Reinforcement Learning Environments for Optimal Control.

This module provides RL environments for training neural network policies
on optimal control problems.

Environments:
- OptimalControlEnv: Generic optimal control environment
- QuantumControlEnv: Quantum state transfer environment
- ThermodynamicEnv: Thermodynamic process optimization

Author: Nonequilibrium Physics Agents
Date: 2025-09-30
"""

import warnings
from typing import Callable, Tuple, Optional, Dict
import numpy as np

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    warnings.warn(
        "JAX not available. RL environments will use NumPy. "
        "For better performance, install JAX: pip install jax jaxlib"
    )


class OptimalControlEnv:
    """Generic Optimal Control Environment.

    Gym-like environment for optimal control problems.

    Parameters
    ----------
    dynamics : Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        System dynamics f(x, u, t)
    cost : Callable[[np.ndarray, np.ndarray, float], float]
        Running cost L(x, u, t)
    x0 : np.ndarray
        Initial state
    xf : Optional[np.ndarray]
        Target final state (if any)
    duration : float
        Episode duration
    dt : float
        Time step
    control_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        Control bounds (u_min, u_max)
    state_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        State bounds (x_min, x_max)

    Attributes
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Action space dimension
    current_state : np.ndarray
        Current state
    current_time : float
        Current time

    Methods
    -------
    reset()
        Reset environment
    step(action)
        Take action and observe next state, reward
    render()
        Render environment (optional)
    """

    def __init__(
        self,
        dynamics: Callable,
        cost: Callable,
        x0: np.ndarray,
        xf: Optional[np.ndarray] = None,
        duration: float = 10.0,
        dt: float = 0.1,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize optimal control environment."""
        self.dynamics = dynamics
        self.cost = cost
        self.x0 = x0
        self.xf = xf
        self.duration = duration
        self.dt = dt
        self.control_bounds = control_bounds
        self.state_bounds = state_bounds

        self.state_dim = len(x0)
        # Infer action dim from dynamics
        dummy_u = np.zeros(1)
        f_test = self.dynamics(x0, dummy_u, 0.0)
        if len(f_test) == self.state_dim:
            self.action_dim = 1
        else:
            # Try to infer from control bounds
            if control_bounds is not None:
                self.action_dim = len(control_bounds[0])
            else:
                self.action_dim = 1

        self.current_state = None
        self.current_time = None

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns
        -------
        np.ndarray
            Initial state
        """
        self.current_state = self.x0.copy()
        self.current_time = 0.0
        return self.current_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and observe next state and reward.

        Parameters
        ----------
        action : np.ndarray
            Control action, shape (action_dim,)

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict]
            (next_state, reward, done, info)
        """
        # Clip action to bounds
        if self.control_bounds is not None:
            action = np.clip(action, self.control_bounds[0], self.control_bounds[1])

        # Integrate dynamics (simple Euler step)
        f = self.dynamics(self.current_state, action, self.current_time)
        next_state = self.current_state + self.dt * f

        # Clip state to bounds
        if self.state_bounds is not None:
            next_state = np.clip(next_state, self.state_bounds[0], self.state_bounds[1])

        # Compute reward (negative cost)
        reward = -self.cost(self.current_state, action, self.current_time) * self.dt

        # Check if done
        self.current_time += self.dt
        done = self.current_time >= self.duration

        # Add terminal reward if target state specified
        if done and self.xf is not None:
            terminal_cost = np.sum((next_state - self.xf)**2)
            reward -= 10.0 * terminal_cost

        self.current_state = next_state

        info = {'time': self.current_time}

        return next_state, reward, done, info

    def render(self):
        """Render environment (placeholder)."""
        print(f"Time: {self.current_time:.2f}, State: {self.current_state}")


class QuantumControlEnv(OptimalControlEnv):
    """Quantum Control Environment.

    Environment for learning quantum state transfer protocols.

    Parameters
    ----------
    H0 : np.ndarray
        Drift Hamiltonian, shape (n, n)
    control_hamiltonians : list of np.ndarray
        Control Hamiltonians, each shape (n, n)
    psi0 : np.ndarray
        Initial quantum state, shape (n,)
    psi_target : np.ndarray
        Target quantum state, shape (n,)
    duration : float
        Control duration
    dt : float
        Time step
    control_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        Control amplitude bounds
    hbar : float
        Reduced Planck constant

    Methods
    -------
    reset()
        Reset to initial quantum state
    step(action)
        Apply control and evolve quantum state
    get_fidelity()
        Compute fidelity with target state
    """

    def __init__(
        self,
        H0: np.ndarray,
        control_hamiltonians: list,
        psi0: np.ndarray,
        psi_target: np.ndarray,
        duration: float = 5.0,
        dt: float = 0.01,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        hbar: float = 1.0
    ):
        """Initialize quantum control environment."""
        self.H0 = H0
        self.Hk = control_hamiltonians
        self.psi_target = psi_target / np.linalg.norm(psi_target)
        self.hbar = hbar

        n_dim = len(psi0)
        n_controls = len(control_hamiltonians)

        # Convert to real representation: [Re(ψ), Im(ψ)]
        state_dim = 2 * n_dim

        # Dynamics: Schrödinger equation
        def dynamics(x, u, t):
            # Reconstruct complex state
            psi_re = x[:n_dim]
            psi_im = x[n_dim:]

            # Build H(t) = H0 + Σ u_k H_k
            H = H0.copy()
            for k in range(n_controls):
                H += u[k] * self.Hk[k]

            # Compute -i/ħ H ψ
            H_re = H.real
            H_im = H.imag

            Hpsi_re = H_re @ psi_re - H_im @ psi_im
            Hpsi_im = H_re @ psi_im + H_im @ psi_re

            dpsi_re = Hpsi_im / hbar
            dpsi_im = -Hpsi_re / hbar

            return np.concatenate([dpsi_re, dpsi_im])

        # Cost: control effort + infidelity
        def cost(x, u, t):
            control_cost = 0.1 * np.sum(u**2)
            return control_cost

        # Convert initial state to real
        x0_real = np.concatenate([psi0.real, psi0.imag])
        x0_real /= np.linalg.norm(x0_real)

        super().__init__(
            dynamics=dynamics,
            cost=cost,
            x0=x0_real,
            xf=None,
            duration=duration,
            dt=dt,
            control_bounds=control_bounds
        )

        self.n_dim = n_dim

    def get_fidelity(self) -> float:
        """Compute fidelity with target state.

        Returns
        -------
        float
            Fidelity |⟨ψ_target|ψ⟩|²
        """
        # Reconstruct complex state
        psi_re = self.current_state[:self.n_dim]
        psi_im = self.current_state[self.n_dim:]
        psi = psi_re + 1j * psi_im
        psi /= np.linalg.norm(psi)

        # Compute fidelity
        overlap = np.vdot(self.psi_target, psi)
        fidelity = np.abs(overlap)**2

        return fidelity

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and observe next state and reward.

        Parameters
        ----------
        action : np.ndarray
            Control action

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict]
            (next_state, reward, done, info)
        """
        next_state, reward, done, info = super().step(action)

        # Add fidelity-based reward at end
        if done:
            fidelity = self.get_fidelity()
            reward += 100.0 * fidelity  # Large bonus for high fidelity
            info['final_fidelity'] = fidelity

        return next_state, reward, done, info


class ThermodynamicEnv(OptimalControlEnv):
    """Thermodynamic Process Environment.

    Environment for optimizing thermodynamic processes (e.g., heat engine cycles).

    Parameters
    ----------
    protocol_type : str
        Type of protocol: 'compression', 'expansion', 'heating', 'cooling'
    n_particles : int
        Number of particles
    temperature_range : Tuple[float, float]
        Temperature range (T_min, T_max)
    volume_range : Tuple[float, float]
        Volume range (V_min, V_max)
    duration : float
        Process duration
    dt : float
        Time step

    Methods
    -------
    reset()
        Reset to initial thermodynamic state
    step(action)
        Apply control (e.g., change volume) and evolve system
    get_work()
        Compute work done by system
    get_heat()
        Compute heat transferred
    get_efficiency()
        Compute thermodynamic efficiency
    """

    def __init__(
        self,
        protocol_type: str = 'compression',
        n_particles: int = 1000,
        temperature_range: Tuple[float, float] = (300.0, 500.0),
        volume_range: Tuple[float, float] = (1.0, 2.0),
        duration: float = 10.0,
        dt: float = 0.1
    ):
        """Initialize thermodynamic environment."""
        self.protocol_type = protocol_type
        self.n_particles = n_particles
        self.T_range = temperature_range
        self.V_range = volume_range

        # Boltzmann constant (in arbitrary units)
        self.k_B = 1.0

        # State: [Temperature, Volume]
        if protocol_type == 'compression':
            x0 = np.array([temperature_range[0], volume_range[1]])  # Cold, expanded
            xf = np.array([temperature_range[1], volume_range[0]])  # Hot, compressed
        elif protocol_type == 'expansion':
            x0 = np.array([temperature_range[1], volume_range[0]])  # Hot, compressed
            xf = np.array([temperature_range[0], volume_range[1]])  # Cold, expanded
        else:
            x0 = np.array([temperature_range[0], volume_range[0]])
            xf = np.array([temperature_range[1], volume_range[1]])

        # Dynamics: Ideal gas + control
        def dynamics(x, u, t):
            T, V = x
            # Control: rate of volume change and heat input
            dV_dt = u[0]
            dQ_dt = u[1]  # Heat input rate

            # Ideal gas: PV = NkT
            P = n_particles * self.k_B * T / V

            # First law: dU = dQ - PdV
            # For ideal gas: dU = NkdT (constant volume heat capacity)
            dT_dt = (dQ_dt - P * dV_dt) / (n_particles * self.k_B)

            return np.array([dT_dt, dV_dt])

        # Cost: minimize work input + control effort
        def cost(x, u, t):
            T, V = x
            P = n_particles * self.k_B * T / V
            work_rate = P * u[0]  # W = ∫ P dV
            control_cost = 0.01 * np.sum(u**2)
            return work_rate + control_cost

        # Control bounds
        u_min = np.array([-1.0, -10.0])  # dV/dt, dQ/dt
        u_max = np.array([1.0, 10.0])

        # State bounds
        x_min = np.array([temperature_range[0], volume_range[0]])
        x_max = np.array([temperature_range[1], volume_range[1]])

        super().__init__(
            dynamics=dynamics,
            cost=cost,
            x0=x0,
            xf=xf,
            duration=duration,
            dt=dt,
            control_bounds=(u_min, u_max),
            state_bounds=(x_min, x_max)
        )

        self.cumulative_work = 0.0
        self.cumulative_heat = 0.0

    def reset(self) -> np.ndarray:
        """Reset environment."""
        state = super().reset()
        self.cumulative_work = 0.0
        self.cumulative_heat = 0.0
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and observe next state and reward."""
        T, V = self.current_state
        P = self.n_particles * self.k_B * T / V

        # Track work and heat
        self.cumulative_work += P * action[0] * self.dt
        self.cumulative_heat += action[1] * self.dt

        next_state, reward, done, info = super().step(action)

        if done:
            info['total_work'] = self.cumulative_work
            info['total_heat'] = self.cumulative_heat
            if self.cumulative_heat > 0:
                info['efficiency'] = self.cumulative_work / self.cumulative_heat
            else:
                info['efficiency'] = 0.0

        return next_state, reward, done, info


# Convenience exports
__all__ = [
    'OptimalControlEnv',
    'QuantumControlEnv',
    'ThermodynamicEnv',
    'JAX_AVAILABLE',
]
