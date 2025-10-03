"""Real-World Case Studies for Optimal Control.

This module provides complete implementations of real-world optimal control
applications across multiple domains:

1. Robotics: Cart-pole stabilization, quadrotor trajectory, robot arm
2. Energy: Power grid optimization, battery management
3. Finance: Portfolio optimization with transaction costs
4. Process Control: Chemical reactor, HVAC systems
5. Autonomous Systems: Path planning with obstacles

Each case study includes:
- Complete system dynamics
- Cost function formulation
- Constraints (state, control, safety)
- Multiple solver implementations
- Visualization tools
- Performance benchmarking

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# =============================================================================
# Base Case Study Class
# =============================================================================

class CaseStudy(ABC):
    """Abstract base class for case studies.

    Provides common interface for all case studies.
    """

    def __init__(self):
        """Initialize case study."""
        self.name = "Base Case Study"
        self.description = ""
        self.n_states = 0
        self.n_controls = 0

    @abstractmethod
    def dynamics(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """System dynamics dx/dt = f(x, u, t).

        Args:
            x: State vector
            u: Control vector
            t: Time

        Returns:
            State derivative
        """
        pass

    @abstractmethod
    def cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Stage cost l(x, u).

        Args:
            x: State
            u: Control

        Returns:
            Cost value
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get state and control bounds.

        Returns:
            Dictionary with 'state' and 'control' bounds
        """
        pass

    def get_initial_state(self) -> np.ndarray:
        """Get default initial state.

        Returns:
            Initial state
        """
        return np.zeros(self.n_states)

    def get_goal_state(self) -> Optional[np.ndarray]:
        """Get goal state (if applicable).

        Returns:
            Goal state or None
        """
        return None


# =============================================================================
# Cart-Pole Stabilization
# =============================================================================

class CartPoleStabilization(CaseStudy):
    """Cart-pole (inverted pendulum) stabilization.

    System:
        - Cart moves on 1D track
        - Pole attached by hinge to cart
        - Goal: Stabilize pole upright while controlling cart position

    States: [x, x_dot, theta, theta_dot]
        - x: Cart position
        - x_dot: Cart velocity
        - theta: Pole angle (0 = upright)
        - theta_dot: Pole angular velocity

    Control: [F]
        - F: Force applied to cart

    Physics:
        (M + m)x_ddot + m*L*theta_ddot*cos(theta) - m*L*theta_dot^2*sin(theta) = F
        L*theta_ddot + g*sin(theta) = x_ddot*cos(theta)
    """

    def __init__(
        self,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.81,
        friction_cart: float = 0.1,
        friction_pole: float = 0.01
    ):
        """Initialize cart-pole system.

        Args:
            mass_cart: Cart mass (kg)
            mass_pole: Pole mass (kg)
            length: Pole length (m)
            gravity: Gravity constant (m/s^2)
            friction_cart: Cart friction coefficient
            friction_pole: Pole friction coefficient
        """
        super().__init__()
        self.name = "Cart-Pole Stabilization"
        self.description = "Stabilize inverted pendulum on cart"

        self.M = mass_cart
        self.m = mass_pole
        self.L = length
        self.g = gravity
        self.b_cart = friction_cart
        self.b_pole = friction_pole

        self.n_states = 4
        self.n_controls = 1

    def dynamics(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Cart-pole dynamics.

        Args:
            x: [x, x_dot, theta, theta_dot]
            u: [F]
            t: Time (unused)

        Returns:
            State derivative
        """
        pos, vel, theta, theta_dot = x
        F = u[0]

        # Total mass
        m_total = self.M + self.m

        # Sine and cosine
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Equations of motion
        temp = (F - self.b_cart * vel + self.m * self.L * theta_dot**2 * sin_theta) / m_total

        theta_ddot = (self.g * sin_theta - temp * cos_theta - self.b_pole * theta_dot / (self.m * self.L)) / \
                     (self.L * (4/3 - self.m * cos_theta**2 / m_total))

        x_ddot = temp + self.m * self.L * theta_ddot * cos_theta / m_total

        return np.array([vel, x_ddot, theta_dot, theta_ddot])

    def cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Quadratic cost for cart-pole.

        Penalizes:
        - Deviation from upright position
        - Cart position
        - Velocities
        - Control effort

        Args:
            x: State
            u: Control

        Returns:
            Cost
        """
        # State weights
        Q = np.diag([1.0, 0.1, 10.0, 0.1])  # Emphasize angle

        # Control weight
        R = np.array([[0.1]])

        # Goal state (upright at origin)
        x_goal = np.array([0.0, 0.0, 0.0, 0.0])

        # Quadratic cost
        state_cost = (x - x_goal).T @ Q @ (x - x_goal)
        control_cost = u.T @ R @ u

        return state_cost + control_cost

    def get_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get bounds.

        Returns:
            State and control bounds
        """
        return {
            'state': (
                np.array([-3.0, -5.0, -np.pi, -5.0]),  # Lower
                np.array([3.0, 5.0, np.pi, 5.0])       # Upper
            ),
            'control': (
                np.array([-10.0]),  # Lower
                np.array([10.0])    # Upper
            )
        }

    def get_initial_state(self) -> np.ndarray:
        """Get initial state (slightly perturbed from upright).

        Returns:
            Initial state
        """
        return np.array([0.0, 0.0, 0.2, 0.0])  # 0.2 rad tilt

    def get_goal_state(self) -> np.ndarray:
        """Get goal state (upright at origin).

        Returns:
            Goal state
        """
        return np.array([0.0, 0.0, 0.0, 0.0])


# =============================================================================
# Quadrotor Trajectory Tracking
# =============================================================================

class QuadrotorTrajectory(CaseStudy):
    """Quadrotor trajectory tracking in 2D.

    Simplified 2D quadrotor model (planar motion).

    States: [x, z, theta, x_dot, z_dot, theta_dot]
        - x, z: Position (horizontal, vertical)
        - theta: Pitch angle
        - x_dot, z_dot: Velocities
        - theta_dot: Angular velocity

    Controls: [u1, u2]
        - u1, u2: Thrust from left and right rotors
    """

    def __init__(
        self,
        mass: float = 0.5,
        length: float = 0.25,
        inertia: float = 0.01,
        gravity: float = 9.81
    ):
        """Initialize quadrotor.

        Args:
            mass: Quadrotor mass (kg)
            length: Half-distance between rotors (m)
            inertia: Moment of inertia (kg*m^2)
            gravity: Gravity (m/s^2)
        """
        super().__init__()
        self.name = "Quadrotor Trajectory Tracking"
        self.description = "Track reference trajectory with 2D quadrotor"

        self.m = mass
        self.L = length
        self.I = inertia
        self.g = gravity

        self.n_states = 6
        self.n_controls = 2

        # Reference trajectory parameters
        self.trajectory_type = "circle"
        self.trajectory_params = {'radius': 1.0, 'period': 10.0}

    def dynamics(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Quadrotor dynamics.

        Args:
            x: [x, z, theta, x_dot, z_dot, theta_dot]
            u: [u1, u2] - rotor thrusts
            t: Time

        Returns:
            State derivative
        """
        pos_x, pos_z, theta, vel_x, vel_z, omega = x
        u1, u2 = u

        # Total thrust and torque
        thrust = u1 + u2
        torque = self.L * (u1 - u2)

        # Accelerations
        x_ddot = -thrust * np.sin(theta) / self.m
        z_ddot = thrust * np.cos(theta) / self.m - self.g
        theta_ddot = torque / self.I

        return np.array([vel_x, vel_z, omega, x_ddot, z_ddot, theta_ddot])

    def reference_trajectory(self, t: float) -> np.ndarray:
        """Compute reference trajectory.

        Args:
            t: Time

        Returns:
            Reference state [x_ref, z_ref, theta_ref, ...]
        """
        if self.trajectory_type == "circle":
            radius = self.trajectory_params['radius']
            period = self.trajectory_params['period']
            omega = 2 * np.pi / period

            x_ref = radius * np.cos(omega * t)
            z_ref = radius * np.sin(omega * t) + radius + 1.0  # Offset

            x_dot_ref = -radius * omega * np.sin(omega * t)
            z_dot_ref = radius * omega * np.cos(omega * t)

            # Desired pitch angle (tangent to circle)
            theta_ref = np.arctan2(-x_dot_ref, z_dot_ref + self.g)

            return np.array([x_ref, z_ref, theta_ref, x_dot_ref, z_dot_ref, 0.0])

        elif self.trajectory_type == "hover":
            return np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        else:
            return np.zeros(6)

    def cost(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> float:
        """Tracking cost.

        Args:
            x: State
            u: Control
            t: Time (for reference)

        Returns:
            Cost
        """
        # Reference
        x_ref = self.reference_trajectory(t)

        # Weights
        Q = np.diag([10.0, 10.0, 1.0, 1.0, 1.0, 0.1])
        R = np.diag([0.1, 0.1])

        # Tracking error
        error = x - x_ref

        return error.T @ Q @ error + u.T @ R @ u

    def get_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get bounds.

        Returns:
            Bounds
        """
        return {
            'state': (
                np.array([-5.0, 0.0, -np.pi/2, -5.0, -5.0, -5.0]),
                np.array([5.0, 5.0, np.pi/2, 5.0, 5.0, 5.0])
            ),
            'control': (
                np.array([0.0, 0.0]),  # Non-negative thrust
                np.array([2*self.m*self.g, 2*self.m*self.g])  # Max 2x weight each
            )
        }

    def get_initial_state(self) -> np.ndarray:
        """Initial state (hovering).

        Returns:
            Initial state
        """
        return np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


# =============================================================================
# Robot Arm Control
# =============================================================================

class RobotArmControl(CaseStudy):
    """2-link robot arm reaching task.

    Planar robot arm with 2 revolute joints.

    States: [theta1, theta2, omega1, omega2]
        - theta1, theta2: Joint angles
        - omega1, omega2: Joint velocities

    Controls: [tau1, tau2]
        - tau1, tau2: Joint torques
    """

    def __init__(
        self,
        length1: float = 1.0,
        length2: float = 1.0,
        mass1: float = 1.0,
        mass2: float = 1.0,
        gravity: float = 9.81
    ):
        """Initialize robot arm.

        Args:
            length1: Link 1 length (m)
            length2: Link 2 length (m)
            mass1: Link 1 mass (kg)
            mass2: Link 2 mass (kg)
            gravity: Gravity (m/s^2)
        """
        super().__init__()
        self.name = "Robot Arm Control"
        self.description = "2-link planar robot arm reaching"

        self.L1 = length1
        self.L2 = length2
        self.m1 = mass1
        self.m2 = mass2
        self.g = gravity

        # Inertias (point masses at end of links)
        self.I1 = mass1 * length1**2 / 3
        self.I2 = mass2 * length2**2 / 3

        self.n_states = 4
        self.n_controls = 2

        # Goal position in task space
        self.goal_position = np.array([1.5, 0.5])

    def forward_kinematics(self, theta1: float, theta2: float) -> np.ndarray:
        """Compute end-effector position.

        Args:
            theta1: Joint 1 angle
            theta2: Joint 2 angle

        Returns:
            End-effector position [x, y]
        """
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)

        return np.array([x, y])

    def dynamics(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Robot arm dynamics (using Lagrangian mechanics).

        Args:
            x: [theta1, theta2, omega1, omega2]
            u: [tau1, tau2]
            t: Time

        Returns:
            State derivative
        """
        theta1, theta2, omega1, omega2 = x
        tau1, tau2 = u

        # Mass matrix M(theta)
        m11 = self.I1 + self.I2 + self.m2 * self.L1**2 + \
              2 * self.m2 * self.L1 * self.L2 * np.cos(theta2)
        m12 = self.I2 + self.m2 * self.L1 * self.L2 * np.cos(theta2)
        m22 = self.I2

        M = np.array([[m11, m12],
                      [m12, m22]])

        # Coriolis and centrifugal terms C(theta, omega)
        c1 = -self.m2 * self.L1 * self.L2 * np.sin(theta2) * \
             (2 * omega1 * omega2 + omega2**2)
        c2 = self.m2 * self.L1 * self.L2 * np.sin(theta2) * omega1**2

        C = np.array([c1, c2])

        # Gravity terms G(theta)
        g1 = (self.m1 + self.m2) * self.g * self.L1 * np.cos(theta1) + \
             self.m2 * self.g * self.L2 * np.cos(theta1 + theta2)
        g2 = self.m2 * self.g * self.L2 * np.cos(theta1 + theta2)

        G = np.array([g1, g2])

        # Solve for accelerations: M*alpha = tau - C - G
        tau = np.array([tau1, tau2])
        alpha = np.linalg.solve(M, tau - C - G)

        return np.array([omega1, omega2, alpha[0], alpha[1]])

    def cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Cost function for reaching.

        Args:
            x: State
            u: Control

        Returns:
            Cost
        """
        theta1, theta2, omega1, omega2 = x

        # End-effector position
        ee_pos = self.forward_kinematics(theta1, theta2)

        # Distance to goal
        position_error = np.linalg.norm(ee_pos - self.goal_position)**2

        # Velocity penalty
        velocity_cost = 0.1 * (omega1**2 + omega2**2)

        # Control effort
        control_cost = 0.01 * (u[0]**2 + u[1]**2)

        return position_error + velocity_cost + control_cost

    def get_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get bounds.

        Returns:
            Bounds
        """
        return {
            'state': (
                np.array([-np.pi, -np.pi, -5.0, -5.0]),
                np.array([np.pi, np.pi, 5.0, 5.0])
            ),
            'control': (
                np.array([-10.0, -10.0]),
                np.array([10.0, 10.0])
            )
        }

    def get_initial_state(self) -> np.ndarray:
        """Initial state (arm down).

        Returns:
            Initial state
        """
        return np.array([np.pi/2, 0.0, 0.0, 0.0])


# =============================================================================
# Energy System Optimization
# =============================================================================

class EnergySystemOptimization(CaseStudy):
    """Building energy management system.

    Optimizes heating/cooling to maintain comfort while minimizing cost.

    States: [T_indoor]
        - T_indoor: Indoor temperature

    Controls: [P_hvac]
        - P_hvac: HVAC power (positive = heating, negative = cooling)

    External: T_outdoor(t), electricity_price(t)
    """

    def __init__(
        self,
        thermal_mass: float = 1e6,  # J/K
        thermal_resistance: float = 0.01,  # K/W
        hvac_efficiency: float = 3.0,  # COP
        comfort_range: Tuple[float, float] = (20.0, 24.0)
    ):
        """Initialize energy system.

        Args:
            thermal_mass: Building thermal mass (J/K)
            thermal_resistance: Thermal resistance (K/W)
            hvac_efficiency: HVAC coefficient of performance
            comfort_range: Acceptable temperature range (°C)
        """
        super().__init__()
        self.name = "Energy System Optimization"
        self.description = "Building HVAC optimal control"

        self.C = thermal_mass
        self.R = thermal_resistance
        self.COP = hvac_efficiency
        self.T_comfort_min, self.T_comfort_max = comfort_range

        self.n_states = 1
        self.n_controls = 1

        # Time-varying parameters (defaults)
        self.outdoor_temp = lambda t: 10.0 + 5.0 * np.sin(2*np.pi*t/24.0)  # Daily cycle
        self.electricity_price = lambda t: 0.1 + 0.05 * np.sin(2*np.pi*t/24.0 - np.pi/2)  # Peak pricing

    def dynamics(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Thermal dynamics.

        Args:
            x: [T_indoor]
            u: [P_hvac]
            t: Time (hours)

        Returns:
            Temperature rate
        """
        T_indoor = x[0]
        P_hvac = u[0]

        # Outdoor temperature
        T_outdoor = self.outdoor_temp(t)

        # Heat transfer to outdoors
        Q_loss = (T_indoor - T_outdoor) / self.R

        # HVAC heating/cooling
        Q_hvac = self.COP * P_hvac

        # Temperature rate: C*dT/dt = Q_hvac - Q_loss
        dT_dt = (Q_hvac - Q_loss) / self.C

        return np.array([dT_dt])

    def cost(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> float:
        """Cost: electricity + discomfort.

        Args:
            x: State
            u: Control
            t: Time

        Returns:
            Cost
        """
        T_indoor = x[0]
        P_hvac = u[0]

        # Electricity cost
        price = self.electricity_price(t)
        electricity_cost = price * np.abs(P_hvac)

        # Discomfort penalty
        if T_indoor < self.T_comfort_min:
            discomfort = 10.0 * (self.T_comfort_min - T_indoor)**2
        elif T_indoor > self.T_comfort_max:
            discomfort = 10.0 * (T_indoor - self.T_comfort_max)**2
        else:
            discomfort = 0.0

        return electricity_cost + discomfort

    def get_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get bounds.

        Returns:
            Bounds
        """
        return {
            'state': (
                np.array([10.0]),  # Min temp
                np.array([30.0])   # Max temp
            ),
            'control': (
                np.array([-5000.0]),  # Max cooling power (W)
                np.array([5000.0])    # Max heating power (W)
            )
        }

    def get_initial_state(self) -> np.ndarray:
        """Initial state.

        Returns:
            Initial temperature
        """
        return np.array([22.0])  # Comfortable start


# =============================================================================
# Portfolio Optimization
# =============================================================================

class PortfolioOptimization(CaseStudy):
    """Dynamic portfolio optimization with transaction costs.

    States: [w1, ..., wn, cash]
        - wi: Holdings in asset i (shares)
        - cash: Cash position

    Controls: [u1, ..., un]
        - ui: Shares to buy/sell of asset i
    """

    def __init__(
        self,
        n_assets: int = 3,
        returns_mean: Optional[np.ndarray] = None,
        returns_cov: Optional[np.ndarray] = None,
        transaction_cost: float = 0.001
    ):
        """Initialize portfolio.

        Args:
            n_assets: Number of assets
            returns_mean: Expected returns (default: random)
            returns_cov: Covariance matrix (default: random)
            transaction_cost: Transaction cost rate
        """
        super().__init__()
        self.name = "Portfolio Optimization"
        self.description = "Dynamic portfolio with transaction costs"

        self.n_assets = n_assets
        self.transaction_cost = transaction_cost

        # Default parameters
        if returns_mean is None:
            self.mu = np.random.uniform(0.05, 0.15, n_assets)
        else:
            self.mu = returns_mean

        if returns_cov is None:
            # Generate random positive definite covariance
            A = np.random.randn(n_assets, n_assets)
            self.Sigma = A.T @ A * 0.01
        else:
            self.Sigma = returns_cov

        self.n_states = n_assets + 1  # Holdings + cash
        self.n_controls = n_assets  # Buy/sell decisions

        # Asset prices (simplified: constant)
        self.prices = np.ones(n_assets)

    def dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
        t: float = 0.0,
        dt: float = 1.0
    ) -> np.ndarray:
        """Portfolio dynamics (discrete-time).

        Args:
            x: [w1, ..., wn, cash]
            u: [u1, ..., un] - shares to buy/sell
            t: Time
            dt: Time step

        Returns:
            Next state
        """
        holdings = x[:-1]
        cash = x[-1]

        # Execute trades
        transaction_value = np.sum(np.abs(u) * self.prices)
        transaction_fee = self.transaction_cost * transaction_value

        new_holdings = holdings + u
        new_cash = cash - np.sum(u * self.prices) - transaction_fee

        # Asset returns (simplified: deterministic)
        returns = self.mu * dt

        # Update holdings value
        new_holdings = new_holdings * (1 + returns)

        return np.concatenate([new_holdings, [new_cash]])

    def cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Cost: negative expected return with risk penalty.

        Args:
            x: State
            u: Control

        Returns:
            Cost (negative return)
        """
        holdings = x[:-1]
        cash = x[-1]

        # Portfolio value
        portfolio_value = np.sum(holdings * self.prices) + cash

        # Expected return
        weights = (holdings * self.prices) / (portfolio_value + 1e-8)
        expected_return = np.dot(weights, self.mu)

        # Risk (variance)
        variance = weights.T @ self.Sigma @ weights

        # Cost: negative return + risk penalty
        risk_aversion = 0.5
        return -(expected_return - risk_aversion * variance)

    def get_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get bounds.

        Returns:
            Bounds
        """
        return {
            'state': (
                np.concatenate([np.zeros(self.n_assets), [0.0]]),  # No short selling
                np.concatenate([np.ones(self.n_assets) * 1000.0, [10000.0]])
            ),
            'control': (
                -np.ones(self.n_assets) * 100.0,  # Max sell
                np.ones(self.n_assets) * 100.0    # Max buy
            )
        }

    def get_initial_state(self) -> np.ndarray:
        """Initial state (all cash).

        Returns:
            Initial state
        """
        return np.concatenate([np.zeros(self.n_assets), [1000.0]])


# =============================================================================
# Chemical Reactor Control
# =============================================================================

class ChemicalReactorControl(CaseStudy):
    """Continuous stirred tank reactor (CSTR) control.

    Exothermic reaction: A → B

    States: [C_A, T]
        - C_A: Concentration of A (mol/L)
        - T: Temperature (K)

    Controls: [Q, F]
        - Q: Cooling/heating rate (W)
        - F: Feed flow rate (L/s)
    """

    def __init__(
        self,
        volume: float = 1.0,
        rate_constant: float = 1e10,
        activation_energy: float = 8e4,
        heat_reaction: float = -2e5,
        density: float = 1000.0,
        heat_capacity: float = 4184.0
    ):
        """Initialize CSTR.

        Args:
            volume: Reactor volume (L)
            rate_constant: Arrhenius pre-exponential factor (1/s)
            activation_energy: Activation energy (J/mol)
            heat_reaction: Heat of reaction (J/mol)
            density: Liquid density (kg/m^3)
            heat_capacity: Heat capacity (J/kg/K)
        """
        super().__init__()
        self.name = "Chemical Reactor Control"
        self.description = "CSTR temperature and concentration control"

        self.V = volume
        self.k0 = rate_constant
        self.Ea = activation_energy
        self.dH = heat_reaction
        self.rho = density
        self.Cp = heat_capacity

        self.R = 8.314  # Gas constant (J/mol/K)

        # Feed conditions
        self.C_A_feed = 10.0  # mol/L
        self.T_feed = 300.0   # K

        # Desired setpoint
        self.C_A_setpoint = 2.0
        self.T_setpoint = 350.0

        self.n_states = 2
        self.n_controls = 2

    def reaction_rate(self, C_A: float, T: float) -> float:
        """Arrhenius reaction rate.

        Args:
            C_A: Concentration
            T: Temperature

        Returns:
            Reaction rate
        """
        k = self.k0 * np.exp(-self.Ea / (self.R * T))
        return k * C_A

    def dynamics(self, x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """CSTR dynamics.

        Args:
            x: [C_A, T]
            u: [Q, F]
            t: Time

        Returns:
            State derivative
        """
        C_A, T = x
        Q, F = u

        # Reaction rate
        r = self.reaction_rate(C_A, T)

        # Mass balance for A
        dC_A_dt = F / self.V * (self.C_A_feed - C_A) - r

        # Energy balance
        dT_dt = (F / self.V * (self.T_feed - T) +
                 (-self.dH) * r / (self.rho * self.Cp) +
                 Q / (self.V * self.rho * self.Cp))

        return np.array([dC_A_dt, dT_dt])

    def cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Cost: tracking error + input penalty.

        Args:
            x: State
            u: Control

        Returns:
            Cost
        """
        C_A, T = x
        Q, F = u

        # Tracking error
        error_C = (C_A - self.C_A_setpoint)**2
        error_T = ((T - self.T_setpoint) / 100.0)**2  # Normalized

        # Input costs
        cost_Q = 0.01 * Q**2
        cost_F = 0.1 * F**2

        return error_C + error_T + cost_Q + cost_F

    def get_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get bounds.

        Returns:
            Bounds
        """
        return {
            'state': (
                np.array([0.0, 250.0]),     # Min concentration, temp
                np.array([15.0, 450.0])     # Max concentration, temp
            ),
            'control': (
                np.array([-5e5, 0.0]),      # Cooling power, min flow
                np.array([5e5, 5.0])        # Heating power, max flow
            )
        }

    def get_initial_state(self) -> np.ndarray:
        """Initial state (steady-state).

        Returns:
            Initial state
        """
        return np.array([5.0, 330.0])

    def get_goal_state(self) -> np.ndarray:
        """Goal state.

        Returns:
            Setpoint
        """
        return np.array([self.C_A_setpoint, self.T_setpoint])
