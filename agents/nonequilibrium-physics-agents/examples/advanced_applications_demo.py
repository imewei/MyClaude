"""Advanced Applications Demo.

Comprehensive demonstrations of:
1. Multi-objective optimization
2. Robust control
3. Stochastic control
4. Real-world case studies

Each demo is self-contained and can be run independently.

Author: Nonequilibrium Physics Agents
"""

import numpy as np
import matplotlib.pyplot as plt
from applications.multi_objective import (
    WeightedSumMethod, NSGA2Optimizer, MultiObjectiveOptimizer
)
from applications.robust_control import (
    UncertaintySet, UncertaintySetType,
    MinMaxOptimizer, TubeBasedMPC
)
from applications.stochastic_control import (
    RiskMeasure, CVaROptimizer,
    StochasticMPC, SampleAverageApproximation
)
from applications.case_studies import (
    CartPoleStabilization, QuadrotorTrajectory,
    RobotArmControl, EnergySystemOptimization
)


# =============================================================================
# Demo 1: Multi-Objective Optimization
# =============================================================================

def demo_multi_objective():
    """Demo: Multi-objective optimization with Pareto front."""
    print("="*70)
    print("Demo 1: Multi-Objective Optimization")
    print("="*70)
    print("\nProblem: Bi-objective optimization")
    print("  Objective 1: Minimize x^2")
    print("  Objective 2: Minimize (x-2)^2")
    print("  Domain: x ∈ [0, 2]")
    print()

    # Define objectives
    obj1 = lambda x: x[0]**2
    obj2 = lambda x: (x[0] - 2)**2

    bounds = (np.array([0.0]), np.array([2.0]))

    # Method 1: Weighted Sum
    print("Method 1: Weighted Sum")
    ws_method = WeightedSumMethod([obj1, obj2], bounds=bounds)
    front_ws = ws_method.compute_pareto_front(n_points=10)

    print(f"  Found {len(front_ws)} Pareto-optimal solutions")

    # Method 2: NSGA-II
    print("\nMethod 2: NSGA-II (Evolutionary)")
    nsga2 = NSGA2Optimizer(
        [obj1, obj2],
        bounds=bounds,
        population_size=50,
        n_generations=20
    )
    front_nsga2 = nsga2.optimize(verbose=False)

    print(f"  Found {len(front_nsga2)} Pareto-optimal solutions")

    # Visualize
    try:
        obj_matrix_ws = front_ws.get_objectives_matrix()
        obj_matrix_nsga2 = front_nsga2.get_objectives_matrix()

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(obj_matrix_ws[:, 0], obj_matrix_ws[:, 1], label='Weighted Sum', alpha=0.7)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front (Weighted Sum)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(obj_matrix_nsga2[:, 0], obj_matrix_nsga2[:, 1], label='NSGA-II', alpha=0.7, color='red')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front (NSGA-II)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('multi_objective_demo.png', dpi=150)
        print("\n  Plot saved as 'multi_objective_demo.png'")
    except Exception as e:
        print(f"\n  Plotting skipped: {e}")

    print("\n" + "="*70 + "\n")


# =============================================================================
# Demo 2: Robust Control
# =============================================================================

def demo_robust_control():
    """Demo: Robust MPC with tube-based approach."""
    print("="*70)
    print("Demo 2: Robust Model Predictive Control")
    print("="*70)
    print("\nProblem: Control double integrator with disturbances")
    print("  Dynamics: x_{k+1} = Ax_k + Bu_k + w_k")
    print("  Disturbance: w ∈ [-0.1, 0.1]^2")
    print("  Goal: Stabilize to origin")
    print()

    # System matrices (double integrator)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])

    # Cost matrices
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])

    # Uncertainty set
    unc_set = UncertaintySet(
        set_type=UncertaintySetType.BOX,
        dimension=2,
        parameters={'lower': np.array([-0.1, -0.1]),
                   'upper': np.array([0.1, 0.1])}
    )

    print("Creating Tube-based MPC...")
    mpc = TubeBasedMPC(
        A, B, Q, R, unc_set,
        control_constraints=(np.array([-1.0]), np.array([1.0])),
        horizon=10
    )

    print(f"  Ancillary controller K computed")
    print(f"  MRPI set computed")

    # Simulate
    print("\nSimulating closed-loop control...")
    T = 50
    x = np.array([2.0, 0.0])  # Initial state

    states = [x]
    controls = []

    for t in range(T):
        # Compute control
        u = mpc.plan(x)
        controls.append(u)

        # Simulate with random disturbance
        w = np.random.uniform(-0.1, 0.1, size=2)
        x = A @ x + B @ u + w

        states.append(x)

    states = np.array(states)
    controls = np.array(controls)

    print(f"  Simulation complete")
    print(f"  Final state: {states[-1]}")
    print(f"  Final position error: {abs(states[-1, 0]):.4f}")

    # Plot
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(states[:, 0], label='Position')
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time step')
        plt.ylabel('Position')
        plt.title('State Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(states[:, 1], label='Velocity')
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time step')
        plt.ylabel('Velocity')
        plt.title('Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(controls, label='Control')
        plt.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Bounds')
        plt.axhline(-1.0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time step')
        plt.ylabel('Control input')
        plt.title('Control Signal')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('robust_mpc_demo.png', dpi=150)
        print("\n  Plot saved as 'robust_mpc_demo.png'")
    except Exception as e:
        print(f"\n  Plotting skipped: {e}")

    print("\n" + "="*70 + "\n")


# =============================================================================
# Demo 3: Stochastic Control
# =============================================================================

def demo_stochastic_control():
    """Demo: CVaR optimization for risk-aware control."""
    print("="*70)
    print("Demo 3: Risk-Aware Stochastic Control (CVaR)")
    print("="*70)
    print("\nProblem: Control with stochastic disturbances")
    print("  Objective: (u - ξ)^2 where ξ ~ N(0, 1)")
    print("  Risk measure: CVaR at 95% confidence")
    print("  Goal: Find control robust to tail events")
    print()

    # Define stochastic objective
    objective = lambda u, xi: (u[0] - xi[0])**2

    # Gaussian disturbance sampler
    sampler = lambda n: np.random.randn(n, 1)

    # Risk-neutral optimization (expectation)
    print("Risk-neutral optimization (minimize expectation)...")
    saa_neutral = SampleAverageApproximation(
        objective,
        sampler,
        control_bounds=(np.array([-3.0]), np.array([3.0]))
    )

    result_neutral = saa_neutral.solve_saa(
        u0=np.array([1.0]),
        n_samples=1000
    )

    print(f"  Optimal control: {result_neutral['control'][0]:.4f}")
    print(f"  Expected cost: {result_neutral['objective']:.4f}")

    # Risk-averse optimization (CVaR)
    print("\nRisk-averse optimization (minimize CVaR)...")
    cvar_optimizer = CVaROptimizer(
        objective,
        sampler,
        control_bounds=(np.array([-3.0]), np.array([3.0])),
        alpha=0.95
    )

    result_cvar = cvar_optimizer.optimize(
        u0=np.array([1.0]),
        n_samples=1000
    )

    print(f"  Optimal control: {result_cvar['control'][0]:.4f}")
    print(f"  CVaR (95%): {result_cvar['cvar']:.4f}")
    print(f"  Mean cost: {result_cvar['mean_cost']:.4f}")
    print(f"  VaR (95%): {result_cvar['var']:.4f}")

    # Compare distributions
    print("\nComparing cost distributions...")
    n_test = 1000
    test_samples = sampler(n_test)

    costs_neutral = [objective(result_neutral['control'], xi) for xi in test_samples]
    costs_cvar = [objective(result_cvar['control'], xi) for xi in test_samples]

    print(f"  Risk-neutral - Mean: {np.mean(costs_neutral):.4f}, Std: {np.std(costs_neutral):.4f}")
    print(f"  Risk-averse   - Mean: {np.mean(costs_cvar):.4f}, Std: {np.std(costs_cvar):.4f}")

    # Plot distributions
    try:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(costs_neutral, bins=50, alpha=0.7, label='Risk-neutral', density=True)
        plt.hist(costs_cvar, bins=50, alpha=0.7, label='Risk-averse (CVaR)', density=True)
        plt.xlabel('Cost')
        plt.ylabel('Density')
        plt.title('Cost Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(sorted(costs_neutral), np.linspace(0, 1, len(costs_neutral)), label='Risk-neutral')
        plt.plot(sorted(costs_cvar), np.linspace(0, 1, len(costs_cvar)), label='Risk-averse (CVaR)')
        plt.axvline(result_cvar['var'], color='r', linestyle='--', alpha=0.5, label='VaR (95%)')
        plt.xlabel('Cost')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('stochastic_control_demo.png', dpi=150)
        print("\n  Plot saved as 'stochastic_control_demo.png'")
    except Exception as e:
        print(f"\n  Plotting skipped: {e}")

    print("\n" + "="*70 + "\n")


# =============================================================================
# Demo 4: Cart-Pole Stabilization
# =============================================================================

def demo_cart_pole():
    """Demo: Cart-pole stabilization with stochastic MPC."""
    print("="*70)
    print("Demo 4: Cart-Pole Stabilization")
    print("="*70)
    print("\nProblem: Stabilize inverted pendulum on cart")
    print("  States: [position, velocity, angle, angular_velocity]")
    print("  Control: Force on cart")
    print("  Method: Stochastic MPC with process noise")
    print()

    # Create cart-pole system
    cart_pole = CartPoleStabilization()

    print("System parameters:")
    print(f"  Cart mass: {cart_pole.M} kg")
    print(f"  Pole mass: {cart_pole.m} kg")
    print(f"  Pole length: {cart_pole.L} m")

    # Simplified dynamics for MPC (linearized)
    def linearized_dynamics(x, u, w):
        # Simple forward Euler with noise
        dt = 0.05
        x_dot = cart_pole.dynamics(x, u)
        return x + dt * x_dot + w

    # Cost function
    def stage_cost(x, u):
        return cart_pole.cost(x, u) * 0.05  # Scale by dt

    # Gaussian process noise
    noise_std = 0.01
    sampler = lambda n: np.random.randn(n, 4) * noise_std

    # Create stochastic MPC
    print("\nCreating Stochastic MPC controller...")
    bounds = cart_pole.get_bounds()

    mpc = StochasticMPC(
        linearized_dynamics,
        stage_cost,
        sampler,
        horizon=10,
        control_bounds=bounds['control'],
        risk_measure=RiskMeasure.EXPECTATION
    )

    # Simulate
    print("Simulating control...")
    T = 100
    dt = 0.05

    x = cart_pole.get_initial_state()
    print(f"  Initial state: {x}")

    states = [x]
    controls = []

    for t in range(T):
        # Compute control
        u = mpc.plan(x, n_scenarios=20)
        controls.append(u)

        # Simulate with noise
        noise = np.random.randn(4) * noise_std
        x_dot = cart_pole.dynamics(x, u)
        x = x + dt * x_dot + noise

        states.append(x)

        # Check stability
        if abs(x[2]) > 1.0:  # Angle > ~60 degrees
            print(f"  Pole fell at t={t*dt:.2f}s")
            break

    states = np.array(states)
    controls = np.array(controls)

    print(f"  Simulation complete")
    print(f"  Final state: {states[-1]}")
    print(f"  Final angle: {abs(states[-1, 2]) * 180 / np.pi:.2f} degrees")

    # Plot
    try:
        time = np.arange(len(states)) * dt

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(time, states[:, 0])
        plt.xlabel('Time (s)')
        plt.ylabel('Cart Position (m)')
        plt.title('Cart Position')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(time, states[:, 2] * 180 / np.pi)
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Pole Angle (deg)')
        plt.title('Pole Angle')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        if len(controls) > 0:
            plt.plot(time[:-1], controls)
            plt.xlabel('Time (s)')
            plt.ylabel('Force (N)')
            plt.title('Control Input')
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(states[:, 0], states[:, 2] * 180 / np.pi)
        plt.scatter([states[0, 0]], [states[0, 2] * 180 / np.pi],
                   c='g', s=100, marker='o', label='Start', zorder=5)
        plt.scatter([states[-1, 0]], [states[-1, 2] * 180 / np.pi],
                   c='r', s=100, marker='x', label='End', zorder=5)
        plt.xlabel('Cart Position (m)')
        plt.ylabel('Pole Angle (deg)')
        plt.title('Phase Portrait')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('cart_pole_demo.png', dpi=150)
        print("\n  Plot saved as 'cart_pole_demo.png'")
    except Exception as e:
        print(f"\n  Plotting skipped: {e}")

    print("\n" + "="*70 + "\n")


# =============================================================================
# Demo 5: Energy System Optimization
# =============================================================================

def demo_energy_system():
    """Demo: Building energy management with MPC."""
    print("="*70)
    print("Demo 5: Building Energy Management")
    print("="*70)
    print("\nProblem: Optimize HVAC control for comfort + cost")
    print("  State: Indoor temperature")
    print("  Control: HVAC heating/cooling power")
    print("  Objective: Minimize electricity cost + discomfort")
    print("  Constraints: Comfort range [20, 24]°C")
    print()

    # Create energy system
    energy = EnergySystemOptimization()

    print("System parameters:")
    print(f"  Thermal mass: {energy.C:.0f} J/K")
    print(f"  Thermal resistance: {energy.R:.4f} K/W")
    print(f"  HVAC COP: {energy.COP}")
    print(f"  Comfort range: [{energy.T_comfort_min}, {energy.T_comfort_max}]°C")

    # Simulate 24-hour period
    print("\nSimulating 24-hour period...")
    dt = 0.5  # hours
    T_hours = 24
    n_steps = int(T_hours / dt)

    # Simple MPC-like control
    x = energy.get_initial_state()
    time = []
    states = [x[0]]
    controls = []
    outdoor_temps = []
    electricity_prices = []
    costs = []

    for step in range(n_steps):
        t = step * dt

        # Record outdoor conditions
        T_out = energy.outdoor_temp(t)
        price = energy.electricity_price(t)
        outdoor_temps.append(T_out)
        electricity_prices.append(price)

        # Simple proportional control with cost awareness
        T_indoor = x[0]

        # Target temperature (lower when expensive)
        T_target = 22.0 - (price - 0.1) * 10  # Shift target based on price

        # Proportional control
        error = T_target - T_indoor
        u = np.array([5000.0 * error])  # kW

        # Clip to bounds
        bounds = energy.get_bounds()['control']
        u = np.clip(u, bounds[0], bounds[1])

        controls.append(u[0])

        # Compute cost
        cost = energy.cost(x, u, t)
        costs.append(cost)

        # Simulate
        x_dot = energy.dynamics(x, u, t)
        x = x + dt * 3600 * x_dot  # Convert hours to seconds

        time.append(t)
        states.append(x[0])

    print(f"  Simulation complete")
    print(f"  Total cost: ${sum(costs) * dt:.2f}")
    print(f"  Average temperature: {np.mean(states):.2f}°C")

    # Count violations
    violations = sum(1 for T in states if T < energy.T_comfort_min or T > energy.T_comfort_max)
    print(f"  Comfort violations: {violations}/{len(states)} timesteps")

    # Plot
    try:
        time_plot = np.array(time + [time[-1] + dt])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Temperature
        ax = axes[0, 0]
        ax.plot(time_plot, states, label='Indoor', linewidth=2)
        ax.plot(time, outdoor_temps, label='Outdoor', linestyle='--', alpha=0.7)
        ax.axhline(energy.T_comfort_min, color='r', linestyle=':', alpha=0.5, label='Comfort range')
        ax.axhline(energy.T_comfort_max, color='r', linestyle=':', alpha=0.5)
        ax.fill_between([0, 24], energy.T_comfort_min, energy.T_comfort_max, alpha=0.1, color='g')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # HVAC Power
        ax = axes[0, 1]
        ax.plot(time, np.array(controls) / 1000, linewidth=2)  # Convert to kW
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('HVAC Power (kW)')
        ax.set_title('HVAC Control')
        ax.grid(True, alpha=0.3)

        # Electricity Price
        ax = axes[1, 0]
        ax.plot(time, electricity_prices, color='orange', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Price ($/kWh)')
        ax.set_title('Electricity Price')
        ax.grid(True, alpha=0.3)

        # Cost
        ax = axes[1, 1]
        ax.plot(time, np.cumsum(costs) * dt, linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Cumulative Cost ($)')
        ax.set_title('Total Cost')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('energy_system_demo.png', dpi=150)
        print("\n  Plot saved as 'energy_system_demo.png'")
    except Exception as e:
        print(f"\n  Plotting skipped: {e}")

    print("\n" + "="*70 + "\n")


# =============================================================================
# Main Demo Runner
# =============================================================================

def main():
    """Run all demos."""
    print("\n")
    print("="*70)
    print(" "*15 + "ADVANCED APPLICATIONS DEMO SUITE")
    print("="*70)
    print("\nThis suite demonstrates:")
    print("  1. Multi-objective optimization (Pareto fronts)")
    print("  2. Robust control (Tube-based MPC)")
    print("  3. Stochastic control (CVaR optimization)")
    print("  4. Cart-pole stabilization (Case study)")
    print("  5. Energy system optimization (Case study)")
    print("\n" + "="*70 + "\n")

    demos = [
        ("Multi-Objective Optimization", demo_multi_objective),
        ("Robust Control", demo_robust_control),
        ("Stochastic Control", demo_stochastic_control),
        ("Cart-Pole Stabilization", demo_cart_pole),
        ("Energy System Optimization", demo_energy_system),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            print("Continuing to next demo...\n")

    print("="*70)
    print(" "*20 + "ALL DEMOS COMPLETE")
    print("="*70)
    print("\nGenerated plots:")
    print("  - multi_objective_demo.png")
    print("  - robust_mpc_demo.png")
    print("  - stochastic_control_demo.png")
    print("  - cart_pole_demo.png")
    print("  - energy_system_demo.png")
    print("\n")


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run all demos
    main()
