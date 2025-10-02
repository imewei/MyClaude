"""Tests for Advanced Applications.

Comprehensive test suite for:
- Multi-objective optimization
- Robust control
- Stochastic control
- Case studies

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
from applications.multi_objective import (
    ParetoFront, ParetoSolution,
    WeightedSumMethod, EpsilonConstraintMethod,
    NSGA2Optimizer, MultiObjectiveOptimizer
)
from applications.robust_control import (
    UncertaintySet, UncertaintySetType,
    MinMaxOptimizer, RobustOptimizer,
    TubeBasedMPC, HInfinityController
)
from applications.stochastic_control import (
    RiskMeasure, compute_risk,
    CVaROptimizer, RiskAwareOptimizer,
    StochasticMPC, ChanceConstraint,
    SampleAverageApproximation
)
from applications.case_studies import (
    CartPoleStabilization, QuadrotorTrajectory,
    RobotArmControl, EnergySystemOptimization,
    PortfolioOptimization, ChemicalReactorControl
)


# =============================================================================
# Multi-Objective Optimization Tests
# =============================================================================

class TestParetoFront:
    """Test Pareto front data structure."""

    def test_add_solution(self):
        """Test adding solutions to Pareto front."""
        front = ParetoFront()

        sol1 = ParetoSolution(
            decision_variables=np.array([1.0, 2.0]),
            objectives=np.array([1.0, 2.0])
        )
        front.add_solution(sol1)

        assert len(front) == 1

    def test_filter_dominated(self):
        """Test filtering dominated solutions."""
        front = ParetoFront()

        # Add non-dominated solutions
        sol1 = ParetoSolution(
            decision_variables=np.array([1.0]),
            objectives=np.array([1.0, 2.0])
        )
        sol2 = ParetoSolution(
            decision_variables=np.array([2.0]),
            objectives=np.array([2.0, 1.0])
        )

        # Add dominated solution
        sol3 = ParetoSolution(
            decision_variables=np.array([3.0]),
            objectives=np.array([3.0, 3.0])  # Dominated
        )

        front.add_solution(sol1)
        front.add_solution(sol2)
        front.add_solution(sol3)

        front.filter_dominated()

        assert len(front) == 2  # sol3 should be removed

    def test_hypervolume_2d(self):
        """Test hypervolume computation for 2D objectives."""
        front = ParetoFront()

        sol1 = ParetoSolution(
            decision_variables=np.array([1.0]),
            objectives=np.array([1.0, 2.0])
        )
        sol2 = ParetoSolution(
            decision_variables=np.array([2.0]),
            objectives=np.array([2.0, 1.0])
        )

        front.add_solution(sol1)
        front.add_solution(sol2)

        reference = np.array([3.0, 3.0])
        hv = front.compute_hypervolume(reference)

        assert hv > 0


class TestWeightedSumMethod:
    """Test weighted sum scalarization."""

    def test_scalarize(self):
        """Test scalarization with weights."""
        # Simple objectives
        obj1 = lambda x: x[0]**2
        obj2 = lambda x: (x[0] - 1)**2

        method = WeightedSumMethod([obj1, obj2])

        x = np.array([0.5])
        weights = np.array([0.5, 0.5])

        cost = method.scalarize(x, weights)

        assert cost == pytest.approx(0.5 * 0.25 + 0.5 * 0.25)

    def test_optimize_single(self):
        """Test single optimization with weights."""
        obj1 = lambda x: x[0]**2
        obj2 = lambda x: (x[0] - 2)**2

        bounds = (np.array([0.0]), np.array([2.0]))
        method = WeightedSumMethod([obj1, obj2], bounds=bounds)

        weights = np.array([0.5, 0.5])
        solution = method.optimize_single(weights, x0=np.array([1.0]))

        assert solution.metadata['success']
        # Optimum should be at x=1 for equal weights
        assert solution.decision_variables[0] == pytest.approx(1.0, abs=0.1)

    def test_compute_pareto_front(self):
        """Test Pareto front computation."""
        obj1 = lambda x: x[0]**2
        obj2 = lambda x: (x[0] - 2)**2

        bounds = (np.array([0.0]), np.array([2.0]))
        method = WeightedSumMethod([obj1, obj2], bounds=bounds)

        front = method.compute_pareto_front(n_points=5)

        assert len(front) > 0
        assert len(front) <= 5


class TestNSGA2:
    """Test NSGA-II evolutionary algorithm."""

    def test_initialize_population(self):
        """Test population initialization."""
        obj1 = lambda x: x[0]**2 + x[1]**2
        obj2 = lambda x: (x[0] - 1)**2 + (x[1] - 1)**2

        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        nsga2 = NSGA2Optimizer(
            [obj1, obj2],
            bounds=bounds,
            population_size=20,
            n_generations=5
        )

        pop = nsga2.initialize_population()

        assert pop.shape == (20, 2)
        assert np.all(pop >= 0.0)
        assert np.all(pop <= 1.0)

    def test_non_dominated_sort(self):
        """Test non-dominated sorting."""
        obj1 = lambda x: x[0]**2 + x[1]**2
        obj2 = lambda x: (x[0] - 1)**2 + (x[1] - 1)**2

        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        nsga2 = NSGA2Optimizer([obj1, obj2], bounds=bounds, population_size=10)

        # Create test objectives
        objectives = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0],  # Dominated
            [0.5, 2.5],
        ])

        fronts = nsga2.non_dominated_sort(objectives)

        # First front should not include dominated point
        assert 2 not in fronts[0]

    def test_optimize(self):
        """Test full NSGA-II optimization."""
        obj1 = lambda x: x[0]**2 + x[1]**2
        obj2 = lambda x: (x[0] - 1)**2 + (x[1] - 1)**2

        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        nsga2 = NSGA2Optimizer(
            [obj1, obj2],
            bounds=bounds,
            population_size=20,
            n_generations=10
        )

        front = nsga2.optimize()

        assert len(front) > 0


# =============================================================================
# Robust Control Tests
# =============================================================================

class TestUncertaintySet:
    """Test uncertainty set representations."""

    def test_box_uncertainty(self):
        """Test box uncertainty set."""
        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=2,
            parameters={'lower': np.array([-1.0, -1.0]),
                       'upper': np.array([1.0, 1.0])}
        )

        # Test containment
        assert unc_set.contains(np.array([0.0, 0.0]))
        assert unc_set.contains(np.array([1.0, 1.0]))
        assert not unc_set.contains(np.array([2.0, 0.0]))

    def test_ellipsoidal_uncertainty(self):
        """Test ellipsoidal uncertainty set."""
        P = np.eye(2)
        unc_set = UncertaintySet(
            set_type=UncertaintySetType.ELLIPSOIDAL,
            dimension=2,
            parameters={'P': P, 'center': np.zeros(2)}
        )

        # Center should be in set
        assert unc_set.contains(np.zeros(2))

        # Point on boundary
        assert unc_set.contains(np.array([1.0, 0.0]))

    def test_sample_box(self):
        """Test sampling from box uncertainty."""
        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=2,
            parameters={'lower': np.array([0.0, 0.0]),
                       'upper': np.array([1.0, 1.0])}
        )

        samples = unc_set.sample(n_samples=10)

        assert samples.shape == (10, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_get_vertices_box(self):
        """Test getting vertices of box."""
        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=2,
            parameters={'lower': np.array([0.0, 0.0]),
                       'upper': np.array([1.0, 1.0])}
        )

        vertices = unc_set.get_vertices()

        assert vertices.shape == (4, 2)  # 2^2 corners


class TestMinMaxOptimizer:
    """Test min-max robust optimization."""

    def test_evaluate_worst_case(self):
        """Test worst-case evaluation."""
        # Simple quadratic: x^2 + w*x
        objective = lambda u, w: u[0]**2 + w[0] * u[0]

        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=1,
            parameters={'lower': np.array([-1.0]),
                       'upper': np.array([1.0])}
        )

        optimizer = MinMaxOptimizer(objective, None, unc_set)

        u = np.array([1.0])
        worst_cost, worst_w = optimizer.evaluate_worst_case(u, n_samples=20)

        assert worst_cost > 0

    def test_optimize(self):
        """Test min-max optimization."""
        # Objective: (u - w)^2, want u robust to w ∈ [-1, 1]
        objective = lambda u, w: (u[0] - w[0])**2

        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=1,
            parameters={'lower': np.array([-1.0]),
                       'upper': np.array([1.0])}
        )

        bounds = (np.array([-2.0]), np.array([2.0]))
        optimizer = MinMaxOptimizer(objective, bounds, unc_set)

        result = optimizer.optimize(
            u0=np.array([0.5]),
            n_samples=10,
            optimizer_kwargs={'method': 'L-BFGS-B'}
        )

        assert result['success']
        # Optimal should be around u=0 (center of uncertainty)
        assert abs(result['control'][0]) < 0.5


class TestTubeBasedMPC:
    """Test tube-based MPC."""

    def test_initialization(self):
        """Test tube MPC initialization."""
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.eye(1)

        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=2,
            parameters={'lower': np.array([-0.1, -0.1]),
                       'upper': np.array([0.1, 0.1])}
        )

        mpc = TubeBasedMPC(A, B, Q, R, unc_set, horizon=5)

        assert mpc.K is not None
        assert mpc.mrpi_set is not None

    def test_plan(self):
        """Test MPC planning."""
        A = np.array([[1.0, 0.1], [0.0, 0.9]])
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.eye(1)

        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=2,
            parameters={'lower': np.array([-0.01, -0.01]),
                       'upper': np.array([0.01, 0.01])}
        )

        mpc = TubeBasedMPC(
            A, B, Q, R, unc_set,
            control_constraints=(np.array([-1.0]), np.array([1.0])),
            horizon=5
        )

        x0 = np.array([1.0, 0.0])
        u_opt = mpc.plan(x0)

        assert u_opt.shape == (1,)
        assert -1.0 <= u_opt[0] <= 1.0


# =============================================================================
# Stochastic Control Tests
# =============================================================================

class TestRiskMeasures:
    """Test risk measure computations."""

    def test_expectation(self):
        """Test expectation risk measure."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        risk = compute_risk(samples, RiskMeasure.EXPECTATION)

        assert risk == 3.0

    def test_variance(self):
        """Test variance risk measure."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        risk = compute_risk(samples, RiskMeasure.VARIANCE)

        assert risk == pytest.approx(2.0)

    def test_cvar(self):
        """Test CVaR risk measure."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        risk = compute_risk(samples, RiskMeasure.CVAR, alpha=0.8)

        # CVaR should focus on tail (4.0 and 10.0)
        assert risk > 4.0

    def test_mean_variance(self):
        """Test mean-variance risk measure."""
        samples = np.array([1.0, 2.0, 3.0])
        risk = compute_risk(samples, RiskMeasure.MEAN_VARIANCE, lambda_risk=0.5)

        mean = 2.0
        var = 2/3
        expected = mean + 0.5 * var

        assert risk == pytest.approx(expected)


class TestCVaROptimizer:
    """Test CVaR optimization."""

    def test_compute_cvar(self):
        """Test CVaR computation."""
        # Simple objective
        objective = lambda u, xi: (u[0] - xi[0])**2

        # Deterministic sampler
        sampler = lambda n: np.random.uniform(-1, 1, size=(n, 1))

        optimizer = CVaROptimizer(objective, sampler, alpha=0.95)

        u = np.array([0.0])
        samples = sampler(100)

        cvar = optimizer.compute_cvar(u, samples)

        assert cvar > 0

    def test_optimize(self):
        """Test CVaR optimization."""
        # Quadratic with uncertainty: (u - xi)^2
        objective = lambda u, xi: (u[0] - xi[0])**2

        # Uniform uncertainty
        sampler = lambda n: np.random.uniform(-1, 1, size=(n, 1))

        optimizer = CVaROptimizer(
            objective,
            sampler,
            control_bounds=(np.array([-2.0]), np.array([2.0])),
            alpha=0.9
        )

        result = optimizer.optimize(u0=np.array([0.5]), n_samples=100)

        assert result['success']


class TestStochasticMPC:
    """Test stochastic MPC."""

    def test_simulate_scenario(self):
        """Test scenario simulation."""
        # Simple integrator: x+ = x + u + w
        dynamics = lambda x, u, w: x + u + w

        # Quadratic cost
        cost = lambda x, u: x[0]**2 + u[0]**2

        # Gaussian disturbance
        sampler = lambda n: np.random.randn(n, 1) * 0.1

        mpc = StochasticMPC(
            dynamics,
            cost,
            sampler,
            horizon=5,
            control_bounds=(np.array([-1.0]), np.array([1.0]))
        )

        x0 = np.array([1.0])
        u_seq = np.zeros((5, 1))
        disturbances = sampler(5)

        total_cost, traj = mpc.simulate_scenario(x0, u_seq, disturbances)

        assert total_cost > 0
        assert traj.shape == (6, 1)  # horizon + 1

    def test_plan(self):
        """Test stochastic MPC planning."""
        dynamics = lambda x, u, w: x + u + w
        cost = lambda x, u: x[0]**2 + u[0]**2
        sampler = lambda n: np.random.randn(n, 1) * 0.1

        mpc = StochasticMPC(
            dynamics,
            cost,
            sampler,
            horizon=5,
            control_bounds=(np.array([-1.0]), np.array([1.0]))
        )

        x0 = np.array([1.0])
        u_opt = mpc.plan(x0, n_scenarios=10)

        assert u_opt.shape == (1,)
        assert -1.0 <= u_opt[0] <= 1.0


class TestSampleAverageApproximation:
    """Test SAA method."""

    def test_solve_saa(self):
        """Test single SAA solve."""
        # Stochastic objective: E[(u - xi)^2]
        objective = lambda u, xi: (u[0] - xi[0])**2

        # Normal distribution
        sampler = lambda n: np.random.randn(n, 1)

        saa = SampleAverageApproximation(objective, sampler)

        result = saa.solve_saa(u0=np.array([0.5]), n_samples=100)

        assert result['success']
        # Optimal should be near 0 (mean of distribution)
        assert abs(result['control'][0]) < 0.5

    def test_optimize_with_validation(self):
        """Test SAA with validation."""
        objective = lambda u, xi: (u[0] - xi[0])**2
        sampler = lambda n: np.random.randn(n, 1)

        saa = SampleAverageApproximation(
            objective,
            sampler,
            control_bounds=(np.array([-2.0]), np.array([2.0]))
        )

        result = saa.optimize_with_validation(
            u0=np.array([0.5]),
            n_samples_train=100,
            n_samples_val=1000,
            n_replications=3
        )

        assert result['control'] is not None
        assert 'confidence_interval' in result
        assert result['n_replications'] == 3


# =============================================================================
# Case Studies Tests
# =============================================================================

class TestCartPole:
    """Test cart-pole case study."""

    def test_initialization(self):
        """Test cart-pole initialization."""
        cart_pole = CartPoleStabilization()

        assert cart_pole.n_states == 4
        assert cart_pole.n_controls == 1

    def test_dynamics(self):
        """Test cart-pole dynamics."""
        cart_pole = CartPoleStabilization()

        x = np.array([0.0, 0.0, 0.1, 0.0])  # Small angle
        u = np.array([0.0])

        x_dot = cart_pole.dynamics(x, u)

        assert x_dot.shape == (4,)
        # Pole should fall without control
        assert x_dot[3] != 0  # Angular acceleration non-zero

    def test_cost(self):
        """Test cart-pole cost."""
        cart_pole = CartPoleStabilization()

        x_goal = cart_pole.get_goal_state()
        u = np.array([0.0])

        # Cost at goal should be zero
        cost_goal = cart_pole.cost(x_goal, u)
        assert cost_goal == pytest.approx(0.0)

        # Cost away from goal should be positive
        x_away = np.array([1.0, 0.0, 0.5, 0.0])
        cost_away = cart_pole.cost(x_away, u)
        assert cost_away > 0


class TestQuadrotor:
    """Test quadrotor case study."""

    def test_initialization(self):
        """Test quadrotor initialization."""
        quad = QuadrotorTrajectory()

        assert quad.n_states == 6
        assert quad.n_controls == 2

    def test_dynamics(self):
        """Test quadrotor dynamics."""
        quad = QuadrotorTrajectory()

        x = quad.get_initial_state()
        u = np.array([quad.m * quad.g / 2, quad.m * quad.g / 2])  # Hover thrust

        x_dot = quad.dynamics(x, u)

        assert x_dot.shape == (6,)

    def test_reference_trajectory(self):
        """Test reference trajectory generation."""
        quad = QuadrotorTrajectory()
        quad.trajectory_type = "circle"

        x_ref_0 = quad.reference_trajectory(0.0)
        x_ref_5 = quad.reference_trajectory(5.0)

        assert x_ref_0.shape == (6,)
        assert x_ref_5.shape == (6,)
        # Should be different points on circle
        assert not np.allclose(x_ref_0[:2], x_ref_5[:2])


class TestRobotArm:
    """Test robot arm case study."""

    def test_forward_kinematics(self):
        """Test forward kinematics."""
        arm = RobotArmControl()

        # Arm straight out (theta1=0, theta2=0)
        ee = arm.forward_kinematics(0.0, 0.0)

        expected = np.array([arm.L1 + arm.L2, 0.0])
        assert np.allclose(ee, expected)

    def test_dynamics(self):
        """Test robot arm dynamics."""
        arm = RobotArmControl()

        x = np.array([np.pi/2, 0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0])

        x_dot = arm.dynamics(x, u)

        assert x_dot.shape == (4,)

    def test_cost(self):
        """Test reaching cost."""
        arm = RobotArmControl()
        arm.goal_position = np.array([1.0, 0.0])

        # Configuration that reaches goal
        theta1 = 0.0
        theta2 = 0.0
        x = np.array([theta1, theta2, 0.0, 0.0])
        u = np.array([0.0, 0.0])

        cost = arm.cost(x, u)

        # Should be small (goal reachable)
        assert cost >= 0


class TestEnergySystem:
    """Test energy system case study."""

    def test_dynamics(self):
        """Test thermal dynamics."""
        energy = EnergySystemOptimization()

        x = np.array([22.0])  # Comfortable temperature
        u = np.array([0.0])   # No HVAC

        x_dot = energy.dynamics(x, u, t=0.0)

        assert x_dot.shape == (1,)

    def test_cost(self):
        """Test cost function."""
        energy = EnergySystemOptimization()

        # Comfortable temperature, no power
        x_comfort = np.array([22.0])
        u_zero = np.array([0.0])
        cost_comfort = energy.cost(x_comfort, u_zero, t=0.0)

        # Uncomfortable temperature
        x_hot = np.array([30.0])
        cost_hot = energy.cost(x_hot, u_zero, t=0.0)

        # Hot should cost more (discomfort)
        assert cost_hot > cost_comfort


class TestPortfolio:
    """Test portfolio optimization."""

    def test_initialization(self):
        """Test portfolio initialization."""
        portfolio = PortfolioOptimization(n_assets=3)

        assert portfolio.n_states == 4  # 3 assets + cash
        assert portfolio.n_controls == 3

    def test_dynamics(self):
        """Test portfolio dynamics."""
        portfolio = PortfolioOptimization(n_assets=2)

        # Initial: all cash
        x = np.array([0.0, 0.0, 1000.0])

        # Buy some assets
        u = np.array([10.0, 10.0])

        x_next = portfolio.dynamics(x, u, dt=1.0)

        # Cash should decrease
        assert x_next[2] < x[2]

        # Holdings should increase
        assert x_next[0] > 0
        assert x_next[1] > 0


class TestChemicalReactor:
    """Test chemical reactor case study."""

    def test_reaction_rate(self):
        """Test Arrhenius reaction rate."""
        reactor = ChemicalReactorControl()

        # Higher temperature should give higher rate
        rate_low = reactor.reaction_rate(C_A=1.0, T=300.0)
        rate_high = reactor.reaction_rate(C_A=1.0, T=350.0)

        assert rate_high > rate_low

    def test_dynamics(self):
        """Test CSTR dynamics."""
        reactor = ChemicalReactorControl()

        x = reactor.get_initial_state()
        u = np.array([0.0, 1.0])  # No heat, some flow

        x_dot = reactor.dynamics(x, u)

        assert x_dot.shape == (2,)

    def test_cost(self):
        """Test cost function."""
        reactor = ChemicalReactorControl()

        # At setpoint
        x_setpoint = reactor.get_goal_state()
        u = np.array([0.0, 1.0])

        cost_setpoint = reactor.cost(x_setpoint, u)

        # Away from setpoint
        x_away = reactor.get_initial_state()
        cost_away = reactor.cost(x_away, u)

        assert cost_away > cost_setpoint


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test integration between modules."""

    def test_multi_objective_with_case_study(self):
        """Test multi-objective optimization on case study."""
        # Use cart-pole with two objectives
        cart_pole = CartPoleStabilization()

        # Objective 1: Minimize angle deviation
        def obj1(u):
            # Simulate one step
            x0 = cart_pole.get_initial_state()
            x_dot = cart_pole.dynamics(x0, u)
            x1 = x0 + 0.1 * x_dot  # Euler step
            return abs(x1[2])  # Angle

        # Objective 2: Minimize control effort
        def obj2(u):
            return u[0]**2

        bounds = cart_pole.get_bounds()['control']
        optimizer = MultiObjectiveOptimizer([obj1, obj2], bounds=bounds)

        # Compute small Pareto front
        front = optimizer.optimize(method='weighted_sum', n_points=3)

        assert len(front) > 0

    def test_robust_mpc_with_case_study(self):
        """Test robust MPC on linearized system."""
        # Linearized cart-pole around upright
        A = np.array([
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0, 1.0]
        ])
        B = np.array([[0.0], [0.1], [0.0], [0.1]])

        Q = np.diag([1.0, 0.1, 10.0, 0.1])
        R = np.array([[0.1]])

        unc_set = UncertaintySet(
            set_type=UncertaintySetType.BOX,
            dimension=4,
            parameters={'lower': -0.01 * np.ones(4),
                       'upper': 0.01 * np.ones(4)}
        )

        mpc = TubeBasedMPC(
            A, B, Q, R, unc_set,
            control_constraints=(np.array([-10.0]), np.array([10.0])),
            horizon=5
        )

        x0 = np.array([0.0, 0.0, 0.1, 0.0])
        u = mpc.plan(x0)

        assert u.shape == (1,)

    def test_stochastic_control_with_uncertainty(self):
        """Test stochastic control with robust uncertainty."""
        # Simple double integrator with disturbance
        dynamics = lambda x, u, w: np.array([
            x[0] + 0.1 * x[1],
            x[1] + 0.1 * u[0] + w[0]
        ])

        cost = lambda x, u: x[0]**2 + x[1]**2 + 0.1 * u[0]**2

        # Gaussian disturbance
        sampler = lambda n: np.random.randn(n, 1) * 0.1

        mpc = StochasticMPC(
            dynamics,
            cost,
            sampler,
            horizon=5,
            control_bounds=(np.array([-1.0]), np.array([1.0])),
            risk_measure=RiskMeasure.CVAR
        )

        x0 = np.array([1.0, 0.0])
        u = mpc.plan(x0, n_scenarios=20, risk_params={'alpha': 0.9})

        assert u.shape == (1,)
        assert -1.0 <= u[0] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
