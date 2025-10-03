#!/usr/bin/env python3
"""
Performance benchmarking suite for Scientific Computing Agents.

This script runs comprehensive performance benchmarks on all major agents
and generates a detailed performance report.
"""

import time
import json
import numpy as np
from typing import Dict, Any, Callable
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    name: str
    duration: float
    success: bool
    iterations: int
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    metadata: Dict[str, Any]


class BenchmarkSuite:
    """Performance benchmarking suite."""

    def __init__(self):
        self.results = []

    def benchmark_function(
        self,
        name: str,
        func: Callable,
        iterations: int = 10,
        warmup: int = 2,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a single function."""
        logger.info(f"Benchmarking: {name} ({iterations} iterations)")

        # Warmup runs
        for _ in range(warmup):
            try:
                func(**kwargs)
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")

        # Actual benchmark runs
        times = []
        success = True

        for i in range(iterations):
            start = time.perf_counter()
            try:
                func(**kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                logger.error(f"Iteration {i} failed: {e}")
                success = False
                break

        if not times:
            times = [0.0]

        result = BenchmarkResult(
            name=name,
            duration=sum(times),
            success=success,
            iterations=len(times),
            avg_time=np.mean(times),
            min_time=np.min(times),
            max_time=np.max(times),
            std_time=np.std(times),
            metadata=kwargs
        )

        self.results.append(result)
        logger.info(f"  Avg: {result.avg_time*1000:.2f}ms, "
                   f"Min: {result.min_time*1000:.2f}ms, "
                   f"Max: {result.max_time*1000:.2f}ms")

        return result

    def benchmark_ode_solver(self):
        """Benchmark ODE solver agent."""
        from agents.ode_pde_solver_agent import ODEPDESolverAgent

        agent = ODEPDESolverAgent()

        def run_ode_small():
            def dydt(t, y):
                return -y
            agent.process({
                'task': 'solve_ode',
                'equation': dydt,
                'initial_conditions': [1.0],
                't_span': (0, 10),
                't_eval': np.linspace(0, 10, 100)
            })

        def run_ode_large():
            def dydt(t, y):
                return -y
            agent.process({
                'task': 'solve_ode',
                'equation': dydt,
                'initial_conditions': [1.0],
                't_span': (0, 100),
                't_eval': np.linspace(0, 100, 10000)
            })

        self.benchmark_function("ODE Solver (100 points)", run_ode_small)
        self.benchmark_function("ODE Solver (10000 points)", run_ode_large)

    def benchmark_optimization(self):
        """Benchmark optimization agent."""
        from agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        def run_optimization():
            def rosenbrock(x):
                return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

            agent.process({
                'task': 'minimize',
                'function': rosenbrock,
                'x0': np.zeros(5),
                'method': 'L-BFGS-B'
            })

        self.benchmark_function("Optimization (Rosenbrock 5D)", run_optimization)

    def benchmark_linear_algebra(self):
        """Benchmark linear algebra agent."""
        from agents.linear_algebra_agent import LinearAlgebraAgent

        agent = LinearAlgebraAgent()

        def run_solve_small():
            A = np.random.randn(100, 100)
            b = np.random.randn(100)
            agent.process({
                'task': 'solve_linear_system',
                'A': A,
                'b': b
            })

        def run_solve_large():
            A = np.random.randn(1000, 1000)
            b = np.random.randn(1000)
            agent.process({
                'task': 'solve_linear_system',
                'A': A,
                'b': b
            })

        def run_eigenvalues():
            A = np.random.randn(500, 500)
            A = A + A.T  # Make symmetric
            agent.process({
                'task': 'compute_eigenvalues',
                'A': A
            })

        self.benchmark_function("Linear Solve (100x100)", run_solve_small, iterations=20)
        self.benchmark_function("Linear Solve (1000x1000)", run_solve_large, iterations=5)
        self.benchmark_function("Eigenvalues (500x500)", run_eigenvalues, iterations=5)

    def benchmark_integration(self):
        """Benchmark integration agent."""
        from agents.integration_agent import IntegrationAgent

        agent = IntegrationAgent()

        def run_integration_1d():
            def f(x):
                return np.sin(x)
            agent.process({
                'task': 'integrate',
                'function': f,
                'limits': (0, np.pi)
            })

        def run_integration_2d():
            def f(x, y):
                return np.sin(x) * np.cos(y)
            agent.process({
                'task': 'integrate_2d',
                'function': f,
                'x_limits': (0, np.pi),
                'y_limits': (0, np.pi)
            })

        self.benchmark_function("Integration 1D", run_integration_1d, iterations=20)
        self.benchmark_function("Integration 2D", run_integration_2d, iterations=10)

    def benchmark_workflow_orchestration(self):
        """Benchmark workflow orchestration."""
        from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent, WorkflowStep

        orchestrator = WorkflowOrchestrationAgent()

        class DummyAgent:
            def process(self, data):
                time.sleep(0.001)  # Simulate work
                return {'result': data.get('value', 0) * 2}

        def run_sequential():
            agents = [DummyAgent() for _ in range(5)]
            steps = [
                WorkflowStep(
                    step_id=f'step{i}',
                    agent=agents[i],
                    method='process',
                    inputs={'value': i}
                )
                for i in range(5)
            ]
            orchestrator.execute_workflow(steps, parallel=False)

        def run_parallel():
            agents = [DummyAgent() for _ in range(5)]
            steps = [
                WorkflowStep(
                    step_id=f'step{i}',
                    agent=agents[i],
                    method='process',
                    inputs={'value': i}
                )
                for i in range(5)
            ]
            orchestrator.execute_workflow(steps, parallel=True)

        self.benchmark_function("Workflow Sequential (5 steps)", run_sequential)
        self.benchmark_function("Workflow Parallel (5 steps)", run_parallel)

    def run_all_benchmarks(self):
        """Run all benchmarks."""
        logger.info("=" * 60)
        logger.info("Scientific Computing Agents - Performance Benchmarks")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            self.benchmark_ode_solver()
        except Exception as e:
            logger.error(f"ODE benchmarks failed: {e}")

        try:
            self.benchmark_optimization()
        except Exception as e:
            logger.error(f"Optimization benchmarks failed: {e}")

        try:
            self.benchmark_linear_algebra()
        except Exception as e:
            logger.error(f"Linear algebra benchmarks failed: {e}")

        try:
            self.benchmark_integration()
        except Exception as e:
            logger.error(f"Integration benchmarks failed: {e}")

        try:
            self.benchmark_workflow_orchestration()
        except Exception as e:
            logger.error(f"Workflow benchmarks failed: {e}")

        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"Benchmarks completed in {total_time:.2f}s")
        logger.info("=" * 60)

    def generate_report(self, output_file: str = "benchmark_results.json"):
        """Generate benchmark report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_benchmarks': len(self.results),
            'successful': sum(1 for r in self.results if r.success),
            'failed': sum(1 for r in self.results if not r.success),
            'results': [asdict(r) for r in self.results]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Benchmark report saved to {output_file}")

        # Print summary
        logger.info("\nBenchmark Summary:")
        logger.info("-" * 60)
        for result in self.results:
            status = "✓" if result.success else "✗"
            logger.info(f"{status} {result.name:40s} {result.avg_time*1000:8.2f}ms")

        return report


def main():
    """Run benchmarks and generate report."""
    suite = BenchmarkSuite()
    suite.run_all_benchmarks()
    suite.generate_report()


if __name__ == "__main__":
    main()
