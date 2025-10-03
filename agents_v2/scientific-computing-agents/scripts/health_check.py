#!/usr/bin/env python3
"""
Health check script for Scientific Computing Agents.

This script performs comprehensive health checks on the system
and returns appropriate exit codes for monitoring systems.

Exit codes:
    0: System healthy
    1: System unhealthy (critical)
    2: System degraded (warning)
"""

import sys
import time
import logging
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_imports() -> Tuple[bool, str]:
    """Check that all required modules can be imported."""
    try:
        from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent
        from agents.performance_profiler_agent import PerformanceProfilerAgent
        from agents.ode_pde_solver_agent import ODEPDESolverAgent
        return True, "All imports successful"
    except Exception as e:
        return False, f"Import failed: {e}"


def check_orchestrator() -> Tuple[bool, str]:
    """Check that workflow orchestrator can be created and used."""
    try:
        from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent

        orchestrator = WorkflowOrchestrationAgent()

        # Test empty workflow
        result = orchestrator.execute_workflow([])

        if result.success:
            return True, "Orchestrator operational"
        else:
            return False, f"Orchestrator failed: {result.errors}"

    except Exception as e:
        return False, f"Orchestrator check failed: {e}"


def check_profiler() -> Tuple[bool, str]:
    """Check that performance profiler is operational."""
    try:
        from agents.performance_profiler_agent import PerformanceProfilerAgent

        profiler = PerformanceProfilerAgent()

        # Test simple function profiling
        def test_func(n):
            return sum(range(n))

        result = profiler.process({
            'task': 'profile_function',
            'function': test_func,
            'args': [100]
        })

        if result.success:
            return True, "Profiler operational"
        else:
            return False, f"Profiler failed: {result.errors}"

    except Exception as e:
        return False, f"Profiler check failed: {e}"


def check_solver() -> Tuple[bool, str]:
    """Check that ODE solver is operational."""
    try:
        from agents.ode_pde_solver_agent import ODEPDESolverAgent
        import numpy as np

        agent = ODEPDESolverAgent()

        # Test simple ODE: dy/dt = -y, y(0) = 1
        def dydt(t, y):
            return -y

        result = agent.process({
            'task': 'solve_ode',
            'equation': dydt,
            'initial_conditions': [1.0],
            't_span': (0, 1),
            't_eval': np.linspace(0, 1, 10)
        })

        if result.success:
            return True, "ODE solver operational"
        else:
            return False, f"ODE solver failed: {result.errors}"

    except Exception as e:
        return False, f"Solver check failed: {e}"


def check_system_resources() -> Tuple[bool, str]:
    """Check system resource availability."""
    try:
        import psutil

        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            return False, f"Memory critically low: {memory.percent}% used"

        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return False, f"CPU critically high: {cpu_percent}%"

        # Check disk
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            return False, f"Disk critically full: {disk.percent}%"

        return True, f"Resources OK (CPU: {cpu_percent}%, RAM: {memory.percent}%, Disk: {disk.percent}%)"

    except ImportError:
        # psutil not available, skip this check
        return True, "Resource check skipped (psutil not installed)"
    except Exception as e:
        return False, f"Resource check failed: {e}"


def run_health_checks() -> Dict[str, Tuple[bool, str]]:
    """Run all health checks and return results."""
    checks = {
        'imports': check_imports,
        'orchestrator': check_orchestrator,
        'profiler': check_profiler,
        'solver': check_solver,
        'resources': check_system_resources,
    }

    results = {}
    for name, check_func in checks.items():
        logger.info(f"Running check: {name}")
        try:
            success, message = check_func()
            results[name] = (success, message)

            if success:
                logger.info(f"✓ {name}: {message}")
            else:
                logger.error(f"✗ {name}: {message}")
        except Exception as e:
            results[name] = (False, f"Check crashed: {e}")
            logger.error(f"✗ {name}: Check crashed: {e}")

    return results


def main() -> int:
    """Main health check function."""
    logger.info("=" * 60)
    logger.info("Scientific Computing Agents - Health Check")
    logger.info("=" * 60)

    start_time = time.time()
    results = run_health_checks()
    duration = time.time() - start_time

    # Analyze results
    total_checks = len(results)
    passed_checks = sum(1 for success, _ in results.values() if success)
    failed_checks = total_checks - passed_checks

    logger.info("=" * 60)
    logger.info(f"Health Check Summary ({duration:.2f}s)")
    logger.info(f"Total checks: {total_checks}")
    logger.info(f"Passed: {passed_checks}")
    logger.info(f"Failed: {failed_checks}")
    logger.info("=" * 60)

    # Determine exit code
    if failed_checks == 0:
        logger.info("Status: HEALTHY ✓")
        return 0
    elif failed_checks <= total_checks // 3:
        logger.warning("Status: DEGRADED ⚠")
        return 2
    else:
        logger.error("Status: UNHEALTHY ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
