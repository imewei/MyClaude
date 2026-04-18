#!/usr/bin/env python3
"""
limit_recovery_check.py

Automate the analytic-limit recovery pass for a Stage 6 numerical prototype.
Given a simulation callable and an analytic reference for the limiting case,
run both and report whether they agree within tolerance.

Usage (from Python):

    from limit_recovery_check import check_limit_recovery

    def simulation_at_limit():
        # run the prototype with new-physics parameter = 0
        return run_sim(interaction_strength=0.0)

    def analytic_reference():
        # return the known limiting-case result
        return 2 * D * t_array

    result = check_limit_recovery(
        simulation_at_limit,
        analytic_reference,
        tolerance=1e-3,
        metric="relative_l2",
    )
    # result.passed, result.error, result.message

Metrics supported:
    relative_l2: ||sim - ref||_2 / ||ref||_2
    relative_max: max |sim - ref| / max |ref|
    absolute_max: max |sim - ref|
    ks_distance: Kolmogorov-Smirnov distance (for distributions)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class LimitRecoveryResult:
    passed: bool
    error: float
    metric: str
    tolerance: float
    message: str
    sim_shape: tuple | None = None
    ref_shape: tuple | None = None


def _as_array(x):
    import numpy as np
    return np.asarray(x)


def _relative_l2(sim, ref) -> float:
    import numpy as np
    s, r = _as_array(sim), _as_array(ref)
    num = float(np.linalg.norm((s - r).ravel()))
    den = float(np.linalg.norm(r.ravel()))
    if den == 0.0:
        return num
    return num / den


def _relative_max(sim, ref) -> float:
    import numpy as np
    s, r = _as_array(sim), _as_array(ref)
    diff = float(np.max(np.abs(s - r)))
    scale = float(np.max(np.abs(r)))
    if scale == 0.0:
        return diff
    return diff / scale


def _absolute_max(sim, ref) -> float:
    import numpy as np
    return float(np.max(np.abs(_as_array(sim) - _as_array(ref))))


def _ks_distance(sim, ref) -> float:
    try:
        from scipy import stats
    except ImportError:
        raise RuntimeError("scipy required for ks_distance metric")
    return float(stats.ks_2samp(_as_array(sim).ravel(), _as_array(ref).ravel()).statistic)


METRICS: dict[str, Callable] = {
    "relative_l2": _relative_l2,
    "relative_max": _relative_max,
    "absolute_max": _absolute_max,
    "ks_distance": _ks_distance,
}


def check_limit_recovery(
    simulation_callable,
    reference_callable,
    tolerance: float = 1e-3,
    metric: str = "relative_l2",
) -> LimitRecoveryResult:
    """
    Run sim and reference, compute the chosen metric, compare to tolerance.

    Both callables take no arguments and return array-likes with matching
    shape. Shape mismatch is reported as a failure with diagnostic info.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}'; choose from {list(METRICS)}")

    sim = simulation_callable()
    ref = reference_callable()

    sim_arr = _as_array(sim)
    ref_arr = _as_array(ref)

    if sim_arr.shape != ref_arr.shape:
        return LimitRecoveryResult(
            passed=False,
            error=float("inf"),
            metric=metric,
            tolerance=tolerance,
            message=f"shape mismatch: sim {sim_arr.shape} vs ref {ref_arr.shape}",
            sim_shape=sim_arr.shape,
            ref_shape=ref_arr.shape,
        )

    error = METRICS[metric](sim_arr, ref_arr)
    passed = error < tolerance

    if passed:
        message = f"PASS: {metric} = {error:.3e} < tol {tolerance:.3e}"
    else:
        message = (
            f"FAIL: {metric} = {error:.3e} >= tol {tolerance:.3e}; "
            f"investigate sign errors, coefficient errors, or hidden assumptions"
        )

    return LimitRecoveryResult(
        passed=passed,
        error=error,
        metric=metric,
        tolerance=tolerance,
        message=message,
        sim_shape=sim_arr.shape,
        ref_shape=ref_arr.shape,
    )


def _cli() -> int:
    """CLI variant: import user's Python file, look for sim() and ref() callables."""
    parser = argparse.ArgumentParser(description="Limit-recovery check runner")
    parser.add_argument(
        "module",
        type=Path,
        help="Python file defining sim() and ref() callables",
    )
    parser.add_argument("--metric", default="relative_l2", choices=list(METRICS))
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if not args.module.exists():
        print(f"error: {args.module} not found", file=sys.stderr)
        return 2

    import importlib.util
    spec = importlib.util.spec_from_file_location("target", args.module)
    if spec is None or spec.loader is None:
        print("error: could not load target module", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "sim") or not hasattr(mod, "ref"):
        print("error: target module must define sim() and ref() callables", file=sys.stderr)
        return 2

    result = check_limit_recovery(mod.sim, mod.ref, args.tolerance, args.metric)
    print(result.message)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(_cli())
