#!/usr/bin/env python3
"""
convergence_study.py

Run a convergence study for a Stage 6 prototype. Given a callable that produces
an observable as a function of a resolution parameter (timestep, grid size,
particle count), run at multiple resolutions and estimate the converged value
via Richardson extrapolation.

Usage (from a Python script):

    from convergence_study import run_convergence
    from mymod.core import run_prototype  # returns a scalar observable

    resolutions = [1e-2, 5e-3, 2e-3, 1e-3]

    result = run_convergence(
        callable_=run_prototype,
        param_name="dt",
        param_values=resolutions,
        expected_order=1.0,  # first-order integrator
    )
    # result['richardson_estimate'], result['empirical_order'], result['converged']

The script can also be run as a CLI by pointing to a small Python file that
exports a `run(dt)` or `run(n)` function.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConvergenceResult:
    param_values: list[float]
    observable_values: list[float]
    richardson_estimate: float | None
    empirical_order: float | None
    relative_errors: list[float]
    converged: bool
    message: str

    def to_dict(self) -> dict:
        return {
            "param_values": self.param_values,
            "observable_values": self.observable_values,
            "richardson_estimate": self.richardson_estimate,
            "empirical_order": self.empirical_order,
            "relative_errors": self.relative_errors,
            "converged": self.converged,
            "message": self.message,
        }


def run_convergence(
    callable_,
    param_name: str,
    param_values: list[float],
    expected_order: float | None = None,
    convergence_tol: float = 0.05,
) -> ConvergenceResult:
    """
    Run the callable at each resolution in param_values, estimate the converged
    value via Richardson extrapolation on the three finest resolutions.

    callable_ takes a keyword argument named `param_name`.
    """
    sorted_params = sorted(param_values, reverse=True)  # coarsest first
    observed = [float(callable_(**{param_name: p})) for p in sorted_params]

    if len(sorted_params) < 3:
        return ConvergenceResult(
            param_values=sorted_params,
            observable_values=observed,
            richardson_estimate=None,
            empirical_order=None,
            relative_errors=[],
            converged=False,
            message="need at least three resolutions for Richardson extrapolation",
        )

    # Use the three finest
    h = sorted_params[-3:]
    f = observed[-3:]

    # Richardson: assume f(h) = f_true + C h^p
    # From three points: p = log[(f1 - f2)/(f2 - f3)] / log(r) where r = h_i / h_{i+1}
    try:
        if (f[0] - f[1]) * (f[1] - f[2]) <= 0:
            raise ValueError("non-monotonic convergence (or zero difference); order undefined")
        r = h[0] / h[1]
        p_est = abs(
            (f[0] - f[1]) / (f[1] - f[2])
        )
        # log_r |ratio|
        import math
        empirical_order = math.log(p_est) / math.log(r)
        f_true = f[-1] + (f[-1] - f[-2]) / (r**empirical_order - 1)
    except (ValueError, ZeroDivisionError) as exc:
        return ConvergenceResult(
            param_values=sorted_params,
            observable_values=observed,
            richardson_estimate=None,
            empirical_order=None,
            relative_errors=[],
            converged=False,
            message=f"Richardson extrapolation failed: {exc}",
        )

    rel_errors = [abs(fi - f_true) / max(abs(f_true), 1e-30) for fi in observed]

    converged = rel_errors[-1] < convergence_tol
    messages = []
    if expected_order is not None:
        if abs(empirical_order - expected_order) > 0.5:
            messages.append(
                f"empirical order ({empirical_order:.2f}) differs from expected "
                f"({expected_order:.2f}); investigate"
            )
    messages.append(f"finest-resolution relative error: {rel_errors[-1]:.2%}")
    messages.append("converged" if converged else "NOT converged (refine further)")

    return ConvergenceResult(
        param_values=sorted_params,
        observable_values=observed,
        richardson_estimate=f_true,
        empirical_order=empirical_order,
        relative_errors=rel_errors,
        converged=converged,
        message="; ".join(messages),
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Convergence study runner")
    parser.add_argument("module", type=Path, help="Python file exporting a run(param=...) function")
    parser.add_argument("--param-name", default="dt")
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--expected-order", type=float, default=None)
    args = parser.parse_args()

    if not args.module.exists():
        print(f"error: {args.module} not found", file=sys.stderr)
        return 2

    spec = importlib.util.spec_from_file_location("target", args.module)
    if spec is None or spec.loader is None:
        print("error: could not load target module", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "run"):
        print("error: target module must define a run(**kwargs) function", file=sys.stderr)
        return 2

    result = run_convergence(
        mod.run,
        args.param_name,
        args.values,
        expected_order=args.expected_order,
    )

    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(_cli())
