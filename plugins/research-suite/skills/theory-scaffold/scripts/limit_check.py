#!/usr/bin/env python3
"""
limit_check.py

Symbolic limit-recovery check using SymPy.

Given an expression and a limit specification, verify that the expression
reduces to a known reference in that limit.

Usage (from a Python script or an interactive session):

    from limit_check import check_limit
    import sympy as sp

    phi, Pe = sp.symbols('phi Pe', positive=True)
    full_expression = some_derived_result(phi, Pe)
    reference = phi  # known dilute-limit result

    ok, message = check_limit(full_expression, {phi: 0}, reference)
    assert ok, message

The script can also be run as a CLI with a small JSON spec for simple cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import sympy as sp
except ImportError:
    print(
        "error: sympy not installed. Add 'sympy>=1.13' to pyproject.toml or "
        "run 'uv add sympy' in the relevant project.",
        file=sys.stderr,
    )
    sys.exit(2)


def check_limit(
    expression,
    limits: dict,
    reference,
    tol: float = 1e-10,
    direction: str = "+",
) -> tuple[bool, str]:
    """
    Evaluate `expression` in the limit where each variable in `limits` approaches
    its mapped value, then compare to `reference`.

    Parameters
    ----------
    expression : sympy.Expr
        The derived expression.
    limits : dict[sympy.Symbol, sympy.Expr or number]
        Variables and their limit points.
    reference : sympy.Expr
        The expected limiting form.
    tol : float
        Tolerance for numerical difference when symbolic simplification does not
        collapse the residual to zero.
    direction : str
        '+' or '-' for one-sided limits.

    Returns
    -------
    (ok, message)
    """
    limited = expression
    for var, point in limits.items():
        limited = sp.limit(limited, var, point, direction)

    residual = sp.simplify(limited - reference)

    if residual == 0:
        return True, "pass (exact)"

    # Try algebraic simplification
    residual_simplified = sp.simplify(sp.nsimplify(residual))
    if residual_simplified == 0:
        return True, "pass (after nsimplify)"

    # If numeric, compare against tolerance
    try:
        numeric = float(residual.evalf())
        if abs(numeric) < tol:
            return True, f"pass (numeric residual {numeric:.2e})"
    except (TypeError, ValueError):
        pass

    return False, (
        f"fail: limit = {limited}, reference = {reference}, residual = {residual}"
    )


def _cli() -> int:
    """CLI entry point for simple cases defined in a JSON spec.

    JSON schema:
    {
        "symbols": ["phi", "Pe"],
        "expression": "phi + Pe**2 * phi**3",
        "limit": {"Pe": 0},
        "reference": "phi"
    }
    """
    parser = argparse.ArgumentParser(description="Symbolic limit check via SymPy")
    parser.add_argument("spec", type=Path, help="JSON spec file")
    args = parser.parse_args()

    if not args.spec.exists():
        print(f"error: {args.spec} not found", file=sys.stderr)
        return 2

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    symbols = {s: sp.symbols(s) for s in spec["symbols"]}

    def parse(s: str):
        return sp.sympify(s, locals=symbols)

    expression = parse(spec["expression"])
    reference = parse(spec["reference"])
    limits = {symbols[k]: parse(str(v)) for k, v in spec["limit"].items()}

    ok, message = check_limit(expression, limits, reference)
    print(f"{'PASS' if ok else 'FAIL'}: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_cli())
