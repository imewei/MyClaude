#!/usr/bin/env python3
"""
doe_generator.py

Generate a concrete design-of-experiments run table from a factor spec.
Supports full factorial, fractional factorial (2-level), Latin hypercube,
and central composite (response surface) designs.

Usage:
    python doe_generator.py --spec factors.yaml --design full --output runs.csv
    python doe_generator.py --spec factors.yaml --design fractional --fraction 0.5
    python doe_generator.py --spec factors.yaml --design lhs --n 50
    python doe_generator.py --spec factors.yaml --design ccd

factors.yaml format:
    factors:
      - name: phi
        type: continuous           # continuous or categorical
        levels: [0.45, 0.50, 0.55]
      - name: gamma_dot
        type: continuous
        levels: [0.1, 1.0, 10.0]
      - name: batch
        type: categorical
        levels: [A, B, C]
    randomize: true
    replicates: 3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "error: pyyaml required. Add 'pyyaml>=6.0' to pyproject.toml.",
        file=sys.stderr,
    )
    sys.exit(2)


def full_factorial(factors: list[dict]) -> list[dict]:
    """All combinations of all levels."""
    level_lists = [f["levels"] for f in factors]
    names = [f["name"] for f in factors]
    runs = []
    for combo in itertools.product(*level_lists):
        runs.append(dict(zip(names, combo)))
    return runs


def fractional_factorial(factors: list[dict], fraction: float = 0.5) -> list[dict]:
    """Simple fractional factorial for 2-level designs.

    Selects every (1/fraction)-th run from a full factorial. For serious
    fractional factorial work (specific resolution guarantees), use pyDOE2
    or similar; this is a lightweight approximation useful for quick planning.
    """
    all_2level = all(len(f["levels"]) == 2 for f in factors)
    if not all_2level:
        print(
            "warning: fractional factorial works best with 2-level factors; "
            "using every-nth subsampling of full factorial",
            file=sys.stderr,
        )
    full = full_factorial(factors)
    step = max(1, int(1 / fraction))
    return full[::step]


def latin_hypercube(factors: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Latin hypercube sampling: each factor sampled n times, each sample at a distinct level."""
    rng = random.Random(seed)
    runs = []
    factor_samples = {}
    for f in factors:
        if f["type"] == "continuous":
            lo, hi = min(f["levels"]), max(f["levels"])
            # n equally spaced strata
            strata = [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]
            rng.shuffle(strata)
            factor_samples[f["name"]] = strata
        else:
            # Categorical: cycle through levels, shuffle
            levels = list(f["levels"])
            cycled = (levels * ((n // len(levels)) + 1))[:n]
            rng.shuffle(cycled)
            factor_samples[f["name"]] = cycled
    for i in range(n):
        run = {name: factor_samples[name][i] for name in factor_samples}
        runs.append(run)
    return runs


def central_composite(factors: list[dict], alpha: float | None = None) -> list[dict]:
    """Central composite design: factorial cube + axial stars + center.

    Only supports continuous factors. alpha defaults to sqrt(k) (rotatability
    for 2^k design) but is approximate; for exact rotatability use a proper DoE
    library.
    """
    cont = [f for f in factors if f["type"] == "continuous"]
    if len(cont) != len(factors):
        raise ValueError("central composite requires all factors to be continuous")

    k = len(cont)
    if alpha is None:
        alpha = k ** 0.5  # approximate rotatability

    # Normalize: center each factor, use low/high from its levels
    centers = {f["name"]: (min(f["levels"]) + max(f["levels"])) / 2 for f in cont}
    half_ranges = {f["name"]: (max(f["levels"]) - min(f["levels"])) / 2 for f in cont}
    names = [f["name"] for f in cont]

    runs = []

    # Factorial corners at +/-1
    for signs in itertools.product([-1, 1], repeat=k):
        runs.append({
            names[i]: centers[names[i]] + signs[i] * half_ranges[names[i]]
            for i in range(k)
        })

    # Axial stars at +/-alpha for each axis
    for i in range(k):
        for sign in [-1, 1]:
            run = {n: centers[n] for n in names}
            run[names[i]] = centers[names[i]] + sign * alpha * half_ranges[names[i]]
            runs.append(run)

    # Center point
    runs.append({n: centers[n] for n in names})

    return runs


def replicate(runs: list[dict], n: int) -> list[dict]:
    """Return each run repeated n times."""
    out = []
    for run in runs:
        for _ in range(n):
            out.append(dict(run))
    return out


def randomize(runs: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(runs)
    rng.shuffle(shuffled)
    return shuffled


def write_csv(runs: list[dict], path: Path) -> None:
    if not runs:
        print("warning: no runs to write", file=sys.stderr)
        return
    fieldnames = ["run_id"] + list(runs[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, run in enumerate(runs, start=1):
            row = {"run_id": i, **run}
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="DoE run-table generator")
    parser.add_argument("--spec", type=Path, required=True, help="YAML factor spec")
    parser.add_argument(
        "--design",
        choices=["full", "fractional", "lhs", "ccd"],
        required=True,
    )
    parser.add_argument("--fraction", type=float, default=0.5, help="for fractional")
    parser.add_argument("--n", type=int, default=50, help="for lhs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("doe_runs.csv"))
    args = parser.parse_args()

    if not args.spec.exists():
        print(f"error: {args.spec} not found", file=sys.stderr)
        return 2

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    factors = spec["factors"]

    if args.design == "full":
        runs = full_factorial(factors)
    elif args.design == "fractional":
        runs = fractional_factorial(factors, args.fraction)
    elif args.design == "lhs":
        runs = latin_hypercube(factors, args.n, args.seed)
    elif args.design == "ccd":
        runs = central_composite(factors)

    replicates = spec.get("replicates", 1)
    if replicates > 1:
        runs = replicate(runs, replicates)

    if spec.get("randomize", True):
        runs = randomize(runs, args.seed)

    write_csv(runs, args.output)
    print(
        f"{args.design} design: {len(runs)} runs written to {args.output} "
        f"({len(factors)} factors, {replicates} replicate(s), "
        f"randomized={spec.get('randomize', True)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
