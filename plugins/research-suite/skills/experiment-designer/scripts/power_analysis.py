#!/usr/bin/env python3
"""
power_analysis.py

Statistical power analysis for common experimental designs. Given an effect
size, noise estimate, significance level, and desired power, return the
required sample size.

Supports:
- two-sample t-test (equal variance, two-sided)
- one-way ANOVA
- simple linear regression (for slope detection)

Usage (from Python):

    from power_analysis import required_n_two_sample

    n = required_n_two_sample(effect_size=0.5, alpha=0.05, power=0.80)
    # returns required N per group

CLI:

    python power_analysis.py two-sample --effect 0.5 --alpha 0.05 --power 0.80
    python power_analysis.py anova --effect 0.25 --groups 4
    python power_analysis.py regression --effect 0.3 --predictors 1

This is a dependency-minimal implementation using scipy.stats only.
"""

from __future__ import annotations

import argparse
import sys


try:
    from scipy import stats
    import math
except ImportError:
    print(
        "error: scipy required. Add 'scipy>=1.14' to pyproject.toml.",
        file=sys.stderr,
    )
    sys.exit(2)


def required_n_two_sample(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Required N per group for a two-sample t-test.

    effect_size: Cohen's d (difference in means divided by pooled SD)
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return math.ceil(n)


def required_n_anova(
    effect_size: float,
    n_groups: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Required N per group for a one-way ANOVA.

    effect_size: Cohen's f (SD of group means / pooled within-group SD)
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")

    # Iterative approach: increase n until power target is met.
    for n in range(2, 10000):
        df_between = n_groups - 1
        df_within = n_groups * (n - 1)
        if df_within < 1:
            continue
        # Non-centrality parameter
        ncp = effect_size ** 2 * n_groups * n
        f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
        achieved_power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
        if achieved_power >= power:
            return n
    return -1  # not achieved within n=10000


def required_n_regression(
    effect_size: float,
    n_predictors: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Required N for detecting an R^2 of a given effect size in linear regression.

    effect_size: Cohen's f^2 = R^2 / (1 - R^2)
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")

    for n in range(n_predictors + 3, 100000):
        df_num = n_predictors
        df_den = n - n_predictors - 1
        ncp = effect_size * n
        f_crit = stats.f.ppf(1 - alpha, df_num, df_den)
        achieved_power = 1 - stats.ncf.cdf(f_crit, df_num, df_den, ncp)
        if achieved_power >= power:
            return n
    return -1


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Statistical power analysis")
    sub = parser.add_subparsers(dest="test", required=True)

    p_t = sub.add_parser("two-sample", help="two-sample t-test")
    p_t.add_argument("--effect", type=float, required=True, help="Cohen's d")
    p_t.add_argument("--alpha", type=float, default=0.05)
    p_t.add_argument("--power", type=float, default=0.80)

    p_a = sub.add_parser("anova", help="one-way ANOVA")
    p_a.add_argument("--effect", type=float, required=True, help="Cohen's f")
    p_a.add_argument("--groups", type=int, required=True)
    p_a.add_argument("--alpha", type=float, default=0.05)
    p_a.add_argument("--power", type=float, default=0.80)

    p_r = sub.add_parser("regression", help="linear regression")
    p_r.add_argument("--effect", type=float, required=True, help="Cohen's f^2")
    p_r.add_argument("--predictors", type=int, default=1)
    p_r.add_argument("--alpha", type=float, default=0.05)
    p_r.add_argument("--power", type=float, default=0.80)

    args = parser.parse_args()

    if args.test == "two-sample":
        n = required_n_two_sample(args.effect, args.alpha, args.power)
        print(f"Required N per group: {n}")
    elif args.test == "anova":
        n = required_n_anova(args.effect, args.groups, args.alpha, args.power)
        print(f"Required N per group (across {args.groups} groups): {n}")
    elif args.test == "regression":
        n = required_n_regression(args.effect, args.predictors, args.alpha, args.power)
        print(f"Required N total (for {args.predictors} predictor(s)): {n}")

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
