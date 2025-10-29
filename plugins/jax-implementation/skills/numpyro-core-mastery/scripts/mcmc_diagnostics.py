#!/usr/bin/env python3
"""
Comprehensive MCMC diagnostics for NumPyro inference.

Usage:
    python mcmc_diagnostics.py --samples posterior_samples.pkl

Or import:
    from mcmc_diagnostics import diagnose_mcmc
    diagnose_mcmc(mcmc, verbose=True)
"""

import jax.numpy as jnp
import argparse
import pickle


def diagnose_mcmc(mcmc, verbose=True):
    """
    Comprehensive MCMC diagnostics.

    Args:
        mcmc: MCMC object after run()
        verbose: Print detailed diagnostics

    Returns:
        dict: Diagnostic results with warnings and recommendations
    """
    from numpyro.diagnostics import summary

    diagnostics = {
        'converged': True,
        'warnings': [],
        'recommendations': []
    }

    posterior_samples = mcmc.get_samples()
    summary_dict = summary(posterior_samples, prob=0.95)

    if verbose:
        print("="*70)
        print("NUMPYRO MCMC DIAGNOSTICS")
        print("="*70)

    # 1. R-hat check
    if verbose:
        print("\n1. Convergence (R-hat)")

    for param, stats in summary_dict.items():
        r_hat = stats['r_hat']

        if verbose:
            status = "✓" if r_hat < 1.01 else "✗"
            print(f"   {param}: {r_hat:.4f} {status}")

        if r_hat > 1.01:
            diagnostics['converged'] = False
            diagnostics['warnings'].append(f"{param} not converged (R-hat={r_hat:.3f})")

            if r_hat > 1.1:
                diagnostics['recommendations'].extend([
                    f"{param}: Serious convergence issue",
                    "Increase num_warmup or num_samples",
                    "Check trace plots for multimodality"
                ])

    # 2. ESS check
    if verbose:
        print("\n2. Effective Sample Size")

    for param, stats in summary_dict.items():
        n_eff = stats['n_eff']

        if verbose:
            status = "✓" if n_eff > 400 else ("⚠" if n_eff > 100 else "✗")
            print(f"   {param}: {n_eff:.0f} {status}")

        if n_eff < 400:
            diagnostics['warnings'].append(f"{param} low ESS ({n_eff:.0f})")

            if n_eff < 100:
                diagnostics['recommendations'].extend([
                    f"{param}: Very low ESS",
                    "High autocorrelation - run more samples",
                    "Consider reparameterization (non-centered)"
                ])

    # 3. Divergences check
    if verbose:
        print("\n3. Divergences")

    divergences = mcmc.get_extra_fields()['diverging'].sum()

    if verbose:
        print(f"   Total divergences: {divergences}")

    if divergences > 0:
        diagnostics['converged'] = False
        pct = 100 * divergences / (mcmc.num_samples * mcmc.num_chains)
        diagnostics['warnings'].append(f"{divergences} divergences ({pct:.1f}%)")

        if pct > 1:
            diagnostics['recommendations'].extend([
                "Serious: >1% divergences",
                "Increase target_accept_prob to 0.9-0.95",
                "Reparameterize (non-centered for hierarchical)",
                "Use more informative priors"
            ])
        else:
            diagnostics['recommendations'].append(
                "Minor divergences - increase target_accept_prob to 0.9"
            )

    # 4. Summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        if diagnostics['converged'] and not diagnostics['warnings']:
            print("✓ All diagnostics passed - inference reliable!")
        else:
            print(f"⚠ Found {len(diagnostics['warnings'])} issue(s):\n")
            for i, warning in enumerate(diagnostics['warnings'], 1):
                print(f"{i}. {warning}")

            if diagnostics['recommendations']:
                print("\nRecommendations:")
                for rec in set(diagnostics['recommendations']):
                    print(f"  • {rec}")

        print("="*70)

    return diagnostics


def main():
    parser = argparse.ArgumentParser(description='MCMC diagnostics')
    parser.add_argument('--samples', required=True, help='Path to posterior_samples.pkl')
    args = parser.parse_args()

    with open(args.samples, 'rb') as f:
        data = pickle.load(f)

    # Assume data is either mcmc object or posterior_samples dict
    if hasattr(data, 'get_samples'):
        diagnose_mcmc(data)
    else:
        print("Error: Expected MCMC object with get_samples() method")


if __name__ == "__main__":
    main()
