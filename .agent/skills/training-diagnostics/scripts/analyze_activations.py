#!/usr/bin/env python3
"""
Activation Analysis Tool for Neural Network Training

Analyzes activation distributions to detect dead neurons, saturation, and
other activation pathologies. Useful for debugging ReLU, sigmoid, tanh issues.

Usage:
    python analyze_activations.py --model model.pt --data sample.pt
    python analyze_activations.py --model model.pt --data sample.pt --save-plots
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    # allow-torch
import torch
    # allow-torch
import torch.nn as nn
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)


class ActivationHook:
    """Hook to capture activations from intermediate layers."""

    def __init__(self, name: str):
        self.name = name
        self.activations = None

    def __call__(self, module, input, output):
        """Store activation output."""
        self.activations = output.detach()


def register_activation_hooks(model: nn.Module) -> Dict[str, ActivationHook]:
    """
    Register forward hooks on all modules to capture activations.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping module names to ActivationHook objects
    """
    hooks = {}

    for name, module in model.named_modules():
        # Skip container modules
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            continue

        # Skip the root model
        if name == "":
            continue

        hook = ActivationHook(name)
        module.register_forward_hook(hook)
        hooks[name] = hook

    return hooks


def analyze_activation_statistics(activations: torch.Tensor,
                                 layer_name: str,
                                 activation_type: str = "unknown") -> Dict[str, float]:
    """
    Compute statistics for activation tensor.

    Args:
        activations: Activation tensor from forward pass
        layer_name: Name of the layer
        activation_type: Type of activation function (relu, sigmoid, tanh, etc.)

    Returns:
        Dictionary of activation statistics
    """
    acts = activations.detach().cpu().numpy()

    stats = {
        'mean': float(np.mean(acts)),
        'std': float(np.std(acts)),
        'min': float(np.min(acts)),
        'max': float(np.max(acts)),
        'median': float(np.median(acts)),
        'num_zeros': int(np.sum(acts == 0)),
        'num_negative': int(np.sum(acts < 0)),
        'num_positive': int(np.sum(acts > 0)),
        'total_activations': int(acts.size),
        'shape': tuple(activations.shape)
    }

    # Compute percentiles
    stats['p01'] = float(np.percentile(acts, 1))
    stats['p10'] = float(np.percentile(acts, 10))
    stats['p90'] = float(np.percentile(acts, 90))
    stats['p99'] = float(np.percentile(acts, 99))

    # Dead neuron detection (for ReLU)
    if activation_type.lower() in ['relu', 'leakyrelu', 'prelu']:
        zero_percent = 100 * stats['num_zeros'] / stats['total_activations']
        stats['dead_neuron_percent'] = zero_percent

    # Saturation detection (for sigmoid/tanh)
    if activation_type.lower() == 'sigmoid':
        saturated_low = np.sum(acts < 0.1)
        saturated_high = np.sum(acts > 0.9)
        stats['saturation_percent'] = 100 * (saturated_low + saturated_high) / stats['total_activations']
    elif activation_type.lower() == 'tanh':
        saturated_low = np.sum(acts < -0.9)
        saturated_high = np.sum(acts > 0.9)
        stats['saturation_percent'] = 100 * (saturated_low + saturated_high) / stats['total_activations']

    return stats


def diagnose_activation_pathologies(activation_stats: Dict[str, Dict],
                                    model: nn.Module) -> Dict[str, List[str]]:
    """
    Diagnose activation pathologies.

    Args:
        activation_stats: Dictionary mapping layer names to statistics
        model: PyTorch model (to infer activation types)

    Returns:
        Dictionary with 'errors', 'warnings', 'info' lists
    """
    errors = []
    warnings = []
    info = []

    for layer_name, stats in activation_stats.items():
        # Dead ReLU neurons
        if 'dead_neuron_percent' in stats:
            if stats['dead_neuron_percent'] > 50:
                errors.append(f"💀 {layer_name}: {stats['dead_neuron_percent']:.1f}% dead neurons (ReLU)")
                errors.append(f"   → Learning rate may be too high or initialization poor")
            elif stats['dead_neuron_percent'] > 20:
                warnings.append(f"⚠️  {layer_name}: {stats['dead_neuron_percent']:.1f}% dead neurons")

        # Saturated activations (sigmoid/tanh)
        if 'saturation_percent' in stats:
            if stats['saturation_percent'] > 80:
                errors.append(f"📉 {layer_name}: {stats['saturation_percent']:.1f}% saturated activations")
                errors.append(f"   → Consider switching to ReLU or adjusting input scale")
            elif stats['saturation_percent'] > 50:
                warnings.append(f"⚠️  {layer_name}: {stats['saturation_percent']:.1f}% saturated")

        # Very small activations (potential vanishing)
        if stats['max'] < 1e-5:
            warnings.append(f"❄️  {layer_name}: Max activation = {stats['max']:.2e} (very small)")

        # Very large activations (potential exploding)
        if stats['max'] > 1e5:
            warnings.append(f"🔥 {layer_name}: Max activation = {stats['max']:.2e} (very large)")

        # All zeros (dead layer)
        if stats['num_zeros'] == stats['total_activations']:
            errors.append(f"☠️  {layer_name}: ALL activations are zero (dead layer)")

        # Skewed distributions
        if stats['mean'] != 0:
            skew = (stats['mean'] - stats['median']) / (stats['std'] + 1e-10)
            if abs(skew) > 2:
                info.append(f"📊 {layer_name}: Highly skewed distribution (skew={skew:.2f})")

    return {
        'errors': errors,
        'warnings': warnings,
        'info': info
    }


def print_activation_report(activation_stats: Dict[str, Dict],
                           diagnosis: Dict[str, List[str]],
                           verbose: bool = False):
    """Print comprehensive activation analysis report."""

    print("\n" + "="*80)
    print("ACTIVATION DIAGNOSTIC REPORT")
    print("="*80)

    # Print errors
    if diagnosis['errors']:
        print("\n🚨 CRITICAL ISSUES:")
        for error in diagnosis['errors']:
            print(f"  {error}")

    # Print warnings
    if diagnosis['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in diagnosis['warnings']:
            print(f"  {warning}")

    # Print info
    if diagnosis['info']:
        print("\nℹ️  INFORMATION:")
        for info in diagnosis['info']:
            print(f"  {info}")

    # Summary statistics
    print("\n" + "-"*80)
    print("ACTIVATION SUMMARY:")
    print("-"*80)
    print(f"{'Layer':<45} {'Mean':>12} {'Std':>12} {'Dead %':>10}")
    print("-"*80)

    for layer_name, stats in activation_stats.items():
        dead_percent = stats.get('dead_neuron_percent', 0)
        indicator = ""
        if dead_percent > 50:
            indicator = " 💀"
        elif dead_percent > 20:
            indicator = " ⚠️"

        layer_display = layer_name[:43] + ".." if len(layer_name) > 45 else layer_name
        print(f"{layer_display:<45} {stats['mean']:>12.4f} {stats['std']:>12.4f} "
              f"{dead_percent:>9.1f}%{indicator}")

    # Detailed statistics (verbose mode)
    if verbose:
        print("\n" + "-"*80)
        print("DETAILED ACTIVATION STATISTICS:")
        print("-"*80)
        for layer_name, stats in activation_stats.items():
            print(f"\n{layer_name}:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}, Median: {stats['median']:.6f}, Max: {stats['max']:.6f}")
            print(f"  Percentiles: 1%={stats['p01']:.4f}, 10%={stats['p10']:.4f}, "
                  f"90%={stats['p90']:.4f}, 99%={stats['p99']:.4f}")
            print(f"  Zero: {stats['num_zeros']}/{stats['total_activations']} "
                  f"({100*stats['num_zeros']/stats['total_activations']:.1f}%)")
            if 'dead_neuron_percent' in stats:
                print(f"  Dead neurons: {stats['dead_neuron_percent']:.1f}%")
            if 'saturation_percent' in stats:
                print(f"  Saturation: {stats['saturation_percent']:.1f}%")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    if diagnosis['errors']:
        print("\n🔧 IMMEDIATE ACTIONS:")

        if any("dead neurons" in e.lower() for e in diagnosis['errors']):
            print("\n  Dead ReLU Neurons Detected:")
            print("    1. Reduce learning rate (try 10x smaller)")
            print("    2. Use He initialization: nn.init.kaiming_normal_(weights)")
            print("    3. Consider Leaky ReLU instead: nn.LeakyReLU(0.01)")
            print("    4. Add batch normalization before activation")
            print("    5. Check for large negative biases")

        if any("saturated" in e.lower() for e in diagnosis['errors']):
            print("\n  Saturated Activations Detected:")
            print("    1. Switch to ReLU family activations")
            print("    2. Reduce input magnitude (normalize inputs)")
            print("    3. Use batch normalization")
            print("    4. Consider residual connections")

        if any("dead layer" in e.lower() for e in diagnosis['errors']):
            print("\n  Dead Layer Detected:")
            print("    1. CRITICAL: Check weight initialization")
            print("    2. Verify layer is receiving non-zero inputs")
            print("    3. Check for bugs in forward pass")
            print("    4. May need to restart training")

    elif diagnosis['warnings']:
        print("\n⚙️  SUGGESTED IMPROVEMENTS:")
        print("  1. Monitor activation distributions during training")
        print("  2. Consider adding batch normalization")
        print("  3. Review layer-wise statistics above")
        print("  4. May benefit from activation function changes")

    else:
        print("\n✅ Activation distributions appear healthy!")
        print("  - No critical issues detected")
        print("  - Neurons are learning properly")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze neural network activation distributions"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt, .pth)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to sample data (.pt file)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics for all layers"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save activation distribution plots (requires matplotlib)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ACTIVATION ANALYSIS TOOL")
    print("="*80)

    print("\nNote: This is a template script. To use with your model:")
    print("  1. Import your model class")
    print("  2. Instantiate and load: model = YourModel(); model.load_state_dict(...)")
    print("  3. Call: hooks = register_activation_hooks(model)")
    print("  4. Forward pass: output = model(data)")
    print("  5. Analyze: stats = {name: analyze_activation_statistics(hook.activations, name)")
    print("              for name, hook in hooks.items()}")
    print("  6. Diagnose: diagnosis = diagnose_activation_pathologies(stats, model)")
    print("  7. Report: print_activation_report(stats, diagnosis)")

    print("\nThe analysis functions are available for import:")
    print("  from analyze_activations import (")
    print("      register_activation_hooks,")
    print("      analyze_activation_statistics,")
    print("      diagnose_activation_pathologies,")
    print("      print_activation_report")
    print("  )")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
