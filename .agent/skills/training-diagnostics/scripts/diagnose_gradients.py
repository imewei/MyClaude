#!/usr/bin/env python3
"""
Gradient Diagnostic Tool for Neural Network Training

Analyzes gradient flow through neural network layers to detect vanishing or
exploding gradient problems. Provides detailed statistics and visualizations.

Usage:
    python diagnose_gradients.py --checkpoint model.pt --log-dir logs/
    python diagnose_gradients.py --model-class MyModel --data data.pt
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


def compute_gradient_stats(model: nn.Module,
                          data_loader: Optional[torch.utils.data.DataLoader] = None,
                          inputs: Optional[torch.Tensor] = None,
                          targets: Optional[torch.Tensor] = None,
                          criterion: Optional[nn.Module] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive gradient statistics for all model parameters.

    Args:
        model: PyTorch model to analyze
        data_loader: DataLoader with (inputs, targets) pairs
        inputs: Single batch of inputs (if data_loader not provided)
        targets: Single batch of targets (if data_loader not provided)
        criterion: Loss function (defaults to CrossEntropyLoss)

    Returns:
        Dictionary mapping parameter names to gradient statistics
    """
    model.train()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Get single batch
    if data_loader is not None:
        inputs, targets = next(iter(data_loader))
    elif inputs is None or targets is None:
        raise ValueError("Must provide either data_loader or (inputs, targets)")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Collect gradient statistics
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().cpu().numpy()
            grad_stats[name] = {
                'mean': float(np.mean(grad)),
                'std': float(np.std(grad)),
                'min': float(np.min(grad)),
                'max': float(np.max(grad)),
                'norm': float(np.linalg.norm(grad)),
                'abs_mean': float(np.mean(np.abs(grad))),
                'num_zeros': int(np.sum(grad == 0)),
                'num_nan': int(np.sum(np.isnan(grad))),
                'num_inf': int(np.sum(np.isinf(grad))),
                'total_params': int(grad.size),
                'shape': param.shape
            }

    return grad_stats


def diagnose_gradient_pathology(grad_stats: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    """
    Diagnose gradient pathologies (vanishing, exploding, dead neurons).

    Args:
        grad_stats: Gradient statistics from compute_gradient_stats

    Returns:
        Dictionary with 'warnings', 'errors', and 'info' lists
    """
    warnings = []
    errors = []
    info = []

    # Check for NaN or Inf
    for name, stats in grad_stats.items():
        if stats['num_nan'] > 0:
            errors.append(f"❌ {name}: {stats['num_nan']} NaN gradients detected")
        if stats['num_inf'] > 0:
            errors.append(f"❌ {name}: {stats['num_inf']} Inf gradients detected")

    # Check gradient magnitudes
    norms = {name: stats['norm'] for name, stats in grad_stats.items()}

    if len(norms) > 0:
        max_norm = max(norms.values())
        min_norm = min(norms.values())

        # Exploding gradients
        if max_norm > 100:
            errors.append(f"🔥 EXPLODING GRADIENTS: Max norm = {max_norm:.2e} (threshold: 100)")
            exploding_layers = [name for name, norm in norms.items() if norm > 100]
            errors.append(f"   Affected layers: {', '.join(exploding_layers[:5])}")
        elif max_norm > 10:
            warnings.append(f"⚠️  Large gradients: Max norm = {max_norm:.2e} (monitor closely)")

        # Vanishing gradients
        if min_norm < 1e-7 and min_norm > 0:
            errors.append(f"❄️  VANISHING GRADIENTS: Min norm = {min_norm:.2e} (threshold: 1e-7)")
            vanishing_layers = [name for name, norm in norms.items() if norm < 1e-7 and norm > 0]
            errors.append(f"   Affected layers: {', '.join(vanishing_layers[:5])}")
        elif min_norm < 1e-5 and min_norm > 0:
            warnings.append(f"⚠️  Small gradients: Min norm = {min_norm:.2e} (monitor closely)")

        # Check gradient norm ratio (early vs late layers)
        if max_norm > 0 and min_norm > 0:
            ratio = max_norm / min_norm
            if ratio > 1000:
                warnings.append(f"⚠️  Large gradient norm ratio: {ratio:.2e} (max/min)")
                info.append("   → Indicates potential gradient flow issues across depth")

    # Dead neurons (high percentage of zero gradients)
    for name, stats in grad_stats.items():
        zero_percent = 100 * stats['num_zeros'] / stats['total_params']
        if zero_percent > 50:
            warnings.append(f"💀 {name}: {zero_percent:.1f}% zero gradients (dead neurons)")
        elif zero_percent > 20:
            info.append(f"ℹ️  {name}: {zero_percent:.1f}% zero gradients")

    return {
        'errors': errors,
        'warnings': warnings,
        'info': info
    }


def print_gradient_report(grad_stats: Dict[str, Dict[str, float]],
                         diagnosis: Dict[str, List[str]],
                         verbose: bool = False):
    """Print comprehensive gradient analysis report."""

    print("\n" + "="*80)
    print("GRADIENT DIAGNOSTIC REPORT")
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
    print("GRADIENT NORM SUMMARY:")
    print("-"*80)

    norms = [(name, stats['norm']) for name, stats in grad_stats.items()]
    norms.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Layer':<50} {'Gradient Norm':>15}")
    print("-"*80)
    for name, norm in norms[:20]:  # Top 20 layers
        indicator = ""
        if norm > 100:
            indicator = " 🔥"
        elif norm < 1e-7:
            indicator = " ❄️"
        print(f"{name:<50} {norm:>15.6e}{indicator}")

    if len(norms) > 20:
        print(f"... ({len(norms) - 20} more layers)")

    # Detailed statistics (verbose mode)
    if verbose:
        print("\n" + "-"*80)
        print("DETAILED GRADIENT STATISTICS:")
        print("-"*80)
        for name, stats in grad_stats.items():
            print(f"\n{name}:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean: {stats['mean']:.6e}, Std: {stats['std']:.6e}")
            print(f"  Min: {stats['min']:.6e}, Max: {stats['max']:.6e}")
            print(f"  Norm: {stats['norm']:.6e}")
            print(f"  Zeros: {stats['num_zeros']}/{stats['total_params']} "
                  f"({100*stats['num_zeros']/stats['total_params']:.1f}%)")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    if diagnosis['errors']:
        print("\n🔧 IMMEDIATE ACTIONS:")

        if any("EXPLODING" in e for e in diagnosis['errors']):
            print("  1. Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
            print("  2. Reduce learning rate by 10x")
            print("  3. Check weight initialization (should be ~0.01 to 0.1)")
            print("  4. Consider using learning rate warmup")

        if any("VANISHING" in e for e in diagnosis['errors']):
            print("  1. Replace sigmoid/tanh with ReLU activations")
            print("  2. Add residual/skip connections")
            print("  3. Use proper initialization (He for ReLU, Xavier for tanh)")
            print("  4. Consider batch normalization")

        if any("NaN" in e or "Inf" in e for e in diagnosis['errors']):
            print("  1. CRITICAL: Check for division by zero or log(0)")
            print("  2. Reduce learning rate significantly")
            print("  3. Add gradient clipping immediately")
            print("  4. Check input data for NaN/Inf values")

    elif diagnosis['warnings']:
        print("\n⚙️  SUGGESTED IMPROVEMENTS:")
        print("  1. Monitor training closely for instability")
        print("  2. Consider gradient clipping as preventive measure")
        print("  3. Review layer-wise gradient norms above")
        print("  4. May need architecture adjustments if issues persist")

    else:
        print("\n✅ Gradient flow appears healthy!")
        print("  - No critical issues detected")
        print("  - Continue monitoring during training")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose gradient flow problems in neural networks"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.pt, .pth)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to sample data (.pt file with 'inputs' and 'targets')"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to save diagnostic logs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics for all layers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    if not args.checkpoint:
        print("Error: --checkpoint is required")
        print("\nExample usage:")
        print("  python diagnose_gradients.py --checkpoint model.pt --data sample_data.pt")
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print("Checkpoint format: state dict in 'model_state_dict' key")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
        print("Checkpoint format: state dict in 'state_dict' key")
    else:
        print("Warning: Cannot automatically load model. Please load manually.")
        print("This script provides gradient analysis functions you can use:")
        print("  from diagnose_gradients import compute_gradient_stats, diagnose_gradient_pathology")
        sys.exit(1)

    # Note: In practice, you'd need to instantiate your model class here
    print("\nNote: To use this script with your model, you need to:")
    print("  1. Import your model class")
    print("  2. Instantiate the model: model = YourModel()")
    print("  3. Load state dict: model.load_state_dict(model_state)")
    print("  4. Call: compute_gradient_stats(model, data_loader)")
    print("\nFor now, this script serves as a template and provides the analysis functions.")


if __name__ == "__main__":
    main()
