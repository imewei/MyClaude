#!/usr/bin/env python3
"""
Training Run Comparison Tool

Compare multiple training runs to identify differences in hyperparameters,
metrics, and outcomes. Useful for ablation studies and hyperparameter tuning.

Usage:
    python compare_training_runs.py --runs exp1/ exp2/ exp3/
    python compare_training_runs.py --runs exp*/  --metric val_loss
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_training_log(log_path: Path) -> Optional[Dict]:
    """
    Load training log from various formats (JSON, TensorBoard, W&B).

    Args:
        log_path: Path to experiment directory

    Returns:
        Dictionary with training history and config, or None if failed
    """
    # Try JSON format (most common)
    json_files = list(log_path.glob("*.json"))
    for json_file in json_files:
        if "metrics" in json_file.name or "history" in json_file.name:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                return data
            except Exception as e:
                continue

    # Try config.json + metrics.json pattern
    config_file = log_path / "config.json"
    metrics_file = log_path / "metrics.json"

    if config_file.exists() and metrics_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            with open(metrics_file) as f:
                metrics = json.load(f)
            return {"config": config, "metrics": metrics}
        except Exception as e:
            pass

    # Could add TensorBoard, W&B parsing here
    print(f"Warning: Could not load training log from {log_path}")
    return None


def extract_config(run_data: Dict) -> Dict:
    """Extract configuration/hyperparameters from run data."""
    if "config" in run_data:
        return run_data["config"]
    elif "hyperparameters" in run_data:
        return run_data["hyperparameters"]
    elif "args" in run_data:
        return run_data["args"]
    else:
        # Try to find config-like keys
        config = {}
        for key, value in run_data.items():
            if isinstance(value, (int, float, str, bool)):
                config[key] = value
        return config


def extract_metrics(run_data: Dict) -> Dict[str, List[float]]:
    """Extract training metrics time series."""
    if "metrics" in run_data:
        return run_data["metrics"]
    elif "history" in run_data:
        return run_data["history"]
    else:
        # Try to find metric-like keys
        metrics = {}
        for key, value in run_data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    metrics[key] = value
        return metrics


def compare_configs(configs: Dict[str, Dict]) -> Dict:
    """
    Compare configurations across runs to identify differences.

    Args:
        configs: Dictionary mapping run names to config dicts

    Returns:
        Dictionary with 'common' and 'different' keys
    """
    if not configs:
        return {"common": {}, "different": {}}

    # Get all keys
    all_keys = set()
    for config in configs.values():
        all_keys.update(config.keys())

    common = {}
    different = {}

    for key in all_keys:
        values = {}
        for run_name, config in configs.items():
            if key in config:
                values[run_name] = config[key]

        # Check if all values are the same
        unique_values = set(str(v) for v in values.values())
        if len(unique_values) == 1 and len(values) == len(configs):
            common[key] = list(values.values())[0]
        else:
            different[key] = values

    return {"common": common, "different": different}


def compare_metrics(metrics_dict: Dict[str, Dict[str, List[float]]],
                   metric_name: str = "val_loss") -> Dict:
    """
    Compare specific metric across runs.

    Args:
        metrics_dict: Dictionary mapping run names to metrics
        metric_name: Name of metric to compare

    Returns:
        Comparison statistics
    """
    comparison = {}

    for run_name, metrics in metrics_dict.items():
        if metric_name not in metrics:
            comparison[run_name] = {
                "available": False,
                "message": f"Metric '{metric_name}' not found"
            }
            continue

        values = metrics[metric_name]
        comparison[run_name] = {
            "available": True,
            "final": values[-1] if values else None,
            "best": min(values) if "loss" in metric_name.lower() else max(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "num_epochs": len(values),
            "convergence_epoch": _find_convergence_epoch(values, metric_name),
            "values": values
        }

    return comparison


def _find_convergence_epoch(values: List[float], metric_name: str) -> int:
    """Find epoch where metric converged (no improvement for 5 epochs)."""
    if len(values) < 6:
        return len(values) - 1

    is_loss = "loss" in metric_name.lower()
    best_value = values[0]
    no_improvement = 0

    for i, value in enumerate(values):
        improved = (value < best_value) if is_loss else (value > best_value)

        if improved:
            best_value = value
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= 5:
            return i - 4  # Return epoch where last improvement occurred

    return len(values) - 1


def print_comparison_report(configs: Dict[str, Dict],
                           metrics_dict: Dict[str, Dict[str, List[float]]],
                           primary_metric: str = "val_loss"):
    """Print comprehensive comparison report."""

    print("\n" + "="*80)
    print("TRAINING RUN COMPARISON REPORT")
    print("="*80)

    run_names = list(configs.keys())
    print(f"\nComparing {len(run_names)} runs: {', '.join(run_names)}")

    # Configuration comparison
    print("\n" + "-"*80)
    print("CONFIGURATION DIFFERENCES:")
    print("-"*80)

    config_comparison = compare_configs(configs)

    if config_comparison["different"]:
        print(f"\n{'Parameter':<30} {' '.join(f'{name:<15}' for name in run_names)}")
        print("-"*80)

        for param, values in sorted(config_comparison["different"].items()):
            value_strs = [str(values.get(name, "N/A"))[:15] for name in run_names]
            print(f"{param:<30} {' '.join(f'{v:<15}' for v in value_strs)}")
    else:
        print("\n✅ All configurations are identical")

    if config_comparison["common"]:
        print(f"\n📋 Common parameters: {len(config_comparison['common'])} "
              f"(use --verbose to see all)")

    # Metric comparison
    print("\n" + "-"*80)
    print(f"METRIC COMPARISON: {primary_metric}")
    print("-"*80)

    metric_comparison = compare_metrics(metrics_dict, primary_metric)

    print(f"\n{'Run':<20} {'Final':>12} {'Best':>12} {'Mean±Std':>15} {'Converged':>12}")
    print("-"*80)

    for run_name in run_names:
        stats = metric_comparison.get(run_name, {})

        if not stats.get("available", False):
            print(f"{run_name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>15} {'N/A':>12}")
            continue

        final = f"{stats['final']:.4f}"
        best = f"{stats['best']:.4f}"
        mean_std = f"{stats['mean']:.4f}±{stats['std']:.4f}"
        converged = f"Epoch {stats['convergence_epoch']}"

        # Add indicator for best run
        indicator = ""
        if stats['best'] == min(s['best'] for s in metric_comparison.values()
                               if s.get('available', False)) and "loss" in primary_metric.lower():
            indicator = " 🏆"
        elif stats['best'] == max(s['best'] for s in metric_comparison.values()
                                 if s.get('available', False)) and "loss" not in primary_metric.lower():
            indicator = " 🏆"

        print(f"{run_name:<20} {final:>12} {best:>12} {mean_std:>15} {converged:>12}{indicator}")

    # Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS:")
    print("="*80)

    # Find best run
    is_loss = "loss" in primary_metric.lower()
    valid_runs = {name: stats for name, stats in metric_comparison.items()
                  if stats.get("available", False)}

    if valid_runs:
        best_run = min(valid_runs.items(), key=lambda x: x[1]['best']) if is_loss \
                  else max(valid_runs.items(), key=lambda x: x[1]['best'])

        print(f"\n🏆 BEST RUN: {best_run[0]}")
        print(f"   {primary_metric}: {best_run[1]['best']:.4f} "
              f"(converged at epoch {best_run[1]['convergence_epoch']})")

        # Key differences for best run
        if config_comparison["different"]:
            print("\n🔑 KEY HYPERPARAMETERS FOR BEST RUN:")
            for param, values in sorted(config_comparison["different"].items()):
                if best_run[0] in values:
                    print(f"   {param}: {values[best_run[0]]}")

    # Analysis
    print("\n💡 INSIGHTS:")

    # Check for overfitting
    for run_name, stats in metric_comparison.items():
        if stats.get("available"):
            convergence_pct = 100 * stats['convergence_epoch'] / stats['num_epochs']
            if convergence_pct < 50:
                print(f"   ⚠️  {run_name}: Early convergence ({convergence_pct:.0f}% through training)")
                print(f"      → Consider increasing learning rate or model capacity")

    # Check for instability
    for run_name, stats in metric_comparison.items():
        if stats.get("available"):
            # High variance indicates instability
            if stats['std'] > 0.1 * abs(stats['mean']):
                print(f"   ⚠️  {run_name}: High variance in {primary_metric}")
                print(f"      → Training may be unstable, consider reducing learning rate")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple training runs"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to experiment directories"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_loss",
        help="Primary metric to compare (default: val_loss)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed statistics"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("TRAINING RUN COMPARISON TOOL")
    print("="*80)

    print("\nNote: This is a template script. Expected data format:")
    print("\nEach run directory should contain:")
    print("  - config.json: Hyperparameters and settings")
    print("  - metrics.json: Training metrics (loss, accuracy, etc.)")
    print("\nExample structure:")
    print("  experiment1/")
    print("    ├── config.json")
    print("    └── metrics.json")

    print("\nThe comparison functions are available for import:")
    print("  from compare_training_runs import (")
    print("      load_training_log,")
    print("      compare_configs,")
    print("      compare_metrics,")
    print("      print_comparison_report")
    print("  )")

    # Example usage
    print("\n" + "-"*80)
    print("EXAMPLE USAGE:")
    print("-"*80)
    print("\nIn your code:")
    print("""
    configs = {}
    metrics = {}
    for run_dir in run_directories:
        data = load_training_log(run_dir)
        configs[run_dir.name] = extract_config(data)
        metrics[run_dir.name] = extract_metrics(data)

    print_comparison_report(configs, metrics, primary_metric='val_loss')
    """)

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
