#!/usr/bin/env python3
"""
Generate Visual Diagrams for JAX Core Programming

This script generates visual charts and diagrams for:
- Performance comparisons
- Memory optimization
- Speedup charts
- Architecture diagrams

Requirements: matplotlib, numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def setup_plot_style():
    """Configure matplotlib for professional-looking plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def generate_speedup_chart(output_dir):
    """Generate JAX transformation speedup comparison chart"""
    transformations = ['Baseline\n(Python)', 'jit', 'vmap', 'jit+vmap', 'Multi-Device\n(8 GPUs)']
    speedups = [1, 50, 5, 250, 1000]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(transformations, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup}x',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Speedup (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('JAX Transformation Performance Speedups', fontsize=16, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.5, 2000)

    # Add reference lines
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10x speedup')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100x speedup')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'speedup_comparison.png'}")
    plt.close()


def generate_memory_optimization_chart(output_dir):
    """Generate memory optimization techniques comparison"""
    techniques = ['Baseline', 'Remat\n(2-5x)', 'Mixed\nPrecision', 'Gradient\nAccum', 'Multi-Device\n(8 GPUs)']
    memory_reduction = [100, 30, 50, 25, 12.5]  # Percentage of baseline
    compute_overhead = [100, 130, 70, 100, 100]  # Percentage of baseline (lower is better for time)

    x = np.arange(len(techniques))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width/2, memory_reduction, width, label='Memory Usage (%)',
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, compute_overhead, width, label='Compute Time (%)',
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Optimization Technique', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Baseline (%)', fontsize=14, fontweight='bold')
    ax.set_title('Memory Optimization Techniques - Memory vs Compute Trade-offs',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Baseline')

    plt.tight_layout()
    plt.savefig(output_dir / 'memory_optimization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'memory_optimization.png'}")
    plt.close()


def generate_batch_size_scaling(output_dir):
    """Generate batch size scaling efficiency chart"""
    batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    throughput_single = [100, 190, 350, 620, 1100, 1800, 2200, 2400]  # samples/sec
    throughput_multi = [800, 1520, 2800, 4960, 8800, 14400, 17600, 19200]  # 8 GPUs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Throughput
    ax1.plot(batch_sizes, throughput_single, marker='o', linewidth=2.5,
             markersize=8, label='Single GPU', color='#1f77b4')
    ax1.plot(batch_sizes, throughput_multi, marker='s', linewidth=2.5,
             markersize=8, label='8 GPUs (Data Parallel)', color='#ff7f0e')

    ax1.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Throughput (samples/sec)', fontsize=14, fontweight='bold')
    ax1.set_title('Batch Size Scaling - Throughput', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12, loc='upper left')

    # Right plot: Efficiency
    efficiency = np.array(throughput_multi) / (np.array(throughput_single) * 8) * 100
    ax2.plot(batch_sizes, efficiency, marker='D', linewidth=2.5,
             markersize=8, color='#2ca02c')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Scaling')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='90% Efficiency')

    ax2.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Multi-GPU Scaling Efficiency', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'batch_size_scaling.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'batch_size_scaling.png'}")
    plt.close()


def generate_optimizer_comparison(output_dir):
    """Generate optimizer performance comparison"""
    optimizers = ['SGD', 'SGD+Mom', 'Adam', 'AdamW', 'Lion', 'Adafactor']
    convergence_speed = [3, 4, 8, 8.5, 9, 7]  # Relative speed (higher is faster)
    memory_usage = [1, 1, 2, 2, 1.5, 1.2]  # Relative memory (lower is better)
    final_accuracy = [92.3, 93.1, 94.5, 94.8, 94.9, 94.2]  # Percentage

    x = np.arange(len(optimizers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width, convergence_speed, width, label='Convergence Speed',
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, [10 - m for m in memory_usage], width, label='Memory Efficiency (10-usage)',
                   color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, [a - 90 for a in final_accuracy], width, label='Final Accuracy (%-90)',
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Optimizer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Performance (arbitrary units)', fontsize=14, fontweight='bold')
    ax.set_title('Optimizer Comparison - Speed, Memory, and Accuracy',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'optimizer_comparison.png'}")
    plt.close()


def generate_training_progress(output_dir):
    """Generate typical training progress curves"""
    steps = np.linspace(0, 10000, 100)

    # Training loss
    train_loss = 2.5 * np.exp(-steps / 2000) + 0.3 + 0.05 * np.random.randn(100)
    train_loss = np.maximum(train_loss, 0.2)

    # Validation loss
    val_loss = 2.5 * np.exp(-steps / 2000) + 0.4 + 0.08 * np.random.randn(100)
    val_loss = np.maximum(val_loss, 0.3)

    # Learning rate schedule (warmup + cosine decay)
    warmup_steps = 1000
    lr = np.where(steps < warmup_steps,
                  steps / warmup_steps * 0.001,
                  0.001 * (1 + np.cos(np.pi * (steps - warmup_steps) / (10000 - warmup_steps))) / 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curves
    ax1.plot(steps, train_loss, linewidth=2, label='Training Loss', color='#1f77b4')
    ax1.plot(steps, val_loss, linewidth=2, label='Validation Loss', color='#ff7f0e')
    ax1.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Progress - Loss Curves', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Learning rate schedule
    ax2.plot(steps, lr, linewidth=2.5, color='#2ca02c')
    ax2.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Warmup End')
    ax2.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax2.set_title('Learning Rate Schedule (Warmup + Cosine Decay)', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'training_progress.png'}")
    plt.close()


def main():
    """Generate all diagrams"""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating JAX Core Programming Visual Diagrams")
    print("=" * 60 + "\n")

    setup_plot_style()

    try:
        generate_speedup_chart(output_dir)
        generate_memory_optimization_chart(output_dir)
        generate_batch_size_scaling(output_dir)
        generate_optimizer_comparison(output_dir)
        generate_training_progress(output_dir)

        print("\n" + "=" * 60)
        print("✓ All diagrams generated successfully!")
        print(f"Location: {output_dir}")
        print("\nGenerated files:")
        print("  - speedup_comparison.png")
        print("  - memory_optimization.png")
        print("  - batch_size_scaling.png")
        print("  - optimizer_comparison.png")
        print("  - training_progress.png")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Error generating diagrams: {e}")
        print("\nMake sure matplotlib and numpy are installed:")
        print("  pip install matplotlib numpy")


if __name__ == '__main__':
    main()
