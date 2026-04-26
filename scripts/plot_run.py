"""Plot training metrics from CSV logs.

Usage:
    # Single run
    python scripts/plot_run.py outputs/phase1_pretrain/metrics.csv

    # Compare two runs (e.g. Velvet vs AdamW)
    python scripts/plot_run.py outputs/velvet/metrics.csv outputs/adamw/metrics.csv --labels Velvet AdamW

    # Custom output path
    python scripts/plot_run.py outputs/phase1_pretrain/metrics.csv -o my_plot.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt


def read_metrics(csv_path: str) -> dict:
    """Read a metrics CSV into a dict of lists."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(val)
    return data


def plot_single(data: dict, title: str, output_path: str):
    """Plot metrics from a single run."""
    has_velvet = 'beta1' in data

    if has_velvet:
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    steps = data['step']

    # Loss
    ax = axes[0, 0]
    ax.plot(steps, data['loss'], 'b-', linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[0, 1]
    if 'ppl' in data:
        ax.plot(steps, data['ppl'], 'b-', linewidth=0.8)
        ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity (log scale)')
    ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 0]
    ax.plot(steps, data['lr'], 'r-', linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Effective LR')
    ax.grid(True, alpha=0.3)

    # Grad norm (or empty for AdamW)
    ax = axes[1, 1]
    if has_velvet and 'grad_norm' in data:
        ax.plot(steps, data['grad_norm'], 'orange', linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad Norm')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)

    if has_velvet:
        # Beta1 (PGM)
        ax = axes[2, 0]
        ax.plot(steps, data['beta1'], 'g-', linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Beta1')
        ax.set_title('PGM: Effective Beta1')
        ax.grid(True, alpha=0.3)

        # LVS scale
        ax = axes[2, 1]
        ax.plot(steps, data['lvs_scale'], 'm-', linewidth=0.8)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Step')
        ax.set_ylabel('LVS Scale')
        ax.set_title('LVS: LR Scale Factor')
        ax.grid(True, alpha=0.3)

        # Signal strength
        ax = axes[3, 0]
        if 'signal' in data:
            ax.plot(steps, data['signal'], 'c-', linewidth=0.8)
        elif 'r2' in data:
            ax.plot(steps, data['r2'], 'c-', linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Signal')
        ax.set_title('LVS: Signal Strength (EMA gap)')
        ax.grid(True, alpha=0.3)

        # LVS phase
        ax = axes[3, 1]
        if 'lvs_phase' in data:
            ax.plot(steps, data['lvs_phase'], 'k-', linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Phase')
        ax.set_title('LVS: Training Phase (0=early, 1=late)')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_compare(datasets: list, labels: list, output_path: str):
    """Compare loss curves from multiple runs."""
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss comparison
    ax = axes[0]
    for i, (data, label) in enumerate(zip(datasets, labels)):
        color = colors[i % len(colors)]
        ax.plot(data['step'], data['loss'], color=color, linewidth=1.0, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LR comparison
    ax = axes[1]
    for i, (data, label) in enumerate(zip(datasets, labels)):
        color = colors[i % len(colors)]
        ax.plot(data['step'], data['lr'], color=color, linewidth=1.0, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LR Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Run Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot R-Velvet training metrics")
    parser.add_argument("csv_files", nargs='+', help="Path(s) to metrics.csv")
    parser.add_argument("--labels", nargs='*', help="Labels for each run (for comparison)")
    parser.add_argument("-o", "--output", default=None, help="Output image path (default: auto)")
    args = parser.parse_args()

    datasets = [read_metrics(f) for f in args.csv_files]

    if len(datasets) == 1:
        title = Path(args.csv_files[0]).parent.name
        out = args.output or str(Path(args.csv_files[0]).parent / "plot.png")
        plot_single(datasets[0], title, out)
    else:
        labels = args.labels or [Path(f).parent.name for f in args.csv_files]
        out = args.output or "comparison.png"
        plot_compare(datasets, labels, out)


if __name__ == "__main__":
    main()
