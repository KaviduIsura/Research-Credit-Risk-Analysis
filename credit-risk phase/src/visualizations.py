"""
Visualization Module for Model Comparisons and Technical Reporting.
Generates high-resolution comparative figures matching coursework specifications.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import RESULTS_FIGURES_DIR


def plot_author_comparison(reproduced_metrics, dataset_name="German", model_name="MLP", save_dir=RESULTS_FIGURES_DIR):
    """
    Plots a side-by-side grouped bar chart comparing the reproduced metrics against
    the authors' published results from Table 5 (German) or Table 6 (Taiwan).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Paper published benchmarks
    if dataset_name.lower() == "german":
        author_benchmarks = {
            "Accuracy": 0.954,
            "Recall": 0.929,
            "Specificity": 0.968,
            "Precision": 0.945,
            "F1-Score": 0.937
        }
    else:
        author_benchmarks = {
            "Accuracy": 0.821,
            "Recall": 0.745,
            "Specificity": 0.842,
            "Precision": 0.512,
            "F1-Score": 0.610
        }
        
    metrics_keys = ["Accuracy", "Recall", "Specificity", "Precision", "F1-Score"]
    author_vals = [author_benchmarks[k] for k in metrics_keys]
    our_vals = [reproduced_metrics.get(k, 0.0) for k in metrics_keys]
    
    x = np.arange(len(metrics_keys))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, author_vals, width, label="Authors Reported (Paper)", color="#2b5c8f", alpha=0.9)
    bars2 = ax.bar(x + width/2, our_vals, width, label="Our Reproduction (Leak-Free)", color="#e07a5f", alpha=0.9)
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Performance Comparison: Authors' Published vs. Reproduction\n({dataset_name} Credit - {model_name} + SMOTE-ENN)", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    
    # Add values on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    fig_path = save_path / f"comparison_{dataset_name.lower()}_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison figure to: {fig_path}")
    return fig_path
