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
    return fig_path


def plot_three_way_comparison(baseline_metrics, enhanced_metrics, dataset_name="German", model_name="Random Forest", save_dir=RESULTS_FIGURES_DIR):
    """
    Plots a 3-way grouped bar chart:
      1. Authors' Published (Paper)
      2. Baseline Reproduction (Leak-Free Algorithm 1)
      3. Enhanced Model (Phase 1 Methodological Corrections)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_name.lower() == "german":
        author_benchmarks = {"Accuracy": 0.954, "Recall": 0.929, "Precision": 0.945, "Specificity": 0.968, "F1-Score": 0.937}
    else:
        author_benchmarks = {"Accuracy": 0.821, "Recall": 0.745, "Precision": 0.512, "Specificity": 0.842, "F1-Score": 0.610}
        
    metrics_keys = ["Accuracy", "Recall", "Precision", "Specificity", "F1-Score"]
    author_vals = [author_benchmarks[k] for k in metrics_keys]
    base_vals = [baseline_metrics.get(k, 0.0) for k in metrics_keys]
    enh_vals = [enhanced_metrics.get(k, 0.0) for k in metrics_keys]
    
    x = np.arange(len(metrics_keys))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, author_vals, width, label="1. Authors' Published (Paper)", color="#457b9d", alpha=0.9)
    bars2 = ax.bar(x, base_vals, width, label="2. Our Baseline Reproduction", color="#e76f51", alpha=0.9)
    bars3 = ax.bar(x + width, enh_vals, width, label="3. Our Enhanced Pipeline (Corrected)", color="#2a9d8f", alpha=0.95)
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Three-Way Performance Comparison: Published vs. Baseline vs. Enhanced\n({dataset_name} Credit - {model_name})", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars3:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
                    
    plt.tight_layout()
    fig_path = save_path / f"three_way_comparison_{dataset_name.lower()}_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def plot_skewlog_effect(s_before, s_after, feature_name="BILL_AMT1", save_dir=RESULTS_FIGURES_DIR):
    """Plots side-by-side histograms comparing raw vs SkewLog transformed features."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    axes[0].hist(s_before, bins=40, color="#e76f51", edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Original {feature_name}\n(Skewness: {pd.Series(s_before).skew():.2f})", fontsize=11)
    axes[0].set_xlabel("Monetary Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    axes[1].hist(s_after, bins=40, color="#2a9d8f", edgecolor="black", alpha=0.7)
    axes[1].set_title(f"SkewLog Transformed {feature_name}\n(Skewness: {pd.Series(s_after).skew():.2f})", fontsize=11)
    axes[1].set_xlabel("Signed ln(|x| + 1)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig_path = save_path / f"skewlog_{feature_name.lower()}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def plot_threshold_curves(y_true, y_probs, cost_fn=5, cost_fp=1, dataset_name="German", save_dir=RESULTS_FIGURES_DIR):
    """Plots the commercial cost and F1/MCC curves across decision thresholds."""
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    thresholds = np.linspace(0.10, 0.70, 61)
    costs = []
    f1s = []
    mccs = []
    
    for thr in thresholds:
        preds = (y_probs >= thr).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cost = cost_fn * fn + cost_fp * fp
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        mcc = float(matthews_corrcoef(y_true, preds))
        costs.append(cost)
        f1s.append(f1)
        mccs.append(mcc)
        
    best_cost_idx = int(np.argmin(costs))
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(thresholds, costs, color="#e63946", lw=2.2, label=f"Financial Cost ({cost_fn}:1 Matrix)")
    l2 = ax2.plot(thresholds, f1s, color="#457b9d", lw=2, linestyle="--", label="F1-Score")
    l3 = ax2.plot(thresholds, mccs, color="#2a9d8f", lw=2, linestyle="-.", label="MCC")
    
    ax1.axvline(thresholds[best_cost_idx], color="#e63946", linestyle=":", lw=1.8, label=f"Cost-Optimal ({thresholds[best_cost_idx]:.2f})")
    ax1.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default 0.50 Cutoff")
    
    ax1.set_xlabel("Decision Threshold (p*)", fontsize=11)
    ax1.set_ylabel("Total Financial Cost (Lower is Better)", color="#e63946", fontsize=11)
    ax2.set_ylabel("F1 / MCC Metric (Higher is Better)", color="#2a9d8f", fontsize=11)
    ax1.set_title(f"Decision Threshold Optimization Landscape\n({dataset_name} Credit Risk Prediction)", fontsize=12, pad=12)
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig_path = save_path / f"threshold_curves_{dataset_name.lower()}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def plot_fairness_disparities(fairness_list, save_dir=RESULTS_FIGURES_DIR):
    """Plots demographic parity gap and equal opportunity gap across protected attributes."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    attrs = [item["Attribute"] for item in fairness_list]
    dp_gaps = [item["DemographicParityGap"] for item in fairness_list]
    eo_gaps = [item["EqualOpportunityGap"] for item in fairness_list]
    
    x = np.arange(len(attrs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, dp_gaps, width, label="Demographic Parity Gap (Selection Rate)", color="#457b9d", alpha=0.9)
    b2 = ax.bar(x + width/2, eo_gaps, width, label="Equal Opportunity Gap (Recall / TPR)", color="#e76f51", alpha=0.9)
    
    ax.axhline(0.10, color="red", linestyle="--", label="Four-Fifths / 10% Disparity Threshold")
    ax.set_ylabel("Disparity Gap (|Group A - Group B|)", fontsize=11)
    ax.set_title("Trustworthy AI Demographic Fairness Audit\n(Protected Attributes: Gender & Age)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(max(dp_gaps + eo_gaps) * 1.3, 0.25))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    
    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    fig_path = save_path / "fairness_audit_disparities.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path
