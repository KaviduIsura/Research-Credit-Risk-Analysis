"""
Generates side-by-side visual comparison between authors' published numbers and our leak-free results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_comparison_plot(verdicts_path="results/verdicts.csv", output_path="results/figures/authors_vs_ours_comparison.png"):
    if not os.path.exists(verdicts_path):
        print(f"Error: {verdicts_path} not found.")
        return
    
    vt = pd.read_csv(verdicts_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = ["accuracy", "recall", "specificity", "precision", "f1"]

    for i, ds in enumerate(["German", "Taiwan"]):
        ax = axes[i]
        sub = vt[vt.dataset == ds]
        sub = sub.set_index("metric").reindex(metrics).reset_index()
        
        x = np.arange(len(metrics))
        width = 0.35
        
        author_label = "Authors' Published (Table 5)" if ds == "German" else "Authors' Published (Table 6)"
        rects1 = ax.bar(x - width/2, sub["authors"], width, label=author_label, color="#d62728", alpha=0.85)
        rects2 = ax.bar(x + width/2, sub["mine"], width, label="Our Leak-Free Primary Pipeline", color="#1f77b4", alpha=0.85)
        
        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title(f"{ds} Credit: Authors vs. Our Pipeline", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f77b4")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Comparison plot successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_comparison_plot()
