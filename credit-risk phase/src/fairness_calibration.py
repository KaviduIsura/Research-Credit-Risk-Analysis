"""
Trustworthy AI: Probability Calibration and Demographic Fairness Audit.
Implements Phase 1 ethical and reliability requirements:
  - Brier score and Platt scaling probability calibration.
  - Demographic Parity Difference & Equal Opportunity Difference (Gender and Age).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from src.config import RESULTS_FIGURES_DIR


# ---------------------------------------------------------------------------
# 1. Probability Calibration Audit
# ---------------------------------------------------------------------------

def audit_calibration(y_true, y_probs, model_name="Enhanced Model", save_dir=RESULTS_FIGURES_DIR):
    """
    Computes Brier score and plots the reliability calibration curve.
    Brier Score: Mean squared difference between predicted probabilities and actual binary outcomes.
    Lower Brier score indicates superior calibration.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    brier = float(brier_score_loss(y_true, y_probs))
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#2b5c8f", label=f"{model_name} (Brier = {brier:.4f})")
    
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Actual Default Rate)", fontsize=11)
    ax.set_title(f"Probability Calibration Reliability Curve\n({model_name})", fontsize=12, pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig_path = save_path / f"calibration_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    
    return {"BrierScore": round(brier, 4), "FigurePath": str(fig_path)}


# ---------------------------------------------------------------------------
# 2. Demographic Fairness Audit (Gender & Age)
# ---------------------------------------------------------------------------

def audit_demographic_fairness(y_true, y_pred, sensitive_attribute, attr_name="Gender"):
    """
    Evaluates ethical fairness metrics across sensitive groups:
      - Selection Rate: P(Pred=1 | Group)
      - Demographic Parity Difference: max(SR) - min(SR) (Ideal = 0.0)
      - True Positive Rate (Recall): P(Pred=1 | True=1, Group)
      - Equal Opportunity Difference: |TPR_A - TPR_B| (Ideal = 0.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(sensitive_attribute)
    
    unique_groups = np.unique(groups)
    stats_by_group = {}
    
    for g in unique_groups:
        mask = (groups == g)
        n_group = mask.sum()
        
        # Selection Rate (Approval / Denial rate)
        sr = float(y_pred[mask].mean()) if n_group > 0 else 0.0
        
        # True Positive Rate (Recall on actual defaults)
        def_mask = mask & (y_true == 1)
        tpr = float(y_pred[def_mask].mean()) if def_mask.sum() > 0 else 0.0
        
        stats_by_group[str(g)] = {
            "Count": int(n_group),
            "SelectionRate": round(sr, 3),
            "Recall_TPR": round(tpr, 3)
        }
        
    sr_vals = [v["SelectionRate"] for v in stats_by_group.values()]
    tpr_vals = [v["Recall_TPR"] for v in stats_by_group.values()]
    
    demographic_parity_gap = max(sr_vals) - min(sr_vals)
    equal_opportunity_gap = max(tpr_vals) - min(tpr_vals)
    
    summary = {
        "Attribute": attr_name,
        "GroupsEvaluated": list(stats_by_group.keys()),
        "DemographicParityGap": round(demographic_parity_gap, 3),
        "EqualOpportunityGap": round(equal_opportunity_gap, 3),
        "FairnessCompliance": "Acceptable (Gap < 0.10)" if max(demographic_parity_gap, equal_opportunity_gap) < 0.10 else "Potential Disparity Detected",
        "GroupDetails": stats_by_group
    }
    return summary
