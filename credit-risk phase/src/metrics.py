"""
Metric calculation, custom financial cost functions, and scoring interfaces.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    brier_score_loss, accuracy_score, balanced_accuracy_score, make_scorer
)
from src.config import COST_FN, COST_FP

def specificity(y_true, y_pred):
    """True Negative Rate: TN / (TN + FP)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def misclass_cost(y_true, y_pred):
    """
    German credit asymmetric cost function:
    Misclassifying a defaulting borrower (FN) costs 5 units.
    Rejecting a good applicant (FP) costs 1 unit.
    Returns average misclassification cost per applicant.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (COST_FN * fn + COST_FP * fp) / len(y_true)

def get_scoring(ds):
    """Returns comprehensive dictionary of cross_validate scorers."""
    scorers = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "specificity": make_scorer(specificity),
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": "f1",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "mcc": make_scorer(matthews_corrcoef),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "brier": "neg_brier_score"
    }
    if ds == "German":
        scorers["cost"] = make_scorer(misclass_cost, greater_is_better=False)
    return scorers

def thr_scorer(obj):
    """Scorer factory for in-fold decision threshold optimization."""
    if obj == "cost":
        return make_scorer(misclass_cost, greater_is_better=False)
    if obj == "mcc":
        return make_scorer(matthews_corrcoef)
    if obj == "f1":
        return make_scorer(f1_score)
    raise ValueError(f"Unknown threshold tuning objective: {obj}")

def best_threshold(y, p, obj):
    """Finds optimal decision threshold across a grid of 97 candidate cutoffs."""
    grid = np.linspace(0.02, 0.98, 97)
    if obj == "cost":
        vals = [-misclass_cost(y, (p >= t).astype(int)) for t in grid]
    elif obj == "mcc":
        vals = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
    else:
        vals = [f1_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(vals))])
