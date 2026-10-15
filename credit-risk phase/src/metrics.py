"""
Evaluation Metrics for Credit Risk Prediction.
Strictly implements Equations (19) to (23) from Section 3.7 of the research paper:
  - Accuracy (Eq. 19)
  - Precision (Eq. 20)
  - Recall / Sensitivity (Eq. 21)
  - Specificity (Eq. 22)
  - F1-Score (Eq. 23)
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true, y_pred):
    """
    Compute Accuracy, Recall, Specificity, Precision, and F1-Score
    based on the binary confusion matrix:
      TP: True Positives (Default correctly predicted)
      TN: True Negatives (Good/Non-default correctly predicted)
      FP: False Positives (Good incorrectly predicted as Default)
      FN: False Negatives (Default incorrectly predicted as Good)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "Accuracy": float(accuracy),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1-Score": float(f1)
    }


def aggregate_fold_metrics(fold_metrics_list):
    """
    Aggregates a list of metric dicts across K cross-validation folds.
    Returns:
      mean_dict: Mean value for each metric.
      std_dict: Standard deviation for each metric.
    """
    keys = ["Accuracy", "Recall", "Specificity", "Precision", "F1-Score"]
    mean_dict = {}
    std_dict = {}
    
    for k in keys:
        vals = [m[k] for m in fold_metrics_list]
        mean_dict[k] = float(np.mean(vals))
        std_dict[k] = float(np.std(vals))
        
    return mean_dict, std_dict
