"""
Statistical Significance Testing Module.
Implements the Nadeau-Bengio (2003) Corrected Resampled t-Test for Cross-Validation:
  - Corrects for sample dependence across overlapping cross-validation training folds.
  - Formally tests H0: Performance(Enhanced) == Performance(Baseline).
"""

import numpy as np
from scipy import stats
import pandas as pd


def nadeau_bengio_ttest(diffs, n_train, n_test):
    """
    Computes Nadeau-Bengio corrected resampled t-test:
      diffs: Array of fold-by-fold differences (Enhanced - Paper/Baseline).
      n_train: Number of training samples per fold.
      n_test: Number of validation samples per fold.
      
    Correction factor:
      var_corrected = (1/K + n_test/n_train) * sample_variance(diffs)
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    K = len(diffs)
    if K <= 1:
        return 0.0, 1.0
        
    mean_diff = np.mean(diffs)
    sample_var = np.var(diffs, ddof=1)
    
    correction = (1.0 / K) + (float(n_test) / float(n_train))
    denom = np.sqrt(correction * sample_var)
    
    if denom == 0 or np.isnan(denom):
        return 0.0, 1.0
        
    t_stat = mean_diff / denom
    df = K - 1
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df))
    
    return float(t_stat), float(p_value)


def evaluate_significance_matrix(baseline_fold_metrics, enhanced_fold_metrics, n_train, n_test, metric_keys=["F1-Score", "Accuracy", "Recall", "MCC"]):
    """
    Computes significance tests across a set of metrics.
    Returns a formatted DataFrame of differences, t-statistics, and p-values.
    """
    rows = []
    for k in metric_keys:
        b_vals = [m[k] for m in baseline_fold_metrics]
        e_vals = [m[k] for m in enhanced_fold_metrics]
        diffs = np.array(e_vals) - np.array(b_vals)
        
        t_stat, p_val = nadeau_bengio_ttest(diffs, n_train, n_test)
        
        rows.append({
            "Metric": k,
            "Baseline Mean": round(float(np.mean(b_vals)), 3),
            "Enhanced Mean": round(float(np.mean(e_vals)), 3),
            "Difference": round(float(np.mean(diffs)), 3),
            "t-Statistic": round(t_stat, 2),
            "p-Value": round(p_val, 4),
            "Significant (p < 0.05)": "Yes" if p_val < 0.05 else "No"
        })
        
    return pd.DataFrame(rows)
