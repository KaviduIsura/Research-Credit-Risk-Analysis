"""
Stage E: Model Calibration Assessment and Demographic Fairness Audit.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import SEED, PRIMARY, THR_OBJECTIVE, RESULTS_DIR, FIGURES_DIR
from src.pipelines import make_pipe
from src.metrics import best_threshold

def get_oof_probs(ds, calibrated, data_dict, prior_odds, seed=SEED):
    """Generates out-of-fold predicted probabilities using stratified cross-validation."""
    X, y, _, _ = data_dict[ds]
    pm, pv = PRIMARY[ds]
    est = make_pipe(ds, pm, pv, data_dict, prior_odds, seed=seed)
    if calibrated:
        est = CalibratedClassifierCV(est, method="sigmoid", cv=3)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    probs = cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return probs

def group_report(y, pred, group_series, attr_name):
    """Calculates subgroup selection rates, TPR, and FPR for fairness evaluation."""
    rows = []
    for val in sorted(pd.unique(group_series)):
        mask = (group_series == val)
        yt, yp = y[mask], pred[mask]
        n_pos = max(int((yt == 1).sum()), 1)
        n_neg = max(int((yt == 0).sum()), 1)
        
        rows.append(dict(
            attribute=attr_name,
            group=str(val),
            n=int(mask.sum()),
            base_rate=round(float(yt.mean()), 3),
            selection_rate=round(float(yp.mean()), 3),
            TPR=round(float(((yp == 1) & (yt == 1)).sum() / n_pos), 3),
            FPR=round(float(((yp == 1) & (yt == 0)).sum() / n_neg), 3)
        ))
    return pd.DataFrame(rows)

def get_group_definitions(ds, data_dict):
    """Extracts sensitive attributes for demographic auditing."""
    X = data_dict[ds][0]
    if ds == "German":
        age_group = np.where(X["age"] < 25, "Age < 25", "Age >= 25")
        foreign = np.where(X["foreign_worker"].astype(str).str.contains("A201|yes|1", case=False), "Foreign", "Domestic")
        return [("German: Age (<25)", age_group), ("German: Foreign Worker", foreign)]
    else:
        sex = np.where(X["SEX"].astype(str) == "1", "Male", "Female")
        age_group = np.where(X["AGE"] < 30, "Age < 30", "Age >= 30")
        return [("Taiwan: Sex", sex), ("Taiwan: Age (<30)", age_group)]

def audit_calibration_and_fairness(data_dict, prior_odds, datasets=None, results_dir=None, figures_dir=None, seed=SEED):
    """
    Executes complete Stage E evaluation:
    1. Brier score loss and calibration curves (raw vs Platt sigmoid calibrated).
    2. Demographic fairness audits across protected groups.
    """
    dsets = datasets or ["German", "Taiwan"]
    rdir = results_dir or RESULTS_DIR
    fdir = figures_dir or FIGURES_DIR

    cal_rows, audit_rows = [], []
    fig, axes = plt.subplots(1, len(dsets), figsize=(5.5 * len(dsets), 4.2), squeeze=False)

    for i, ds in enumerate(dsets):
        ax = axes[0, i]
        y = data_dict[ds][1].values
        
        p_raw = get_oof_probs(ds, False, data_dict, prior_odds, seed=seed)
        p_cal = get_oof_probs(ds, True, data_dict, prior_odds, seed=seed)
        
        b_raw = brier_score_loss(y, p_raw)
        b_cal = brier_score_loss(y, p_cal)
        
        cal_rows.append(dict(dataset=ds, model=PRIMARY[ds][0], variant=PRIMARY[ds][1],
                             brier_raw=round(b_raw, 4), brier_calibrated=round(b_cal, 4)))
        
        frac_raw, mean_raw = calibration_curve(y, p_raw, n_bins=8, strategy="uniform")
        frac_cal, mean_cal = calibration_curve(y, p_cal, n_bins=8, strategy="uniform")
        
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(mean_raw, frac_raw, "s-", color="#d62728", label=f"Raw (Brier={b_raw:.3f})")
        ax.plot(mean_cal, frac_cal, "o-", color="#2ca02c", label=f"Calibrated (Brier={b_cal:.3f})")
        ax.set_title(f"{ds} Credit — Reliability Curve", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Fairness audit at optimal decision threshold
        t_opt = best_threshold(y, p_raw, THR_OBJECTIVE[ds])
        preds = (p_raw >= t_opt).astype(int)
        
        for name, groups in get_group_definitions(ds, data_dict):
            rep = group_report(y, preds, groups, name)
            for _, r in rep.iterrows():
                audit_rows.append(r.to_dict())

    plt.tight_layout()
    plt.savefig(os.path.join(fdir, "calibration_curves.png"), dpi=200)
    plt.close()

    cal_df = pd.DataFrame(cal_rows)
    audit_df = pd.DataFrame(audit_rows)
    
    cal_df.to_csv(os.path.join(rdir, "calibration.csv"), index=False)
    audit_df.to_csv(os.path.join(rdir, "fairness_gaps.csv"), index=False)
    
    return cal_df, audit_df
