"""
Stage D: Controlled Data Leakage Experiment.
Empirically quantifies the metric inflation caused by global pre-split resampling vs leak-free in-fold resampling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, matthews_corrcoef
from imblearn.combine import SMOTEENN

from src.config import SEED, RESULTS_DIR, FIGURES_DIR
from src.preprocessors import pre_ordinal
from src.pipelines import make_model, make_pipe

def run_leakage_experiment(ds, data_dict, prior_odds, model="RandomForest", seed=SEED):
    """
    Evaluates a model under:
    1. Leaky setup: SMOTE-ENN fitted globally over entire dataset prior to cross-validation splits.
    2. Proper leak-free setup: SMOTE-ENN fitted strictly within training folds only.
    """
    X, y, cat, num = data_dict[ds]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    sc = {
        "f1": "f1",
        "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0),
        "mcc": make_scorer(matthews_corrcoef)
    }

    # 1. Leaky global resampling
    Z = pre_ordinal(cat, num, scale=False).fit_transform(X)
    Zr, yr = SMOTEENN(random_state=seed).fit_resample(Z, y)
    clf_leaky = make_model(model, "paper_style", ds, prior_odds, seed=seed)
    leaky_cv = cross_validate(clf_leaky, Zr, yr, cv=cv, scoring=sc, n_jobs=-1)

    # 2. Methodologically proper in-fold resampling
    pipe_proper = make_pipe(ds, model, "paper_style", data_dict, prior_odds, seed=seed)
    proper_cv = cross_validate(pipe_proper, X, y, cv=cv, scoring=sc, n_jobs=-1)

    row = {"dataset": ds, "model": model}
    for k in sc:
        row[f"leaky_{k}"] = round(float(leaky_cv[f"test_{k}"].mean()), 3)
        row[f"in-fold_{k}"] = round(float(proper_cv[f"test_{k}"].mean()), 3)
    row["F1 inflation"] = round(row["leaky_f1"] - row["in-fold_f1"], 3)
    return row

def run_all_leakage(data_dict, prior_odds, datasets=None, results_dir=None, figures_dir=None, seed=SEED):
    """Runs leakage experiment for all datasets and generates summary table and visualization."""
    dsets = datasets or ["German", "Taiwan"]
    rdir = results_dir or RESULTS_DIR
    fdir = figures_dir or FIGURES_DIR
    
    rows = []
    for ds in dsets:
        res = run_leakage_experiment(ds, data_dict, prior_odds, model="RandomForest", seed=seed)
        rows.append(res)
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rdir, "leakage_experiment.csv"), index=False)
    
    # Generate visualization
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(dsets))
    w = 0.35
    
    ax.bar(x - w/2, df["in-fold_f1"], width=w, label="Proper In-Fold Resampling", color="#1f77b4")
    ax.bar(x + w/2, df["leaky_f1"], width=w, label="Leaky Pre-Split Resampling", color="#d62728")
    
    for i in range(len(dsets)):
        inf = df["in-fold_f1"].iloc[i]
        lky = df["leaky_f1"].iloc[i]
        diff = df["F1 inflation"].iloc[i]
        ax.text(i - w/2, inf + 0.02, f"{inf:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w/2, lky + 0.02, f"{lky:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.annotate(f"+{diff:.3f} inflation", xy=(i + w/2, lky), xytext=(i, lky + 0.08),
                    ha="center", fontsize=9, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
        
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, fontsize=11, fontweight="bold")
    ax.set_ylabel("Binary F1-Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Empirical Impact of Pre-Split Data Leakage vs In-Fold Resampling (Random Forest)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    fig_path = os.path.join(fdir, "data_leakage_impact.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    return df
