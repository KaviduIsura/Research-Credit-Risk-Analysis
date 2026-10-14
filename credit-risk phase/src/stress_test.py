"""
Stage F: Noise Perturbation Stress-Testing.
Tests hypothesis for coursework Section 3d: evaluates degradation slope under feature noise.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score

from src.config import SEED, RESULTS_DIR, FIGURES_DIR
from src.pipelines import make_pipe

def run_noise_test_model(ds, model_name, data_dict, prior_odds, variant="fixed_none",
                         levels=(0, 0.1, 0.25, 0.5, 1.0), seed=SEED):
    """
    Evaluates metric degradation when Gaussian noise scaled to training SD
    is added to validation continuous features only.
    """
    X, y, cat, num = data_dict[ds]
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []

    for tr, te in cv.split(X, y):
        pipe = make_pipe(ds, model_name, variant, data_dict, prior_odds, seed=seed)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        sd = X.iloc[tr][num].astype(float).std().values
        sd = np.where(sd == 0, 1.0, sd)

        for lvl in levels:
            Xn = X.iloc[te].copy()
            noise = rng.normal(0, 1, (len(Xn), len(num))) * sd * lvl
            Xn[num] = Xn[num].astype(float).values + noise
            preds = pipe.predict(Xn)
            
            rows.append(dict(
                dataset=ds, model=model_name, variant=variant, noise_level=lvl,
                mcc=matthews_corrcoef(y.iloc[te], preds),
                f1=f1_score(y.iloc[te], preds, zero_division=0)
            ))

    res = (pd.DataFrame(rows)
           .groupby(["dataset", "model", "variant", "noise_level"])
           .agg({"mcc": ["mean", "std"], "f1": ["mean", "std"]}))
    
    flat = res.reset_index()
    flat.columns = ["dataset", "model", "variant", "noise_level", "mcc_mean", "mcc_std", "f1_mean", "f1_std"]
    for c in ["mcc_mean", "mcc_std", "f1_mean", "f1_std"]:
        flat[c] = flat[c].round(3)
    return flat

def run_all_stress_tests(data_dict, prior_odds, datasets=None, models=None, results_dir=None, figures_dir=None, seed=SEED):
    """Executes noise perturbation tests across models and generates comparison plots."""
    dsets = datasets or ["German", "Taiwan"]
    test_models = models or ["XGBoost", "RandomForest", "LogReg"]
    rdir = results_dir or RESULTS_DIR
    fdir = figures_dir or FIGURES_DIR

    all_dfs = []
    for ds in dsets:
        for m in test_models:
            df_m = run_noise_test_model(ds, m, data_dict, prior_odds, variant="fixed_none", seed=seed)
            all_dfs.append(df_m)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(rdir, "noise_stress_test.csv"), index=False)

    # Plot degradation curves
    fig, axes = plt.subplots(1, len(dsets), figsize=(6 * len(dsets), 4.5), squeeze=False)
    for i, ds in enumerate(dsets):
        ax = axes[0, i]
        sub = combined[combined.dataset == ds]
        for m in test_models:
            m_data = sub[sub.model == m]
            ax.plot(m_data["noise_level"], m_data["mcc_mean"], marker="o", label=m, linewidth=2)
            ax.fill_between(
                m_data["noise_level"],
                m_data["mcc_mean"] - m_data["mcc_std"],
                m_data["mcc_mean"] + m_data["mcc_std"],
                alpha=0.15
            )
        ax.set_title(f"{ds} Credit — Noise Stress Degradation", fontsize=11, fontweight="bold")
        ax.set_xlabel("Gaussian Noise Level (Fraction of Training SD)")
        ax.set_ylabel("Matthews Correlation Coefficient (MCC)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(fdir, "noise_stress_degradation.png"), dpi=200)
    plt.close()

    return combined
