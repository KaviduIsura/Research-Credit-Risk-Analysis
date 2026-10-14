"""
Stage C: Formal benchmark comparison and verdict analysis against authors' reported metrics.
"""

import os
import pandas as pd
from src.config import RESULTS_DIR, PRIMARY

def load_authors_benchmarks(results_dir=None):
    """Loads and filters the authors' best reported headline numbers from authors_reported.csv."""
    rdir = results_dir or RESULTS_DIR
    path = os.path.join(rdir, "authors_reported.csv")
    if not os.path.exists(path):
        alt = os.path.join("results", "authors_reported.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"authors_reported.csv not found at {path}")

    authors_raw = pd.read_csv(path, comment="#")
    authors_best = {}
    
    # German best: MLP + SMOTE-ENN; Taiwan best: Random Forest + SMOTE-ENN
    targets = [("German", "MLP"), ("Taiwan", "Random Forest")]
    for ds, m_name in targets:
        match = authors_raw[(authors_raw.dataset == ds) & 
                            (authors_raw.resampling == "SMOTE-ENN") & 
                            (authors_raw.model == m_name)]
        if not match.empty:
            row = match.iloc[0]
            authors_best[ds] = {
                "model": f"{row['model']} + SMOTE-ENN (Authors Table {row['table']})",
                "f1": float(row["f1_score"]),
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "specificity": float(row["specificity"]),
                "accuracy": float(row["accuracy"])
            }
    return authors_best

def compute_verdicts(R, datasets=None, results_dir=None):
    """
    Evaluates primary pipeline against authors' published benchmarks.
    Applies objective decision rule:
      delta = mine - authors
      band = max(0.02, fold_sd)
      delta > band   -> 'Mine better'
      delta < -band  -> 'Authors better'
      otherwise      -> 'Comparable'
    """
    dsets = datasets or ["German", "Taiwan"]
    authors_best = load_authors_benchmarks(results_dir)
    verdict_rows = []
    replication_rows = []

    for ds in dsets:
        pm, pv = PRIMARY[ds]
        r = R.get((ds, pm, pv, "tuned_thr", "f1"))
        a = authors_best.get(ds)
        
        if r is None or a is None:
            continue

        for m in ["f1", "recall", "precision", "specificity", "accuracy"]:
            if m not in a or a[m] is None:
                continue
            mine_val = r[f"{m}_mean"]
            sd_val = r[f"{m}_std"]
            author_val = a[m]
            delta = mine_val - author_val
            band = max(0.02, sd_val)

            if delta > band:
                v = "Mine better"
            elif delta < -band:
                v = "Authors better"
            else:
                v = "Comparable"

            verdict_rows.append(dict(
                dataset=ds, metric=m,
                mine=round(mine_val, 3),
                authors=round(author_val, 3),
                delta=round(delta, 3),
                band=round(band, 3),
                verdict=v
            ))

        # Check replication of authors' exact architecture under paper-style pipeline
        rep_model = "MLP" if ds == "German" else "RandomForest"
        rp = R.get((ds, rep_model, "paper_style", "default", None))
        if rp:
            replication_rows.append(dict(
                dataset=ds,
                pipeline=f"{rep_model} paper_style (default)",
                f1_binary=round(rp["f1_mean"], 3),
                f1_macro=round(rp.get("f1_macro_mean", 0), 3),
                f1_weighted=round(rp.get("f1_weighted_mean", 0), 3),
                mcc=round(rp["mcc_mean"], 3),
                authors_f1=a["f1"]
            ))

    verdict_df = pd.DataFrame(verdict_rows)
    repl_df = pd.DataFrame(replication_rows)
    return verdict_df, repl_df
