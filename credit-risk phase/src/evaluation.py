"""
Cross-validation engine, nested tuning, fold caching, and statistical significance testing.
"""

import os
import glob
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV, cross_validate
)

try:
    from sklearn.model_selection import TunedThresholdClassifierCV
    HAS_THR = True
except ImportError:
    HAS_THR = False

from src.config import (
    SEED, QUICK_TEST, RESULTS_DIR, OUTER_FOLDS, REPEATS, OUTER_TUNED,
    INNER_FOLDS, THR_FOLDS, N_ITER
)
from src.pipelines import make_pipe, PARAMS
from src.metrics import get_scoring, thr_scorer

SIGN = {"brier": -1, "cost": -1}

def get_cache_path(ds, model, variant, mode, obj, quick_test=None, results_dir=None):
    """Generates unique file path for caching fold results."""
    q = QUICK_TEST if quick_test is None else quick_test
    rdir = results_dir or RESULTS_DIR
    return os.path.join(rdir, f"{ds}__{model}__{variant}__{mode}__{obj}__q{int(q)}.json")

def get_cv_splitter(ds, tuned=False, quick_test=None, seed=SEED):
    """Returns repeated stratified CV splitter."""
    q = QUICK_TEST if quick_test is None else quick_test
    if tuned:
        splits = OUTER_TUNED[ds]
        return RepeatedStratifiedKFold(n_splits=splits, n_repeats=1, random_state=seed)
    splits = OUTER_FOLDS
    repeats = REPEATS[ds]
    return RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)

def build_estimator(ds, model, variant, mode, obj, data_dict, prior_odds, quick_test=None, seed=SEED):
    """Builds base pipeline or nested hyperparameter + threshold tuned classifier."""
    q = QUICK_TEST if quick_test is None else quick_test
    n_iter = N_ITER[ds]
    pipe = make_pipe(ds, model, variant, data_dict, prior_odds, seed=seed)
    
    if mode == "tuned_thr":
        inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            pipe, PARAMS[model], n_iter=n_iter, scoring="average_precision",
            cv=inner, random_state=seed, n_jobs=-1, refit=True
        )
        if HAS_THR:
            return TunedThresholdClassifierCV(
                search, scoring=thr_scorer(obj), cv=THR_FOLDS, refit=True, random_state=seed
            )
        return search
        
    return pipe

def run_experiment(ds, model, variant, mode="default", obj=None, data_dict=None, prior_odds=None,
                   quick_test=None, results_dir=None, seed=SEED):
    """
    Executes a single model configuration across outer CV folds.
    Caches outcomes to JSON; skips computation if cache file already exists.
    """
    q = QUICK_TEST if quick_test is None else quick_test
    rdir = results_dir or RESULTS_DIR
    path = get_cache_path(ds, model, variant, mode, obj, quick_test=q, results_dir=rdir)
    
    if os.path.exists(path):
        return json.load(open(path))

    if data_dict is None or prior_odds is None:
        raise ValueError("data_dict and prior_odds must be provided to run_experiment.")

    X, y, _, _ = data_dict[ds]
    est = build_estimator(ds, model, variant, mode, obj, data_dict, prior_odds, quick_test=q, seed=seed)
    tuned = (mode == "tuned_thr")
    cv = get_cv_splitter(ds, tuned=tuned, quick_test=q, seed=seed)
    scoring = get_scoring(ds)
    
    t0 = time.time()
    res = cross_validate(
        est, X, y, cv=cv, scoring=scoring,
        n_jobs=1 if tuned else -1, return_estimator=tuned
    )
    
    out = dict(dataset=ds, model=model, variant=variant, mode=mode, objective=obj, folds={})
    for m in scoring:
        v = SIGN.get(m, 1) * np.asarray(res[f"test_{m}"])
        out[f"{m}_mean"] = float(v.mean())
        out[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out["folds"][m] = [float(a) for a in v]
        
    if tuned:
        bp, thr = [], []
        for e in res["estimator"]:
            inner = e.estimator_ if hasattr(e, "best_threshold_") else e
            bp.append(getattr(inner, "best_params_", {}))
            if hasattr(e, "best_threshold_"):
                thr.append(float(e.best_threshold_))
        out["best_params"] = bp
        out["thresholds"] = thr
        
    out["seconds"] = round(time.time() - t0, 1)
    
    with open(path, "w") as f:
        json.dump(out, f, default=str)
        
    print(f"[{ds}] {model:12s} {variant:14s} {mode:9s} {str(obj):5s} "
          f"MCC={out['mcc_mean']:.3f}  F1={out['f1_mean']:.3f}  PR-AUC={out['pr_auc_mean']:.3f}  ({out['seconds']}s)")
    return out

def load_all_results(results_dir=None, quick_test=None):
    """Loads all cached experiment JSON records into a dictionary."""
    rdir = results_dir or RESULTS_DIR
    q = QUICK_TEST if quick_test is None else quick_test
    records = {}
    pattern = os.path.join(rdir, f"*__q{int(q)}.json")
    for p in glob.glob(pattern):
        r = json.load(open(p))
        records[(r["dataset"], r["model"], r["variant"], r["mode"], r["objective"])] = r
    return records

def format_results_table(R, ds, mode, obj=None, cols=("recall", "specificity", "precision", "f1", "mcc", "pr_auc", "roc_auc", "brier")):
    """Formats experiment records into a sorted, presentation-ready pandas DataFrame."""
    rows = []
    for (d_, m, v, md_, ob), r in R.items():
        if d_ == ds and md_ == mode and (obj is None or ob == obj):
            row = {"model": m, "variant": v, "objective": ob}
            for c in list(cols) + (["cost"] if ds == "German" else []):
                if f"{c}_mean" in r:
                    row[c] = f"{r[c + '_mean']:.3f} ± {r[c + '_std']:.3f}"
            row["_mcc"] = r.get("mcc_mean", 0)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_mcc", ascending=False).drop(columns="_mcc").reset_index(drop=True)

def corrected_resampled_ttest(diffs, test_train_ratio):
    """
    Nadeau & Bengio (2003) corrected resampled t-test for cross-validation comparisons.
    Accounts for overlapping training samples across CV folds.
    """
    k = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0 or k < 2:
        return np.nan, np.nan
    t = np.mean(diffs) / np.sqrt((1 / k + test_train_ratio) * var)
    p_val = float(2 * stats.t.sf(abs(t), k - 1))
    return float(t), p_val

def compare_significance(R, datasets, models_used, outer_folds=OUTER_FOLDS):
    """Generates comparison table and corrected p-values between fixed variants and paper_style."""
    rows = []
    ratio = 1.0 / max(outer_folds - 1, 1)
    for ds in datasets:
        for m in models_used[ds]:
            base = R.get((ds, m, "paper_style", "default", None))
            for v in ["fixed_none", "fixed_smotenc", "fixed_smoteenn"]:
                cur = R.get((ds, m, v, "default", None))
                if base is None or cur is None:
                    continue
                for metric in ["mcc", "pr_auc", "f1"]:
                    d = np.array(cur["folds"][metric]) - np.array(base["folds"][metric])
                    t, p = corrected_resampled_ttest(d, ratio)
                    rows.append(dict(
                        dataset=ds, model=m, variant=v, metric=metric,
                        paper_style=round(float(np.mean(base["folds"][metric])), 3),
                        corrected=round(float(np.mean(cur["folds"][metric])), 3),
                        diff=round(float(d.mean()), 3),
                        t_stat=round(t, 3) if not np.isnan(t) else np.nan,
                        p_value=round(p, 4) if not np.isnan(p) else np.nan
                    ))
    return pd.DataFrame(rows)
