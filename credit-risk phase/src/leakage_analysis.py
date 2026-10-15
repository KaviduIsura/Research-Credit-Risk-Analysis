"""
Controlled Data Leakage Audit Module.
Demonstrates the mathematical mechanism behind the authors' published German MLP F1 = 0.937:
  - Leaky Pipeline: Applying SMOTE-ENN to the entire dataset BEFORE cross-validation split (pre-split oversampling).
  - Leak-Free Pipeline (Algorithm 1): Applying SMOTE-ENN strictly WITHIN training folds.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import SEED
from src.metrics import compute_metrics, aggregate_fold_metrics
from src.resampling import get_resampler, apply_resampling
from src.models import create_model


def run_leakage_comparison(X, y, cat_cols, num_cols, model_name="MLP", n_splits=5, seed=SEED):
    """
    Directly compares:
      1. Leak-Free Pipeline (Resampling applied inside CV folds)
      2. Leaky Pipeline (Resampling applied before CV split - paper artifact)
    """
    print(f"\n=======================================================")
    print(f"CONTROLLED DATA LEAKAGE AUDIT ({model_name} on German Dataset)")
    print(f"=======================================================")
    
    # 1. LEAK-FREE PIPELINE
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    leak_free_metrics = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ])
        X_train_trans = pre.fit_transform(X_train)
        X_val_trans = pre.transform(X_val)
        
        resampler = get_resampler("SMOTE-ENN", seed=seed)
        X_train_res, y_train_res = apply_resampling(resampler, X_train_trans, y_train)
        
        clf = create_model(model_name, seed=seed)
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_val_trans)
        leak_free_metrics.append(compute_metrics(y_val, y_pred))
        
    mean_lf, std_lf = aggregate_fold_metrics(leak_free_metrics)
    print(f"Leak-Free Pipeline (Valid)   -> F1: {mean_lf['F1-Score']:.3f} | Accuracy: {mean_lf['Accuracy']:.3f} | Recall: {mean_lf['Recall']:.3f}")

    # 2. LEAKY PIPELINE (Pre-split oversampling)
    pre_full = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    X_full_trans = pre_full.fit_transform(X)
    resampler_full = get_resampler("SMOTE-ENN", seed=seed)
    X_leaky_res, y_leaky_res = apply_resampling(resampler_full, X_full_trans, y)
    
    skf_leaky = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    leaky_metrics = []
    
    for train_idx, val_idx in skf_leaky.split(X_leaky_res, y_leaky_res):
        X_tr, X_va = X_leaky_res[train_idx], X_leaky_res[val_idx]
        y_tr, y_va = y_leaky_res.iloc[train_idx], y_leaky_res.iloc[val_idx]
        
        clf = create_model(model_name, seed=seed)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)
        leaky_metrics.append(compute_metrics(y_va, y_pred))
        
    mean_lk, std_lk = aggregate_fold_metrics(leaky_metrics)
    print(f"Leaky Pipeline (Pre-split)   -> F1: {mean_lk['F1-Score']:.3f} | Accuracy: {mean_lk['Accuracy']:.3f} | Recall: {mean_lk['Recall']:.3f}")

    comparison_df = pd.DataFrame([
        {"Pipeline": "Leak-Free (In-Fold CV)", "Accuracy": round(mean_lf["Accuracy"], 3), "Recall": round(mean_lf["Recall"], 3), "Precision": round(mean_lf["Precision"], 3), "Specificity": round(mean_lf["Specificity"], 3), "F1-Score": round(mean_lf["F1-Score"], 3)},
        {"Pipeline": "Leaky (Pre-Split SMOTE-ENN)", "Accuracy": round(mean_lk["Accuracy"], 3), "Recall": round(mean_lk["Recall"], 3), "Precision": round(mean_lk["Precision"], 3), "Specificity": round(mean_lk["Specificity"], 3), "F1-Score": round(mean_lk["F1-Score"], 3)},
        {"Pipeline": "Authors Reported (Table 5)", "Accuracy": 0.954, "Recall": 0.929, "Precision": 0.945, "Specificity": 0.968, "F1-Score": 0.937}
    ])
    
    return comparison_df
