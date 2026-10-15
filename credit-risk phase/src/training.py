"""
Cross-Validation and Training Engine.
Strictly implements Algorithm 1 from Section 3.6 of the research paper:
  - 10-Fold Stratified Cross-Validation (Outer Loop)
  - Resampling strictly applied within training folds to avoid data leakage
  - Tracking of Accuracy, Recall, Specificity, Precision, and F1-Score
  - Dynamic execution timer and Remaining Time (ETA) calculation via tqdm
"""

import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import SEED, OUTER_FOLDS, INNER_FOLDS
from src.metrics import compute_metrics, aggregate_fold_metrics
from src.resampling import get_resampler, apply_resampling
from src.models import create_model


def format_duration(seconds):
    """Format seconds into readable Xm Ys or Xs."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs:02d}s"


def build_preprocessor(cat_cols, num_cols):
    """
    Constructs a ColumnTransformer to scale continuous features
    and one-hot encode categorical features.
    """
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        
    return ColumnTransformer(transformers=transformers)


def run_experiment(
    X,
    y,
    cat_cols,
    num_cols,
    model_name,
    resampling_method,
    n_outer_folds=OUTER_FOLDS,
    seed=SEED,
    verbose=True
):
    """
    Executes Algorithm 1 across n_outer_folds with timing feedback:
      1. Stratified K-fold outer split.
      2. In-fold feature transformation (scaling + one-hot).
      3. In-fold resampling (None, SMOTE, ENN, SMOTE-ENN).
      4. Model training on resampled training fold.
      5. Evaluation on untouched validation fold.
      6. Returns aggregated metrics (mean, std) and last trained model for SHAP.
    """
    skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    
    t0 = time.time()
    last_trained_model = None
    last_X_val_trans = None
    last_y_val = None
    last_preprocessor = None
    
    if verbose:
        print(f"--> Training: Model='{model_name}' | Resampling='{resampling_method}' ({n_outer_folds} Folds)")
        
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        # Step 1: Split D into D_train and D_val
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Step 2: Fit preprocessor strictly on D_train, transform D_train and D_val
        preprocessor = build_preprocessor(cat_cols, num_cols)
        X_train_trans = preprocessor.fit_transform(X_train)
        X_val_trans = preprocessor.transform(X_val)
        
        # Step 3: Apply resampling technique R strictly on D_train
        resampler = get_resampler(resampling_method, seed=seed + fold)
        X_train_res, y_train_res = apply_resampling(resampler, X_train_trans, y_train)
        
        # Step 4: Instantiate and train model M
        model = create_model(model_name, seed=seed + fold)
        model.fit(X_train_res, y_train_res)
        
        # Step 5: Evaluate model M on untouched D_val
        y_pred = model.predict(X_val_trans)
        metrics = compute_metrics(y_val, y_pred)
        fold_metrics.append(metrics)
        
        # Save last fold artifacts for SHAP explainability
        last_trained_model = model
        last_X_val_trans = X_val_trans
        last_y_val = y_val
        last_preprocessor = preprocessor

    elapsed = time.time() - t0
    mean_metrics, std_metrics = aggregate_fold_metrics(fold_metrics)
    
    if verbose:
        print(f"    Completed in {format_duration(elapsed)} | F1: {mean_metrics['F1-Score']:.3f} ± {std_metrics['F1-Score']:.3f} | Acc: {mean_metrics['Accuracy']:.3f} | Rec: {mean_metrics['Recall']:.3f}")
        
    return {
        "model": model_name,
        "resampling": resampling_method,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "fold_metrics": fold_metrics,
        "last_model": last_trained_model,
        "last_X_val": last_X_val_trans,
        "last_y_val": last_y_val,
        "last_preprocessor": last_preprocessor,
        "elapsed_seconds": elapsed
    }


def run_benchmark_matrix(
    dataset_name,
    X,
    y,
    cat_cols,
    num_cols,
    models=None,
    resampling_methods=None,
    n_outer_folds=OUTER_FOLDS,
    seed=SEED
):
    """
    Runs a full matrix of Models x Resampling methods on a dataset,
    providing an interactive tqdm progress bar and dynamic Remaining Time (ETA) calculation.
    """
    if models is None:
        from src.config import PRIMARY_MODELS
        models = PRIMARY_MODELS
        
    if resampling_methods is None:
        from src.config import RESAMPLING_METHODS
        resampling_methods = RESAMPLING_METHODS
        
    total_experiments = len(resampling_methods) * len(models)
    results_rows = []
    
    print(f"\n=======================================================")
    print(f"BENCHMARK RUN: {dataset_name} Credit Dataset ({len(X):,} instances)")
    print(f"Total Model Configurations: {total_experiments} ({len(resampling_methods)} Resampling x {len(models)} Models)")
    print(f"Cross-Validation: {n_outer_folds}-Fold Stratified CV per configuration")
    print(f"=======================================================\n")
    
    matrix_start_time = time.time()
    current_idx = 0
    
    with tqdm(total=total_experiments, desc=f"Training {dataset_name}", unit="model") as pbar:
        for res_name in resampling_methods:
            for m_name in models:
                current_idx += 1
                
                # Update progress bar description
                pbar.set_description(f"[{current_idx}/{total_experiments}] {dataset_name} -> {m_name} ({res_name})")
                
                res = run_experiment(
                    X=X,
                    y=y,
                    cat_cols=cat_cols,
                    num_cols=num_cols,
                    model_name=m_name,
                    resampling_method=res_name,
                    n_outer_folds=n_outer_folds,
                    seed=seed,
                    verbose=True
                )
                
                m = res["mean_metrics"]
                results_rows.append({
                    "Dataset": dataset_name,
                    "Resampling": res_name,
                    "Model": m_name,
                    "Accuracy": round(m["Accuracy"], 3),
                    "Recall": round(m["Recall"], 3),
                    "Specificity": round(m["Specificity"], 3),
                    "Precision": round(m["Precision"], 3),
                    "F1-Score": round(m["F1-Score"], 3),
                    "Time": format_duration(res["elapsed_seconds"])
                })
                
                # Compute elapsed & remaining time
                total_elapsed = time.time() - matrix_start_time
                avg_time_per_model = total_elapsed / current_idx
                remaining_models = total_experiments - current_idx
                est_remaining_seconds = avg_time_per_model * remaining_models
                
                # Print explicit timer status
                if remaining_models > 0:
                    print(f"    --> Progress: {current_idx}/{total_experiments} | Elapsed: {format_duration(total_elapsed)} | Est. Remaining Time (ETA): {format_duration(est_remaining_seconds)}\n")
                else:
                    print(f"    --> Completed all {total_experiments} models in {format_duration(total_elapsed)}!\n")
                
                pbar.set_postfix({
                    "Last F1": f"{m['F1-Score']:.3f}",
                    "ETA": format_duration(est_remaining_seconds) if remaining_models > 0 else "0s"
                })
                pbar.update(1)
                
    df_results = pd.DataFrame(results_rows)
    return df_results
