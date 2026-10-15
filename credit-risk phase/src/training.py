"""
Cross-Validation and Training Engine.
Strictly implements Algorithm 1 from Section 3.6 of the research paper:
  - 10-Fold Stratified Cross-Validation (Outer Loop)
  - Nested 5-Fold Stratified Cross-Validation for Hyperparameter Tuning (Inner Loop)
  - Resampling strictly applied within training folds to avoid data leakage
  - Tracking of Accuracy, Recall, Specificity, Precision, and F1-Score
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import SEED, OUTER_FOLDS, INNER_FOLDS
from src.metrics import compute_metrics, aggregate_fold_metrics
from src.resampling import get_resampler, apply_resampling
from src.models import create_model


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
    Executes Algorithm 1:
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
        print(f"--> Running: Model='{model_name}' | Resampling='{resampling_method}' | Folds={n_outer_folds}")
        
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
        print(f"    Done in {elapsed:.1f}s | F1: {mean_metrics['F1-Score']:.3f} ± {std_metrics['F1-Score']:.3f} | Acc: {mean_metrics['Accuracy']:.3f} | Rec: {mean_metrics['Recall']:.3f}")
        
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
    producing a formatted DataFrame matching Table 5 or Table 6 of the paper.
    """
    if models is None:
        from src.config import PRIMARY_MODELS
        models = PRIMARY_MODELS
        
    if resampling_methods is None:
        from src.config import RESAMPLING_METHODS
        resampling_methods = RESAMPLING_METHODS
        
    results_rows = []
    
    print(f"\n=======================================================")
    print(f"BENCHMARK RUN: {dataset_name} Credit Dataset")
    print(f"Models: {models}")
    print(f"Resampling: {resampling_methods}")
    print(f"=======================================================")
    
    for res_name in resampling_methods:
        for m_name in models:
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
                "Time(s)": round(res["elapsed_seconds"], 1)
            })
            
    df_results = pd.DataFrame(results_rows)
    return df_results
