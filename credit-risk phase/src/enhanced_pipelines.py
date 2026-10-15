"""
Enhanced Modeling Pipelines & In-Fold Threshold Optimization.
Implements Phase 1 methodological enhancements:
  1. Categorical-aware SMOTE-NC (Nominal VDM Distance + Continuous Euclidean)
  2. Cost-sensitive class weighting (scale_pos_weight / balanced_subsample)
  3. In-fold decision threshold tuning (German 5:1 Cost Matrix, Taiwan MCC)
  4. Extended metric suite: F1, Accuracy, Precision, Recall, Specificity, MCC, PR-AUC, Financial Cost
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN

from src.config import SEED
from src.enhanced_preprocessors import (
    build_enhanced_preprocessor,
    build_ordinal_scaled_preprocessor
)
from src.metrics import compute_metrics
from src.training import format_duration


# ---------------------------------------------------------------------------
# 1. Cost-Sensitive Estimator Factory
# ---------------------------------------------------------------------------

def make_enhanced_model(model_name, dataset_name, variant="enhanced_none", seed=SEED):
    """
    Instantiates models with cost-sensitive class weights if variant == 'enhanced_none'.
      - German prior odds: ~700 / 300 = 2.33
      - Taiwan prior odds: ~23364 / 6636 = 3.52
    """
    prior_odds = {"German": 2.333, "Taiwan": 3.521}.get(dataset_name, 1.0)
    weighted = (variant == "enhanced_none")
    
    if model_name in ["Random Forest", "RandomForest", "rf"]:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed,
            n_jobs=-1
        )
    elif model_name in ["XGBoost", "xgboost", "xgb"]:
        return XGBClassifier(
            learning_rate=0.05,
            max_depth=5,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=prior_odds if weighted else 1.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )
    elif model_name in ["MLP", "Multilayer Perceptron", "mlp"]:
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=64,
            max_iter=100,
            early_stopping=True,
            random_state=seed
        )
    elif model_name in ["Logistic Regression", "LogReg", "lr"]:
        return LogisticRegression(
            solver="liblinear",
            class_weight="balanced" if weighted else None,
            random_state=seed,
            max_iter=1000
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ---------------------------------------------------------------------------
# 2. Enhanced Pipeline Factory (SMOTE-NC & In-Fold Preprocessing)
# ---------------------------------------------------------------------------

def make_enhanced_pipeline(model_name, dataset_name, variant, cat_cols, num_cols, seed=SEED):
    """
    Builds strict in-fold imblearn Pipeline:
      - 'enhanced_none': In-fold SkewLog + Scaling + OneHot -> Cost-sensitive model (No synthetic sampling).
      - 'enhanced_smotenc': In-fold SkewLog + Scaling + SMOTE-NC on nominal categoricals -> OneHot -> Model.
      - 'enhanced_smoteenn': In-fold SkewLog + Scaling + OneHot -> SMOTE-ENN -> Model.
    """
    clf = make_enhanced_model(model_name, dataset_name, variant=variant, seed=seed)
    n_num = len(num_cols)
    n_cat = len(cat_cols)
    
    if variant == "enhanced_none":
        return ImbPipeline([
            ("pre", build_enhanced_preprocessor(cat_cols, num_cols, apply_skewlog=True)),
            ("clf", clf)
        ])
        
    elif variant == "enhanced_smoteenn":
        return ImbPipeline([
            ("pre", build_enhanced_preprocessor(cat_cols, num_cols, apply_skewlog=True)),
            ("res", SMOTEENN(random_state=seed)),
            ("clf", clf)
        ])
        
    elif variant == "enhanced_smotenc":
        # Categorical feature indices after ordinal preprocessor
        cat_indices = list(range(n_num, n_num + n_cat))
        
        pre1 = build_ordinal_scaled_preprocessor(cat_cols, num_cols, scale=True)
        resampler = SMOTENC(categorical_features=cat_indices, random_state=seed)
        
        # One-hot encode the categorical columns after SMOTE-NC
        pre2 = ColumnTransformer(
            transformers=[("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_indices)],
            remainder="passthrough"
        )
        
        return ImbPipeline([
            ("pre1", pre1),
            ("res", resampler),
            ("pre2", pre2),
            ("clf", clf)
        ])
    else:
        raise ValueError(f"Unknown enhanced variant: {variant}")


# ---------------------------------------------------------------------------
# 3. In-Fold Decision Threshold Tuning
# ---------------------------------------------------------------------------

def compute_financial_cost(y_true, y_pred, cost_fn=5, cost_fp=1):
    """German 5:1 Asymmetric Cost Matrix: Cost = 5 * FN + 1 * FP."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(cost_fn * fn + cost_fp * fp)


def find_optimal_threshold(y_true, y_probs, objective="cost", cost_fn=5, cost_fp=1):
    """
    Searches decision threshold p* in [0.10, 0.70] to optimize:
      - 'cost': Minimize financial cost (5 * FN + 1 * FP)
      - 'mcc': Maximize Matthews Correlation Coefficient
      - 'f1': Maximize F1-score
    """
    thresholds = np.linspace(0.10, 0.70, 61)
    best_score = float("inf") if objective == "cost" else -float("inf")
    best_thr = 0.50
    
    for thr in thresholds:
        y_pred = (y_probs >= thr).astype(int)
        
        if objective == "cost":
            score = compute_financial_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)
            if score < best_score:
                best_score = score
                best_thr = thr
        elif objective == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_thr = thr
        elif objective == "f1":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if f1 > best_score:
                best_score = f1
                best_thr = thr
                
    return best_thr


def compute_extended_metrics(y_true, y_pred, y_probs=None, cost_fn=5, cost_fp=1):
    """Computes full metric suite including MCC, PR-AUC, ROC-AUC, and Financial Cost."""
    base_m = compute_metrics(y_true, y_pred)
    
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cost = compute_financial_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)
    
    pr_auc_val = 0.0
    roc_auc_val = 0.0
    if y_probs is not None:
        try:
            precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_probs)
            pr_auc_val = float(auc(recall_arr, precision_arr))
            roc_auc_val = float(roc_auc_score(y_true, y_probs))
        except Exception:
            pass
            
    base_m.update({
        "MCC": round(mcc, 3),
        "PR-AUC": round(pr_auc_val, 3),
        "ROC-AUC": round(roc_auc_val, 3),
        "FinancialCost": cost
    })
    return base_m


# ---------------------------------------------------------------------------
# 4. Enhanced Cross-Validation & Benchmark Runner
# ---------------------------------------------------------------------------

def run_enhanced_cv(
    X,
    y,
    cat_cols,
    num_cols,
    model_name="Random Forest",
    dataset_name="German",
    variant="enhanced_none",
    tune_threshold=True,
    objective="cost",
    cost_fn=5,
    cost_fp=1,
    n_splits=10,
    seed=SEED,
    verbose=True
):
    """
    Executes in-fold cross-validation of enhanced pipeline:
      1. Stratified 10-fold split.
      2. In-fold threshold tuning on training split (if tune_threshold=True).
      3. In-fold pipeline fit (Scaling + SkewLog + Resampling + Model).
      4. Evaluation on untouched validation split.
      5. Full tracking of OOF probabilities, predictions, and fold metrics.
    """
    t0 = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_metrics = []
    tuned_thresholds = []
    oof_probs = np.zeros(len(y), dtype=np.float64)
    oof_preds = np.zeros(len(y), dtype=np.int32)
    
    if verbose:
        print(f"--> Enhanced CV: Dataset='{dataset_name}' | Model='{model_name}' | Variant='{variant}' | TuneThr={tune_threshold} ({objective})")
        
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # In-fold threshold tuning (strictly on training fold)
        if tune_threshold:
            X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
                X_train, y_train, test_size=0.20, stratify=y_train, random_state=seed + fold
            )
            inner_pipe = make_enhanced_pipeline(
                model_name, dataset_name, variant, cat_cols, num_cols, seed=seed + fold
            )
            inner_pipe.fit(X_inner_tr, y_inner_tr)
            inner_probs = inner_pipe.predict_proba(X_inner_val)[:, 1]
            thr = find_optimal_threshold(
                y_inner_val, inner_probs, objective=objective, cost_fn=cost_fn, cost_fp=cost_fp
            )
        else:
            thr = 0.50
            
        tuned_thresholds.append(round(float(thr), 3))
        
        # Main fold fit
        pipe = make_enhanced_pipeline(
            model_name, dataset_name, variant, cat_cols, num_cols, seed=seed + fold
        )
        pipe.fit(X_train, y_train)
        
        probs = pipe.predict_proba(X_val)[:, 1]
        preds = (probs >= thr).astype(int)
        
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds
        
        m = compute_extended_metrics(y_val, preds, probs, cost_fn=cost_fn, cost_fp=cost_fp)
        m["Fold"] = fold
        m["Threshold"] = round(float(thr), 3)
        fold_metrics.append(m)
        
    elapsed = time.time() - t0
    
    # Compute mean and std
    keys = ["Accuracy", "Recall", "Specificity", "Precision", "F1-Score", "MCC", "PR-AUC", "ROC-AUC", "FinancialCost"]
    mean_metrics = {k: round(float(np.mean([fm[k] for fm in fold_metrics])), 3) for k in keys}
    std_metrics = {k: round(float(np.std([fm[k] for fm in fold_metrics], ddof=1)), 3) for k in keys}
    
    if verbose:
        print(f"    Completed in {format_duration(elapsed)} | F1: {mean_metrics['F1-Score']:.3f} ± {std_metrics['F1-Score']:.3f} | Cost: {mean_metrics['FinancialCost']:.0f} | MCC: {mean_metrics['MCC']:.3f} | AvgThr: {np.mean(tuned_thresholds):.2f}\n")
        
    return {
        "dataset": dataset_name,
        "model": model_name,
        "variant": variant,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "fold_metrics": fold_metrics,
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "tuned_thresholds": tuned_thresholds,
        "elapsed_seconds": elapsed
    }
