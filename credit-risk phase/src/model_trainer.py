"""
Model training and serialization service for production deployment.
Trains and exports primary production models to models/ directory.
"""

import os
import joblib
import pandas as pd
from src.config import PRIMARY, SEED
from src.data_loader import get_data_bundle
from src.pipelines import make_pipe

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_export_production_models():
    """Trains primary leak-free models on datasets and exports them for low-latency serving."""
    print("Loading datasets for production model training...")
    DATA, _, _, PRIOR_ODDS = get_data_bundle(quick_test=False)
    
    # 1. German Primary Model (XGBoost + fixed_none, Cost-Sensitive Threshold)
    print("Training German Credit production model (XGBoost)...")
    X_g, y_g, _, _ = DATA["German"]
    pipe_german = make_pipe("German", "XGBoost", "fixed_none", DATA, PRIOR_ODDS, seed=SEED)
    pipe_german.fit(X_g, y_g)
    
    german_artifact = {
        "pipeline": pipe_german,
        "dataset": "German",
        "model_name": "XGBoost + fixed_none",
        "threshold": 0.35,  # In-fold optimal cost-sensitive threshold (5:1 matrix)
        "objective": "cost_sensitive_5_to_1"
    }
    g_path = os.path.join(MODELS_DIR, "german_primary_xgb.joblib")
    joblib.dump(german_artifact, g_path)
    print(f"Saved German production model to: {g_path}")

    # 2. Taiwan Primary Model (XGBoost + fixed_none, MCC-Tuned Threshold)
    print("Training Taiwan Credit production model (XGBoost)...")
    X_t, y_t, _, _ = DATA["Taiwan"]
    pipe_taiwan = make_pipe("Taiwan", "XGBoost", "fixed_none", DATA, PRIOR_ODDS, seed=SEED)
    pipe_taiwan.fit(X_t, y_t)
    
    taiwan_artifact = {
        "pipeline": pipe_taiwan,
        "dataset": "Taiwan",
        "model_name": "XGBoost + fixed_none",
        "threshold": 0.46,  # In-fold optimal MCC threshold
        "objective": "mcc_optimal"
    }
    t_path = os.path.join(MODELS_DIR, "taiwan_primary_xgb.joblib")
    joblib.dump(taiwan_artifact, t_path)
    print(f"Saved Taiwan production model to: {t_path}")

    return g_path, t_path

if __name__ == "__main__":
    train_and_export_production_models()
