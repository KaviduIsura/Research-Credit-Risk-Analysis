"""
FastAPI Production Microservice for Credit Risk Scoring and Metric Analytics.
"""

import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import GermanApplicant, TaiwanClient, PredictionResponse
from src.config import RESULTS_DIR

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load production models into memory during service startup."""
    g_path = os.path.join(MODELS_DIR, "german_primary_xgb.joblib")
    t_path = os.path.join(MODELS_DIR, "taiwan_primary_xgb.joblib")
    
    if os.path.exists(g_path):
        MODELS["German"] = joblib.load(g_path)
    if os.path.exists(t_path):
        MODELS["Taiwan"] = joblib.load(t_path)
        
    print(f"Production models loaded: {list(MODELS.keys())}")
    yield
    MODELS.clear()

app = FastAPI(
    title="Credit Risk Analytics & Scoring API",
    description="Enterprise-grade production service for credit default risk scoring, cost-sensitive underwriting, and academic audit validation.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def read_root():
    return {
        "service": "Credit Risk Scoring API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "models_loaded": list(MODELS.keys())
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "models_ready": len(MODELS) == 2
    }

@app.post("/predict/german", response_model=PredictionResponse, tags=["Scoring"])
def predict_german_credit(applicant: GermanApplicant):
    """
    Score a German Credit loan applicant.
    Applies the German 5:1 asymmetric financial cost matrix cutoff (threshold = 0.35).
    """
    if "German" not in MODELS:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="German credit model not loaded.")
    
    artifact = MODELS["German"]
    pipe = artifact["pipeline"]
    threshold = artifact["threshold"]
    
    input_df = pd.DataFrame([applicant.model_dump()])
    prob = float(pipe.predict_proba(input_df)[0, 1])
    decision = 1 if prob >= threshold else 0
    
    if prob >= 0.55:
        risk_level = "High Default Risk"
    elif prob >= threshold:
        risk_level = "Moderate Risk (Cost-Sensitive Threshold Rejection)"
    else:
        risk_level = "Low Default Risk"

    return PredictionResponse(
        dataset="German Credit",
        model_name=artifact["model_name"],
        default_probability=round(prob, 4),
        decision=decision,
        decision_label="Reject Loan (Default Warning)" if decision == 1 else "Approve Loan (Creditworthy)",
        risk_level=risk_level,
        threshold_applied=threshold,
        optimization_objective="Asymmetric Financial Cost (5x FN Loss)"
    )

@app.post("/predict/taiwan", response_model=PredictionResponse, tags=["Scoring"])
def predict_taiwan_default(client: TaiwanClient):
    """
    Score a Taiwan Credit Card client default risk.
    Applies the Matthews Correlation Coefficient (MCC) optimal cutoff (threshold = 0.46).
    """
    if "Taiwan" not in MODELS:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Taiwan credit model not loaded.")
    
    artifact = MODELS["Taiwan"]
    pipe = artifact["pipeline"]
    threshold = artifact["threshold"]
    
    input_df = pd.DataFrame([client.model_dump()])
    prob = float(pipe.predict_proba(input_df)[0, 1])
    decision = 1 if prob >= threshold else 0
    
    if prob >= 0.60:
        risk_level = "High Default Risk"
    elif prob >= threshold:
        risk_level = "Moderate Default Risk"
    else:
        risk_level = "Low Default Risk"

    return PredictionResponse(
        dataset="Taiwan Credit Card",
        model_name=artifact["model_name"],
        default_probability=round(prob, 4),
        decision=decision,
        decision_label="High Risk (Default Warning)" if decision == 1 else "Normal (Non-Default)",
        risk_level=risk_level,
        threshold_applied=threshold,
        optimization_objective="Matthews Correlation Coefficient (MCC Optimal)"
    )

@app.get("/metrics/summary", tags=["Analytics & Audit"])
def get_metrics_summary():
    """Retrieve full model benchmark evaluation matrix."""
    path = os.path.join(RESULTS_DIR, "all_results.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="all_results.csv not found.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/metrics/verdicts", tags=["Analytics & Audit"])
def get_author_verdicts():
    """Retrieve head-to-head comparison verdict against published paper."""
    path = os.path.join(RESULTS_DIR, "verdicts.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="verdicts.csv not found.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/metrics/leakage", tags=["Analytics & Audit"])
def get_leakage_results():
    """Retrieve empirical evidence of data leakage inflation."""
    path = os.path.join(RESULTS_DIR, "leakage_experiment.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="leakage_experiment.csv not found.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/metrics/fairness", tags=["Analytics & Audit"])
def get_fairness_audit():
    """Retrieve demographic fairness audit metrics across sensitive attributes."""
    path = os.path.join(RESULTS_DIR, "fairness_gaps.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="fairness_gaps.csv not found.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
