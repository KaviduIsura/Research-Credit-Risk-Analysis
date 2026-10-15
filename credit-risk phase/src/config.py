"""
Global configuration settings for Credit Risk Analysis.
"""

import os
import sys

SEED = 42

# Execution mode toggle: True for fast verification, False for full production runs
QUICK_TEST = False

DATASETS = ["German"]

MODELS_USED = {
    "German": ["LogReg", "RandomForest", "XGBoost", "MLP"],
    "Taiwan": ["LogReg", "RandomForest", "XGBoost", "MLP"]
}

VARIANTS = ["paper_style", "fixed_none", "fixed_smotenc", "fixed_smoteenn"]

# Primary declared pipeline (prevents best-of-many selection bias)
PRIMARY = {
    "German": ("XGBoost", "fixed_none"),
    "Taiwan": ("XGBoost", "fixed_none")
}

CANDIDATES = {
    "German": [
        ("XGBoost", "fixed_none"),
        ("RandomForest", "fixed_smotenc"),
        ("MLP", "fixed_smotenc")
    ],
    "Taiwan": [
        ("XGBoost", "fixed_none"),
        ("RandomForest", "fixed_smotenc")
    ]
}

THR_OBJECTIVE = {"German": "cost", "Taiwan": "mcc"}

# German 5:1 cost matrix: misclassifying default costs 5x more than rejecting a good applicant
COST_FN = 5
COST_FP = 1
TOL = 0.02

# Cross-validation folds configuration
if QUICK_TEST:
    OUTER_FOLDS = 3
    REPEATS = {"German": 1, "Taiwan": 1}
    OUTER_TUNED = {"German": 3, "Taiwan": 3}
    INNER_FOLDS = 2
    THR_FOLDS = 2
    N_ITER = {"German": 2, "Taiwan": 2}
    QUICK_ROWS = 3000
else:
    OUTER_FOLDS = 10
    REPEATS = {"German": 3, "Taiwan": 1}
    OUTER_TUNED = {"German": 10, "Taiwan": 5}
    INNER_FOLDS = 3
    THR_FOLDS = 3
    N_ITER = {"German": 20, "Taiwan": 6}
    QUICK_ROWS = None

def get_data_dir():
    """Locate data/raw directory relative to current working directory or package."""
    candidates = [
        "data/raw",
        "../data/raw",
        "../../data/raw",
        os.path.join(os.path.dirname(__file__), "..", "data", "raw"),
        "/content/data"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "german.data")):
            return os.path.abspath(c)
    return os.path.abspath("data/raw")

def get_results_dir():
    """Locate results directory relative to data directory or current working directory."""
    candidates = [
        "results",
        "../results",
        "../../results",
        os.path.join(os.path.dirname(__file__), "..", "results")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    res = os.path.abspath("results")
    os.makedirs(res, exist_ok=True)
    return res

DATA_DIR = get_data_dir()
RESULTS_DIR = get_results_dir()
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
