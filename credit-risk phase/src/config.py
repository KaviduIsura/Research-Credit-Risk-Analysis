"""
Global Configuration for Credit Risk Analysis & Research Paper Replication.
Paper: Performance Evaluation of Machine Learning and Deep Learning Models for Credit Risk Prediction
Authors: Irvine Mapfumo & Thokozani Shongwe (2026), JRFM 19(3), 210.

Protocol: Full Dataset Training (100% of German & Taiwan datasets, Zero Subsampling)
"""

from pathlib import Path

# Random seed for reproducibility (Section 3.6 of the paper)
SEED = 42

# Datasets analysed in the paper
DATASETS = ["German", "Taiwan"]

# Resampling methods evaluated in the paper (Section 3.4)
RESAMPLING_METHODS = ["No Resampling", "SMOTE", "ENN", "SMOTE-ENN"]

# All 10 models evaluated in the paper (Section 3.2, 3.3, and Table 4)
MODELS_ALL = [
    "MLP",
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
    "SVM",
    "Decision Tree",
    "AdaBoost",
    "CNN",
    "LSTM",
    "GRU"
]

# Primary models evaluated across all benchmark matrices
PRIMARY_MODELS = ["MLP", "Random Forest", "XGBoost", "Logistic Regression"]

# Full Production Cross-Validation Parameters (Algorithm 1 & Section 3.6)
# Strictly 10 outer folds across 100% of the dataset
OUTER_FOLDS = 10
INNER_FOLDS = 5
N_ITER_SEARCH = 10

# Model Epochs (Table 4)
EPOCHS_MLP = 80             # Table 4: 80 epochs with early stopping
EPOCHS_DL = 50

# Path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure output directories exist
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Published authors' benchmark results from Table 5 & Table 6 for comparison
PAPER_BENCHMARKS = {
    "German": {
        "MLP_SMOTE-ENN": {"Accuracy": 0.954, "Recall": 0.929, "Specificity": 0.968, "Precision": 0.945, "F1-Score": 0.937},
        "CNN_SMOTE-ENN": {"Accuracy": 0.934, "Recall": 0.893, "Specificity": 0.958, "Precision": 0.926, "F1-Score": 0.909},
        "LSTM_SMOTE-ENN": {"Accuracy": 0.912, "Recall": 0.887, "Specificity": 0.941, "Precision": 0.920, "F1-Score": 0.903},
        "GRU_SMOTE-ENN": {"Accuracy": 0.895, "Recall": 0.865, "Specificity": 0.924, "Precision": 0.893, "F1-Score": 0.879},
        "Random Forest_SMOTE-ENN": {"Accuracy": 0.680, "Recall": 0.738, "Specificity": 0.542, "Precision": 0.794, "F1-Score": 0.765},
        "XGBoost_SMOTE-ENN": {"Accuracy": 0.670, "Recall": 0.702, "Specificity": 0.593, "Precision": 0.805, "F1-Score": 0.750},
        "Logistic Regression_SMOTE-ENN": {"Accuracy": 0.570, "Recall": 0.582, "Specificity": 0.542, "Precision": 0.752, "F1-Score": 0.656},
        "SVM_SMOTE-ENN": {"Accuracy": 0.615, "Recall": 0.624, "Specificity": 0.593, "Precision": 0.786, "F1-Score": 0.696},
        "Decision Tree_SMOTE-ENN": {"Accuracy": 0.630, "Recall": 0.667, "Specificity": 0.542, "Precision": 0.777, "F1-Score": 0.718},
        "AdaBoost_SMOTE-ENN": {"Accuracy": 0.660, "Recall": 0.688, "Specificity": 0.593, "Precision": 0.802, "F1-Score": 0.741}
    },
    "Taiwan": {
        "Random Forest_SMOTE-ENN": {"Accuracy": 0.821, "Recall": 0.745, "Specificity": 0.842, "Precision": 0.512, "F1-Score": 0.610},
        "XGBoost_SMOTE-ENN": {"Accuracy": 0.708, "Recall": 0.632, "Specificity": 0.729, "Precision": 0.395, "F1-Score": 0.486},
        "MLP_SMOTE-ENN": {"Accuracy": 0.651, "Recall": 0.714, "Specificity": 0.633, "Precision": 0.353, "F1-Score": 0.472},
        "Logistic Regression_SMOTE-ENN": {"Accuracy": 0.593, "Recall": 0.746, "Specificity": 0.550, "Precision": 0.317, "F1-Score": 0.445},
        "SVM_SMOTE-ENN": {"Accuracy": 0.647, "Recall": 0.741, "Specificity": 0.621, "Precision": 0.354, "F1-Score": 0.479},
        "Decision Tree_SMOTE-ENN": {"Accuracy": 0.660, "Recall": 0.601, "Specificity": 0.677, "Precision": 0.343, "F1-Score": 0.436},
        "AdaBoost_SMOTE-ENN": {"Accuracy": 0.652, "Recall": 0.733, "Specificity": 0.629, "Precision": 0.356, "F1-Score": 0.479},
        "CNN_SMOTE-ENN": {"Accuracy": 0.606, "Recall": 0.701, "Specificity": 0.580, "Precision": 0.318, "F1-Score": 0.438},
        "LSTM_SMOTE-ENN": {"Accuracy": 0.647, "Recall": 0.767, "Specificity": 0.613, "Precision": 0.357, "F1-Score": 0.487},
        "GRU_SMOTE-ENN": {"Accuracy": 0.679, "Recall": 0.720, "Specificity": 0.667, "Precision": 0.377, "F1-Score": 0.495}
    }
}
