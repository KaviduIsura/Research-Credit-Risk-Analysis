"""
Credit Risk Analysis & Research Paper Replication Package.
Paper: Performance Evaluation of Machine Learning and Deep Learning Models for Credit Risk Prediction (JRFM 2026).
"""

from src.config import SEED, QUICK_TEST, DATASETS, RESAMPLING_METHODS, MODELS_ALL
from src.data_loader import load_german_data, load_taiwan_data, load_dataset
from src.models import create_model
from src.resampling import get_resampler, apply_resampling
from src.metrics import compute_metrics, aggregate_fold_metrics
from src.training import run_experiment, run_benchmark_matrix
from src.interpretability import explain_with_shap

__all__ = [
    "SEED",
    "QUICK_TEST",
    "DATASETS",
    "RESAMPLING_METHODS",
    "MODELS_ALL",
    "load_german_data",
    "load_taiwan_data",
    "load_dataset",
    "create_model",
    "get_resampler",
    "apply_resampling",
    "compute_metrics",
    "aggregate_fold_metrics",
    "run_experiment",
    "run_benchmark_matrix",
    "explain_with_shap"
]
