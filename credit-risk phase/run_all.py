"""
Standalone Runner for Credit Risk Research Paper Replication.
Trains models across 100% of both datasets (German 1,000 instances and Taiwan 30,000 instances)
using full 10-Fold Stratified Cross-Validation with dynamic Remaining Time (ETA) tracking.

Usage:
  python run_all.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import src
import src.leakage_analysis as la
import src.visualizations as viz
from src.config import OUTER_FOLDS, RESULTS_TABLES_DIR, RESULTS_FIGURES_DIR


def main():
    print("\n" + "=" * 65)
    print("  CREDIT RISK ANALYSIS: RESEARCH PAPER REPLICATION (JRFM 2026)")
    print("  Full Dataset Protocol: 100% Data, Zero Subsampling")
    print(f"  Cross-Validation: {OUTER_FOLDS}-Fold Stratified Cross-Validation")
    print("=" * 65)

    # -------------------------------------------------------------
    # Step 1: Load Full Datasets
    # -------------------------------------------------------------
    print("\n[Stage 1/5] Ingesting 100% of Both Benchmark Datasets...")
    X_german, y_german, cat_german, num_german = src.load_german_data()
    X_taiwan, y_taiwan, cat_taiwan, num_taiwan = src.load_taiwan_data()

    print(f"  --> German Credit Dataset: {len(X_german):,} rows, {X_german.shape[1]} features")
    print(f"      Good: {(y_german==0).sum():,} ({100*(y_german==0).mean():.1f}%) | Default: {(y_german==1).sum():,} ({100*(y_german==1).mean():.1f}%)")
    print(f"  --> Taiwan Credit Dataset: {len(X_taiwan):,} rows, {X_taiwan.shape[1]} features")
    print(f"      Non-default: {(y_taiwan==0).sum():,} ({100*(y_taiwan==0).mean():.1f}%) | Default: {(y_taiwan==1).sum():,} ({100*(y_taiwan==1).mean():.1f}%)")

    models_to_eval = ["MLP", "Random Forest", "XGBoost", "Logistic Regression"]
    resampling_to_eval = ["No Resampling", "SMOTE", "ENN", "SMOTE-ENN"]

    # -------------------------------------------------------------
    # Step 2: German Credit Benchmark (Replicating Table 5)
    # -------------------------------------------------------------
    print(f"\n[Stage 2/5] Running German Credit 10-Fold Benchmark Matrix...")
    df_german = src.run_benchmark_matrix(
        dataset_name="German",
        X=X_german,
        y=y_german,
        cat_cols=cat_german,
        num_cols=num_german,
        models=models_to_eval,
        resampling_methods=resampling_to_eval,
        n_outer_folds=OUTER_FOLDS
    )
    table5_path = RESULTS_TABLES_DIR / "reproduced_table5_german.csv"
    df_german.to_csv(table5_path, index=False)
    print(f"  [SAVED] German benchmark table exported to: {table5_path}")

    # -------------------------------------------------------------
    # Step 3: Taiwan Credit Benchmark (Replicating Table 6)
    # -------------------------------------------------------------
    print(f"\n[Stage 3/5] Running Taiwan Credit 10-Fold Benchmark Matrix (30,000 rows)...")
    df_taiwan = src.run_benchmark_matrix(
        dataset_name="Taiwan",
        X=X_taiwan,
        y=y_taiwan,
        cat_cols=cat_taiwan,
        num_cols=num_taiwan,
        models=models_to_eval,
        resampling_methods=resampling_to_eval,
        n_outer_folds=OUTER_FOLDS
    )
    table6_path = RESULTS_TABLES_DIR / "reproduced_table6_taiwan.csv"
    df_taiwan.to_csv(table6_path, index=False)
    print(f"  [SAVED] Taiwan benchmark table exported to: {table6_path}")

    # -------------------------------------------------------------
    # Step 4: Controlled Data Leakage Audit (German Credit)
    # -------------------------------------------------------------
    print("\n[Stage 4/5] Running Controlled Data Leakage Audit on German Credit...")
    df_leakage = la.run_leakage_comparison(
        X=X_german,
        y=y_german,
        cat_cols=cat_german,
        num_cols=num_german,
        model_name="MLP",
        n_splits=10
    )
    leakage_path = RESULTS_TABLES_DIR / "data_leakage_experiment.csv"
    df_leakage.to_csv(leakage_path, index=False)
    print(f"  [SAVED] Data leakage audit exported to: {leakage_path}")

    # -------------------------------------------------------------
    # Step 5: Comparative Charts & SHAP Explainability
    # -------------------------------------------------------------
    print("\n[Stage 5/5] Generating Author Comparison Plots & SHAP Summary...")
    german_mlp = df_german[(df_german["Model"]=="MLP") & (df_german["Resampling"]=="SMOTE-ENN")].iloc[0].to_dict()
    taiwan_rf = df_taiwan[(df_taiwan["Model"]=="Random Forest") & (df_taiwan["Resampling"]=="SMOTE-ENN")].iloc[0].to_dict()
    
    viz.plot_author_comparison(german_mlp, dataset_name="German", model_name="MLP")
    viz.plot_author_comparison(taiwan_rf, dataset_name="Taiwan", model_name="Random Forest")

    # Run single Taiwan fold for SHAP summary plot (Figure 2)
    exp_taiwan = src.run_experiment(
        X=X_taiwan,
        y=y_taiwan,
        cat_cols=cat_taiwan,
        num_cols=num_taiwan,
        model_name="Random Forest",
        resampling_method="SMOTE-ENN",
        n_outer_folds=3,
        verbose=False
    )
    feature_names = exp_taiwan["last_preprocessor"].get_feature_names_out()
    src.explain_with_shap(
        model=exp_taiwan["last_model"],
        X_val=exp_taiwan["last_X_val"],
        feature_names=feature_names,
        dataset_name="Taiwan",
        model_name="Random Forest"
    )

    print("\n" + "=" * 65)
    print("  TRAINING & BENCHMARK COMPLETE!")
    print(f"  All output tables:  {RESULTS_TABLES_DIR}")
    print(f"  All output figures: {RESULTS_FIGURES_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
