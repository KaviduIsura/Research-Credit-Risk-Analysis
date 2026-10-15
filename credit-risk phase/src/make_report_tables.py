"""
src/make_report_tables.py

Reads ONLY CSV files in results/ and generates results/report_tables.md containing
all markdown-formatted tables required for the technical review report.

Rules strictly adhered to:
1. Reads ONLY CSV files in results/.
2. Formats all numeric columns to 3 decimals.
3. Never hardcodes numbers by hand.
4. Never rounds numbers before display.
5. Documents the source CSV file name under every generated table.
"""

import os
import pandas as pd
import numpy as np


def fmt_val(v, decimals=3):
    """Formats float or int values to string without prior rounding."""
    if pd.isna(v) or v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{decimals}f}"
    if isinstance(v, (int, np.integer)):
        return str(v)
    try:
        f = float(v)
        return f"{f:.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def df_to_markdown(df, float_cols=None, int_cols=None, decimals=3):
    """Converts DataFrame to GitHub-flavored Markdown table with 3 decimal precision."""
    df_out = df.copy()
    for col in df_out.columns:
        if float_cols is not None and col in float_cols:
            df_out[col] = df_out[col].apply(lambda x: fmt_val(x, decimals))
        elif int_cols is not None and col in int_cols:
            df_out[col] = df_out[col].apply(lambda x: str(int(x)) if pd.notna(x) else "-")
        elif pd.api.types.is_float_dtype(df_out[col]):
            df_out[col] = df_out[col].apply(lambda x: fmt_val(x, decimals))
        elif pd.api.types.is_integer_dtype(df_out[col]):
            df_out[col] = df_out[col].apply(lambda x: str(int(x)) if pd.notna(x) else "-")
        else:
            df_out[col] = df_out[col].fillna("-").astype(str)
    
    # Generate markdown table string
    headers = list(df_out.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df_out.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def generate_report_tables(results_dir="results", output_file="results/report_tables.md"):
    rdir = os.path.abspath(results_dir)
    out_path = os.path.abspath(output_file)
    
    md_sections = []
    md_sections.append("# Complete Phase 2 Technical Review Report Tables\n")
    md_sections.append(
        "> **Note:** All tables in this document are automatically extracted directly from the "
        "production cross-validation CSV outputs in `results/`. Values are formatted to 3 decimal places "
        "with complete methodological provenance documented under each table.\n"
    )
    
    table_index = []

    # -------------------------------------------------------------------------
    # (a) Authors' Published Numbers from authors_reported.csv
    # -------------------------------------------------------------------------
    path_a = os.path.join(rdir, "authors_reported.csv")
    if os.path.exists(path_a):
        df_a = pd.read_csv(path_a, comment="#")
        
        # German table (Table 5)
        df_a_ger = df_a[df_a["dataset"] == "German"].copy()
        cols_a = ["resampling", "model", "accuracy", "recall", "specificity", "precision", "f1_score"]
        df_a_ger_disp = df_a_ger[cols_a].rename(columns={
            "resampling": "Resampling Method",
            "model": "Model",
            "accuracy": "Accuracy",
            "recall": "Recall",
            "specificity": "Specificity",
            "precision": "Precision",
            "f1_score": "F1 Score (Minority)"
        })
        md_sections.append("## Table 1a: Authors' Published Results — German Credit Dataset (Table 5, p. 15)")
        md_sections.append(df_to_markdown(df_a_ger_disp))
        md_sections.append("\n*Source: results/authors_reported.csv*\n")
        table_index.append("Table 1a: Authors' Published Results — German Credit (Table 5)")

        # Taiwan table (Table 6)
        df_a_tai = df_a[df_a["dataset"] == "Taiwan"].copy()
        df_a_tai_disp = df_a_tai[cols_a].rename(columns={
            "resampling": "Resampling Method",
            "model": "Model",
            "accuracy": "Accuracy",
            "recall": "Recall",
            "specificity": "Specificity",
            "precision": "Precision",
            "f1_score": "F1 Score (Minority)"
        })
        md_sections.append("## Table 1b: Authors' Published Results — Taiwan Credit Dataset (Table 6, p. 16)")
        md_sections.append(df_to_markdown(df_a_tai_disp))
        md_sections.append("\n*Source: results/authors_reported.csv*\n")
        table_index.append("Table 1b: Authors' Published Results — Taiwan Credit (Table 6)")

    # -------------------------------------------------------------------------
    # (b) verdicts.csv
    # -------------------------------------------------------------------------
    path_b = os.path.join(rdir, "verdicts.csv")
    if os.path.exists(path_b):
        df_b = pd.read_csv(path_b)
        df_b_disp = df_b.rename(columns={
            "dataset": "Dataset",
            "metric": "Metric",
            "mine": "Our Primary Pipeline",
            "authors": "Authors' Published",
            "delta": "Difference (Ours - Authors)",
            "band": "Uncertainty Band (±2.5 SD)",
            "verdict": "Formal Verdict"
        })
        md_sections.append("## Table 2: Benchmark Comparison Verdicts (Our Primary Pipeline vs. Authors' Published)")
        md_sections.append(df_to_markdown(df_b_disp))
        md_sections.append("\n*Source: results/verdicts.csv*\n")
        table_index.append("Table 2: Benchmark Comparison Verdicts (verdicts.csv)")

    # -------------------------------------------------------------------------
    # (c) fix_vs_paper_style_tests.csv restricted to MCC
    # -------------------------------------------------------------------------
    path_c = os.path.join(rdir, "fix_vs_paper_style_tests.csv")
    if os.path.exists(path_c):
        df_c = pd.read_csv(path_c)
        df_c_mcc = df_c[df_c["metric"] == "mcc"].copy()
        cols_c = ["dataset", "model", "variant", "paper_style", "corrected", "diff", "t_stat", "p_value"]
        df_c_disp = df_c_mcc[cols_c].rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "variant": "Methodological Fix Variant",
            "paper_style": "Paper-Style MCC",
            "corrected": "Corrected MCC",
            "diff": "Delta MCC",
            "t_stat": "t-statistic",
            "p_value": "p-value (Corrected t-test)"
        })
        md_sections.append("## Table 3: Statistical Significance Tests: Corrected vs. Paper-Style Pipeline (MCC)")
        md_sections.append(df_to_markdown(df_c_disp))
        md_sections.append("\n*Source: results/fix_vs_paper_style_tests.csv*\n")
        table_index.append("Table 3: Statistical Significance Tests (MCC only)")

    # -------------------------------------------------------------------------
    # (d) Stage A matrix per dataset from all_results.csv
    # -------------------------------------------------------------------------
    path_d = os.path.join(rdir, "all_results.csv")
    if os.path.exists(path_d):
        df_d = pd.read_csv(path_d)
        df_d_stage_a = df_d[df_d["mode"] == "default"].copy()

        for ds, t_num in [("German", "4a"), ("Taiwan", "4b")]:
            df_ds = df_d_stage_a[df_d_stage_a["dataset"] == ds].copy()
            if df_ds.empty:
                continue
            
            # Pivot Matrix: Model x Variant for MCC Mean
            piv_mcc = df_ds.pivot(index="model", columns="variant", values="mcc_mean")
            piv_mcc = piv_mcc[["paper_style", "fixed_none", "fixed_smotenc", "fixed_smoteenn"]]
            piv_mcc_disp = piv_mcc.reset_index().rename(columns={"model": "Model"})
            
            md_sections.append(f"## Table {t_num}1: {ds} Credit — Stage A Model × Variant MCC Matrix")
            md_sections.append(df_to_markdown(piv_mcc_disp))
            md_sections.append("\n*Source: results/all_results.csv*\n")
            table_index.append(f"Table {t_num}1: {ds} Credit Stage A MCC Matrix")

            # Full Stage A Metrics Table
            cols_disp = [
                "model", "variant", "accuracy_mean", "balanced_accuracy_mean",
                "recall_mean", "specificity_mean", "precision_mean", "f1_mean",
                "mcc_mean", "pr_auc_mean", "roc_auc_mean", "brier_mean"
            ]
            cols_disp = [c for c in cols_disp if c in df_ds.columns]
            df_ds_disp = df_ds[cols_disp].rename(columns={
                "model": "Model",
                "variant": "Pipeline Variant",
                "accuracy_mean": "Accuracy",
                "balanced_accuracy_mean": "Balanced Acc",
                "recall_mean": "Recall",
                "specificity_mean": "Specificity",
                "precision_mean": "Precision",
                "f1_mean": "F1",
                "mcc_mean": "MCC",
                "pr_auc_mean": "PR-AUC",
                "roc_auc_mean": "ROC-AUC",
                "brier_mean": "Brier Score"
            })
            md_sections.append(f"## Table {t_num}2: {ds} Credit — Stage A Full Benchmark Metrics (Mean)")
            md_sections.append(df_to_markdown(df_ds_disp))
            md_sections.append("\n*Source: results/all_results.csv*\n")
            table_index.append(f"Table {t_num}2: {ds} Credit Stage A Full Benchmark Table")

    # -------------------------------------------------------------------------
    # (e) leakage_experiment.csv
    # -------------------------------------------------------------------------
    path_e = os.path.join(rdir, "leakage_experiment.csv")
    if os.path.exists(path_e):
        df_e = pd.read_csv(path_e)
        df_e_disp = df_e.rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "leaky_f1": "Leaky F1",
            "in-fold_f1": "In-Fold F1",
            "F1 inflation": "F1 Inflation (Δ)",
            "leaky_mcc": "Leaky MCC",
            "in-fold_mcc": "In-Fold MCC",
            "leaky_recall": "Leaky Recall",
            "in-fold_recall": "In-Fold Recall",
            "leaky_precision": "Leaky Precision",
            "in-fold_precision": "In-Fold Precision"
        })
        md_sections.append("## Table 5: Controlled Data Leakage Experiment (Pre-Split vs. In-Fold Resampling)")
        md_sections.append(df_to_markdown(df_e_disp))
        md_sections.append("\n*Source: results/leakage_experiment.csv*\n")
        table_index.append("Table 5: Data Leakage Experiment (leakage_experiment.csv)")

    # -------------------------------------------------------------------------
    # (f) calibration.csv
    # -------------------------------------------------------------------------
    path_f = os.path.join(rdir, "calibration.csv")
    if os.path.exists(path_f):
        df_f = pd.read_csv(path_f)
        df_f_disp = df_f.rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "variant": "Variant",
            "brier_raw": "Raw Brier Score",
            "brier_calibrated": "Isotonic Calibrated Brier"
        })
        md_sections.append("## Table 6: Probability Calibration Audit (Isotonic Regression Brier Reduction)")
        md_sections.append(df_to_markdown(df_f_disp))
        md_sections.append("\n*Source: results/calibration.csv*\n")
        table_index.append("Table 6: Probability Calibration Audit (calibration.csv)")

    # -------------------------------------------------------------------------
    # (g) fairness_gaps.csv
    # -------------------------------------------------------------------------
    path_g = os.path.join(rdir, "fairness_gaps.csv")
    if os.path.exists(path_g):
        df_g = pd.read_csv(path_g)
        df_g_disp = df_g.rename(columns={
            "attribute": "Sensitive Attribute",
            "group": "Demographic Group",
            "n": "Sample Size (N)",
            "base_rate": "Base Default Rate",
            "selection_rate": "Selection Rate (Predicted Default)",
            "TPR": "True Positive Rate (TPR)",
            "FPR": "False Positive Rate (FPR)"
        })
        md_sections.append("## Table 7: Demographic Fairness & Disparity Audit")
        md_sections.append(df_to_markdown(df_g_disp, int_cols=["n"]))
        md_sections.append("\n*Source: results/fairness_gaps.csv*\n")
        table_index.append("Table 7: Demographic Fairness Audit (fairness_gaps.csv)")

    # -------------------------------------------------------------------------
    # (h) noise_stress_test.csv for every model and dataset
    # -------------------------------------------------------------------------
    path_h = os.path.join(rdir, "noise_stress_test.csv")
    if os.path.exists(path_h):
        df_h = pd.read_csv(path_h)
        
        # German Stress Table
        df_h_ger = df_h[df_h["dataset"] == "German"].copy()
        df_h_ger_disp = df_h_ger[["model", "variant", "noise_level", "mcc_mean", "mcc_std", "f1_mean", "f1_std"]].rename(columns={
            "model": "Model",
            "variant": "Variant",
            "noise_level": "Noise σ (Std Dev)",
            "mcc_mean": "MCC Mean",
            "mcc_std": "MCC SD",
            "f1_mean": "F1 Mean",
            "f1_std": "F1 SD"
        })
        md_sections.append("## Table 8a: Noise Perturbation Stress Test — German Credit Dataset")
        md_sections.append(df_to_markdown(df_h_ger_disp))
        md_sections.append("\n*Source: results/noise_stress_test.csv*\n")
        table_index.append("Table 8a: Noise Stress Test — German Credit")

        # Taiwan Stress Table
        df_h_tai = df_h[df_h["dataset"] == "Taiwan"].copy()
        df_h_tai_disp = df_h_tai[["model", "variant", "noise_level", "mcc_mean", "mcc_std", "f1_mean", "f1_std"]].rename(columns={
            "model": "Model",
            "variant": "Variant",
            "noise_level": "Noise σ (Std Dev)",
            "mcc_mean": "MCC Mean",
            "mcc_std": "MCC SD",
            "f1_mean": "F1 Mean",
            "f1_std": "F1 SD"
        })
        md_sections.append("## Table 8b: Noise Perturbation Stress Test — Taiwan Credit Dataset")
        md_sections.append(df_to_markdown(df_h_tai_disp))
        md_sections.append("\n*Source: results/noise_stress_test.csv*\n")
        table_index.append("Table 8b: Noise Stress Test — Taiwan Credit")

        # Combined Pivot Matrix: MCC degradation across noise levels
        piv_stress = df_h.pivot_table(index=["dataset", "model"], columns="noise_level", values="mcc_mean").reset_index()
        piv_stress.columns = [f"Noise {c}" if isinstance(c, (float, int)) else c for c in piv_stress.columns]
        md_sections.append("## Table 8c: Noise Degradation Summary Matrix (MCC across Noise Levels)")
        md_sections.append(df_to_markdown(piv_stress))
        md_sections.append("\n*Source: results/noise_stress_test.csv*\n")
        table_index.append("Table 8c: Noise Degradation Summary Matrix")

    # Write output markdown file
    full_content = "\n\n".join(md_sections) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_content)
    
    print(f"Successfully generated {len(table_index)} report tables in: {out_path}")
    return table_index


if __name__ == "__main__":
    generate_report_tables()
