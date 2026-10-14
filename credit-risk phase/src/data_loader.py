"""
Data loading, validation, and hygiene auditing module.
Handles German Credit and Taiwan Credit Card Default datasets.
"""

import os
import glob
import pandas as pd
import numpy as np
from src.config import DATA_DIR, SEED, QUICK_TEST, QUICK_ROWS

CLEAN_LOG = []

GERMAN_COLS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "target"
]

def load_german_credit(data_dir=None):
    """
    Load and clean the German Credit dataset (1,000 instances, 20 attributes).
    Target: 1 = Good credit (0), 2 = Bad credit/Default (1).
    """
    ddir = data_dir or DATA_DIR
    path = os.path.join(ddir, "german.data")
    if not os.path.exists(path):
        raise FileNotFoundError(f"German credit file not found at: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLS)
    
    # Target remapping: 2 (bad) -> 1, 1 (good) -> 0
    y = (df.pop("target") == 2).astype(int)
    
    missing_cnt = int(df.isna().sum().sum())
    CLEAN_LOG.append(("German", "Missing values detected", missing_cnt))
    
    dup_cnt = int(df.duplicated().sum())
    CLEAN_LOG.append(("German", "Duplicate rows removed", dup_cnt))
    
    keep = ~df.duplicated()
    df, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in df.columns if c not in cat]
    
    return df, y, cat, num

def load_taiwan_credit(data_dir=None, quick_test=None, quick_rows=None, seed=None):
    """
    Load and clean the Taiwan Credit Card Default dataset (30,000 instances, 23 attributes).
    Applies data hygiene: removes 35 duplicates, collapses undocumented codes, renames PAY_0 -> PAY_1.
    """
    ddir = data_dir or DATA_DIR
    q_test = QUICK_TEST if quick_test is None else quick_test
    q_rows = QUICK_ROWS if quick_rows is None else quick_rows
    rnd = SEED if seed is None else seed

    candidates = glob.glob(os.path.join(ddir, "*default*credit*"))
    if not candidates:
        raise FileNotFoundError(f"Taiwan credit dataset file not found in: {ddir}")
    path = candidates[0]

    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path, header=1)
    df = df.drop(columns=[c for c in df.columns if str(c).strip().upper() == "ID"])
    
    target_col = [c for c in df.columns if "default" in str(c).lower()][0]
    y = df.pop(target_col).astype(int)
    
    # Chronological consistency fix: PAY_0 is the September repayment status
    df = df.rename(columns={"PAY_0": "PAY_1"})
    
    missing_cnt = int(df.isna().sum().sum())
    CLEAN_LOG.append(("Taiwan", "Missing values detected", missing_cnt))
    
    # Recode undocumented categorical levels as noted in Phase 1 report
    for col, bad_codes, to_val in [("EDUCATION", [0, 5, 6], 4), ("MARRIAGE", [0], 3)]:
        count = int(df[col].isin(bad_codes).sum())
        CLEAN_LOG.append(("Taiwan", f"{col} undocumented codes {bad_codes} recoded to {to_val} ('Others')", count))
        df[col] = df[col].replace({b: to_val for b in bad_codes})
        
    y_dup = pd.concat([df, y], axis=1).duplicated()
    dup_count = int(y_dup.sum())
    CLEAN_LOG.append(("Taiwan", "Duplicate rows removed", dup_count))
    keep = ~y_dup
    df, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    
    cat = ["SEX", "EDUCATION", "MARRIAGE"]
    for c in cat:
        df[c] = df[c].astype(str)
    num = [c for c in df.columns if c not in cat]
    
    if q_test and q_rows and len(df) > q_rows:
        idx = df.sample(q_rows, random_state=rnd).index
        df, y = df.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
        CLEAN_LOG.append(("Taiwan", f"Quick-test subsample applied (rows: {q_rows})", len(df)))
        
    return df, y, cat, num

def get_data_bundle(quick_test=None):
    """Loads both datasets and returns data dict, clean audit dataframe, and summary table."""
    data = {
        "German": load_german_credit(),
        "Taiwan": load_taiwan_credit(quick_test=quick_test)
    }
    
    audit_df = pd.DataFrame(CLEAN_LOG, columns=["dataset", "step", "rows/values affected"])
    
    summary_rows = []
    for ds, (X, y, cat, num) in data.items():
        summary_rows.append({
            "dataset": ds,
            "rows": len(X),
            "features": X.shape[1],
            "categorical": len(cat),
            "numeric": len(num),
            "minority_default_pct": round(100 * y.mean(), 2)
        })
    summary_df = pd.DataFrame(summary_rows).set_index("dataset")
    
    prior_odds = {ds: float((1 - data[ds][1].mean()) / data[ds][1].mean()) for ds in data}
    
    return data, audit_df, summary_df, prior_odds
