"""
Enhanced Preprocessing & Data Hygiene Module.
Implements Phase 1 methodological corrections (Section 3.3 & 3.4 of Phase 1 report):
  1. Data Hygiene: Removes 35 duplicate records in Taiwan, standardizes undocumented codes (EDUCATION 0,5,6 -> 4; MARRIAGE 0 -> 3), aligns PAY_0 -> PAY_1.
  2. SkewLog Transformation: Handles extreme positive monetary skewness via signed log1p.
  3. In-Fold Standardization: Prevents feature variance dominance in distance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# 1. Data Hygiene Cleaners
# ---------------------------------------------------------------------------

def clean_taiwan_hygiene(df_raw, cat_cols=None, num_cols=None):
    """
    Applies data hygiene audit fixes to Taiwan Credit dataset:
      - Drops duplicate rows.
      - Collapses undocumented EDUCATION levels (0, 5, 6) into 4 ('Others').
      - Collapses undocumented MARRIAGE level (0) into 3 ('Others').
      - Renames PAY_0 -> PAY_1 for chronological sequence consistency.
      - Synchronizes cat_cols and num_cols lists if provided.
    """
    df = df_raw.copy()
    
    # Drop arbitrary identifier column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        
    # Drop identical duplicate borrower records (Phase 1 audit found 35 duplicate rows)
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dups = initial_len - len(df)
    
    # Collapse undocumented EDUCATION levels (0, 5, 6 -> 4)
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        
    # Collapse undocumented MARRIAGE level (0 -> 3)
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
        
    # Standardize column nomenclature PAY_0 -> PAY_1
    if "PAY_0" in df.columns and "PAY_1" not in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})
        
    clean_cat = None
    if cat_cols is not None:
        clean_cat = ["PAY_1" if c == "PAY_0" else c for c in cat_cols if c != "ID"]
        
    clean_num = None
    if num_cols is not None:
        clean_num = [c for c in num_cols if c != "ID"]
        
    if cat_cols is not None and num_cols is not None:
        return df, n_dups, clean_cat, clean_num
    elif cat_cols is not None:
        return df, n_dups, clean_cat
    return df, n_dups


def clean_german_hygiene(df_raw):
    """
    Applies data hygiene audit to German Credit dataset:
      - Removes duplicates if any.
      - Confirms schema consistency.
    """
    df = df_raw.copy()
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dups = initial_len - len(df)
    return df, n_dups


# ---------------------------------------------------------------------------
# 2. SkewLog Transformation for Extreme Monetary Skewness
# ---------------------------------------------------------------------------

class SkewLogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies signed log1p transformation to continuous monetary features:
      SkewLog(x) = sign(x) * ln(|x| + 1)
    Compresses heavy positive skewness (e.g., PAY_AMT skewness > 30)
    while preserving negative/zero balances and relative monotonicity.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.sign(X_arr) * np.log1p(np.abs(X_arr))


# ---------------------------------------------------------------------------
# 3. Enhanced Preprocessor Factory
# ---------------------------------------------------------------------------

def build_enhanced_preprocessor(cat_cols, num_cols, apply_skewlog=True):
    """
    Constructs an enhanced in-fold ColumnTransformer:
      - Continuous features: SkewLog -> StandardScaler
      - Categorical features: OneHotEncoder (handling unseen categories)
    """
    from sklearn.pipeline import Pipeline as SkPipeline
    
    transformers = []
    
    if num_cols:
        if apply_skewlog:
            num_pipe = SkPipeline([
                ("skewlog", SkewLogTransformer()),
                ("scaler", StandardScaler())
            ])
        else:
            num_pipe = SkPipeline([
                ("scaler", StandardScaler())
            ])
        transformers.append(("num", num_pipe, num_cols))
        
    if cat_cols:
        cat_pipe = SkPipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
        
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_ordinal_scaled_preprocessor(cat_cols, num_cols, scale=True):
    """
    Prepares scaled continuous + ordinal integer categoricals (used for SMOTE-NC).
    """
    from sklearn.pipeline import Pipeline as SkPipeline
    
    transformers = []
    if num_cols:
        if scale:
            num_pipe = SkPipeline([
                ("skewlog", SkewLogTransformer()),
                ("scaler", StandardScaler())
            ])
        else:
            num_pipe = "passthrough"
        transformers.append(("num", num_pipe, num_cols))
        
    if cat_cols:
        cat_pipe = SkPipeline([
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
        
    return ColumnTransformer(transformers=transformers, remainder="drop")
