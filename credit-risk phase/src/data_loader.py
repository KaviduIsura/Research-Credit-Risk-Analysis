"""
Data Loaders for German Credit and Taiwan Credit Datasets.
Adheres strictly to dataset schemas defined in Section 3.1 (Tables 1, 2, 3) of the research paper.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import DATA_RAW_DIR, SEED

# German Credit feature definitions (Table 1 of paper)
GERMAN_COLUMNS = [
    "StatusOfCheckingAccount",
    "DurationInMonths",
    "CreditHistory",
    "LoanPurpose",
    "CreditAmount",
    "SavingsAccountBonds",
    "EmploymentDuration",
    "InstallmentRatePercentage",
    "PersonalStatusSex",
    "OtherDebtorsGuarantors",
    "ResidenceDuration",
    "PropertyType",
    "AgeInYears",
    "OtherInstallmentPlans",
    "HousingType",
    "ExistingCreditsCount",
    "JobType",
    "Dependents",
    "Telephone",
    "ForeignWorker",
    "Target"
]

GERMAN_NUMERICAL = [
    "DurationInMonths",
    "CreditAmount",
    "InstallmentRatePercentage",
    "ResidenceDuration",
    "AgeInYears",
    "ExistingCreditsCount",
    "Dependents"
]

GERMAN_CATEGORICAL = [col for col in GERMAN_COLUMNS if col not in GERMAN_NUMERICAL and col != "Target"]

# Taiwan Credit feature definitions (Table 2 of paper)
TAIWAN_CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
TAIWAN_NUMERICAL = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]


def load_german_data(data_dir=DATA_RAW_DIR):
    """
    Load and parse the Statlog German Credit Dataset (1,000 instances, 20 features).
    Target mapping:
      Original UCI values: 1 = Good, 2 = Bad (Default).
      Binary mapping: 0 = Good (70%), 1 = Bad/Default (30%).
    """
    filepath = Path(data_dir) / "german.data"
    if not filepath.exists():
        raise FileNotFoundError(f"German dataset not found at {filepath}")

    df = pd.read_csv(filepath, sep=r"\s+", names=GERMAN_COLUMNS, engine="python")
    
    # Binary mapping: 1 (Good) -> 0, 2 (Bad/Default) -> 1
    df["Target"] = df["Target"].map({1: 0, 2: 1})
    
    X = df.drop(columns=["Target"])
    y = df["Target"].astype(int)
    
    return X, y, GERMAN_CATEGORICAL, GERMAN_NUMERICAL


def load_taiwan_data(data_dir=DATA_RAW_DIR, sample_size=None, random_state=SEED):
    """
    Load and parse the Taiwan Credit Card Default Dataset (30,000 instances, 23 features).
    Target mapping:
      'default payment next month': 0 = Non-default (77.9%), 1 = Default (22.1%).
    """
    filepath = Path(data_dir) / "default of credit card clients.xls"
    if not filepath.exists():
        raise FileNotFoundError(f"Taiwan dataset not found at {filepath}")

    # File contains header on row 1 (0-indexed row 1)
    df = pd.read_excel(filepath, header=1)
    
    # Drop arbitrary identifier column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        
    target_col = "default payment next month"
    if target_col not in df.columns:
        # Fallback to last column
        target_col = df.columns[-1]

    # Subsample if sample_size is set (e.g. for QUICK_TEST mode)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    # Feature category resolution
    cat_cols = [c for c in TAIWAN_CATEGORICAL if c in X.columns]
    num_cols = [c for c in TAIWAN_NUMERICAL if c in X.columns]
    
    # Ensure remaining columns are tracked
    for c in X.columns:
        if c not in cat_cols and c not in num_cols:
            num_cols.append(c)

    return X, y, cat_cols, num_cols


def load_dataset(dataset_name, data_dir=DATA_RAW_DIR, sample_size=None):
    """Unified loader dispatch."""
    if dataset_name.lower() == "german":
        return load_german_data(data_dir=data_dir)
    elif dataset_name.lower() == "taiwan":
        return load_taiwan_data(data_dir=data_dir, sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'German' or 'Taiwan'.")
