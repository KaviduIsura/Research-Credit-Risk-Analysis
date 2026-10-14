import json
import os

def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.1"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Cell 0: Title & Overview
    add_md("""# Phase 2 — Technical Application: Reproducible Credit Risk Pipeline
**Module:** Machine Learning and Related Applications (Phase 2 Option A)  
**Deliverable:** Methodologically rigorous baseline reproduction and corrected pipeline architecture for credit risk evaluation on German Credit and Taiwan Credit datasets.

### Environment Compatibility
This notebook is self-contained and engineered to run seamlessly both **locally** (Antigravity IDE / VS Code) and in **Google Colab**.
- All data ingestion routines automatically resolve paths relative to the project root (`data/raw`) or Colab workspace (`/content/data`).
- All feature transformations (signed log1p, scaling, categorical encoding) and resampling steps are strictly embedded inside `imblearn.pipeline.Pipeline` objects to guarantee **zero data leakage** into test folds.""")

    # Cell 1: Setup & Dependencies
    add_code("""# Stage 0: Install dependencies and import libraries
%pip -q install imbalanced-learn xlrd openpyxl xgboost scikit-learn

import os, sys, glob, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import loguniform, uniform, randint

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
print(f"Environment ready | Python {sys.version.split()[0]} | sklearn {sklearn.__version__} | pandas {pd.__version__}")""")

    # Cell 2: Configuration
    add_code("""# Stage 0: Global Configuration (Edit parameters here)
SEED = 42
QUICK_TEST = False       # True = small sample & reduced folds for dry run; False = full evaluation
np.random.seed(SEED)

DATASETS = ["German", "Taiwan"]
MODELS_USED = {
    "German": ["LogReg", "RandomForest", "XGBoost", "MLP"],
    "Taiwan": ["LogReg", "RandomForest", "XGBoost", "MLP"]
}
VARIANTS = ["paper_style", "fixed_none", "fixed_smotenc", "fixed_smoteenn"]

# Cross-validation fold settings
OUTER_FOLDS = 10                                  # Stage A: repeated stratified CV
REPEATS = {"German": 3, "Taiwan": 1}
OUTER_TUNED = {"German": 10, "Taiwan": 5}         # Stage B: nested CV outer folds
INNER_FOLDS = 3                                   # Hyperparameter search folds
THR_FOLDS = 3                                     # Threshold-tuning folds
N_ITER = {"German": 20, "Taiwan": 6}              # Random search candidates
QUICK_ROWS = 3000                                 # Taiwan subsample size in QUICK_TEST

if QUICK_TEST:
    OUTER_FOLDS = 3
    REPEATS = {"German": 1, "Taiwan": 1}
    OUTER_TUNED = {"German": 3, "Taiwan": 3}
    INNER_FOLDS = 2
    THR_FOLDS = 2
    N_ITER = {"German": 2, "Taiwan": 2}

IN_COLAB = "google.colab" in sys.modules

def find_data_dir():
    # Robust path discovery supporting both local repo execution and Google Colab
    candidates = ["data/raw", "../data/raw", "data", "../data", "/content/data"]
    for c in candidates:
        if os.path.exists(os.path.join(c, "german.data")):
            return os.path.abspath(c)
    return os.path.abspath("/content/data" if IN_COLAB else "data/raw")

DATA_DIR = find_data_dir()
RESULTS_DIR = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "results") 
                              if DATA_DIR.endswith(os.path.join("data", "raw")) 
                              else os.path.join(os.path.dirname(DATA_DIR), "results"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Data directory:    {DATA_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Execution mode:    {'QUICK_TEST (dry run)' if QUICK_TEST else 'FULL EVALUATION'}")""")

    # Cell 3: Data discovery / Colab upload fallback
    add_code("""# Data availability check (with Colab upload fallback)
g_ok = os.path.exists(f"{DATA_DIR}/german.data")
t_ok = len(glob.glob(f"{DATA_DIR}/*default*credit*")) > 0

if not (g_ok and t_ok) and IN_COLAB:
    from google.colab import files
    print("Files missing in Colab environment. Please upload german.data and the Taiwan .xls/.csv file:")
    for name, content in files.upload().items():
        open(f"{DATA_DIR}/{name}", "wb").write(content)
    g_ok = os.path.exists(f"{DATA_DIR}/german.data")
    t_ok = len(glob.glob(f"{DATA_DIR}/*default*credit*")) > 0

print(f"German dataset located: {g_ok} | Taiwan dataset located: {t_ok}")
print("Directory listing in data dir:", os.listdir(DATA_DIR))""")

    # Cell 4: Markdown Data Ingestion Documentation
    add_md("""## Stage 1 — Load, Clean, and Audit Datasets
This stage ingests both datasets under strictly documented data hygiene protocols:
1. **German Credit (Statlog)**:
   - Space-separated format without header (`sep=r"\\s+"`).
   - 20 features (13 categorical, 7 numerical) + target.
   - Original class `2` (bad credit / credit risk) is remapped to `1` as the **positive / minority class**; class `1` (good credit) is mapped to `0`.
2. **Taiwan Credit (Default of Credit Card Clients)**:
   - Parsed from Excel (`.xls`), skipping metadata title row (`header=1`).
   - Drop arbitrary surrogate `ID` column.
   - Column `PAY_0` is renamed to `PAY_1` to correct index irregularity.
   - Target `default payment next month` is mapped to positive class `1`.
   - Undocumented categories recoded: `EDUCATION` codes `0, 5, 6` merged into `4` ('others'); `MARRIAGE` code `0` merged into `3` ('others').
   - Audits and removes duplicate client records across all features + target.""")

    # Cell 5: Load German Data
    add_code("""# Stage 1: Load and Clean German Credit Dataset
CLEAN_LOG = []

GERMAN_COLS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "target"
]

def load_german():
    df = pd.read_csv(f"{DATA_DIR}/german.data", sep=r"\\s+", header=None, names=GERMAN_COLS)
    # Target: 1 = good credit, 2 = bad credit -> map bad credit (2) to positive class 1
    y = (df.pop("target") == 2).astype(int)
    
    # Audit missing values and duplicate rows
    CLEAN_LOG.append(("German", "missing values", int(df.isna().sum().sum())))
    dup_count = int(df.duplicated().sum())
    CLEAN_LOG.append(("German", "duplicate rows removed", dup_count))
    keep = ~df.duplicated()
    df, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    
    # Classify 13 categorical and 7 numerical columns
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in df.columns if c not in cat]
    return df, y, cat, num""")

    # Cell 6: Load Taiwan Data
    add_code("""# Stage 1: Load and Clean Taiwan Credit Dataset
def load_taiwan():
    candidates = glob.glob(f"{DATA_DIR}/*default*credit*")
    if not candidates:
        raise FileNotFoundError(f"Taiwan credit file not found in {DATA_DIR}")
    path = candidates[0]
    
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path, header=1)
    
    # Drop surrogate ID column
    df = df.drop(columns=[c for c in df.columns if str(c).strip().upper() == "ID"])
    
    # Locate and pop target column
    target_col = [c for c in df.columns if "default" in str(c).lower()][0]
    y = df.pop(target_col).astype(int)
    
    # Fix column naming inconsistency: PAY_0 -> PAY_1
    df = df.rename(columns={"PAY_0": "PAY_1"})
    
    # Check for missing values
    CLEAN_LOG.append(("Taiwan", "missing values", int(df.isna().sum().sum())))
    
    # Recode undocumented categorical codes
    for col, bad_codes, to_val in (("EDUCATION", [0, 5, 6], 4), ("MARRIAGE", [0], 3)):
        count = int(df[col].isin(bad_codes).sum())
        CLEAN_LOG.append(("Taiwan", f"{col} undocumented codes {bad_codes} merged into 'others' ({to_val})", count))
        df[col] = df[col].replace({b: to_val for b in bad_codes})
        
    # Deduplication across all features and target
    y_dup = pd.concat([df, y], axis=1).duplicated()
    dup_count = int(y_dup.sum())
    CLEAN_LOG.append(("Taiwan", "duplicate rows removed", dup_count))
    keep = ~y_dup
    df, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    
    # Explicit categorical features (SEX, EDUCATION, MARRIAGE)
    cat = ["SEX", "EDUCATION", "MARRIAGE"]
    for c in cat:
        df[c] = df[c].astype(str)
    num = [c for c in df.columns if c not in cat]
    
    # Subsampling hook for quick test (applied after recording full-data cleaning stats)
    if QUICK_TEST and len(df) > QUICK_ROWS:
        idx = df.sample(QUICK_ROWS, random_state=SEED).index
        df, y = df.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
        
    return df, y, cat, num""")

    # Cell 7: Execute loaders & display cleaning log / summary
    add_code("""# Execute data ingestion and display audit logs
LOADERS = {"German": load_german, "Taiwan": load_taiwan}
DATA = {ds: LOADERS[ds]() for ds in DATASETS}

print("================== DATA CLEANING AUDIT LOG ==================")
clean_df = pd.DataFrame(CLEAN_LOG, columns=["dataset", "step", "rows/values affected"])
display(clean_df)

print("\\n================== DATASET SUMMARY ==================")
summary_rows = []
for ds, (X, y, cat, num) in DATA.items():
    summary_rows.append({
        "dataset": ds,
        "rows": len(X),
        "features": X.shape[1],
        "categorical": len(cat),
        "numeric": len(num),
        "minority_pct": round(100 * y.mean(), 2)
    })
display(pd.DataFrame(summary_rows).set_index("dataset"))

# Compute class prior odds for XGBoost scale_pos_weight
PRIOR_ODDS = {ds: float((1 - DATA[ds][1].mean()) / DATA[ds][1].mean()) for ds in DATA}""")

    # Cell 8: Markdown Pipeline Design Justification
    add_md("""## Stage 2 — Modelling Pipelines: Rationale and Design Choices

In the reviewed paper (Ala'raj et al., JRFM 2024), critical preprocessing choices are left unstated, and standard practices that produce optimistic results are introduced. To evaluate the true impact of methodology, we construct four strictly contained `imblearn.pipeline.Pipeline` variants:

### 1. `paper_style` (Replication of Authors' Implicit Pipeline)
- **Categoricals**: Ordinal-encoded arbitrarily ($0, 1, 2, \\dots$).
- **Numerics**: No scaling or distribution transformation.
- **Resampling**: Standard `SMOTEENN` applied globally across all features.
- **Critique / Flaw**: Imposing arbitrary integer distance metrics on nominal categories ($A121=1, A122=2, \\dots$) causes standard SMOTE and ENN to interpolate points along imaginary linear geometric continuums between unordered categories (e.g. creating synthetic clients with nonsensical blended demographic codes).

### 2. `fixed_none` (Methodological Baseline with Cost-Sensitive Weighting)
- **Categoricals**: Standard `OneHotEncoder(handle_unknown="ignore")`.
- **Numerics**: A custom in-fold `SkewLog(threshold=1.0)` transformer that applies signed log1p ($\text{sign}(x) \cdot \log(1 + |x|)$) exclusively to training features with absolute sample skewness $> 1$, followed by `StandardScaler`.
- **Class Imbalance**: Managed via cost-sensitive learning (`class_weight="balanced"` in LogReg, `class_weight="balanced_subsample"` in RandomForest, and `scale_pos_weight=PRIOR_ODDS` in XGBoost) **without synthetic sample generation**.
- **Advantage**: Preserves genuine decision boundaries and eliminates synthetic noise artifacts.

### 3. `fixed_smotenc` (Categorical-Aware Resampling)
- **Categoricals & Numerics**: SkewLog + StandardScaler on continuous features; categoricals passed as nominal integers to `SMOTENC`.
- **Resampling**: `SMOTENC` uses the Value Difference Metric (VDM) and frequency-based category selection rather than Euclidean interpolation.
- **Post-Resampling**: Nominal categories are one-hot encoded (`pre2`) before passing into the classifier.
- **Advantage**: Prevents fractional/linear distortion of categorical identities while addressing class imbalance.

### 4. `fixed_smoteenn` (Cleaned Preprocessing + SMOTE-ENN)
- **Pre-processing**: Full `fixed_none` pipeline (One-Hot + SkewLog + StandardScaler).
- **Resampling**: `SMOTEENN` executed strictly in-fold on the scaled, one-hot encoded feature matrix.
- **Advantage**: Isolates the exact effect of SMOTE-ENN filtering when scaling and encoding flaws are resolved.""")

    # Cell 9: SkewLog Transformer & Preprocessing Blocks
    add_code("""# Stage 2: Custom In-Fold Transformers and Preprocessors

class SkewLog(BaseEstimator, TransformerMixin):
    \"\"\"
    Applies signed log1p: sign(x) * log1p(|x|) to numeric features whose absolute
    sample skewness exceeds threshold on the training fold only.
    Guarantees no distribution leakage from validation folds.
    \"\"\"
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        sk = stats.skew(X, axis=0, nan_policy="omit")
        self.skewed_ = np.abs(sk) > self.threshold
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if hasattr(self, "skewed_") and self.skewed_.any():
            X[:, self.skewed_] = np.sign(X[:, self.skewed_]) * np.log1p(np.abs(X[:, self.skewed_]))
        return X

def num_block():
    \"\"\"Numeric preprocessing: In-fold SkewLog followed by StandardScaler.\"\"\"
    return ImbPipeline([("skew", SkewLog(threshold=1.0)), ("scale", StandardScaler())])

def pre_fixed(cat, num):
    \"\"\"Clean preprocessing: Scaled & log-transformed numerics + OneHot categoricals.\"\"\"
    return ColumnTransformer([
        ("num", num_block(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
    ], sparse_threshold=0)

def pre_ordinal(cat, num, scale=False):
    \"\"\"Paper-style preprocessing: Ordinal-encoded categoricals, optional scaling.\"\"\"
    return ColumnTransformer([
        ("num", num_block() if scale else "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)
    ], sparse_threshold=0)""")

    # Cell 10: Model Factory & Pipeline Assembly
    add_code("""# Stage 2: Model Instantiation and Pipeline Assembly

def make_model(name, variant, ds):
    \"\"\"Instantiates classifiers with appropriate cost/class-weight adjustments.\"\"\"
    weighted = (variant == "fixed_none")
    if name == "LogReg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if weighted else None,
            random_state=SEED
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample" if weighted else None,
            random_state=SEED,
            n_jobs=1
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=PRIOR_ODDS[ds] if weighted else 1.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
            verbosity=0
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=300,
            random_state=SEED
        )
    raise ValueError(f"Unknown model: {name}")

def make_pipe(ds, model_name, variant):
    \"\"\"
    Builds the specified pipeline variant inside an imblearn Pipeline.
    Guarantees all fitting and resampling occurs strictly in-fold.
    \"\"\"
    _, _, cat, num = DATA[ds]
    clf = make_model(model_name, variant, ds)
    n_num, n_cat = len(num), len(cat)
    
    if variant == "paper_style":
        steps = [
            ("pre", pre_ordinal(cat, num, scale=False)),
            ("res", SMOTEENN(random_state=SEED)),
            ("clf", clf)
        ]
    elif variant == "fixed_none":
        steps = [
            ("pre", pre_fixed(cat, num)),
            ("clf", clf)
        ]
    elif variant == "fixed_smoteenn":
        steps = [
            ("pre", pre_fixed(cat, num)),
            ("res", SMOTEENN(random_state=SEED)),
            ("clf", clf)
        ]
    elif variant == "fixed_smotenc":
        cat_idx = list(range(n_num, n_num + n_cat))
        steps = [
            ("pre1", pre_ordinal(cat, num, scale=True)),
            ("res", SMOTENC(categorical_features=cat_idx, random_state=SEED)),
            ("pre2", ColumnTransformer(
                [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx)],
                remainder="passthrough", sparse_threshold=0
            )),
            ("clf", clf)
        ]
    else:
        raise ValueError(f"Unknown variant: {variant}")
        
    return ImbPipeline(steps)

print("Pipeline builder validated for variants:", VARIANTS)
print("Supported models:", list(MODELS_USED["German"]))""")

    # Cell 11: Hyperparameter Search Spaces
    add_code("""# Stage 2: Hyperparameter Search Spaces for RandomizedSearchCV

PARAMS = {
    "LogReg": {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__penalty": ["l2"]
    },
    "RandomForest": {
        "clf__n_estimators": randint(100, 400),
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2", 0.5]
    },
    "XGBoost": {
        "clf__n_estimators": randint(100, 400),
        "clf__max_depth": randint(2, 7),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_weight": randint(1, 8)
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "clf__alpha": loguniform(1e-5, 1e-1),
        "clf__learning_rate_init": loguniform(1e-4, 1e-2)
    }
}

print("Hyperparameter search spaces declared for all 4 models:")
for m, p in PARAMS.items():
    print(f"  {m:12s}: {list(p.keys())}")""")

    # Cell 12: Pipeline verification / sanity check
    add_code("""# Pipeline Smoke Test: Verify that all 4 models and 4 variants build cleanly
print("Verifying pipeline construction across datasets, models, and variants...")
for ds in DATASETS:
    for m in MODELS_USED[ds]:
        for v in VARIANTS:
            pipe = make_pipe(ds, m, v)
            assert isinstance(pipe, ImbPipeline)
print("All 32 pipeline configurations successfully constructed without errors!")""")

    target_path = os.path.abspath("notebooks/Phase2_agent_build.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated {target_path} ({len(nb['cells'])} cells)")

if __name__ == "__main__":
    build_notebook()
