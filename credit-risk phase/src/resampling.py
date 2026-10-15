"""
Data Resampling Modules for Imbalanced Credit Risk Data.
Strictly implements the 4 resampling protocols from Section 3.4 of the research paper:
  1. No Resampling (Baseline)
  2. SMOTE (Section 3.4.1)
  3. ENN (Section 3.4.2)
  4. SMOTE-ENN (Section 3.4.3)
"""

from imblearn.over_sampling import SMOTE
try:
    from imblearn.under_sampling import EditedNearestNeighbors
except ImportError:
    from imblearn.under_sampling import EditedNearestNeighbours as EditedNearestNeighbors

from imblearn.combine import SMOTEENN
from src.config import SEED


def get_resampler(method_name, seed=SEED):
    """
    Returns the appropriate resampling estimator.
    Resampling is strictly applied within training splits to prevent data leakage (Algorithm 1).
    """
    clean_name = method_name.strip()
    
    if clean_name in ["No Resampling", "None", "none"]:
        return None
    elif clean_name == "SMOTE":
        return SMOTE(random_state=seed)
    elif clean_name == "ENN":
        return EditedNearestNeighbors()
    elif clean_name == "SMOTE-ENN":
        return SMOTEENN(random_state=seed)
    else:
        raise ValueError(f"Unknown resampling method: '{method_name}'. Choose from 'No Resampling', 'SMOTE', 'ENN', 'SMOTE-ENN'.")


def apply_resampling(resampler, X, y):
    """
    Applies the resampler to features X and labels y.
    If resampler is None, returns X, y untouched.
    """
    if resampler is None:
        return X, y
    return resampler.fit_resample(X, y)
