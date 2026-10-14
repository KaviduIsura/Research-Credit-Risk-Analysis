"""
Pipeline factory and estimator constructors.
Builds leak-free imbalanced-learn pipelines for paper replication and corrected architectures.
"""

from scipy.stats import loguniform, uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN

from src.config import SEED
from src.preprocessors import pre_fixed, pre_ordinal

def make_model(name, variant, ds, prior_odds, seed=SEED):
    """Instantiate base classifier with cost-sensitive class weights if variant == 'fixed_none'."""
    weighted = (variant == "fixed_none")
    odds = prior_odds.get(ds, 1.0)
    
    if name == "LogReg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if weighted else None,
            random_state=seed
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed,
            n_jobs=1
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=odds if weighted else 1.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=300,
            random_state=seed
        )
    raise ValueError(f"Unknown model name: {name}")

def make_pipe(ds, model_name, variant, data_dict, prior_odds, seed=SEED):
    """
    Constructs an imblearn.pipeline.Pipeline for the given dataset, model, and pipeline style.
    Guarantees strict containment: resampling and preprocessing occur only inside training folds.
    """
    _, _, cat, num = data_dict[ds]
    clf = make_model(model_name, variant, ds, prior_odds, seed=seed)
    n_num, n_cat = len(num), len(cat)

    if variant == "paper_style":
        # Paper replication: unscaled numerics, ordinal-encoded categoricals, SMOTE-ENN
        steps = [
            ("pre", pre_ordinal(cat, num, scale=False)),
            ("res", SMOTEENN(random_state=seed)),
            ("clf", clf)
        ]
    elif variant == "fixed_none":
        # Corrected leak-free pipeline with cost-sensitive weights, no synthetic oversampling
        steps = [
            ("pre", pre_fixed(cat, num)),
            ("clf", clf)
        ]
    elif variant == "fixed_smoteenn":
        # Corrected preprocessing (scaled + one-hot) followed by in-fold SMOTE-ENN
        steps = [
            ("pre", pre_fixed(cat, num)),
            ("res", SMOTEENN(random_state=seed)),
            ("clf", clf)
        ]
    elif variant == "fixed_smotenc":
        # Categorical-aware SMOTE-NC: scaled continuous variables, nominal categorical handling
        cat_idx = list(range(n_num, n_num + n_cat))
        steps = [
            ("pre1", pre_ordinal(cat, num, scale=True)),
            ("res", SMOTENC(categorical_features=cat_idx, random_state=seed)),
            ("pre2", ColumnTransformer(
                [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx)],
                remainder="passthrough", sparse_threshold=0
            )),
            ("clf", clf)
        ]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ImbPipeline(steps)

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
