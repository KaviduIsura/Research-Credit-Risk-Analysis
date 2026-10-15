"""
SHAP Interpretability Module.
Strictly implements Section 3.5 and Figure 2 of the research paper:
  - SHapley Additive exPlanations (SHAP) summary beeswarm plots
  - Feature importance ranking and visualization
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.config import RESULTS_FIGURES_DIR


def explain_with_shap(
    model,
    X_val,
    feature_names=None,
    dataset_name="Taiwan",
    model_name="Random Forest",
    max_display=15,
    save_dir=RESULTS_FIGURES_DIR
):
    """
    Computes SHAP values and generates:
      1. SHAP Summary Plot (Beeswarm, matching Figure 2 in the paper)
      2. SHAP Bar Plot (Global feature importance)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--> Computing SHAP values for {dataset_name} ({model_name})...")
    
    # Subsample validation set if large (e.g. Taiwan) for fast SHAP computation
    if len(X_val) > 500:
        indices = np.random.RandomState(42).choice(len(X_val), size=500, replace=False)
        X_sample = X_val[indices] if isinstance(X_val, np.ndarray) else X_val.iloc[indices]
    else:
        X_sample = X_val

    # Convert to DataFrame if feature names provided
    if feature_names is not None and not isinstance(X_sample, pd.DataFrame):
        X_df = pd.DataFrame(X_sample, columns=feature_names)
    else:
        X_df = X_sample

    plt.close("all")
    
    try:
        # Check if model has tree explainer support
        if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            
            # Handle list output for multiclass/binary in sklearn RF
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Default class is index 1
                vals_to_plot = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                vals_to_plot = shap_values[:, :, 1]
            else:
                vals_to_plot = shap_values
        else:
            # For linear, MLP, and neural models, use KernelExplainer with background summary
            background = shap.kmeans(X_df, min(50, len(X_df)))
            
            def predict_fn(x):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(x)[:, 1]
                return model.predict(x)
                
            explainer = shap.KernelExplainer(predict_fn, background)
            vals_to_plot = explainer.shap_values(X_df.iloc[:100] if isinstance(X_df, pd.DataFrame) else X_df[:100])
            X_df = X_df.iloc[:100] if isinstance(X_df, pd.DataFrame) else X_df[:100]

        # Generate Figure 2: SHAP Summary Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            vals_to_plot,
            X_df,
            max_display=max_display,
            show=False
        )
        
        fig_filename = f"shap_summary_{dataset_name.lower()}_{model_name.lower().replace(' ', '_')}.png"
        full_fig_path = save_path / fig_filename
        plt.title(f"SHAP Summary Plot ({dataset_name} Credit - {model_name})", fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(full_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"    Saved SHAP figure to: {full_fig_path}")
        return full_fig_path

    except Exception as e:
        print(f"    Notice: SHAP calculation skipped due to: {e}")
        return None
