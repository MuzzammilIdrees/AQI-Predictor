import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import settings
from src.feature_engineering import get_feature_columns

# Import to check model type
try:
    from src.models import NeuralNetworkRegressor
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    NeuralNetworkRegressor = None


def load_model(model_path=None):
    path = model_path or settings.model_path
    with open(path, "rb") as f:
        return pickle.load(f)


def load_shap(shap_path=None):
    """Load SHAP explainer. Returns None if loading fails (e.g., version mismatch).
    
    Also returns version info if available for compatibility checking.
    """
    try:
        import shap
        import sys
        path = shap_path or settings.shap_path
        with open(path, "rb") as f:
            payload = pickle.load(f)
        
        # Check version compatibility
        saved_shap_version = payload.get("shap_version", "unknown")
        saved_python_version = payload.get("python_version", "unknown")
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # If major Python version differs, explainer may not work
        if saved_python_version != "unknown" and saved_python_version != current_python:
            print(f"Warning: SHAP saved with Python {saved_python_version}, running on {current_python}")
        
        return payload
    except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError, Exception) as e:
        # SHAP explainers may fail to unpickle across Python versions/environments
        print(f"SHAP load failed: {e}")
        return None


def predict(df_features: pd.DataFrame, model=None) -> pd.DataFrame:
    model = model or load_model()
    feature_cols = get_feature_columns(df_features)
    X = df_features[feature_cols].copy()
    # Impute at inference to avoid NaNs reaching models (e.g., Ridge)
    X = X.ffill().bfill()
    for col in X.columns:
        if X[col].isna().all():
            X[col] = 0
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)

    # Handle neural network models that need numpy arrays
    if DEEP_LEARNING_AVAILABLE and NeuralNetworkRegressor is not None and isinstance(model, NeuralNetworkRegressor):
        preds = model.predict(X.values)
    else:
        preds = model.predict(X)
    
    out = df_features.copy()
    out["prediction"] = preds
    out["hazard_level"] = out["prediction"].apply(tag_hazard)
    return out


def tag_hazard(aqi: float) -> str:
    thresholds = settings.hazard_thresholds
    if aqi >= thresholds[2]:
        return "Very Unhealthy"
    if aqi >= thresholds[1]:
        return "Unhealthy"
    if aqi >= thresholds[0]:
        return "Moderate"
    return "Good"


def top_shap_contributors(explainer_payload, df_features: pd.DataFrame, max_features: int = 8) -> List[Tuple[str, float]]:
    explainer = explainer_payload["explainer"]
    feature_cols = explainer_payload["feature_cols"]
    if df_features.empty:
        return []
    
    try:
        X_subset = df_features[feature_cols]
        # For KernelExplainer (used by neural networks), use a small sample
        if hasattr(explainer, 'shap_values'):
            if len(X_subset) > 100:
                X_subset = X_subset.sample(min(100, len(X_subset)), random_state=42)
            shap_vals = explainer.shap_values(X_subset.values)
        else:
            shap_vals = explainer.shap_values(X_subset)
        
        # Handle different output shapes
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0] if len(shap_vals) > 0 else shap_vals
        
        # Handle empty or degenerate outputs defensively
        if getattr(shap_vals, "size", 0) == 0:
            return []
        mean_abs = np.abs(shap_vals).mean(axis=0)
        if np.isnan(mean_abs).all() or len(mean_abs) != len(feature_cols):
            return []
        pairs = list(zip(feature_cols, mean_abs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:max_features]
    except Exception:
        # If SHAP fails for any reason, return empty
        return []


def get_lime_explanation(model, df_features: pd.DataFrame, instance_idx: int = -1, 
                         num_features: int = 8) -> List[Tuple[str, float]]:
    """Generate LIME explanation for a single prediction.
    
    Args:
        model: Trained model with predict method
        df_features: DataFrame with feature columns
        instance_idx: Index of instance to explain (-1 for last)
        num_features: Number of top features to return
        
    Returns:
        List of (feature_name, contribution) tuples
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
        
        feature_cols = get_feature_columns(df_features)
        X = df_features[feature_cols].copy()
        
        # Impute missing values
        X = X.ffill().bfill()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data=X.values,
            feature_names=feature_cols,
            mode='regression',
            verbose=False
        )
        
        # Get instance to explain
        instance = X.iloc[instance_idx].values
        
        # Generate explanation
        exp = explainer.explain_instance(
            instance, 
            model.predict,
            num_features=num_features
        )
        
        # Extract feature contributions
        contributions = exp.as_list()
        # Parse LIME output format: [('feature > value', weight), ...]
        result = []
        for feature_desc, weight in contributions:
            # Extract just the feature name (before any comparison operators)
            for col in feature_cols:
                if col in feature_desc:
                    result.append((col, abs(weight)))
                    break
        
        return result
        
    except ImportError:
        print("LIME not installed. Install with: pip install lime")
        return []
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        return []
