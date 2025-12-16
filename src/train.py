import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import settings
from src.feature_engineering import get_feature_columns

# Import deep learning models (optional - graceful degradation if TensorFlow not available)
try:
    from src.models import NeuralNetworkRegressor
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


@dataclass
class TrainResult:
    model_name: str
    metrics: Dict[str, float]
    model_path: Path
    shap_path: Path


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_models(df: pd.DataFrame) -> TrainResult:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["target_aqi"].copy()

    # Impute to ensure no NaNs reach models.
    X = X.ffill().bfill()
    # Column-wise median/zero fallback
    for col in X.columns:
        if X[col].isna().all():
            X[col] = 0
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)

    y = y.ffill().bfill()
    if y.isna().all():
        # All targets missing; bail early with a clear error.
        raise RuntimeError("Target column is entirely NaN; check source AQI fields in feature store.")
    median_target = y.median()
    y = y.fillna(median_target if pd.notna(median_target) else 0)

    # Drop any rows that still contain NaN in features or target after imputation.
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < 10:
        raise RuntimeError(
            f"Not enough samples after cleaning (got {len(X)}). Increase backfill days or add more cities."
        )

    test_size = 0.2 if len(X) >= 20 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42
    )

    # Variety of models: statistical (Ridge, Lasso), tree-based (RF, GBR), and deep learning (MLP)
    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1, random_state=42),
    }
    
    # Add deep learning model if available and enough data
    if DEEP_LEARNING_AVAILABLE and len(X_train) >= 50:
        try:
            mlp = NeuralNetworkRegressor(
                input_dim=len(feature_cols),
                hidden_layers=(64, 32, 16),
                dropout=0.2,
                learning_rate=0.001,
                random_state=42
            )
            candidates["neural_network_mlp"] = mlp
        except Exception as e:
            # Skip neural network if it fails (e.g., TensorFlow not installed or other issues)
            print(f"Warning: Could not initialize neural network: {e}")

    results = []
    for name, model in candidates.items():
        try:
            # Fit model (with special handling for neural networks)
            if isinstance(model, NeuralNetworkRegressor):
                model.fit(X_train.values, y_train.values, epochs=50, batch_size=32, verbose=0)
                preds = model.predict(X_test.values)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            
            metrics = evaluate(y_test.values if hasattr(y_test, 'values') else y_test, preds)
            results.append((name, model, metrics))
            print(f"  {name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
        except Exception as e:
            print(f"  Warning: {name} failed: {e}. Skipping...")
            continue

    best_name, best_model, best_metrics = sorted(results, key=lambda x: x[2]["rmse"])[0]

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.model_path, "wb") as f:
        pickle.dump(best_model, f)

    explainer = _build_explainer(best_model, X_train)
    import sys
    shap_payload = {
        "explainer": explainer, 
        "feature_cols": feature_cols,
        "shap_version": shap.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "model_name": best_name
    }
    with open(settings.shap_path, "wb") as f:
        pickle.dump(shap_payload, f)

    return TrainResult(
        model_name=best_name,
        metrics=best_metrics,
        model_path=settings.model_path,
        shap_path=settings.shap_path,
    )


def _build_explainer(model, X_train):
    """Build SHAP explainer with fallbacks for compatibility."""
    # Tree-based models use TreeExplainer
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        try:
            return shap.TreeExplainer(model)
        except Exception as e:
            print(f"TreeExplainer failed: {e}, falling back to KernelExplainer")
    
    # Neural networks use KernelExplainer (sample-based, slower but works)
    if DEEP_LEARNING_AVAILABLE and isinstance(model, NeuralNetworkRegressor):
        background = X_train.values[:min(100, len(X_train))]
        return shap.KernelExplainer(
            lambda x: model.predict(x).reshape(-1, 1),
            background
        )
    
    # Linear models - try LinearExplainer without deprecated params, fallback to KernelExplainer
    try:
        # Try without the deprecated feature_dependence parameter
        return shap.LinearExplainer(model, X_train)
    except Exception as e:
        print(f"LinearExplainer failed: {e}, falling back to KernelExplainer")
        # Fallback to KernelExplainer which works universally
        background = shap.sample(X_train, min(100, len(X_train)))
        return shap.KernelExplainer(model.predict, background)


