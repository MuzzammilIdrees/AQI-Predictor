"""
Model Performance Tests - Test ML model quality and predictions.
"""
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.predict import load_model, tag_hazard
from src.feature_store import FeatureStore
from src.metrics_store import MetricsStore


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def model():
    """Load the trained model."""
    try:
        return load_model()
    except Exception:
        pytest.skip("Model not available")


@pytest.fixture
def sample_input():
    """Create sample input for predictions."""
    return pd.DataFrame([{
        "pm2_5": 50.0,
        "pm10": 80.0,
        "temperature_2m": 25.0,
        "relative_humidity_2m": 60.0,
        "wind_speed_10m": 5.0,
        "hour": 12,
        "day_of_week": 3,
        "pm2_5_roll_3h": 50.0,
        "pm2_5_roll_6h": 50.0,
        "pm10_roll_3h": 80.0,
        "pm10_roll_6h": 80.0,
        "temp_roll_6h": 25.0,
        "humidity_roll_6h": 60.0,
        "pm2_5_lag_1h": 50.0,
        "pm10_lag_1h": 80.0,
        "us_aqi": 100.0,
        "european_aqi": 75.0,
    }])


# ============================================================================
# Model Loading Tests
# ============================================================================
class TestModelLoading:
    """Test model loading and basic functionality."""
    
    def test_model_exists(self):
        """Test that model file exists."""
        if not settings.model_path.exists():
            pytest.skip("Model file not found")
        assert settings.model_path.exists()
    
    def test_model_loads(self):
        """Test that model loads without error."""
        if not settings.model_path.exists():
            pytest.skip("Model file not found")
        model = load_model()
        assert model is not None
    
    def test_model_has_predict(self, model):
        """Test that model has predict method."""
        assert hasattr(model, 'predict')
        assert callable(model.predict)


# ============================================================================
# Prediction Tests
# ============================================================================
class TestPredictions:
    """Test model predictions."""
    
    def test_prediction_returns_array(self, model, sample_input):
        """Test that prediction returns an array."""
        predictions = model.predict(sample_input)
        assert predictions is not None
        assert len(predictions) == len(sample_input)
    
    def test_prediction_is_numeric(self, model, sample_input):
        """Test that predictions are numeric."""
        predictions = model.predict(sample_input)
        assert np.issubdtype(predictions.dtype, np.number)
    
    def test_prediction_in_valid_range(self, model, sample_input):
        """Test that predictions are in valid AQI range."""
        predictions = model.predict(sample_input)
        # AQI should typically be between 0 and 500
        assert all(predictions >= 0), "Predictions should be non-negative"
        assert all(predictions <= 1000), "Predictions seem unreasonably high"
    
    def test_prediction_deterministic(self, model, sample_input):
        """Test that same input gives same output."""
        pred1 = model.predict(sample_input)
        pred2 = model.predict(sample_input)
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_different_inputs_give_different_outputs(self, model):
        """Test that different inputs produce different predictions."""
        input1 = pd.DataFrame([{
            "pm2_5": 20.0, "pm10": 30.0, "temperature_2m": 25.0,
            "relative_humidity_2m": 50.0, "wind_speed_10m": 5.0,
            "hour": 12, "day_of_week": 3,
            "pm2_5_roll_3h": 20.0, "pm2_5_roll_6h": 20.0,
            "pm10_roll_3h": 30.0, "pm10_roll_6h": 30.0,
            "temp_roll_6h": 25.0, "humidity_roll_6h": 50.0,
            "pm2_5_lag_1h": 20.0, "pm10_lag_1h": 30.0,
            "us_aqi": 50.0, "european_aqi": 40.0,
        }])
        
        input2 = pd.DataFrame([{
            "pm2_5": 150.0, "pm10": 200.0, "temperature_2m": 35.0,
            "relative_humidity_2m": 80.0, "wind_speed_10m": 2.0,
            "hour": 18, "day_of_week": 5,
            "pm2_5_roll_3h": 150.0, "pm2_5_roll_6h": 150.0,
            "pm10_roll_3h": 200.0, "pm10_roll_6h": 200.0,
            "temp_roll_6h": 35.0, "humidity_roll_6h": 80.0,
            "pm2_5_lag_1h": 150.0, "pm10_lag_1h": 200.0,
            "us_aqi": 200.0, "european_aqi": 150.0,
        }])
        
        pred1 = model.predict(input1)[0]
        pred2 = model.predict(input2)[0]
        
        # Higher pollution should generally give higher AQI
        assert pred2 > pred1, "Higher pollution should give higher AQI prediction"


# ============================================================================
# Hazard Level Tests
# ============================================================================
class TestHazardLevels:
    """Test hazard level tagging."""
    
    def test_hazard_good(self):
        """Test good AQI level."""
        level = tag_hazard(30)
        assert "good" in level.lower() or level == "Good"
    
    def test_hazard_moderate(self):
        """Test moderate AQI level."""
        level = tag_hazard(75)
        assert "moderate" in level.lower()
    
    def test_hazard_unhealthy(self):
        """Test unhealthy AQI level."""
        level = tag_hazard(175)
        assert "unhealthy" in level.lower()
    
    def test_hazard_hazardous(self):
        """Test hazardous AQI level."""
        level = tag_hazard(350)
        assert "hazard" in level.lower() or "very" in level.lower()


# ============================================================================
# Metrics Tests
# ============================================================================
class TestMetrics:
    """Test metrics storage and retrieval."""
    
    def test_metrics_store_creation(self, tmp_path):
        """Test that metrics store can be created."""
        store = MetricsStore(tmp_path / "test_metrics.csv")
        assert store is not None
    
    def test_metrics_logging(self, tmp_path):
        """Test logging metrics."""
        store = MetricsStore(tmp_path / "test_metrics.csv")
        store.log(
            model_name="test_model",
            metrics={"rmse": 10.5, "mae": 8.0, "r2": 0.85},
            sample_count=100,
            feature_count=15
        )
        
        df = store.load()
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["model_name"] == "test_model"
    
    def test_metrics_accumulation(self, tmp_path):
        """Test that metrics accumulate across logs."""
        store = MetricsStore(tmp_path / "test_metrics.csv")
        
        for i in range(3):
            store.log(
                model_name=f"model_{i}",
                metrics={"rmse": 10 + i, "mae": 8 + i, "r2": 0.85},
                sample_count=100,
                feature_count=15
            )
        
        df = store.load()
        assert len(df) == 3


# ============================================================================
# Performance Threshold Tests
# ============================================================================
class TestPerformanceThresholds:
    """Test that model meets performance thresholds."""
    
    def test_r2_threshold(self):
        """Test that model R² meets minimum threshold."""
        store = MetricsStore(settings.metrics_history_path)
        df = store.load()
        
        if df is None or df.empty:
            pytest.skip("No metrics history available")
        
        latest_r2 = df.iloc[-1]["r2"]
        min_r2 = 0.5  # Minimum acceptable R²
        
        assert latest_r2 >= min_r2, f"R² ({latest_r2:.3f}) below threshold ({min_r2})"
    
    def test_rmse_threshold(self):
        """Test that model RMSE is within acceptable range."""
        store = MetricsStore(settings.metrics_history_path)
        df = store.load()
        
        if df is None or df.empty:
            pytest.skip("No metrics history available")
        
        latest_rmse = df.iloc[-1]["rmse"]
        max_rmse = 50.0  # Maximum acceptable RMSE
        
        assert latest_rmse <= max_rmse, f"RMSE ({latest_rmse:.2f}) above threshold ({max_rmse})"


# ============================================================================
# Run Tests
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
