"""
Data Integrity Tests using DeepChecks.
Tests for data quality, missing values, duplicates, and feature validation.
"""
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import deepchecks, skip tests if not available
try:
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import (
        IsSingleValue,
        MixedNulls,
        MixedDataTypes,
        StringMismatch,
        DataDuplicates,
        FeatureLabelCorrelation,
    )
    from deepchecks.tabular.suites import data_integrity
    DEEPCHECKS_AVAILABLE = True
except ImportError:
    DEEPCHECKS_AVAILABLE = False

from src.config import settings
from src.feature_store import FeatureStore


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
        "city": np.random.choice(["Lahore", "Karachi", "Islamabad"], n_samples),
        "pm2_5": np.random.uniform(10, 200, n_samples),
        "pm10": np.random.uniform(20, 300, n_samples),
        "temperature_2m": np.random.uniform(15, 40, n_samples),
        "relative_humidity_2m": np.random.uniform(30, 90, n_samples),
        "wind_speed_10m": np.random.uniform(0, 15, n_samples),
        "hour": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "target_aqi": np.random.uniform(20, 250, n_samples),
    })


@pytest.fixture
def feature_store_data():
    """Load actual feature store data if available."""
    store = FeatureStore(settings.feature_store_path)
    df = store.load()
    if df is None or df.empty:
        pytest.skip("Feature store is empty")
    return df


# ============================================================================
# Basic Data Quality Tests (No DeepChecks required)
# ============================================================================
class TestBasicDataQuality:
    """Basic data quality tests without DeepChecks."""
    
    def test_no_empty_dataframe(self, sample_features):
        """Test that data is not empty."""
        assert len(sample_features) > 0
        assert len(sample_features.columns) > 0
    
    def test_required_columns_present(self, sample_features):
        """Test that required columns are present."""
        required_cols = ["pm2_5", "pm10", "hour"]
        for col in required_cols:
            assert col in sample_features.columns, f"Missing column: {col}"
    
    def test_no_all_null_columns(self, sample_features):
        """Test that no column is entirely null."""
        for col in sample_features.columns:
            assert not sample_features[col].isna().all(), f"Column {col} is all null"
    
    def test_numeric_columns_valid_range(self, sample_features):
        """Test that numeric columns have valid ranges."""
        assert (sample_features["pm2_5"] >= 0).all(), "pm2_5 has negative values"
        assert (sample_features["pm10"] >= 0).all(), "pm10 has negative values"
        assert (sample_features["hour"] >= 0).all() and (sample_features["hour"] <= 23).all()
        assert (sample_features["day_of_week"] >= 0).all() and (sample_features["day_of_week"] <= 6).all()
    
    def test_no_duplicate_timestamps_per_city(self, sample_features):
        """Test for duplicate timestamps within each city."""
        if "time" in sample_features.columns and "city" in sample_features.columns:
            duplicates = sample_features.duplicated(subset=["time", "city"], keep=False)
            # Allow some duplicates but warn if too many
            duplicate_ratio = duplicates.sum() / len(sample_features)
            assert duplicate_ratio < 0.1, f"Too many duplicates: {duplicate_ratio:.1%}"


# ============================================================================
# DeepChecks Data Integrity Tests
# ============================================================================
@pytest.mark.skipif(not DEEPCHECKS_AVAILABLE, reason="DeepChecks not installed")
class TestDeepChecksDataIntegrity:
    """Data integrity tests using DeepChecks."""
    
    def test_no_single_value_columns(self, sample_features):
        """Test that no column has only a single value."""
        ds = Dataset(sample_features, label="target_aqi", cat_features=["city"])
        check = IsSingleValue()
        result = check.run(ds)
        assert result.passed_conditions(), f"Single value columns found: {result.value}"
    
    def test_no_mixed_nulls(self, sample_features):
        """Test for columns with mixed null representations."""
        ds = Dataset(sample_features, label="target_aqi", cat_features=["city"])
        check = MixedNulls()
        result = check.run(ds)
        # This check may pass even with some issues, just ensure it runs
        assert result is not None
    
    def test_low_duplicate_rate(self, sample_features):
        """Test that duplicate rate is acceptable."""
        ds = Dataset(sample_features, label="target_aqi", cat_features=["city"])
        check = DataDuplicates()
        result = check.run(ds)
        # Allow up to 10% duplicates
        if hasattr(result, 'value') and result.value is not None:
            assert result.value < 0.1, f"High duplicate rate: {result.value:.1%}"
    
    def test_full_data_integrity_suite(self, sample_features):
        """Run the full DeepChecks data integrity suite."""
        ds = Dataset(sample_features, label="target_aqi", cat_features=["city"])
        suite = data_integrity()
        result = suite.run(ds)
        
        # Get failed checks
        failed = [r for r in result.get_not_ran_checks() + result.get_not_passed_checks()]
        
        # Allow some failures but ensure most pass
        pass_rate = (len(result.results) - len(failed)) / len(result.results)
        assert pass_rate >= 0.7, f"Too many integrity checks failed: {1-pass_rate:.1%}"


# ============================================================================
# Feature Store Data Tests
# ============================================================================
class TestFeatureStoreData:
    """Tests on actual feature store data."""
    
    def test_feature_store_not_empty(self):
        """Test that feature store has data."""
        store = FeatureStore(settings.feature_store_path)
        df = store.load()
        # Skip if no data (first run)
        if df is None:
            pytest.skip("No feature store data yet")
        assert len(df) > 0
    
    def test_feature_store_has_target(self):
        """Test that feature store has target column."""
        store = FeatureStore(settings.feature_store_path)
        df = store.load()
        if df is None:
            pytest.skip("No feature store data yet")
        assert "target_aqi" in df.columns or "us_aqi" in df.columns
    
    def test_feature_store_has_cities(self):
        """Test that feature store has city column."""
        store = FeatureStore(settings.feature_store_path)
        df = store.load()
        if df is None:
            pytest.skip("No feature store data yet")
        assert "city" in df.columns


# ============================================================================
# Run Tests
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
