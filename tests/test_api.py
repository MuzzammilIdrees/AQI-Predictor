"""
API Tests - Test FastAPI endpoints.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


client = TestClient(app)


# ============================================================================
# Health & Info Endpoint Tests
# ============================================================================
class TestHealthEndpoints:
    """Test health and info endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "AQI Predictor API"
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert "timestamp" in data
    
    def test_docs_accessible(self):
        """Test that API docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200


# ============================================================================
# Prediction Endpoint Tests
# ============================================================================
class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_predict_valid_city(self):
        """Test prediction for a valid city."""
        response = client.post(
            "/predict",
            json={"city": "Lahore", "forecast_hours": 24}
        )
        # May fail if model not loaded, but should return valid response
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            assert data["data"]["city"] == "Lahore"
    
    def test_predict_invalid_city(self):
        """Test prediction for an invalid city returns error."""
        response = client.post(
            "/predict",
            json={"city": "InvalidCity123", "forecast_hours": 24}
        )
        assert response.status_code in [400, 503]
    
    def test_predict_invalid_forecast_hours(self):
        """Test that invalid forecast hours are rejected."""
        response = client.post(
            "/predict",
            json={"city": "Lahore", "forecast_hours": 1000}  # > 168 max
        )
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict(self):
        """Test batch prediction endpoint."""
        response = client.post(
            "/predict/batch",
            json={
                "cities": ["Lahore", "Karachi"],
                "forecast_hours": 24
            }
        )
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            assert "total_cities" in data
    
    def test_feature_predict(self):
        """Test direct feature prediction endpoint."""
        response = client.post(
            "/predict/features",
            json={
                "pm2_5": 45.5,
                "pm10": 78.2,
                "temperature_2m": 25.0,
                "relative_humidity_2m": 65.0,
                "wind_speed_10m": 3.5,
                "hour": 14,
                "day_of_week": 2
            }
        )
        assert response.status_code in [200, 503]


# ============================================================================
# Input Validation Tests
# ============================================================================
class TestInputValidation:
    """Test input validation."""
    
    def test_missing_required_field(self):
        """Test that missing required fields are caught."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_negative_pm_values(self):
        """Test that negative PM values are rejected."""
        response = client.post(
            "/predict/features",
            json={
                "pm2_5": -10,  # Negative - should fail
                "pm10": 50,
                "hour": 12,
                "day_of_week": 1
            }
        )
        assert response.status_code == 422
    
    def test_invalid_hour(self):
        """Test that invalid hour values are rejected."""
        response = client.post(
            "/predict/features",
            json={
                "pm2_5": 50,
                "pm10": 50,
                "hour": 25,  # Invalid - should be 0-23
                "day_of_week": 1
            }
        )
        assert response.status_code == 422


# ============================================================================
# File Upload Tests
# ============================================================================
class TestFileUpload:
    """Test file upload endpoint."""
    
    def test_upload_non_csv(self):
        """Test that non-CSV files are rejected."""
        response = client.post(
            "/predict/file",
            files={"file": ("test.txt", b"some content", "text/plain")}
        )
        assert response.status_code == 400
    
    def test_upload_valid_csv(self):
        """Test uploading a valid CSV file."""
        csv_content = b"city\nLahore\nKarachi"
        response = client.post(
            "/predict/file",
            files={"file": ("cities.csv", csv_content, "text/csv")}
        )
        assert response.status_code in [200, 503]


# ============================================================================
# Run Tests
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
