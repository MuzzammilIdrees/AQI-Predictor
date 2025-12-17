"""
Pydantic schemas for API request/response validation.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Request Schemas
# ============================================================================

class PredictionRequest(BaseModel):
    """Single city prediction request."""
    city: str = Field(..., description="City name for AQI prediction", example="Lahore")
    forecast_hours: int = Field(default=72, ge=1, le=168, description="Hours to forecast (1-168)")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Lahore",
                "forecast_hours": 72
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple cities."""
    cities: List[str] = Field(..., description="List of city names", min_length=1, max_length=25)
    forecast_hours: int = Field(default=72, ge=1, le=168)

    class Config:
        json_schema_extra = {
            "example": {
                "cities": ["Lahore", "Karachi", "Islamabad"],
                "forecast_hours": 48
            }
        }


class FeatureInput(BaseModel):
    """Direct feature input for prediction (advanced users)."""
    pm2_5: float = Field(..., ge=0, description="PM2.5 concentration")
    pm10: float = Field(..., ge=0, description="PM10 concentration")
    temperature_2m: Optional[float] = Field(None, description="Temperature in Celsius")
    relative_humidity_2m: Optional[float] = Field(None, ge=0, le=100, description="Humidity %")
    wind_speed_10m: Optional[float] = Field(None, ge=0, description="Wind speed m/s")
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")

    class Config:
        json_schema_extra = {
            "example": {
                "pm2_5": 45.5,
                "pm10": 78.2,
                "temperature_2m": 25.0,
                "relative_humidity_2m": 65.0,
                "wind_speed_10m": 3.5,
                "hour": 14,
                "day_of_week": 2
            }
        }


# ============================================================================
# Response Schemas
# ============================================================================

class PredictionResult(BaseModel):
    """Single prediction result."""
    timestamp: datetime
    predicted_aqi: float
    hazard_level: str
    confidence_lower: float
    confidence_upper: float


class CityPrediction(BaseModel):
    """Prediction results for a single city."""
    city: str
    current_aqi: float
    current_hazard_level: str
    predictions: List[PredictionResult]
    model_name: str
    generated_at: datetime


class PredictionResponse(BaseModel):
    """API response for prediction request."""
    success: bool
    data: CityPrediction
    message: str = "Prediction successful"


class BatchPredictionResponse(BaseModel):
    """API response for batch prediction request."""
    success: bool
    data: List[CityPrediction]
    total_cities: int
    message: str = "Batch prediction successful"


class FeaturePredictionResponse(BaseModel):
    """API response for direct feature prediction."""
    success: bool
    predicted_aqi: float
    hazard_level: str
    model_name: str
    message: str = "Feature prediction successful"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    features_count: int
    training_samples: int
    metrics: Dict[str, float]
    last_trained: Optional[datetime]
    supported_cities: List[str]


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = False
    error: str
    detail: Optional[str] = None
