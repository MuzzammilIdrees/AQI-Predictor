"""
FastAPI Application - AQI Predictor API
Serves real-time AQI predictions via REST endpoints.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data_fetch import fetch_air_quality
from src.feature_engineering import build_features
from src.predict import load_model, predict, tag_hazard

from api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    FeatureInput,
    PredictionResponse,
    BatchPredictionResponse,
    FeaturePredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    CityPrediction,
    PredictionResult,
)

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("aqi-api")

# ============================================================================
# Global State
# ============================================================================
model = None
model_info = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, model_info
    logger.info("Starting AQI Predictor API...")
    
    try:
        model = load_model()
        model_info = {
            "model_name": type(model).__name__,
            "loaded_at": datetime.utcnow(),
        }
        logger.info(f"Model loaded: {model_info['model_name']}")
    except Exception as e:
        logger.warning(f"Model not loaded on startup: {e}")
        model = None
    
    yield
    
    logger.info("Shutting down AQI Predictor API...")


# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(
    title="AQI Predictor API",
    description="""
    ## Air Quality Index Prediction API
    
    This API provides real-time AQI predictions for cities in Pakistan and worldwide.
    
    ### Features:
    - 🌆 **City Predictions**: Get AQI forecasts for any supported city
    - 📊 **Batch Processing**: Predict multiple cities at once
    - 📁 **File Upload**: Upload CSV files for bulk predictions
    - 🔬 **Feature Input**: Direct feature-based predictions
    
    ### Supported Cities (Pakistan):
    Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad, Multan, Peshawar, Quetta, and more.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


# ============================================================================
# Health & Info Endpoints
# ============================================================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "AQI Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.utcnow()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=model_info.get("model_name", "Unknown"),
        model_type=type(model).__name__,
        features_count=17,  # Based on feature engineering
        training_samples=model_info.get("training_samples", 0),
        metrics=model_info.get("metrics", {}),
        last_trained=model_info.get("loaded_at"),
        supported_cities=list(settings.city_coords.keys())
    )


# ============================================================================
# Prediction Endpoints
# ============================================================================
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_city(request: PredictionRequest):
    """
    Get AQI prediction for a single city.
    
    - **city**: Name of the city (e.g., "Lahore", "Karachi")
    - **forecast_hours**: Number of hours to forecast (default: 72)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.city not in settings.city_coords:
        raise HTTPException(
            status_code=400,
            detail=f"City '{request.city}' not supported. Supported cities: {list(settings.city_coords.keys())}"
        )
    
    try:
        # Fetch data and make predictions
        raw = fetch_air_quality(request.city, past_days=1, forecast_days=4)
        features = build_features(raw)
        preds = predict(features)
        
        # Build response
        predictions = []
        for _, row in preds.iterrows():
            predictions.append(PredictionResult(
                timestamp=row['time'],
                predicted_aqi=round(row['prediction'], 2),
                hazard_level=row['hazard_level'],
                confidence_lower=round(row['prediction'] * 0.85, 2),
                confidence_upper=round(row['prediction'] * 1.15, 2),
            ))
        
        # Limit to requested hours
        predictions = predictions[:request.forecast_hours]
        
        current_aqi = predictions[-1].predicted_aqi if predictions else 0
        
        return PredictionResponse(
            success=True,
            data=CityPrediction(
                city=request.city,
                current_aqi=current_aqi,
                current_hazard_level=tag_hazard(current_aqi),
                predictions=predictions,
                model_name=model_info.get("model_name", "Unknown"),
                generated_at=datetime.utcnow()
            ),
            message=f"Generated {len(predictions)} hour forecast for {request.city}"
        )
        
    except Exception as e:
        logger.error(f"Prediction error for {request.city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Get AQI predictions for multiple cities.
    
    - **cities**: List of city names
    - **forecast_hours**: Number of hours to forecast (default: 72)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    errors = []
    
    for city in request.cities:
        if city not in settings.city_coords:
            errors.append(f"City '{city}' not supported")
            continue
        
        try:
            raw = fetch_air_quality(city, past_days=1, forecast_days=4)
            features = build_features(raw)
            preds = predict(features)
            
            predictions = []
            for _, row in preds.head(request.forecast_hours).iterrows():
                predictions.append(PredictionResult(
                    timestamp=row['time'],
                    predicted_aqi=round(row['prediction'], 2),
                    hazard_level=row['hazard_level'],
                    confidence_lower=round(row['prediction'] * 0.85, 2),
                    confidence_upper=round(row['prediction'] * 1.15, 2),
                ))
            
            current_aqi = predictions[-1].predicted_aqi if predictions else 0
            
            results.append(CityPrediction(
                city=city,
                current_aqi=current_aqi,
                current_hazard_level=tag_hazard(current_aqi),
                predictions=predictions,
                model_name=model_info.get("model_name", "Unknown"),
                generated_at=datetime.utcnow()
            ))
            
        except Exception as e:
            errors.append(f"{city}: {str(e)}")
            logger.error(f"Batch prediction error for {city}: {e}")
    
    message = f"Processed {len(results)} cities successfully"
    if errors:
        message += f". Errors: {'; '.join(errors)}"
    
    return BatchPredictionResponse(
        success=True,
        data=results,
        total_cities=len(results),
        message=message
    )


@app.post("/predict/features", response_model=FeaturePredictionResponse, tags=["Predictions"])
async def predict_from_features(features: FeatureInput):
    """
    Make prediction from raw feature values.
    
    Advanced endpoint for users who want to provide their own feature values.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Build feature array
        feature_values = {
            "pm2_5": features.pm2_5,
            "pm10": features.pm10,
            "temperature_2m": features.temperature_2m or 25.0,
            "relative_humidity_2m": features.relative_humidity_2m or 50.0,
            "wind_speed_10m": features.wind_speed_10m or 5.0,
            "hour": features.hour,
            "day_of_week": features.day_of_week,
            # Add rolling features with placeholder values
            "pm2_5_roll_3h": features.pm2_5,
            "pm2_5_roll_6h": features.pm2_5,
            "pm10_roll_3h": features.pm10,
            "pm10_roll_6h": features.pm10,
            "temp_roll_6h": features.temperature_2m or 25.0,
            "humidity_roll_6h": features.relative_humidity_2m or 50.0,
            "pm2_5_lag_1h": features.pm2_5,
            "pm10_lag_1h": features.pm10,
            "us_aqi": features.pm2_5 * 2.0,  # Approximate
            "european_aqi": features.pm2_5 * 1.5,  # Approximate
        }
        
        df = pd.DataFrame([feature_values])
        prediction = float(model.predict(df)[0])
        
        return FeaturePredictionResponse(
            success=True,
            predicted_aqi=round(prediction, 2),
            hazard_level=tag_hazard(prediction),
            model_name=model_info.get("model_name", "Unknown"),
            message="Prediction from features successful"
        )
        
    except Exception as e:
        logger.error(f"Feature prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file", tags=["Predictions"])
async def predict_from_file(file: UploadFile = File(...)):
    """
    Upload a CSV file for batch predictions.
    
    The CSV should have columns: city (or pm2_5, pm10, etc. for direct feature prediction)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        if 'city' in df.columns:
            # City-based predictions
            cities = df['city'].unique().tolist()
            results = []
            
            for city in cities:
                if city in settings.city_coords:
                    try:
                        raw = fetch_air_quality(city, past_days=1, forecast_days=1)
                        features = build_features(raw)
                        preds = predict(features)
                        current_aqi = float(preds['prediction'].iloc[-1])
                        results.append({
                            "city": city,
                            "predicted_aqi": round(current_aqi, 2),
                            "hazard_level": tag_hazard(current_aqi)
                        })
                    except Exception as e:
                        results.append({"city": city, "error": str(e)})
                else:
                    results.append({"city": city, "error": "City not supported"})
            
            return {"success": True, "predictions": results, "total": len(results)}
        
        else:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'city' column"
            )
            
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run with Uvicorn
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
