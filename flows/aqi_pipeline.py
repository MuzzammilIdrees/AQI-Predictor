"""
Prefect Flow - AQI Pipeline Orchestration
Complete ML pipeline with data ingestion, feature engineering, training, and evaluation.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data_fetch import fetch_air_quality
from src.feature_engineering import build_features
from src.feature_store import FeatureStore
from src.train import train_models
from src.metrics_store import MetricsStore


# ============================================================================
# Notification Helper
# ============================================================================
def send_notification(message: str, status: str = "info"):
    """Send notification via Discord webhook (if configured)."""
    import os
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        return
    
    try:
        import requests
        emoji = "✅" if status == "success" else "❌" if status == "error" else "ℹ️"
        payload = {
            "content": f"{emoji} **AQI Pipeline** - {message}",
            "username": "AQI Bot"
        }
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        print(f"Notification failed: {e}")


# ============================================================================
# Prefect Tasks
# ============================================================================
@task(
    name="Ingest City Data",
    description="Fetch AQI data from Open-Meteo API for a single city",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def ingest_city_data(city: str, past_days: int = 2, forecast_days: int = 4):
    """Fetch air quality data for a single city with retry logic."""
    logger = get_run_logger()
    logger.info(f"Fetching data for {city}...")
    
    try:
        raw = fetch_air_quality(city, past_days=past_days, forecast_days=forecast_days)
        logger.info(f"Fetched {len(raw)} rows for {city}")
        return raw
    except Exception as e:
        logger.error(f"Failed to fetch data for {city}: {e}")
        raise


@task(
    name="Engineer Features",
    description="Transform raw data into ML features",
    retries=2,
    retry_delay_seconds=10
)
def engineer_features(raw_data, city: str):
    """Build features from raw API data."""
    logger = get_run_logger()
    logger.info(f"Engineering features for {city}...")
    
    features = build_features(raw_data)
    logger.info(f"Created {len(features)} feature rows with {len(features.columns)} columns")
    return features


@task(
    name="Store Features",
    description="Append features to the feature store"
)
def store_features(features, store_path: Path):
    """Save features to the feature store."""
    logger = get_run_logger()
    
    store = FeatureStore(store_path)
    store.append(features)
    
    total_df = store.load()
    total_rows = len(total_df) if total_df is not None else 0
    logger.info(f"Feature store now has {total_rows} total rows")
    
    return total_rows


@task(
    name="Load Feature Store",
    description="Load all features from the store"
)
def load_features(store_path: Path):
    """Load features from the feature store."""
    logger = get_run_logger()
    
    store = FeatureStore(store_path)
    df = store.load()
    
    if df is None or df.empty:
        raise ValueError("Feature store is empty. Run data ingestion first.")
    
    logger.info(f"Loaded {len(df)} rows from feature store")
    return df


@task(
    name="Train Model",
    description="Train ML models and select the best one",
    retries=2,
    retry_delay_seconds=60
)
def train_model(features_df):
    """Train models and return the best result."""
    logger = get_run_logger()
    logger.info(f"Training models on {len(features_df)} samples...")
    
    result = train_models(features_df)
    
    logger.info(f"Best model: {result.model_name}")
    logger.info(f"Metrics - RMSE: {result.metrics['rmse']:.4f}, MAE: {result.metrics['mae']:.4f}, R²: {result.metrics['r2']:.4f}")
    
    return result


@task(
    name="Evaluate Model",
    description="Evaluate model performance and check thresholds"
)
def evaluate_model(train_result, min_r2: float = 0.5, max_rmse: float = 50.0):
    """Evaluate if model meets quality thresholds."""
    logger = get_run_logger()
    
    r2 = train_result.metrics['r2']
    rmse = train_result.metrics['rmse']
    
    passed = True
    messages = []
    
    if r2 < min_r2:
        passed = False
        messages.append(f"R² ({r2:.3f}) below threshold ({min_r2})")
    else:
        messages.append(f"R² check passed: {r2:.3f} >= {min_r2}")
    
    if rmse > max_rmse:
        passed = False
        messages.append(f"RMSE ({rmse:.2f}) above threshold ({max_rmse})")
    else:
        messages.append(f"RMSE check passed: {rmse:.2f} <= {max_rmse}")
    
    for msg in messages:
        logger.info(msg)
    
    return {
        "passed": passed,
        "r2": r2,
        "rmse": rmse,
        "messages": messages,
        "model_name": train_result.model_name
    }


@task(
    name="Save Metrics",
    description="Log metrics to history for tracking"
)
def save_metrics(train_result, features_count: int, sample_count: int):
    """Save training metrics to history."""
    logger = get_run_logger()
    
    metrics_store = MetricsStore(settings.metrics_history_path)
    metrics_store.log(
        model_name=train_result.model_name,
        metrics=train_result.metrics,
        sample_count=sample_count,
        feature_count=features_count
    )
    
    logger.info(f"Metrics saved to {settings.metrics_history_path}")
    return True


# ============================================================================
# Prefect Flows
# ============================================================================
@flow(
    name="Data Ingestion Flow",
    description="Fetch AQI data for specified cities"
)
def data_ingestion_flow(
    cities: Optional[List[str]] = None,
    past_days: int = 2,
    forecast_days: int = 4
):
    """
    Ingest AQI data for multiple cities.
    
    Args:
        cities: List of city names. Defaults to Pakistan cities.
        past_days: Days of historical data to fetch
        forecast_days: Days of forecast data to fetch
    """
    logger = get_run_logger()
    
    if cities is None:
        # Default to Pakistan cities
        cities = [
            "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad",
            "Multan", "Peshawar", "Quetta", "Sialkot", "Gujranwala"
        ]
    
    logger.info(f"Starting data ingestion for {len(cities)} cities")
    send_notification(f"Starting data ingestion for {len(cities)} cities")
    
    total_rows = 0
    successful_cities = []
    failed_cities = []
    
    for city in cities:
        try:
            # Fetch and process
            raw = ingest_city_data(city, past_days, forecast_days)
            features = engineer_features(raw, city)
            rows = store_features(features, settings.feature_store_path)
            
            total_rows = rows
            successful_cities.append(city)
            
        except Exception as e:
            logger.error(f"Failed to process {city}: {e}")
            failed_cities.append(city)
    
    result = {
        "total_cities": len(cities),
        "successful": len(successful_cities),
        "failed": len(failed_cities),
        "total_rows": total_rows,
        "failed_cities": failed_cities
    }
    
    status = "success" if not failed_cities else "warning"
    send_notification(
        f"Ingestion complete: {len(successful_cities)}/{len(cities)} cities, {total_rows} total rows",
        status
    )
    
    logger.info(f"Ingestion complete: {result}")
    return result


@flow(
    name="Model Training Flow",
    description="Train and evaluate ML model"
)
def training_flow(min_r2: float = 0.5, max_rmse: float = 50.0):
    """
    Train model on available features.
    
    Args:
        min_r2: Minimum acceptable R² score
        max_rmse: Maximum acceptable RMSE
    """
    logger = get_run_logger()
    logger.info("Starting model training flow")
    send_notification("Starting model training")
    
    try:
        # Load features
        features_df = load_features(settings.feature_store_path)
        
        # Train model
        train_result = train_model(features_df)
        
        # Evaluate
        eval_result = evaluate_model(train_result, min_r2, max_rmse)
        
        # Save metrics
        save_metrics(train_result, len(features_df.columns), len(features_df))
        
        if eval_result["passed"]:
            send_notification(
                f"Training successful! Model: {eval_result['model_name']}, R²: {eval_result['r2']:.3f}",
                "success"
            )
        else:
            send_notification(
                f"Training completed but quality checks failed: {eval_result['messages']}",
                "warning"
            )
        
        return eval_result
        
    except Exception as e:
        logger.error(f"Training flow failed: {e}")
        send_notification(f"Training failed: {e}", "error")
        raise


@flow(
    name="Full AQI Pipeline",
    description="Complete pipeline: ingestion → training → evaluation"
)
def full_pipeline_flow(
    cities: Optional[List[str]] = None,
    past_days: int = 2,
    forecast_days: int = 4,
    min_r2: float = 0.5,
    max_rmse: float = 50.0
):
    """
    Run the complete AQI prediction pipeline.
    
    This flow orchestrates:
    1. Data ingestion for specified cities
    2. Feature engineering and storage
    3. Model training and evaluation
    4. Metrics logging
    """
    logger = get_run_logger()
    logger.info("=" * 50)
    logger.info("STARTING FULL AQI PIPELINE")
    logger.info("=" * 50)
    
    send_notification("🚀 Starting full AQI pipeline")
    
    try:
        # Step 1: Data Ingestion
        ingestion_result = data_ingestion_flow(cities, past_days, forecast_days)
        
        if ingestion_result["successful"] == 0:
            raise ValueError("No cities were successfully processed")
        
        # Step 2: Training
        training_result = training_flow(min_r2, max_rmse)
        
        # Final result
        result = {
            "ingestion": ingestion_result,
            "training": training_result,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        send_notification(
            f"✅ Pipeline complete! {ingestion_result['successful']} cities, "
            f"R²: {training_result['r2']:.3f}, RMSE: {training_result['rmse']:.2f}",
            "success"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        send_notification(f"❌ Pipeline failed: {e}", "error")
        raise


# ============================================================================
# CLI Entry Point
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AQI Pipeline Flows")
    parser.add_argument(
        "--flow",
        choices=["ingest", "train", "full"],
        default="full",
        help="Which flow to run"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Cities to process (default: Pakistan cities)"
    )
    
    args = parser.parse_args()
    
    if args.flow == "ingest":
        data_ingestion_flow(cities=args.cities)
    elif args.flow == "train":
        training_flow()
    else:
        full_pipeline_flow(cities=args.cities)
