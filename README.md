# AQI Predictor (local-ready)

End-to-end pipeline to fetch air-quality forecasts, build features, train a model, and serve 3‑day AQI predictions via Streamlit. Data source uses the free Open-Meteo Air Quality API (no API key). The app is designed for local/container deployment.

## Quickstart
- Python 3.10+ recommended.
- Create a virtual env, then install deps: `pip install -r requirements.txt`.
- Run a fast smoke: `python scripts/backfill.py --city "Delhi" --days 15` then `python scripts/run_training_job.py`.
- Serve locally: `streamlit run app.py`.

## Project layout
- `app.py` — Streamlit UI, live fetch + 3-day forecast, hazard alerts, SHAP explanations.
- `src/data_fetch.py` — pulls hourly air-quality + weather from Open-Meteo.
- `src/feature_engineering.py` — builds time and trend features, targets.
- `src/feature_store.py` — simple parquet-backed feature store (serverless friendly).
- `src/train.py` — trains/evaluates models (RandomForestRegressor baseline + Ridge).
- `src/predict.py` — loads model, makes forecasts, hazard tagging.
- `scripts/backfill.py` — backfills historical data for training.
- `scripts/run_feature_job.py` — one-off feature ingest (for cron/GitHub Actions).
- `scripts/run_training_job.py` — daily training job (for cron/GitHub Actions).
- `.github/workflows/pipeline.yml` — example CI that runs feature + training jobs on schedules.
- `Dockerfile` — container for the app (run locally or on any container host).

## Minimal usage flow
1) Backfill features/targets  
`python scripts/backfill.py --city "Delhi" --days 30`

2) Train + evaluate  
`python scripts/run_training_job.py`

3) Serve the app  
`streamlit run app.py`

## Local deployment
- Run locally: `streamlit run app.py` (after backfill + training).
- Container: `docker build -t aqi-predictor .` then `docker run -p 8501:8501 aqi-predictor`.

## CI/CD (serverless-friendly)
- GitHub Actions workflow (`pipeline.yml`) runs hourly feature job and daily training job. Adjust cron as needed.
- Jobs persist artifacts (`data/features.parquet`, `models/latest_model.pkl`) via workflow artifacts; swap with S3/Hub storage if desired.

## Models
The system trains multiple models and selects the best one:
- **Statistical models**: Ridge Regression, Lasso Regression
- **Tree-based models**: Random Forest, Gradient Boosting
- **Deep learning**: Multi-Layer Perceptron (MLP) Neural Network (requires TensorFlow)

All models are evaluated using RMSE, MAE, and R² metrics. The best model (lowest RMSE) is saved and used for predictions.

## Feature importance
- SHAP explainers are used for model interpretation:
  - TreeExplainer for tree-based models (RF, GBR)
  - LinearExplainer for linear models (Ridge, Lasso)
  - KernelExplainer for neural networks (MLP)
- App shows top features contributing to predictions.

## Alerts
- App raises badges when predicted AQI exceeds standard thresholds (100, 150, 200).

## Notes / Extensibility
- **Deep Learning**: Neural network models require TensorFlow. If TensorFlow is not available, the system gracefully falls back to traditional ML models (RF, Ridge, etc.).
- Data source is forecast-based; if you prefer historical observed AQI, swap `data_fetch.fetch_air_quality()` with an API that returns observations (e.g., OpenAQ) — the feature interface stays the same.
- Feature store uses CSV format for portability; replace with Hopsworks/Vertex AI Feature Store by re-implementing `FeatureStore` methods.

