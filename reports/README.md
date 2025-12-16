# AQI Predictor - Project Report

## Air Quality Index Prediction System using Machine Learning

**Author:** Muzzammil Idrees  
**Date:** December 2025  
**Repository:** [github.com/MuzzammilIdrees/AQI-Predictor](https://github.com/MuzzammilIdrees/AQI-Predictor)

---

## Executive Summary

This project implements an end-to-end Air Quality Index (AQI) prediction system using a serverless machine learning architecture. The system fetches real-time air quality and weather data, engineers predictive features, trains multiple ML models, and serves 72-hour AQI forecasts through an interactive web dashboard.

**Key Achievements:**
- ✅ Real-time data ingestion from Open-Meteo APIs
- ✅ Automated feature engineering pipeline
- ✅ Multi-model comparison (4+ algorithms)
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Interactive Streamlit dashboard with SHAP explainability
- ✅ Docker containerization for deployment

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Open-Meteo      │    │ Open-Meteo      │                     │
│  │ Air Quality API │    │ Weather API     │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
└───────────┼──────────────────────┼──────────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE PIPELINE                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Data Fetch  │ → │ Feature     │ → │ Feature     │           │
│  │             │   │ Engineering │   │ Store (CSV) │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Model       │ → │ Evaluation  │ → │ Model       │           │
│  │ Training    │   │ (RMSE/MAE)  │   │ Registry    │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   WEB APPLICATION                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Streamlit   │ ← │ Prediction  │ ← │ SHAP        │           │
│  │ Dashboard   │   │ Engine      │   │ Explainer   │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Pipeline

### 2.1 Data Sources

| API | Endpoint | Data Provided |
|-----|----------|---------------|
| Open-Meteo Air Quality | `air-quality-api.open-meteo.com` | PM2.5, PM10, US AQI, European AQI |
| Open-Meteo Weather | `api.open-meteo.com` | Temperature, Humidity, Wind Speed |

### 2.2 Feature Engineering

**Time-based Features:**
- Hour of day (0-23)
- Day of month (1-31)
- Month (1-12)
- Day of week (0-6)

**Derived Features:**
- `pm25_roll6`: 6-hour rolling average of PM2.5
- `pm25_roll24`: 24-hour rolling average of PM2.5
- `pm10_roll6`: 6-hour rolling average of PM10
- `pm10_roll24`: 24-hour rolling average of PM10
- `pm25_change_6h`: 6-hour rate of change in PM2.5
- `pm10_change_6h`: 6-hour rate of change in PM10

### 2.3 Feature Store

The feature store is implemented as a lightweight CSV-based storage system suitable for serverless deployment:

```python
class FeatureStore:
    def append(df): # Append new features
    def load():     # Load all features
```

**Note:** For production scale, this can be upgraded to Hopsworks or Vertex AI Feature Store.

---

## 3. Model Training

### 3.1 Models Evaluated

| Model | Type | Key Parameters |
|-------|------|----------------|
| Random Forest | Tree-based | n_estimators=200, max_depth=12 |
| Gradient Boosting | Tree-based | n_estimators=100, learning_rate=0.1 |
| Ridge Regression | Linear | alpha=1.0 |
| Lasso Regression | Linear | alpha=0.1 |

### 3.2 Evaluation Metrics

All models are evaluated using:
- **RMSE** (Root Mean Square Error) - Primary selection metric
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

### 3.3 Model Selection

The model with the lowest RMSE on the test set is automatically selected and saved to the model registry.

---

## 4. Explainability

### 4.1 SHAP (SHapley Additive exPlanations)

The system uses SHAP values to explain model predictions:

- **TreeExplainer**: For Random Forest and Gradient Boosting models
- **LinearExplainer**: For Ridge and Lasso regression
- **KernelExplainer**: Fallback for any model type

**Visualization:** Interactive bar chart showing top feature contributions in the dashboard.

### 4.2 LIME (Local Interpretable Model-agnostic Explanations)

LIME provides instance-level explanations for individual predictions.

---

## 5. CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# Hourly: Feature ingestion
- cron: "0 * * * *"

# Daily: Model retraining  
- cron: "0 0 * * *"
```

**Pipeline Jobs:**
1. `feature_job`: Fetches data, engineers features, stores in feature store
2. `train_job`: Loads features, trains models, saves best model

---

## 6. Web Application

### 6.1 Dashboard Features

| Tab | Description |
|-----|-------------|
| **Forecast** | 72-hour AQI predictions with interactive Plotly chart |
| **Health Advisory** | Dynamic health recommendations based on AQI level |
| **SHAP Analysis** | Feature importance visualization |
| **Data** | Raw data table with CSV download |

### 6.2 Key UI Components

- **AQI Gauge Meter**: Real-time visual indicator
- **Hazard Alerts**: Warnings for unhealthy predictions
- **Pollutant Charts**: PM2.5, PM10, Temperature, Humidity trends
- **City Selector**: Support for multiple global cities

---

## 7. Deployment

### 7.1 Docker

```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### 7.2 Streamlit Cloud

The application is deployed on Streamlit Cloud with automatic redeployment on git push.

**Live URL:** [Deployed on Streamlit Cloud]

---

## 8. Results & Performance

### Model Performance (Typical Results)

| Metric | Value |
|--------|-------|
| RMSE | ~15-25 AQI units |
| MAE | ~10-20 AQI units |
| R² | ~0.7-0.85 |

*Note: Actual values depend on city and data availability*

---

## 9. Future Improvements

1. **Deep Learning**: Re-enable TensorFlow/PyTorch models for improved accuracy
2. **More Cities**: Expand to additional global cities
3. **Longer Forecasts**: Extend predictions beyond 72 hours
4. **Historical Analysis**: Add historical trend comparison
5. **Alert System**: Email/SMS notifications for hazardous levels

---

## 10. Repository Structure

```
AQI-Predictor/
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── .github/workflows/     # CI/CD pipeline
│   └── pipeline.yml
├── src/
│   ├── config.py          # Settings and city coordinates
│   ├── data_fetch.py      # API data fetching
│   ├── feature_engineering.py  # Feature creation
│   ├── feature_store.py   # Feature storage
│   ├── train.py           # Model training
│   ├── predict.py         # Inference
│   └── models.py          # Neural network definitions
├── scripts/
│   ├── backfill.py        # Historical data backfill
│   ├── run_feature_job.py # Feature pipeline runner
│   └── run_training_job.py # Training pipeline runner
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis
├── data/                  # Feature store data
├── models/                # Trained model artifacts
└── reports/               # Generated reports and plots
```

---

## Conclusion

This project successfully demonstrates a complete ML engineering pipeline for AQI prediction, from data ingestion to interactive visualization. The serverless architecture ensures scalability and low operational overhead, while the automated CI/CD pipeline maintains model freshness with hourly data updates.

---

*Report generated: December 2025*
