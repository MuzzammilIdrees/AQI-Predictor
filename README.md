# 🌬️ AQI Predictor

[![CI/CD Pipeline](https://github.com/MuzzammilIdrees/AQI-Predictor/actions/workflows/pipeline.yml/badge.svg)](https://github.com/MuzzammilIdrees/AQI-Predictor/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

A full-stack ML Engineering system for Air Quality Index (AQI) prediction, featuring real-time predictions, automated pipelines, and production-grade deployment.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Prefect Orchestration](#-prefect-orchestration)
- [Testing](#-testing)
- [Docker Deployment](#-docker-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Supported Cities](#-supported-cities)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **FastAPI REST API** | Real-time predictions via REST endpoints |
| **Streamlit Dashboard** | Interactive visualization and forecasting |
| **Prefect Orchestration** | Automated ML pipeline with retry logic |
| **DeepChecks Testing** | Automated ML model validation |
| **Docker Compose** | Multi-service containerization |
| **CI/CD Pipeline** | Automated testing, training, and deployment |
| **20 Pakistan Cities** | Full coverage of major Pakistani cities |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD                      │
│  ┌─────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌───────┐ │
│  │  Lint   │→ │  Test  │→ │ Ingest │→ │ Train  │→ │ Build │ │
│  └─────────┘  └────────┘  └────────┘  └────────┘  └───────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   FastAPI API   │  │    Streamlit    │  │   Prefect   │  │
│  │    :8000        │  │     :8501       │  │   Worker    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/MuzzammilIdrees/AQI-Predictor.git
cd AQI-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Services

```bash
# Run FastAPI server
uvicorn api.main:app --reload --port 8000

# Run Streamlit dashboard (in another terminal)
streamlit run app.py

# Or use Docker Compose
docker-compose up
```

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model metadata |
| `POST` | `/predict` | Single city prediction |
| `POST` | `/predict/batch` | Multiple cities prediction |
| `POST` | `/predict/features` | Direct feature prediction |
| `POST` | `/predict/file` | CSV file upload |

### Example Request

```bash
# Single city prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Lahore", "forecast_hours": 24}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"cities": ["Lahore", "Karachi", "Islamabad"], "forecast_hours": 48}'
```

### Interactive Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔄 Prefect Orchestration

### Available Flows

```python
from flows.aqi_pipeline import full_pipeline_flow, data_ingestion_flow, training_flow

# Run complete pipeline
full_pipeline_flow()

# Run individual flows
data_ingestion_flow(cities=["Lahore", "Karachi"])
training_flow()
```

### CLI Usage

```bash
# Run full pipeline
python -m flows.aqi_pipeline --flow full

# Run data ingestion only
python -m flows.aqi_pipeline --flow ingest --cities Lahore Karachi

# Run training only
python -m flows.aqi_pipeline --flow train
```

### Flow Features
- ✅ Automatic retries (3 attempts)
- ✅ Error handling
- ✅ Discord/Slack notifications
- ✅ Task caching (1 hour)

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=html

# Run specific test files
pytest tests/test_api.py -v
pytest tests/test_model.py -v
pytest tests/test_data_integrity.py -v
```

### Test Categories

| Test File | Description |
|-----------|-------------|
| `test_api.py` | API endpoint tests |
| `test_model.py` | Model prediction tests |
| `test_data_integrity.py` | DeepChecks data validation |

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up api
docker-compose up dashboard

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI REST API |
| `dashboard` | 8501 | Streamlit Dashboard |
| `prefect-worker` | - | Prefect flow runner |

## ⚙️ CI/CD Pipeline

The GitHub Actions pipeline includes:

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   Lint   │ → │   Test   │ → │  Ingest  │ → │  Train   │ → │  Build   │
│(flake8)  │   │(pytest)  │   │(20 cities)│   │(ML tests)│   │(Docker)  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### Triggers
- 🕐 Scheduled: Every 6 hours
- 🔄 Manual: workflow_dispatch
- 📝 On push to main
- 🔀 On pull request

## 🌍 Supported Cities

### Pakistan (20 cities)
| | | | |
|---|---|---|---|
| Karachi | Lahore | Islamabad | Rawalpindi |
| Faisalabad | Multan | Peshawar | Quetta |
| Sialkot | Gujranwala | Hyderabad | Bahawalpur |
| Sargodha | Sukkur | Larkana | Sheikhupura |
| Mirpur Khas | Rahim Yar Khan | Gujrat | Jhang |

### International
Delhi, New York, London, Beijing, Sydney

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| RMSE | < 20 |
| MAE | < 15 |
| R² Score | > 0.85 |

## 📁 Project Structure

```
aqi-predictor/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   └── schemas.py         # Pydantic models
├── flows/                  # Prefect orchestration
│   └── aqi_pipeline.py    # ML pipeline flows
├── src/                    # Core ML logic
│   ├── config.py          # Configuration
│   ├── data_fetch.py      # Data fetching
│   ├── feature_engineering.py
│   ├── train.py           # Model training
│   └── predict.py         # Predictions
├── tests/                  # Test suite
│   ├── test_api.py
│   ├── test_model.py
│   └── test_data_integrity.py
├── .github/workflows/      # CI/CD
├── docker-compose.yml      # Container orchestration
├── Dockerfile              # FastAPI container
├── Dockerfile.streamlit    # Dashboard container
├── app.py                  # Streamlit app
└── requirements.txt        # Dependencies
```

## 📝 License

MIT License

## 👤 Author

**Muzzammil Idrees**

- GitHub: [@MuzzammilIdrees](https://github.com/MuzzammilIdrees)
