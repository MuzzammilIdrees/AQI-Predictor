import argparse

from src.config import settings
from src.data_fetch import fetch_air_quality
from src.feature_engineering import build_features
from src.feature_store import FeatureStore


def main():
    parser = argparse.ArgumentParser(description="Hourly feature ingestion")
    parser.add_argument("--city", default=settings.default_city)
    parser.add_argument("--past-days", type=int, default=2)
    parser.add_argument("--forecast-days", type=int, default=4)
    args = parser.parse_args()

    raw = fetch_air_quality(args.city, past_days=args.past_days, forecast_days=args.forecast_days)
    features = build_features(raw)
    store = FeatureStore(settings.feature_store_path)
    store.append(features)
    print(f"Ingested {len(features)} rows for {args.city}")


if __name__ == "__main__":
    main()

