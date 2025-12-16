import argparse
import datetime as dt

from src.config import settings
from src.data_fetch import fetch_air_quality
from src.feature_engineering import build_features
from src.feature_store import FeatureStore


def main():
    parser = argparse.ArgumentParser(description="Backfill historical features")
    parser.add_argument("--city", default=settings.default_city)
    parser.add_argument("--days", type=int, default=2, help="Number of past days to backfill (API allows up to 2)")
    args = parser.parse_args()

    # Open-Meteo supports only small historical windows; fetch_air_quality caps this internally.
    raw = fetch_air_quality(args.city, past_days=args.days, forecast_days=1)
    features = build_features(raw)
    store = FeatureStore(settings.feature_store_path)
    store.append(features)
    print(f"Backfilled {len(features)} rows for {args.city} into {settings.feature_store_path}")


if __name__ == "__main__":
    main()

