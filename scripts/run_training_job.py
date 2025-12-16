from src.config import settings
from src.feature_store import FeatureStore
from src.train import train_models


def main():
    store = FeatureStore(settings.feature_store_path)
    df = store.load()
    if df is None or df.empty:
        raise RuntimeError("Feature store is empty. Run backfill or feature job first.")
    result = train_models(df)
    print(f"Trained {result.model_name} | RMSE: {result.metrics['rmse']:.2f} | MAE: {result.metrics['mae']:.2f} | R2: {result.metrics['r2']:.3f}")
    print(f"Saved model to {result.model_path}")


if __name__ == "__main__":
    main()

