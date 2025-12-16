from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    feature_store_path: Path = Path("data/features.csv")
    model_path: Path = Path("models/latest_model.pkl")
    shap_path: Path = Path("models/latest_shap.pkl")
    metrics_history_path: Path = Path("reports/metrics_history.csv")
    default_city: str = "Delhi"
    # City coordinates for quick use; extend as needed.
    city_coords: dict = None
    hazard_thresholds = [100, 150, 200]

    def __post_init__(self):
        if self.city_coords is None:
            self.city_coords = {
                "Delhi": (28.7041, 77.1025),
                "New York": (40.7128, -74.0060),
                "London": (51.5072, -0.1276),
                "Beijing": (39.9042, 116.4074),
                "Sydney": (-33.8688, 151.2093),
            }


settings = Settings()

