from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    feature_store_path: Path = Path("data/features.csv")
    model_path: Path = Path("models/latest_model.pkl")
    shap_path: Path = Path("models/latest_shap.pkl")
    metrics_history_path: Path = Path("reports/metrics_history.csv")
    default_city: str = "Lahore"
    # City coordinates for quick use; extend as needed.
    city_coords: dict = None
    hazard_thresholds = [100, 150, 200]

    def __post_init__(self):
        if self.city_coords is None:
            self.city_coords = {
                # Pakistan Cities
                "Karachi": (24.8607, 67.0011),
                "Lahore": (31.5497, 74.3436),
                "Islamabad": (33.6844, 73.0479),
                "Rawalpindi": (33.5651, 73.0169),
                "Faisalabad": (31.4504, 73.1350),
                "Multan": (30.1575, 71.5249),
                "Peshawar": (34.0151, 71.5249),
                "Quetta": (30.1798, 66.9750),
                "Sialkot": (32.4945, 74.5229),
                "Gujranwala": (32.1877, 74.1945),
                "Hyderabad": (25.3960, 68.3578),
                "Bahawalpur": (29.3544, 71.6911),
                "Sargodha": (32.0836, 72.6711),
                "Sukkur": (27.7052, 68.8574),
                "Larkana": (27.5570, 68.2028),
                "Sheikhupura": (31.7167, 73.9850),
                "Mirpur Khas": (25.5276, 69.0159),
                "Rahim Yar Khan": (28.4202, 70.2952),
                "Gujrat": (32.5742, 74.0789),
                "Jhang": (31.2781, 72.3317),
                # International Cities
                "Delhi": (28.7041, 77.1025),
                "New York": (40.7128, -74.0060),
                "London": (51.5072, -0.1276),
                "Beijing": (39.9042, 116.4074),
                "Sydney": (-33.8688, 151.2093),
            }


settings = Settings()

