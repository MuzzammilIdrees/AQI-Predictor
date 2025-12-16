from pathlib import Path
from typing import Optional

import pandas as pd


class FeatureStore:
    def __init__(self, path: Path):
        # Use CSV format for better compatibility (no pyarrow dependency)
        self.path = Path(str(path).replace('.parquet', '.csv'))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, df: pd.DataFrame) -> None:
        existing = None
        if self.path.exists():
            existing = pd.read_csv(self.path, parse_dates=['time'])
        combined = pd.concat([existing, df], ignore_index=True) if existing is not None else df
        combined = combined.drop_duplicates(subset=["time", "city"]).sort_values("time")
        combined.to_csv(self.path, index=False)

    def load(self) -> Optional[pd.DataFrame]:
        if not self.path.exists():
            return None
        return pd.read_csv(self.path, parse_dates=['time'])

