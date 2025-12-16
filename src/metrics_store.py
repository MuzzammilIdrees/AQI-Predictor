"""Metrics Store - Track model training metrics over time."""
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd


class MetricsStore:
    """Append-only store for model training metrics."""

    COLUMNS = [
        "timestamp",
        "model_name",
        "rmse",
        "mae",
        "r2",
        "sample_count",
        "feature_count",
    ]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        model_name: str,
        metrics: Dict[str, float],
        sample_count: int,
        feature_count: int,
    ) -> None:
        """Append a new training run's metrics."""
        file_exists = self.path.exists()

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "rmse": round(metrics.get("rmse", 0), 4),
            "mae": round(metrics.get("mae", 0), 4),
            "r2": round(metrics.get("r2", 0), 4),
            "sample_count": sample_count,
            "feature_count": feature_count,
        }

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def load(self) -> Optional[pd.DataFrame]:
        """Load all historical metrics."""
        if not self.path.exists():
            return None
        df = pd.read_csv(self.path, parse_dates=["timestamp"])
        return df.sort_values("timestamp", ascending=True)

    def get_latest(self) -> Optional[Dict]:
        """Get the most recent training run metrics."""
        df = self.load()
        if df is None or df.empty:
            return None
        return df.iloc[-1].to_dict()

    def get_trend(self, metric: str = "rmse", n_runs: int = 5) -> Optional[str]:
        """Determine if a metric is improving, degrading, or stable.
        
        Returns: 'improving', 'degrading', or 'stable'
        """
        df = self.load()
        if df is None or len(df) < 2:
            return None

        recent = df.tail(n_runs)[metric].values
        
        # For RMSE and MAE, lower is better
        # For R², higher is better
        if metric in ("rmse", "mae"):
            if recent[-1] < recent[0] * 0.95:  # 5% improvement threshold
                return "improving"
            elif recent[-1] > recent[0] * 1.05:  # 5% degradation threshold
                return "degrading"
        else:  # r2
            if recent[-1] > recent[0] * 1.05:
                return "improving"
            elif recent[-1] < recent[0] * 0.95:
                return "degrading"
        
        return "stable"

    def get_summary_stats(self) -> Optional[Dict]:
        """Get summary statistics across all training runs."""
        df = self.load()
        if df is None or df.empty:
            return None

        return {
            "total_runs": len(df),
            "first_run": df["timestamp"].min().isoformat() if pd.notna(df["timestamp"].min()) else None,
            "last_run": df["timestamp"].max().isoformat() if pd.notna(df["timestamp"].max()) else None,
            "best_rmse": float(df["rmse"].min()),
            "best_r2": float(df["r2"].max()),
            "avg_rmse": float(df["rmse"].mean()),
            "avg_r2": float(df["r2"].mean()),
            "most_common_model": df["model_name"].mode().iloc[0] if not df["model_name"].mode().empty else None,
        }
