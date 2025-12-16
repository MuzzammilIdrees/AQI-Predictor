import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["time"].dt.hour
    out["day"] = out["time"].dt.day
    out["month"] = out["time"].dt.month
    out["dayofweek"] = out["time"].dt.dayofweek
    return out


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pm25_roll6"] = out["pm2_5"].rolling(6, min_periods=1).mean()
    out["pm25_roll24"] = out["pm2_5"].rolling(24, min_periods=1).mean()
    out["pm10_roll6"] = out["pm10"].rolling(6, min_periods=1).mean()
    out["pm10_roll24"] = out["pm10"].rolling(24, min_periods=1).mean()
    out["pm25_change_6h"] = out["pm2_5"].diff().rolling(6, min_periods=1).mean()
    out["pm10_change_6h"] = out["pm10"].diff().rolling(6, min_periods=1).mean()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("time").reset_index(drop=True)
    out = add_time_features(out)
    out = add_trend_features(out)
    # Fill pollutant gaps to avoid losing all rows
    for col in ["pm2_5", "pm10", "us_aqi", "european_aqi"]:
        if col in out:
            out[col] = out[col].interpolate().bfill().ffill()

    # Target selection with fallbacks
    out["target_aqi"] = out["us_aqi"].fillna(out["european_aqi"])
    # If AQI is still missing, fallback to scaled pm2_5 as a proxy
    if out["target_aqi"].isna().any() and "pm2_5" in out:
        out.loc[out["target_aqi"].isna(), "target_aqi"] = out.loc[out["target_aqi"].isna(), "pm2_5"]

    # Final numeric fills to eliminate lingering NaNs (e.g., from diff/rolling)
    numeric_cols = out.select_dtypes(include="number").columns
    for col in numeric_cols:
        out[col] = out[col].ffill().bfill()
        if out[col].isna().all():
            out[col] = 0
        else:
            out[col] = out[col].fillna(out[col].median())

    # Ensure target has no NaNs
    if out["target_aqi"].isna().any():
        median_target = out["target_aqi"].median()
        out["target_aqi"] = out["target_aqi"].fillna(median_target if pd.notna(median_target) else 0)

    # Drop rows where target is still missing (should be rare after fills)
    out = out.dropna(subset=["target_aqi"])
    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"time", "city", "target_aqi"}
    return [c for c in df.columns if c not in exclude]

