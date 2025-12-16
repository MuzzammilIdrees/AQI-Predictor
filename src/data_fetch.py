import datetime as dt
from typing import Optional, Tuple

import pandas as pd
import requests

from src.config import settings


OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def get_coords(city: str) -> Tuple[float, float]:
    coords = settings.city_coords.get(city)
    if coords:
        return coords
    raise ValueError(f"City '{city}' not configured. Add it in config.city_coords.")


def fetch_air_quality(
    city: str,
    past_days: int = 3,
    forecast_days: int = 4,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
) -> pd.DataFrame:
    """Fetch hourly air-quality and weather data.

    Note: Open-Meteo air quality API supports at most 2 past days. We cap to avoid 400 errors.
    """
    lat, lon = get_coords(city)
    max_past = min(past_days, 2)

    # Fetch air quality data
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10",
            "pm2_5",
            "european_aqi",
            "us_aqi",
        ],
        "past_days": max_past,
        "forecast_days": forecast_days,
    }

    aq_response = requests.get(OPEN_METEO_AQ_URL, params=aq_params, timeout=20)
    aq_response.raise_for_status()
    aq_payload = aq_response.json()

    aq_hourly = aq_payload.get("hourly", {})
    df = pd.DataFrame(aq_hourly)
    if df.empty:
        raise RuntimeError("No data returned from Open-Meteo Air Quality API.")
    
    df["time"] = pd.to_datetime(df["time"])
    
    # Fetch weather data (temperature, humidity, wind)
    try:
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
            ],
            "past_days": max_past,
            "forecast_days": forecast_days,
        }
        
        weather_response = requests.get(OPEN_METEO_WEATHER_URL, params=weather_params, timeout=20)
        weather_response.raise_for_status()
        weather_payload = weather_response.json()
        
        weather_hourly = weather_payload.get("hourly", {})
        weather_df = pd.DataFrame(weather_hourly)
        
        if not weather_df.empty:
            weather_df["time"] = pd.to_datetime(weather_df["time"])
            # Merge weather data with air quality data
            df = df.merge(weather_df, on="time", how="left")
    except Exception as e:
        # If weather fetch fails, continue without it
        print(f"Warning: Could not fetch weather data: {e}")
    
    df["city"] = city
    return df


