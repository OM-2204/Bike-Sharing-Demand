import numpy as np
import pandas as pd

TARGET = "cnt"

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe predictors from the hourly bike-sharing data."""
    data = df.copy()
    data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce")
    data["hour"] = data["hr"].astype(int)
    data["day_of_month"] = data["dteday"].dt.day
    data["day_of_year"] = data["dteday"].dt.dayofyear
    data["is_weekend"] = data["weekday"].isin([0, 6]).astype(int)
    data["is_rush_hour"] = (
        (data["workingday"] == 1) &
        (data["hr"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hr"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hr"] / 24)
    data["month_sin"] = np.sin(2 * np.pi * data["mnth"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["mnth"] / 12)
    return data

def get_feature_columns():
    categorical = [
        "season", "yr", "mnth", "holiday", "weekday",
        "workingday", "weathersit", "hr", "is_weekend",
        "is_rush_hour"
    ]
    numeric = [
        "temp", "atemp", "hum", "windspeed",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_of_month", "day_of_year"
    ]
    return categorical, numeric
