import pandas as pd

def quality_report(df: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_min": str(pd.to_datetime(df["dteday"]).min().date()),
        "date_max": str(pd.to_datetime(df["dteday"]).max().date()),
        "target_min": float(df["cnt"].min()),
        "target_max": float(df["cnt"].max()),
        "negative_target_rows": int((df["cnt"] < 0).sum()),
        "casual_registered_mismatch_rows": int(
            ((df["casual"] + df["registered"]) != df["cnt"]).sum()
        )
    }
    return report
