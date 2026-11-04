import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

def run_eda(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Average rentals by hour
    hourly = df.groupby("hr", as_index=False)["cnt"].mean()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=hourly, x="hr", y="cnt", color="#3979A8")
    plt.title("Average Bike Rentals by Hour")
    plt.xlabel("Hour of day")
    plt.ylabel("Average rentals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_demand_by_hour.png"), dpi=160)
    plt.close()

    # Average rentals by weekday
    weekday = df.groupby("weekday", as_index=False)["cnt"].mean()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=weekday, x="weekday", y="cnt", color="#4C9A76")
    plt.title("Average Bike Rentals by Weekday")
    plt.xlabel("Weekday (0=Sunday)")
    plt.ylabel("Average rentals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_demand_by_weekday.png"), dpi=160)
    plt.close()

    # Monthly trend
    monthly = df.groupby(["yr", "mnth"], as_index=False)["cnt"].mean()
    monthly["period"] = monthly["yr"].astype(str) + "-" + monthly["mnth"].astype(str).str.zfill(2)
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x="period", y="cnt", marker="o")
    plt.xticks(rotation=60)
    plt.title("Average Hourly Rentals by Month")
    plt.xlabel("Year-month (dataset year code)")
    plt.ylabel("Average hourly rentals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_demand.png"), dpi=160)
    plt.close()

    # Weather relationship
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(len(df), 2500), random_state=42),
                    x="temp", y="cnt", hue="weathersit", alpha=0.55)
    plt.title("Temperature and Bike Rentals")
    plt.xlabel("Normalized temperature")
    plt.ylabel("Hourly rentals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temperature_vs_demand.png"), dpi=160)
    plt.close()

    # Working day vs non-working day
    wd = df.groupby("workingday", as_index=False)["cnt"].mean()
    plt.figure(figsize=(7, 5))
    sns.barplot(data=wd, x="workingday", y="cnt", color="#B77B45")
    plt.title("Average Rentals: Non-working Day vs Working Day")
    plt.xlabel("Working day (0=no, 1=yes)")
    plt.ylabel("Average hourly rentals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workingday_comparison.png"), dpi=160)
    plt.close()

    # User segments
    segments = df[["casual", "registered"]].mean().rename("average_hourly_rentals").reset_index()
    segments.columns = ["user_type", "average_hourly_rentals"]
    plt.figure(figsize=(7, 5))
    sns.barplot(data=segments, x="user_type", y="average_hourly_rentals", color="#8064A2")
    plt.title("Average Hourly Rentals by User Segment")
    plt.xlabel("User segment")
    plt.ylabel("Average hourly rentals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "user_segments.png"), dpi=160)
    plt.close()

def test_rush_hour_hypothesis(df: pd.DataFrame) -> dict:
    """
    H0: Mean daily average hourly rentals in weekday rush-hour slots
        equals the mean during weekday non-rush-hour slots.
    H1: Rush-hour mean is higher.
    Paired comparison by date controls for day-to-day demand level.
    """
    data = df.copy()
    data["dteday"] = pd.to_datetime(data["dteday"])
    weekday = data[data["workingday"] == 1].copy()
    weekday["rush"] = weekday["hr"].isin([7, 8, 9, 17, 18, 19])

    daily = weekday.groupby(["dteday", "rush"])["cnt"].mean().unstack()
    daily = daily.dropna()
    rush = daily[True]
    nonrush = daily[False]

    statistic, p_two_sided = ttest_rel(rush, nonrush, nan_policy="omit")
    # One-sided alternative: rush-hour mean > non-rush-hour mean.
    p_one_sided = p_two_sided / 2 if statistic > 0 else 1 - p_two_sided / 2
    diff = float((rush - nonrush).mean())

    return {
        "hypothesis": "Weekday rush-hour average hourly rentals are higher than weekday non-rush-hour rentals.",
        "null_hypothesis": "There is no difference in paired daily average hourly rentals.",
        "alternative_hypothesis": "Rush-hour average hourly rentals are higher.",
        "rush_hours": "07:00-09:00 and 17:00-19:00",
        "test": "Paired t-test across working days; one-sided alternative",
        "number_of_days": int(len(daily)),
        "mean_rush_hour_rentals": float(rush.mean()),
        "mean_nonrush_hour_rentals": float(nonrush.mean()),
        "mean_paired_difference": diff,
        "t_statistic": float(statistic),
        "p_value_one_sided": float(p_one_sided),
        "alpha": 0.05,
        "decision": "Reject H0" if p_one_sided < 0.05 and diff > 0 else "Fail to reject H0"
    }
