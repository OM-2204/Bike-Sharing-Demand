import os, json
import pandas as pd
from src.data_quality import quality_report
from src.analysis import run_eda, test_rush_hour_hypothesis
from src.train import train_models

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
PLOTS = os.path.join(OUT, "plots")
MODELS = os.path.join(ROOT, "models")
os.makedirs(OUT, exist_ok=True)

hour_path = os.path.join(DATA, "hour.csv")
day_path = os.path.join(DATA, "day.csv")

hour = pd.read_csv(hour_path)
day = pd.read_csv(day_path)

quality = {
    "hour.csv": quality_report(hour),
    "day.csv": quality_report(day)
}
with open(os.path.join(OUT, "data_quality_report.json"), "w") as f:
    json.dump(quality, f, indent=2)

run_eda(hour, PLOTS)

hypothesis = test_rush_hour_hypothesis(hour)
with open(os.path.join(OUT, "hypothesis_test.json"), "w") as f:
    json.dump(hypothesis, f, indent=2)

results = train_models(hour_path, MODELS, OUT)

print("\n=== DATA QUALITY ===")
print(json.dumps(quality, indent=2))
print("\n=== HYPOTHESIS TEST ===")
print(json.dumps(hypothesis, indent=2))
print("\n=== MODEL RESULTS ===")
print(json.dumps(results, indent=2))
print(f"\nOutputs saved in: {OUT}")
