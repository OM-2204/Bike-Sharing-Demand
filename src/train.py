import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

from src.features import prepare_features, get_feature_columns

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def build_pipeline(model, categorical, numeric):
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric)
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

def chronological_split(data):
    data = data.sort_values("dteday").reset_index(drop=True)
    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]

def train_models(hour_csv, model_dir, output_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raw = pd.read_csv(hour_csv)
    raw["dteday"] = pd.to_datetime(raw["dteday"])
    data = prepare_features(raw)

    train, val, test = chronological_split(data)
    categorical, numeric = get_feature_columns()
    features = categorical + numeric

    # Explicitly exclude instant, dteday, casual, registered and cnt from X.
    X_train, y_train = train[features], train["cnt"]
    X_val, y_val = val[features], val["cnt"]
    X_test, y_test = test[features], test["cnt"]

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=160, max_depth=24, min_samples_leaf=2,
            n_jobs=2, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250, max_depth=7, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=42,
            n_jobs=2
        )
    }

    validation_results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator, categorical, numeric)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        validation_results[name] = metrics(y_val, pred)
        fitted[name] = pipe

    best_name = min(validation_results, key=lambda k: validation_results[k]["MAE"])

    # Refit selected model on train + validation, then evaluate once on held-out test.
    train_val = pd.concat([train, val], ignore_index=True)
    final_model = build_pipeline(candidates[best_name], categorical, numeric)
    final_model.fit(train_val[features], train_val["cnt"])
    test_pred = final_model.predict(X_test)
    test_results = metrics(y_test, test_pred)

    joblib.dump({
        "model": final_model,
        "features": features,
        "categorical": categorical,
        "numeric": numeric,
        "best_model_name": best_name,
        "target": "cnt"
    }, os.path.join(model_dir, "best_model.joblib"))

    comparison = {
        "validation_metrics": validation_results,
        "selected_model": best_name,
        "held_out_test_metrics": test_results,
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
            "strategy": "Chronological 70% / 15% / 15%"
        }
    }
    with open(os.path.join(output_dir, "model_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    pd.DataFrame({
        "actual_cnt": y_test.to_numpy(),
        "predicted_cnt": test_pred,
        "absolute_error": np.abs(y_test.to_numpy() - test_pred)
    }).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    return comparison
