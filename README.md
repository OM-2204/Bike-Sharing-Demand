# Bike Sharing Demand Intelligence

**Hourly bike-rental demand analysis, forecasting, and operational
decision support**

[![Live
Demo](https://img.shields.io/badge/Live%20Demo-Render-46a2f1?style=for-the-badge&logo=render)](https://bike-sharing-demand-di0h.onrender.com/)

A machine-learning project that analyzes historical bike-sharing
activity and predicts hourly rental demand. It combines exploratory data
analysis, a statistical hypothesis test, regression modelling, and a
Streamlit interface to help operations teams understand demand patterns
and plan for busy periods.

> **Live application:** https://bike-sharing-demand-di0h.onrender.com/

------------------------------------------------------------------------

## 1. Project Overview

Bike-sharing operators need to make sure bicycles are available when and
where customers want them. Demand changes throughout the day and is
influenced by factors such as time, season, weather, and whether a day
is a working day.

This project uses historical Capital Bikeshare data from Washington,
D.C. (2011--2012) to:

-   Explore hourly and daily rental patterns.
-   Identify demand differences across hours, weekdays, seasons, and
    weather conditions.
-   Test a business hypothesis about weekday rush-hour demand.
-   Train regression models to estimate hourly rental counts.
-   Provide an interactive Streamlit application for exploring
    predictions.

The analysis is intended to support **hourly demand planning**. The
dataset does not contain station-level inventory or location
information, so the model does not determine the exact number of
bicycles required at each station or the total fleet size.

## 2. Business Problem

The business question is:

**How can a bike-sharing operator better anticipate hourly demand so
that operational planning reflects periods of higher and lower usage?**

A reliable demand estimate can help an operations team:

-   Recognize recurring peak-demand periods.
-   Compare expected demand under different weather and calendar
    conditions.
-   Plan staffing, maintenance, and operational readiness.
-   Identify when more detailed station-level monitoring may be needed.

The model predicts rentals per hour; it does not directly predict unmet
demand or simultaneous bicycle requirements.

## 3. Dataset

The project uses the hourly (`hour.csv`) and daily (`day.csv`) files
from the Capital Bikeshare dataset.

  File         Description                    Records
  ------------ ---------------------------- ---------
  `hour.csv`   Hourly rental observations      17,379
  `day.csv`    Daily rental observations          731

**Coverage:** January 2011 to December 2012\
**Geography:** Washington, D.C., United States\
**Primary modelling file:** `hour.csv`

### Key fields

  Field                                Description
  ------------------------------------ ------------------------------------------------
  `dteday`                             Calendar date
  `season`                             Season category
  `yr`, `mnth`, `hr`                   Year, month, and hour
  `holiday`, `weekday`, `workingday`   Calendar indicators
  `weathersit`                         Weather situation category
  `temp`, `atemp`                      Normalized temperature and feeling temperature
  `hum`                                Normalized humidity
  `windspeed`                          Normalized wind speed
  `casual`                             Rentals by casual users
  `registered`                         Rentals by registered users
  `cnt`                                Total rentals; prediction target

The target is `cnt`, the total number of rentals in an hour.

## 4. Data Quality and Preparation

Initial data checks covered missing values, duplicate records, negative
target values, and consistency between the target and its user
components.

The checks performed for the supplied dataset found:

-   No missing cells in the inspected data.
-   No duplicate records in the inspected data.
-   No negative rental-count values.
-   No mismatch between `cnt` and `casual + registered`.

### Feature engineering

Calendar and cyclical features were prepared to help represent recurring
demand patterns, including:

-   Weekend indicator.
-   Rush-hour indicator for working-day morning and evening periods.
-   Day of month and day of year.
-   Sine and cosine transformations for hour and month.

### Leakage prevention

`casual` and `registered` were excluded from the predictors because
their sum defines the target `cnt`. The `instant` index was also
excluded from model features.

Categorical and numerical columns are preprocessed separately using a
scikit-learn pipeline. Categorical variables are imputed and one-hot
encoded; numerical variables are imputed using the median.

## 5. Exploratory Data Analysis

The analysis examines demand patterns across:

-   Hour of day.
-   Weekday and working-day status.
-   Month and season.
-   Temperature and weather conditions.
-   Casual and registered user segments.

Some descriptive patterns observed in the dataset:

-   Demand is elevated during weekday commuting periods, particularly
    around 8 AM and 5--6 PM.
-   Average rentals are substantially lower under the most adverse
    weather category than under clear or partly cloudy conditions.

These are historical associations in the dataset and should not be
interpreted as proof that a particular factor causes demand to change.

## 6. Hypothesis Testing

**Business hypothesis:** Average rental demand during weekday rush hours
is higher than during non-rush hours.

-   **Null hypothesis (H₀):** Mean daily rush-hour demand is equal to
    mean daily non-rush-hour demand.
-   **Alternative hypothesis (H₁):** Mean daily rush-hour demand is
    higher.
-   **Test:** One-sided paired t-test using working days.
-   **Significance level:** 0.05.

### Result

  Measure                                               Result
  ------------------------------ -----------------------------
  Average rush-hour demand                              394.92
  Average non-rush-hour demand                          124.33
  Mean paired difference                                270.59
  Working days analysed                                    499
  p-value                          Approximately 1.23 × 10⁻²⁰⁶

At the 5% significance level, the null hypothesis is rejected. In this
historical sample, weekday rush-hour demand is significantly higher than
non-rush-hour demand.

**Interpretation note:** This is an observational comparison. Time
trends and dependence between observations may affect the assumptions of
a paired t-test; the result should be treated as evidence of a
historical pattern, not a causal conclusion.

## 7. Machine Learning Approach

The task is framed as a supervised regression problem:

-   **Input:** Calendar, time, and weather-related features.
-   **Output:** Estimated number of rentals in an hour.
-   **Primary evaluation metric:** Mean Absolute Error (MAE).

### Models evaluated

1.  Linear Regression --- baseline model.
2.  Random Forest Regressor.
3.  XGBoost Regressor.

A chronological split was used to better reflect forecasting on future
observations:

-   70% training set.
-   15% validation set.
-   15% held-out test set.

The validation set was used for model selection. The selected model was
then evaluated on the held-out test period.

### Validation results

  Model                   MAE     RMSE      R²
  ------------------- ------- -------- -------
  Linear Regression     99.64   124.17   0.692
  Random Forest         46.02    71.94   0.897
  XGBoost               45.73    68.40   0.907

### Final test results --- XGBoost

  Metric        Result
  -------- -----------
  MAE        **42.39**
  RMSE       **66.20**
  R²         **0.905**

**Interpretation:** The model's test MAE of 42.39 means its predictions
differ from actual hourly rental counts by about 42 rentals per hour on
average in absolute terms. The R² of 0.905 indicates that the model
explains approximately 90.5% of the variation in the held-out target
values. R² is not prediction accuracy.

## 8. Streamlit Application

The project includes an interactive Streamlit application.

### Application capabilities

-   Accepts date, hour, season, weather, temperature, humidity, and
    wind-speed inputs.
-   Generates an hourly rental-demand estimate using the trained model.
-   Displays model evaluation metrics.
-   Provides a lightweight question interface for a limited set of
    supported demand questions.

The question interface is rule-based and recognizes selected question
patterns. It is not a general-purpose, LLM-powered conversational
assistant.

**Try the live application:**\
https://bike-sharing-demand-di0h.onrender.com/

The free Render service may take some time to wake up after a period of
inactivity.

## 9. Repository Structure

``` text
Bike-Sharing-Demand/
│
├── app.py                      # Streamlit application
├── run_analysis.py             # End-to-end analysis and training entry point
├── requirements.txt            # Python dependencies
├── README.md
│
├── data/
│   ├── hour.csv                # Hourly bike-sharing data
│   ├── day.csv                 # Daily bike-sharing data
│   └── Readme.txt
│
├── src/
│   ├── __init__.py
│   ├── features.py              # Feature engineering and preprocessing
│   ├── data_quality.py          # Data validation checks
│   ├── analysis.py              # Exploratory analysis and hypothesis test
│   └── train.py                 # Model training and evaluation
│
├── models/
│   └── best_model.joblib        # Saved trained model
│
└── outputs/
    ├── model_results.json       # Model evaluation results
    └── ...                      # Generated plots and prediction outputs
```

The `models/` and `outputs/` artifacts are generated by the analysis
workflow and are required by the deployed application.

## 10. Run Locally

### Requirements

-   Python 3.10 or 3.11 recommended.
-   pip.
-   Git (optional, for cloning the repository).

### Installation

Clone the repository:

``` bash
git clone https://github.com/OM-2204/Bike-Sharing-Demand.git
cd Bike-Sharing-Demand
```

Create and activate a virtual environment:

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

### Generate model and analysis artifacts

Run the analysis workflow from the repository root:

``` bash
python run_analysis.py
```

This workflow performs the analysis and model-training steps and
generates the model and output artifacts used by the application.

### Start the Streamlit app

``` bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually:

``` text
http://localhost:8501
```

If the repository already includes the generated model and result files,
the app can use those artifacts without rerunning training.

## 11. Deployment

The application is deployed on Render as a Python web service.

**Live URL:** https://bike-sharing-demand-di0h.onrender.com/

Typical Render configuration:

**Build command**

``` bash
pip install -r requirements.txt
```

**Start command**

``` bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

The repository must include `app.py`, `requirements.txt`, the `src/`
package, and the saved model and results artifacts expected by the app.

## 12. Business Recommendations

Based on the historical analysis and model results:

1.  Use hourly demand estimates to identify recurring peak periods.
2.  Prioritize operational readiness around weekday commuting hours.
3.  Consider lower-demand periods when planning maintenance activities.
4.  Treat model estimates as decision support and monitor actual demand
    against predictions.
5.  Collect station-level operational data before making
    station-specific redistribution or fleet-purchase decisions.

### Important limitations

The supplied dataset does not include:

-   Station identifiers or geographic coordinates.
-   Real-time or historical bicycle availability by station.
-   Trip duration or bicycle occupancy over time.
-   Failed rental attempts or customers unable to find a bicycle.
-   Rebalancing costs or operational constraints.

Consequently, hourly rental counts cannot be directly translated into
the number of bicycles simultaneously required. The current model
supports hourly demand estimation, not an exact fleet-sizing or
station-rebalancing optimization system.

## 13. Future Improvements

Potential extensions include:

-   Integrating station-level trip and inventory data.
-   Estimating unmet demand using failed rental or availability records.
-   Testing additional time-series forecasting approaches.
-   Monitoring prediction error as new observations become available.
-   Developing a station-level rebalancing optimization model once
    operational data is available.

## 14. Technology Stack

-   **Language:** Python
-   **Data processing:** Pandas, NumPy
-   **Visualization:** Matplotlib, Seaborn
-   **Statistical testing:** SciPy
-   **Machine learning:** Scikit-learn, XGBoost
-   **Model persistence:** Joblib
-   **Application:** Streamlit
-   **Deployment:** Render
-   **Version control:** Git, GitHub

## 15. Author

**Om Prakash Mallick**

Project: **Bike Sharing Demand Intelligence**\
Focus areas: Data Analytics · Statistical Hypothesis Testing · Machine
Learning · Demand Forecasting · Streamlit Deployment

------------------------------------------------------------------------

*This project was developed as a case-study exercise. All reported
findings and model metrics refer to the supplied historical dataset and
the described evaluation procedure.*
