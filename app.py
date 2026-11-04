import os, re, json, joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from src.features import prepare_features

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")
RESULTS_PATH = os.path.join(ROOT, "outputs", "model_results.json")

st.set_page_config(page_title="Bike Demand Intelligence", page_icon="🚲", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%);
        padding: 1.6rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(14,165,233,.25);
    }
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1.02rem;}
    .big-card {
        background: linear-gradient(135deg, #ecfeff 0%, #f0fdf4 100%);
        border: 1px solid #99f6e4; border-radius: 18px; padding: 1.4rem 1.6rem; text-align: center;
    }
    .big-card .label {color: #0f766e; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; font-size: .8rem;}
    .big-card .value {font-size: 3.4rem; font-weight: 800; color: #134e4a; line-height: 1.1;}
    .big-card .sub {color: #475569; font-size: .95rem;}
    .pill {display: inline-block; padding: .25rem .8rem; border-radius: 999px; font-weight: 600; font-size: .85rem; margin-top: .5rem;}
    .pill.low {background: #dcfce7; color: #166534;}
    .pill.mid {background: #fef9c3; color: #854d0e;}
    .pill.high {background: #fee2e2; color: #991b1b;}
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px; padding: .8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🚲 Bike Sharing Demand Intelligence</h1>
      <p>Forecast hourly rental demand for smarter operations planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. First run: `python run_analysis.py`")
    st.stop()


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model = artifact["model"]

SEASONS = {1: "❄️ Winter (Jan–Mar)", 2: "🌱 Spring (Apr–Jun)", 3: "☀️ Summer (Jul–Sep)", 4: "🍂 Fall (Oct–Dec)"}
WEATHER = {
    1: "☀️ Clear / partly cloudy",
    2: "🌫️ Mist / cloudy",
    3: "🌦️ Light rain / snow",
    4: "⛈️ Heavy rain / snow",
}
# In the UCI Bike Sharing data, seasons follow calendar quarters (Dec 19 is season 4)
MONTH_TO_SEASON = {m: (m - 1) // 3 + 1 for m in range(1, 13)}


def build_row(dt, hour, season, weather, temp, hum, wind, holiday):
    weekday = int((dt.dayofweek + 1) % 7)  # UCI encoding: Sunday=0
    workingday = int(weekday not in [0, 6] and not holiday)
    return {
        "dteday": dt, "season": season, "yr": 1 if dt.year >= 2012 else 0, "mnth": dt.month,
        "hr": hour, "holiday": int(holiday), "weekday": weekday, "workingday": workingday,
        "weathersit": weather, "temp": temp, "atemp": temp, "hum": hum, "windspeed": wind,
        "casual": 0, "registered": 0, "cnt": 0,
    }


def predict_rows(rows):
    df = pd.DataFrame(rows)
    feats = prepare_features(df)[artifact["features"]]
    return np.maximum(0, model.predict(feats))


def hour_label(h):
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("⚙️ Scenario")
    date = st.date_input("📅 Date", value=pd.Timestamp("2012-12-19").date())
    dt = pd.Timestamp(date)
    hour = st.slider("🕒 Hour of day", 0, 23, 8, format="%d:00")
    holiday = st.checkbox("🎉 Public holiday", value=False)

    st.markdown("---")
    season_default = MONTH_TO_SEASON[dt.month]
    season = st.selectbox(
        "Season", list(SEASONS), index=season_default - 1,
        format_func=lambda x: SEASONS[x], key=f"season_{dt.month}",
        help="Auto-selected from the date; change it if you like.",
    )
    weather = st.selectbox("Weather", list(WEATHER), format_func=lambda x: WEATHER[x])

    st.markdown("---")
    st.caption("Conditions (shown in real-world units)")
    temp_c = st.slider("🌡️ Temperature (°C)", -8, 39, 20)
    hum_pct = st.slider("💧 Humidity (%)", 0, 100, 60)
    wind_kmh = st.slider("💨 Wind speed (km/h)", 0, 67, 13)

    # Convert to the dataset's normalized scale (UCI: temp (t+8)/47, wind /67, hum /100)
    temp = (temp_c + 8) / 47
    humidity = hum_pct / 100
    windspeed = wind_kmh / 67

row = build_row(dt, hour, season, weather, temp, humidity, windspeed, holiday)
prediction = float(predict_rows([row])[0])

# 24-hour curve for the chosen conditions
curve_rows = [build_row(dt, h, season, weather, temp, humidity, windspeed, holiday) for h in range(24)]
curve = predict_rows(curve_rows)
curve_df = pd.DataFrame({"Hour": [hour_label(h) for h in range(24)], "Rentals": curve})
peak_h = int(np.argmax(curve))
quiet_h = int(np.argmin(curve))

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "💬 Ask a question", "🧪 Model quality"])

# ---------- Tab 1: Forecast ----------
with tab1:
    st.caption(f"{dt.strftime('%A, %d %B %Y')} · {hour_label(hour)} · {WEATHER[weather]} · {temp_c}°C")

    ratio = prediction / max(curve.max(), 1)
    if ratio < 0.33:
        level, cls = "Low demand", "low"
    elif ratio < 0.66:
        level, cls = "Moderate demand", "mid"
    else:
        level, cls = "High demand", "high"

    left, right = st.columns([1, 1.6])
    with left:
        st.markdown(
            f"""
            <div class="big-card">
              <div class="label">Estimated rentals at {hour_label(hour)}</div>
              <div class="value">{prediction:,.0f}</div>
              <div class="sub">bikes rented during this hour</div>
              <span class="pill {cls}">{level}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Peak hour", hour_label(peak_h), f"{curve[peak_h]:,.0f} rentals")
        m2.metric("Quietest hour", hour_label(quiet_h), f"{curve[quiet_h]:,.0f} rentals")
        m3, m4 = st.columns(2)
        m3.metric("Daily total (est.)", f"{curve.sum():,.0f}")
        m4.metric("vs. daily average", f"{prediction - curve.mean():+,.0f}")
    with right:
        st.markdown("**Predicted demand across the day**")
        order = [hour_label(h) for h in range(24)]
        base = alt.Chart(curve_df).encode(
            x=alt.X("Hour:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Rentals:Q", title="Rentals per hour"),
            tooltip=["Hour", alt.Tooltip("Rentals:Q", format=",.0f")],
        )
        area = base.mark_area(
            line={"color": "#0ea5e9"}, interpolate="monotone",
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#0ea5e9", offset=0),
                       alt.GradientStop(color="rgba(14,165,233,0.05)", offset=1)],
                x1=1, x2=1, y1=0, y2=1,
            ),
        )
        selected = (
            alt.Chart(curve_df[curve_df["Hour"] == hour_label(hour)])
            .mark_point(size=160, filled=True, color="#f97316")
            .encode(x=alt.X("Hour:N", sort=order), y="Rentals:Q")
        )
        st.altair_chart((area + selected).properties(height=330), use_container_width=True)
        st.caption("Same date and weather, every hour. Adjust the sidebar to compare scenarios.")

    with st.expander("ℹ️ What this forecast does (and doesn't) tell you"):
        st.write(
            "This prototype estimates hourly rental demand from historical patterns. "
            "It does not estimate station-level shortages or simultaneous fleet size because "
            "station inventory, trip duration and failed rental requests are not in the supplied data. "
            "The number is rentals during the selected hour, not bikes needed at once."
        )

# ---------- Tab 2: Natural-language questions ----------
with tab2:
    st.subheader("Ask about demand")
    st.caption("Uses the scenario currently set in the sidebar.")

    chips = st.columns(4)
    examples = ["Demand at 8 AM?", "Demand at 5 PM?", "When is the peak?", "How does weather matter?"]
    for col, ex in zip(chips, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["q_text"] = ex

    question = st.text_input(
        "Or type your own question", key="q_text",
        placeholder="Example: What is the predicted demand at 8 AM?",
    )

    if question:
        q = question.lower()
        match = re.search(r"(\d{1,2})\s*(am|pm)?", q)
        q_hour = None
        if match:
            q_hour = int(match.group(1))
            suffix = match.group(2)
            if suffix == "pm" and q_hour != 12:
                q_hour += 12
            elif suffix == "am" and q_hour == 12:
                q_hour = 0

        if "peak" in q or "highest" in q:
            st.success(
                f"For {dt.strftime('%d %b %Y')} with the current weather settings, demand is expected to peak at "
                f"**{hour_label(peak_h)}** (about **{curve[peak_h]:,.0f}** rentals)."
            )
        elif q_hour is not None and 0 <= q_hour <= 23:
            q_pred = float(curve[q_hour])
            st.success(f"Estimated demand at **{hour_label(q_hour)}** on {dt.strftime('%d %b %Y')}: **{q_pred:,.0f} rentals**.")
            st.caption("It estimates hourly rentals, not simultaneous bike inventory.")
        elif "weather" in q:
            st.info(
                "Weather is a predictor through weather type, temperature, humidity and wind speed. "
                "Change those in the sidebar and watch the forecast update."
            )
        else:
            st.warning(
                "Try asking about a specific hour (e.g. 'demand at 5 PM'), the peak, or weather impact. "
                "This is a lightweight rule-based layer, not a general-purpose LLM."
            )

# ---------- Tab 3: Model evaluation ----------
with tab3:
    st.subheader("How reliable is the model?")
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        st.write(f"Selected model: **{results['selected_model']}**")
        m = results["held_out_test_metrics"]
        a, b, c = st.columns(3)
        a.metric("Test MAE", f"{m['MAE']:.2f}", help="Average absolute error, in rentals per hour")
        b.metric("Test RMSE", f"{m['RMSE']:.2f}", help="Like MAE but penalises large misses more")
        c.metric("Test R²", f"{m['R2']:.3f}", help="Share of demand variation explained (1.0 is perfect)")
        st.caption("Metrics come from a held-out test set the model never saw during training.")
    else:
        st.warning("Run the analysis pipeline to generate model evaluation results.")