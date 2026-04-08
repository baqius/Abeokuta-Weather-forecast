import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths if needed
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH  = "GRU_model.pth"
DATA_FOLDER = "weather_data"
LOOKBACK    = 30
HORIZON     = 7


# ══════════════════════════════════════════════════════════════════════════════
# SEASON HELPER
# ══════════════════════════════════════════════════════════════════════════════

SEASONS = {
    (1,  2):  ("Dry / Harmattan", "☀️"),
    (3,  3):  ("Early Rains",     "🌦️"),
    (4,  7):  ("Peak Wet Season", "🌧️"),
    (8,  9):  ("Late Rains",      "⛅"),
    (10, 10): ("Short Dry Spell", "🌤️"),
    (11, 12): ("Dry Season",      "🔆"),
}

def get_season(month: int):
    for (s, e), (name, icon) in SEASONS.items():
        if s <= month <= e:
            return name, icon
    return "Unknown", "🌡️"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_weather_data(days: int = 60):
    files = sorted(glob.glob(f"{DATA_FOLDER}/*.csv"))
    if not files:
        st.error(f"No CSV files found in `{DATA_FOLDER}/`.")
        return None
    try:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df[["datetime", "temp", "humidity", "precip", "windspeed", "conditions"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["temp"]     = pd.to_numeric(df["temp"], errors="coerce")
        df = df.dropna(subset=["temp"]).set_index("datetime").sort_index()
        return df.tail(days)
    except KeyError as e:
        st.error(f"Missing column in CSV: {e}.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GRU MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, horizon=7):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        B  = x.size(0)
        h0 = x.new_zeros(self.gru.num_layers, B, self.gru.hidden_size)
        out, _ = self.gru(x, h0)
        return self.head(out[:, -1, :])


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: `{MODEL_PATH}`.")
        return None
    model = GRUModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def run_forecast(model, recent_temps: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    scaler.fit(recent_temps.reshape(-1, 1))
    scaled = scaler.transform(recent_temps.reshape(-1, 1))
    x = torch.tensor(scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1), dtype=torch.float32)
    with torch.no_grad():
        preds_scaled = model(x).numpy().flatten()
    return scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()


def f_to_c(f):
    return round((f - 32) * 5 / 9, 1)


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_recommendations(temps_c: list, month: int) -> list:
    avg = np.mean(temps_c)
    mx  = max(temps_c)
    recs = []
    season, _ = get_season(month)

    if mx >= 36:
        recs.append({"icon": "🥵", "title": "Extreme Heat Alert",
            "body": f"Peak forecast reaches {mx}°C. Avoid outdoor exertion 11am–4pm.",
            "colour": "#FF4B4B"})
    elif mx >= 33:
        recs.append({"icon": "🌡️", "title": "High Temperature Advisory",
            "body": f"Temperatures up to {mx}°C expected. Stay hydrated.",
            "colour": "#FF8C00"})

    if avg <= 26:
        recs.append({"icon": "😌", "title": "Pleasant Weather Window",
            "body": f"Average of {avg:.1f}°C — great for outdoor activities.",
            "colour": "#00A86B"})

    if season in ("Peak Wet Season", "Late Rains", "Early Rains"):
        recs.append({"icon": "☔", "title": "Rainy Season Tip",
            "body": "Carry an umbrella. Roads near Ake/Kemta may flood.",
            "colour": "#1E90FF"})
        recs.append({"icon": "🌾", "title": "Farming Advisory",
            "body": "Good for planting cassava, maize, and yam. Watch for waterlogging.",
            "colour": "#228B22"})

    if season in ("Dry / Harmattan", "Dry Season"):
        recs.append({"icon": "🌬️", "title": "Harmattan Precaution",
            "body": "Dry, dusty winds expected. Use a face mask outdoors.",
            "colour": "#D2691E"})
        recs.append({"icon": "💧", "title": "Stay Hydrated",
            "body": "Drink at least 2–3 litres of water daily.",
            "colour": "#1E90FF"})

    if avg >= 30:
        recs.append({"icon": "🏥", "title": "Health Reminder",
            "body": "Heat stress can aggravate hypertension. Keep elderly and children cool.",
            "colour": "#9B59B6"})

    recs.append({"icon": "⚡", "title": "Energy Tip",
        "body": "Charge devices early in the day before peak heat hours.",
        "colour": "#F4C542"})

    recs.append({"icon": "🚗", "title": "Travel Planning",
        "body": "Best travel: 6–9am and 5–7pm to avoid peak heat.",
        "colour": "#17A589"})

    return recs


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Abeokuta 7-Day Forecast", page_icon="🌤️", layout="centered")

st.markdown("""
<style>
    /* Constrain the main content block */
    .block-container {
        max-width: 860px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    .main-title { font-size: 1.7rem; font-weight: 800; color: #1E3A5F; }
    .subtitle   { font-size: 0.9rem; color: #555; margin-top: -8px; }
    .sec-hdr    { font-size: 1.05rem; font-weight: 700; color: #1E3A5F; margin: 14px 0 6px; }
    .rec-card   { border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; border-left: 5px solid; }
    .rec-title  { font-weight: 700; font-size: 0.9rem; }
    .rec-body   { font-size: 0.82rem; margin-top: 3px; color: #333; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-title">🌤️ Abeokuta 7-Day Weather Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Abeokuta, Ogun State, Nigeria</div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        f"<br><div style='text-align:right; color:#777; font-size:0.8rem;'>"
        f"Updated: {datetime.now().strftime('%d %b %Y, %H:%M')}</div>",
        unsafe_allow_html=True)

st.divider()

# Load model & data
model = load_model()
if model is None:
    st.stop()

with st.spinner("Loading weather data…"):
    weather_df = load_weather_data(days=60)

if weather_df is None or len(weather_df) < LOOKBACK:
    st.error(f"Need at least {LOOKBACK} days of data. Check your CSV files.")
    st.stop()

# Run forecast
temps_recent = weather_df["temp"].values.astype(float)
preds_f      = run_forecast(model, temps_recent)
preds_c      = [f_to_c(t) for t in preds_f]

today        = datetime.today().date()
fc_dates     = [today + timedelta(days=i + 1) for i in range(HORIZON)]
day_names    = [d.strftime("%A")    for d in fc_dates]
date_strs    = [d.strftime("%d %b") for d in fc_dates]
season_name, season_icon = get_season(today.month)

# Current conditions
last = weather_df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌡️ Last Temp",   f"{f_to_c(last['temp'])}°C", f"{last['temp']:.1f}°F")
c2.metric("💧 Humidity",    f"{last.get('humidity', 'N/A')}%")
c3.metric("🌬️ Wind Speed",  f"{last.get('windspeed', 'N/A')} km/h")
c4.metric(f"{season_icon} Season", season_name)

st.divider()

# 7-day forecast cards
st.markdown('<div class="sec-hdr">📅 7-Day Temperature Forecast</div>', unsafe_allow_html=True)
cols = st.columns(7)
for i, col in enumerate(cols):
    t = preds_c[i]
    bg, border = ("#FFF0F0","#FF4B4B") if t>=36 else ("#FFF8EC","#FF8C00") if t>=33 else ("#F0FFF4","#00A86B") if t<=26 else ("#F0F4FA","#2C7BE5")
    col.markdown(f"""
        <div style='background:{bg}; border-top:4px solid {border};
                    border-radius:10px; padding:10px 4px; text-align:center;'>
            <div style='font-size:0.72rem; color:#666;'>{date_strs[i]}</div>
            <div style='font-weight:700; font-size:0.85rem; color:#1E3A5F; margin:2px 0;'>{day_names[i][:3]}</div>
            <div style='font-size:1.5rem; font-weight:800; color:#1E3A5F;'>{t}°C</div>
            <div style='font-size:0.7rem; color:#888;'>{preds_f[i]:.1f}°F</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Trend chart
st.markdown('<div class="sec-hdr">📈 Temperature Trend</div>', unsafe_allow_html=True)
chart_df = pd.DataFrame({"Temperature (°C)": preds_c},
                         index=[d.strftime("%a %d %b") for d in fc_dates])
st.line_chart(chart_df, use_container_width=True, height=220)

# Historical context
with st.expander("🗓️ Last 30 days history"):
    hist_df = weather_df[["temp"]].tail(30).copy()
    hist_df["Temp (°C)"] = hist_df["temp"].apply(f_to_c)
    hist_df.index = hist_df.index.strftime("%d %b")
    st.line_chart(hist_df[["Temp (°C)"]], use_container_width=True, height=180)

st.divider()

# Recommendations
st.markdown('<div class="sec-hdr">💡 Recommendations & Advice</div>', unsafe_allow_html=True)
recs = get_recommendations(preds_c, today.month)
rcol1, rcol2 = st.columns(2)
for i, rec in enumerate(recs):
    target = rcol1 if i % 2 == 0 else rcol2
    target.markdown(f"""
        <div class="rec-card" style='border-color:{rec["colour"]}; background:{rec["colour"]}18;'>
            <div class="rec-title">{rec["icon"]} {rec["title"]}</div>
            <div class="rec-body">{rec["body"]}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# Full table
with st.expander("📋 Full forecast table"):
    tbl = pd.DataFrame({
        "Day":        day_names,
        "Date":       date_strs,
        "Temp (°C)":  preds_c,
        "Temp (°F)":  [round(t, 1) for t in preds_f],
        "Feels Like": ["Hot" if t >= 33 else "Warm" if t >= 28 else "Pleasant" for t in preds_c],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)
