import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="RSI Mean Reversion Research",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# CLEAN UI
# ======================================================
st.markdown("""
<style>
[data-testid="stFileUploader"] small { display: none !important; }

html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system;
    background-color: #f8fafc;
}

.block-container { padding: 1rem 2rem; }

.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

.title { font-size: 36px; font-weight: 700; color: #111827; }
.subtitle { font-size: 15px; color: #6b7280; }
.metric-label { font-size: 13px; color: #6b7280; }
.metric-value { font-size: 17px; font-weight: 600; color: #111827; }
.section-title { font-size: 20px; font-weight: 600; color: #111827; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="card">
    <div class="title">RSI Mean Reversion Research Tool</div>
    <div class="subtitle">
        Quantitative analysis of RSI mean reversion on daily price data
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# METRICS
# ======================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='card'><div class='metric-label'>Indicator</div><div class='metric-value'>Wilder RSI (14)</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><div class='metric-label'>Signal Logic</div><div class='metric-value'>First Cross, Non-Overlapping</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><div class='metric-label'>Holding Windows</div><div class='metric-value'>5 / 30 / 60 Days</div></div>", unsafe_allow_html=True)

# ======================================================
# DATA INPUT
# ======================================================
st.markdown("""
<div class="card">
    <div class="section-title">Data Input</div>
    <div class="subtitle">
        Upload daily historical price data (CSV only).<br><br>
        <strong>Recommended source:</strong> Investing.com (Daily timeframe)
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = stl = st.file_uploader("", type=["csv"])
if uploaded_file is None:
    st.stop()

# ======================================================
# LOAD & CLEAN DATA
# ======================================================
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower()

date_col = next(c for c in df.columns if "date" in c)
price_col = next(c for c in df.columns if c in ["price", "close"])

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
df[price_col] = df[price_col].astype(str).str.replace(",", "").astype(float)

df = df.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)

# ======================================================
# WILDER RSI
# ======================================================
delta = df[price_col].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

# ======================================================
# BACKTEST (FIXED — RSI LEVEL INCLUDED)
# ======================================================
THRESHOLDS = [30, 25, 20]
HOLD_DAYS = [5, 30, 60]

def backtest(level):
    trades = []
    last_exit = -1

    for i in range(1, len(df)):
        crossed = df.loc[i-1, "RSI"] >= level and df.loc[i, "RSI"] < level
        if crossed and i > last_exit:
            entry_price = df.loc[i, price_col]
            entry_rsi = round(df.loc[i, "RSI"], 2)

            row = {
                "RSI Level": level,
                "Entry Date": df.loc[i, date_col],
                "Entry Price": round(entry_price, 2),
                "Entry RSI": entry_rsi
            }

            for d in HOLD_DAYS:
                row[f"Return {d}D (%)"] = (
                    round((df.loc[i+d, price_col] / entry_price - 1) * 100, 2)
                    if i + d < len(df) else np.nan
                )

            trades.append(row)
            last_exit = i + max(HOLD_DAYS)

    return pd.DataFrame(trades)

# ======================================================
# RUN BACKTEST
# ======================================================
results = pd.concat([backtest(t) for t in THRESHOLDS], ignore_index=True)

if results.empty:
    st.warning("No RSI mean-reversion signals detected.")
    st.stop()

# ======================================================
# SUMMARY
# ======================================================
summary = []
for t in THRESHOLDS:
    tr = results[results["RSI Level"] == t]
    s = {"RSI <": t, "Trades": len(tr)}
    for d in HOLD_DAYS:
        col = f"Return {d}D (%)"
        s[f"Avg {d}D %"] = round(tr[col].mean(), 2)
        s[f"WinRate {d}D %"] = round((tr[col] > 0).mean() * 100, 2)
    summary.append(s)

st.markdown("<div class='section-title'>Summary Statistics</div>", unsafe_allow_html=True)
st.dataframe(pd.DataFrame(summary), use_container_width=True)

# ======================================================
# EXPORT (NOW INCLUDES RSI LEVEL)
# ======================================================
st.download_button(
    "Download Full Trade-Level Results",
    results.to_csv(index=False).encode("utf-8"),
    file_name="rsi_mean_reversion_trades.csv"
)

st.caption("Research tool. CSV input recommended from Investing.com.")
