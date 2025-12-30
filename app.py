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
# RESPONSIVE STYLES
# ======================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system;
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: clamp(0.75rem, 3vw, 2rem);
    padding-right: clamp(0.75rem, 3vw, 2rem);
}

.card {
    background: #ffffff;
    border-radius: 14px;
    padding: clamp(14px, 3vw, 22px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}

.title {
    font-size: clamp(24px, 6vw, 40px);
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #111827;
}

.subtitle {
    font-size: clamp(14px, 3.5vw, 16px);
    color: #6b7280;
    margin-top: 6px;
}

.metric-label {
    font-size: clamp(12px, 3vw, 13px);
    color: #6b7280;
}

.metric-value {
    font-size: clamp(15px, 4vw, 18px);
    font-weight: 600;
    color: #111827;
}

.section-title {
    font-size: clamp(18px, 4.5vw, 22px);
    font-weight: 600;
    margin-bottom: 6px;
    color: #111827;
}
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
cols = st.columns(3)

metrics = [
    ("Indicator", "Wilder RSI (14)"),
    ("Signal Logic", "First Cross, Non-Overlapping"),
    ("Holding Windows", "5 / 30 / 60 Days"),
]

for col, (label, value) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# DATA INPUT
# ======================================================
st.markdown("""
<div class="card">
    <div class="section-title">Data Input</div>
    <div class="subtitle">
        Upload daily historical price data (CSV).<br><br>
        <strong>Recommended source:</strong> Investing.com — Daily timeframe
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["csv"],
    help="Upload CSV downloaded from Investing.com (Daily)"
)

if uploaded_file is None:
    st.stop()

# ======================================================
# LOAD CSV (INVESTING.COM)
# ======================================================
try:
    df = pd.read_csv(uploaded_file)
except Exception:
    st.error("CSV file could not be read.")
    st.stop()

df.columns = df.columns.str.strip().str.lower()

date_col = next((c for c in df.columns if "date" in c), None)
price_col = next((c for c in df.columns if c in ["price", "close"]), None)

if date_col is None or price_col is None:
    st.error("Invalid CSV format. Use Investing.com daily export.")
    st.stop()

# ======================================================
# CLEAN DATA
# ======================================================
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

df[price_col] = (
    df[price_col]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

df = (
    df.dropna(subset=[date_col, price_col])
    .sort_values(date_col)
    .reset_index(drop=True)
)

# ======================================================
# WILDER RSI (STRICT)
# ======================================================
RSI_PERIOD = 14

delta = df[price_col].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()

df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

# ======================================================
# BACKTEST
# ======================================================
THRESHOLDS = [30, 25, 20]
HOLD_DAYS = [5, 30, 60]

def backtest(level):
    trades = []
    last_exit = -1

    for i in range(1, len(df)):
        crossed = df.loc[i-1, "RSI"] >= level and df.loc[i, "RSI"] < level
        if crossed and i > last_exit:
            entry = df.loc[i, price_col]
            row = {
                "Entry Date": df.loc[i, date_col],
                "Entry Price": round(entry, 2),
                "RSI": round(df.loc[i, "RSI"], 2),
                "Signal": f"RSI < {level}"
            }
            for d in HOLD_DAYS:
                row[f"Return {d}D (%)"] = (
                    round((df.loc[i+d, price_col] / entry - 1) * 100, 2)
                    if i + d < len(df) else np.nan
                )
            trades.append(row)
            last_exit = i + max(HOLD_DAYS)

    return pd.DataFrame(trades)

all_trades = []
summary = []

for t in THRESHOLDS:
    tr = backtest(t)
    all_trades.append(tr)

    s = {"RSI <": t, "Trades": len(tr)}
    for d in HOLD_DAYS:
        col = f"Return {d}D (%)"
        s[f"Avg {d}D %"] = round(tr[col].mean(), 2)
        s[f"WinRate {d}D %"] = round((tr[col] > 0).mean() * 100, 2)
    summary.append(s)

results_df = pd.concat(all_trades, ignore_index=True)
summary_df = pd.DataFrame(summary)

if results_df.empty:
    st.warning("No RSI mean-reversion signals detected.")
    st.stop()

# ======================================================
# OUTPUT
# ======================================================
st.markdown("<div class='section-title'>Summary Statistics</div>", unsafe_allow_html=True)
st.dataframe(summary_df, use_container_width=True)

csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Full Trade-Level Results",
    data=csv,
    file_name="rsi_mean_reversion_trades.csv"
)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Research tool. CSV input recommended from Investing.com.")
