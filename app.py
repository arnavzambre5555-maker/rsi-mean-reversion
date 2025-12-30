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
# STYLES
# ======================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system;
}
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.04);
}
.title {
    font-size: 40px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.subtitle {
    font-size: 16px;
    color: #6b7280;
}
.metric-label {
    color: #6b7280;
    font-size: 13px;
}
.metric-value {
    font-size: 18px;
    font-weight: 600;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 6px;
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

st.markdown("")

# ======================================================
# METRICS
# ======================================================
c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("Indicator", "Wilder RSI (14)"),
    ("Signal Logic", "First Cross, Non-Overlapping"),
    ("Holding Windows", "5 / 30 / 60 Days"),
    ("Data Source", "Investing.com (Daily)")
]

for col, (label, value) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ======================================================
# DATA INPUT
# ======================================================
st.markdown("""
<div class="card">
    <div class="section-title">Data Input</div>
    <div class="subtitle">
        Upload daily historical data exported from Investing.com
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv", "xlsx"])
if uploaded_file is None:
    st.stop()

# ======================================================
# LOAD DATA (INVESTING.COM ASSUMPTION)
# ======================================================
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception:
    st.error("File could not be read.")
    st.stop()

df.columns = df.columns.str.strip().str.lower()

# ======================================================
# COLUMN DETECTION
# ======================================================
date_col = next((c for c in df.columns if "date" in c), None)
price_col = next((c for c in df.columns if c in ["price", "close"]), None)

if date_col is None or price_col is None:
    st.error("Required columns not found. Use Investing.com daily export.")
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
# BACKTEST (UNCHANGED RESEARCH LOGIC)
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

            trade = {
                "Entry Date": df.loc[i, date_col],
                "Entry Price": round(entry_price, 2),
                "RSI": round(df.loc[i, "RSI"], 2),
                "Signal": f"RSI < {level}"
            }

            for d in HOLD_DAYS:
                trade[f"Return {d}D (%)"] = (
                    round((df.loc[i + d, price_col] / entry_price - 1) * 100, 2)
                    if i + d < len(df) else np.nan
                )

            trades.append(trade)
            last_exit = i + max(HOLD_DAYS)

    return pd.DataFrame(trades)

all_trades = []
summary_rows = []

for t in THRESHOLDS:
    trades = backtest(t)
    all_trades.append(trades)

    row = {"RSI <": t, "Trades": len(trades)}
    for d in HOLD_DAYS:
        col = f"Return {d}D (%)"
        row[f"Avg {d}D %"] = round(trades[col].mean(), 2)
        row[f"WinRate {d}D %"] = round((trades[col] > 0).mean() * 100, 2)

    summary_rows.append(row)

results_df = pd.concat(all_trades, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

if results_df.empty:
    st.warning("No RSI mean-reversion signals detected.")
    st.stop()

# ======================================================
# FRONTEND OUTPUT (SUMMARY ONLY)
# ======================================================
st.markdown("<div class='section-title'>Summary Statistics</div>", unsafe_allow_html=True)
st.dataframe(summary_df, use_container_width=True)

# ======================================================
# DOWNLOAD FULL RESULTS
# ======================================================
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
st.caption("Research tool. Daily data sourced from Investing.com.")
