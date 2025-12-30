import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

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
    ("Input Formats", "CSV • XLSX • XLS")
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
# FILE INPUT
# ======================================================
st.markdown("""
<div class="card">
    <div class="section-title">Data Input</div>
    <div class="subtitle">
        Upload daily historical data. Excel files may be edited and reused safely.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.stop()

# ======================================================
# LOAD FILE
# ======================================================
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
except Exception:
    st.error("File could not be read.")
    st.stop()

df.columns = df.columns.str.strip().str.lower()

# ======================================================
# AUTO-DETECT COLUMNS
# ======================================================
date_candidates = [c for c in df.columns if "date" in c or "time" in c]
price_keywords = ["close", "price", "px_last", "last"]
price_candidates = [c for c in df.columns if any(k in c for k in price_keywords)]

if not date_candidates or not price_candidates:
    st.error("Required fields not detected.")
    st.stop()

DATE_COL = date_candidates[0]
PRICE_COL = price_candidates[0]

# ======================================================
# CLEAN DATA
# ======================================================
df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")

df[PRICE_COL] = (
    df[PRICE_COL]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.replace("$", "", regex=False)
)

df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")

df = (
    df.dropna(subset=[DATE_COL, PRICE_COL])
    .sort_values(DATE_COL)
    .reset_index(drop=True)
)

# ======================================================
# WILDER RSI (STRICT)
# ======================================================
RSI_PERIOD = 14
delta = df[PRICE_COL].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()

df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

# ======================================================
# BACKTEST ENGINE (UNCHANGED LOGIC)
# ======================================================
THRESHOLDS = [30, 25, 20]
HOLD_DAYS = [5, 30, 60]

def backtest(threshold):
    trades = []
    last_exit = -1

    for i in range(1, len(df)):
        crossed = df.loc[i-1, "RSI"] >= threshold and df.loc[i, "RSI"] < threshold

        if crossed and i > last_exit:
            entry_price = df.loc[i, PRICE_COL]

            trade = {
                "Entry Date": df.loc[i, DATE_COL],
                "Entry Price": round(entry_price, 2),
                "RSI": round(df.loc[i, "RSI"], 2),
                "Signal": f"RSI < {threshold}"
            }

            for d in HOLD_DAYS:
                trade[f"Return {d}D (%)"] = (
                    round((df.loc[i + d, PRICE_COL] / entry_price - 1) * 100, 2)
                    if i + d < len(df) else np.nan
                )

            trades.append(trade)
            last_exit = i + max(HOLD_DAYS)

    return pd.DataFrame(trades)

# ======================================================
# RUN BACKTEST
# ======================================================
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
# DOWNLOAD FULL TRADES
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
st.caption("Quantitative research application. CSV and Excel inputs supported.")
