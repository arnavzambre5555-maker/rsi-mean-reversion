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
# GLOBAL STYLES
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

for col, label, value in [
    (c1, "Indicator", "Wilder RSI (14)"),
    (c2, "Signal Logic", "First Cross Only"),
    (c3, "Holding Windows", "30 / 60 Days"),
    (c4, "Input Formats", "CSV • XLSX • XLS")
]:
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
        Upload historical daily data. Edited Excel files are fully supported.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.stop()

# ======================================================
# LOAD DATA
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

# ======================================================
# NORMALIZATION
# ======================================================
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "")
)

date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
price_col = next((c for c in df.columns if any(k in c for k in ["close", "price", "last"])), None)

if date_col is None or price_col is None:
    st.error("Required fields not detected.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

df[price_col] = (
    df[price_col]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.replace("$", "", regex=False)
)

df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

df = (
    df.dropna(subset=[date_col, price_col])
    .sort_values(date_col)
    .reset_index(drop=True)
)

# ======================================================
# WILDER RSI
# ======================================================
def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["rsi"] = wilder_rsi(df[price_col])

# ======================================================
# BACKTEST — FIRST CROSS ONLY
# ======================================================
THRESHOLDS = [30, 25, 20]
HOLD_DAYS = [30, 60]

signals = []

for level in THRESHOLDS:
    fired = False
    for i in range(1, len(df)):
        if fired:
            break
        if df.loc[i-1, "rsi"] >= level and df.loc[i, "rsi"] < level:
            for h in HOLD_DAYS:
                if i + h < len(df):
                    signals.append({
                        "RSI Level": level,
                        "Entry Date": df.loc[i, date_col].date(),
                        "Entry Price": round(df.loc[i, price_col], 2),
                        "Holding Days": h,
                        "Exit Price": round(df.loc[i+h, price_col], 2),
                        "Return %": round(
                            (df.loc[i+h, price_col] / df.loc[i, price_col] - 1) * 100, 2
                        )
                    })
            fired = True

results_df = pd.DataFrame(signals)

if results_df.empty:
    st.warning("No valid RSI mean-reversion signals detected.")
    st.stop()

# ======================================================
# OUTPUT
# ======================================================
st.markdown("<div class='section-title'>Trade Signals</div>", unsafe_allow_html=True)
st.dataframe(results_df, use_container_width=True)

summary_df = (
    results_df
    .groupby(["RSI Level", "Holding Days"])
    .agg(
        Trades=("Return %", "count"),
        Avg_Return=("Return %", "mean"),
        Win_Rate=("Return %", lambda x: (x > 0).mean() * 100)
    )
    .round(2)
    .reset_index()
)

st.markdown("<div class='section-title'>Summary Statistics</div>", unsafe_allow_html=True)
st.dataframe(summary_df, use_container_width=True)

# ======================================================
# EXPORT
# ======================================================
csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Results CSV",
    data=csv,
    file_name="rsi_mean_reversion_results.csv"
)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Quantitative research tool. Excel and CSV inputs supported.")
