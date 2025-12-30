import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="RSI Mean Reversion Research Tool",
    page_icon="📉",
    layout="wide"
)

# ======================================================
# CONSTANTS
# ======================================================
RSI_PERIOD = 14
THRESHOLDS = [30, 25, 20]
HOLD_DAYS = [5, 30, 60]

# ======================================================
# STYLING
# ======================================================
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    h1 { font-size: 2.4rem; }
    h2 { margin-top: 2rem; }
    .metric-box {
        background-color: #f5f7fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# HEADER
# ======================================================
st.title("RSI Mean Reversion Research Tool")
st.caption(
    "Quantitative research utility for studying RSI-based mean reversion "
    "using daily equity price data."
)

# ======================================================
# LIMITATIONS / NOTICE
# ======================================================
with st.expander("⚠️ Data Requirements & Limitations", expanded=True):
    st.markdown(
        """
**Accepted Data Source**
- CSV files downloaded from **Investing.com** or **NSE India**
- Daily timeframe only

**Mandatory Columns (any ONE naming works)**
- Date column: `Date`
- Price column: `Close`, `Price`, or `Close Price`

**Not Supported**
- Bloomberg exports
- Excel files (`.xlsx`)
- Intraday data
- Adjusted corporate action backfills

**Disclaimer**
This is a **research-only tool**.  
No investment advice. Results depend entirely on data quality.
"""
    )

# ======================================================
# FILE UPLOAD
# ======================================================
file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="Use CSV from Investing.com or NSE only"
)

# ======================================================
# HELPERS
# ======================================================
def clean_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    date_candidates = ["date"]
    price_candidates = ["close", "price", "close price"]

    date_col = next((c for c in date_candidates if c in df_raw.columns), None)
    price_col = next((c for c in price_candidates if c in df_raw.columns), None)

    if date_col is None or price_col is None:
        st.error("Required Date / Close column not found. Use Investing.com or NSE CSV.")
        st.stop()

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(df_raw[date_col], dayfirst=True, errors="coerce")

    df["close"] = (
        df_raw[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(df) < 200:
        st.error("Dataset too small for meaningful RSI analysis.")
        st.stop()

    return df


def wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    trades = []
    last_exit = -1

    for i in range(1, len(df)):
        crossed = df.loc[i - 1, "rsi"] >= threshold and df.loc[i, "rsi"] < threshold

        if crossed and i > last_exit:
            entry_price = df.loc[i, "close"]

            trade = {
                "Entry Date": df.loc[i, "date"],
                "Entry Price": round(entry_price, 2),
                "RSI": round(df.loc[i, "rsi"], 2),
                "Signal": f"RSI < {threshold}",
            }

            for d in HOLD_DAYS:
                if i + d < len(df):
                    trade[f"Return {d}D (%)"] = round(
                        (df.loc[i + d, "close"] / entry_price - 1) * 100, 2
                    )
                else:
                    trade[f"Return {d}D (%)"] = np.nan

            trades.append(trade)
            last_exit = i + max(HOLD_DAYS)

    return pd.DataFrame(trades)


# ======================================================
# MAIN LOGIC
# ======================================================
if file is not None:
    df_raw = pd.read_csv(file)
    df = clean_input(df_raw)
    df["rsi"] = wilder_rsi(df["close"], RSI_PERIOD)

    all_trades = []
    summary_rows = []

    for t in THRESHOLDS:
        trades = backtest(df, t)

        if not trades.empty:
            all_trades.append(trades)

            row = {
                "RSI Threshold": f"< {t}",
                "Trades": len(trades),
            }

            for d in HOLD_DAYS:
                col = f"Return {d}D (%)"
                row[f"Avg {d}D %"] = round(trades[col].mean(), 2)
                row[f"Win % {d}D"] = round((trades[col] > 0).mean() * 100, 2)

            summary_rows.append(row)

    if not all_trades:
        st.warning("No RSI mean-reversion signals detected in this dataset.")
        st.stop()

    results_df = pd.concat(all_trades).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows)

    # ==================================================
    # OUTPUT
    # ==================================================
    st.subheader("📌 Trade Signals")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("📊 Summary Statistics")
    st.dataframe(summary_df, use_container_width=True)

    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv,
        file_name="rsi_mean_reversion_results.csv",
        mime="text/csv",
    )

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption(
    "Educational quantitative research tool. "
    "No investment advice. Use only clean daily CSV data."
)
