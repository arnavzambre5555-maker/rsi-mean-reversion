
import streamlit as st
import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================
RSI_PERIOD = 14
THRESHOLDS = [30, 25, 20]
HOLD_DAYS = [5, 30, 60]   # 5D = ~1 week

# =============================
# HELPERS
# =============================
def normalize_input(df_raw):
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    date_candidates = ["date", "trading date", "timestamp"]
    price_candidates = ["close", "price", "close price", "last", "adj close"]

    date_col = next((c for c in date_candidates if c in df_raw.columns), None)
    price_col = next((c for c in price_candidates if c in df_raw.columns), None)

    if date_col is None or price_col is None:
        st.error("Required Date or Price column not found")
        st.stop()

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(df_raw[date_col], dayfirst=True, errors="coerce")
    df["close"] = (
        df_raw[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    df = (
        df.dropna(subset=["date", "close"])
          .sort_values("date")
          .reset_index(drop=True)
    )

    return df

def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def backtest(df, threshold):
    trades = []
    last_exit = -1

    for i in range(1, len(df)):
        crossed = df.loc[i-1, "rsi"] >= threshold and df.loc[i, "rsi"] < threshold

        if crossed and i > last_exit:
            entry_price = df.loc[i, "close"]

            trade = {
                "Entry Date": df.loc[i, "date"],
                "Entry Price": round(entry_price, 2),
                "RSI": round(df.loc[i, "rsi"], 2),
                "Signal": f"RSI < {threshold}"
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

# =============================
# UI
# =============================
st.set_page_config(page_title="RSI Mean Reversion Tool")
st.title("RSI Mean Reversion Tool")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df_raw = pd.read_csv(file)
    df = normalize_input(df_raw)

    df["rsi"] = wilder_rsi(df["close"], RSI_PERIOD)

    all_trades = []
    summary = []

    for t in THRESHOLDS:
        trades = backtest(df, t)

        if len(trades) > 0:
            all_trades.append(trades)

            row = {
                "RSI <": t,
                "Trades": len(trades)
            }

            for d in HOLD_DAYS:
                col = f"Return {d}D (%)"
                row[f"Avg {d}D %"] = round(trades[col].mean(), 2)
                row[f"WinRate {d}D %"] = round((trades[col] > 0).mean() * 100, 2)

            summary.append(row)

    if all_trades:
        results = pd.concat(all_trades).reset_index(drop=True)

        st.subheader("Signals")
        st.dataframe(results)

        st.subheader("Summary")
        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            data=csv,
            file_name="rsi_mean_reversion_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("No RSI threshold signals found in this dataset.")
