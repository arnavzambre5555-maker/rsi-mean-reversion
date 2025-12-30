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
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system; }
.card { background:#fff; border-radius:14px; padding:18px 22px;
        box-shadow:0 6px 24px rgba(0,0,0,0.04); }
.title { font-size:40px; font-weight:700; letter-spacing:-0.5px; }
.subtitle { font-size:16px; color:#6b7280; }
.metric-label { color:#6b7280; font-size:13px; }
.metric-value { font-size:18px; font-weight:600; }
.section-title { font-size:22px; font-weight:600; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="card">
  <div class="title">RSI Mean Reversion Research Tool</div>
  <div class="subtitle">Quantitative RSI mean-reversion analysis on daily price data</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ======================================================
# METRICS
# ======================================================
c1,c2,c3,c4 = st.columns(4)
metrics = [
    ("Indicator","Wilder RSI (14)"),
    ("Signal Logic","First Cross, Non-Overlapping"),
    ("Holding Windows","5 / 30 / 60 Days"),
    ("Input Formats","CSV • XLSX • XLS")
]
for col,(l,v) in zip([c1,c2,c3,c4],metrics):
    with col:
        st.markdown(f"""
        <div class="card">
          <div class="metric-label">{l}</div>
          <div class="metric-value">{v}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ======================================================
# FILE INPUT
# ======================================================
st.markdown("""
<div class="card">
  <div class="section-title">Data Input</div>
  <div class="subtitle">Upload daily historical data. Excel structure is auto-detected.</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv","xlsx","xls"])
if uploaded_file is None:
    st.stop()

# ======================================================
# ROBUST FILE LOADER
# ======================================================
def load_any_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)

    # Excel: try progressively
    for skip in range(0, 10):
        try:
            df = pd.read_excel(file, skiprows=skip)
            if df.shape[1] >= 2:
                return df
        except:
            pass

    raise ValueError("Excel structure unreadable")

try:
    df = load_any_file(uploaded_file)
except Exception:
    st.error("Unable to parse file structure.")
    st.stop()

# ======================================================
# NORMALIZE COLUMNS
# ======================================================
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "")
)

# ======================================================
# AUTO-DETECT DATE & PRICE
# ======================================================
date_cols = [c for c in df.columns if "date" in c or "time" in c]
price_keys = ["close","price","px_last","last","adj_close"]
price_cols = [c for c in df.columns if any(k in c for k in price_keys)]

if not date_cols or not price_cols:
    st.error("Required date/price fields not found.")
    st.stop()

DATE_COL = date_cols[0]
PRICE_COL = price_cols[0]

# ======================================================
# CLEAN DATA
# ======================================================
df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")

df[PRICE_COL] = (
    df[PRICE_COL]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹","",regex=False)
    .str.replace("$","",regex=False)
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
# BACKTEST (UNCHANGED LOGIC)
# ======================================================
THRESHOLDS = [30,25,20]
HOLD_DAYS = [5,30,60]

def backtest(level):
    trades=[]
    last_exit=-1
    for i in range(1,len(df)):
        cross = df.loc[i-1,"RSI"]>=level and df.loc[i,"RSI"]<level
        if cross and i>last_exit:
            entry=df.loc[i,PRICE_COL]
            row={
                "Entry Date":df.loc[i,DATE_COL],
                "Entry Price":round(entry,2),
                "RSI":round(df.loc[i,"RSI"],2),
                "Signal":f"RSI < {level}"
            }
            for d in HOLD_DAYS:
                row[f"Return {d}D (%)"] = (
                    round((df.loc[i+d,PRICE_COL]/entry-1)*100,2)
                    if i+d<len(df) else np.nan
                )
            trades.append(row)
            last_exit=i+max(HOLD_DAYS)
    return pd.DataFrame(trades)

all_trades=[]
summary=[]

for t in THRESHOLDS:
    tr=backtest(t)
    all_trades.append(tr)
    s={"RSI <":t,"Trades":len(tr)}
    for d in HOLD_DAYS:
        col=f"Return {d}D (%)"
        s[f"Avg {d}D %"]=round(tr[col].mean(),2)
        s[f"WinRate {d}D %"]=round((tr[col]>0).mean()*100,2)
    summary.append(s)

results_df=pd.concat(all_trades,ignore_index=True)
summary_df=pd.DataFrame(summary)

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
st.caption("Quantitative research application. Robust Excel and CSV ingestion.")
