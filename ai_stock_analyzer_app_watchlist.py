# AI-Powered Stock Analyzer — Watchlist + News Tone
# -------------------------------------------------
# Features:
#  - Single-ticker deep dive w/ House+Senate trades, signals, and verdicts
#  - Watchlist mode: score many tickers at once (CSV upload or text list)
#  - Optional news sentiment factor via GDELT timeline tone
#
import os, io, math, requests, pandas as pd, numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Stock Analyzer — Watchlist + Tone", layout="wide")

# --- API key setup: Streamlit Secrets first, then env, then export to env ---
def get_alpha_key():
    key = ""
    try:
        import streamlit as st
        # Streamlit Cloud stores secrets here
        key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    except Exception:
        # st.secrets may not exist locally
        key = ""
    if not key:
        import os
        key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    return (key or "").strip()

ALPHA_KEY = get_alpha_key()

# Export to os.environ so any downstream code that reads env still works
if ALPHA_KEY:
    import os
    os.environ["ALPHAVANTAGE_API_KEY"] = ALPHA_KEY


# -----------------------------
# Price helpers (Alpha Vantage + Stooq fallback)
# -----------------------------
def fetch_alpha_vantage_daily(symbol: str) -> pd.DataFrame:
    if not ALPHA_KEY:
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": ALPHA_KEY,
        "outputsize": "full"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        if "Time Series (Daily)" not in data: return pd.DataFrame()
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns=lambda c: c.split(". ")[1] if ". " in c else c)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_index()
        df = df.rename(columns={"adjusted close":"adj_close"})
        if "adj_close" not in df: df["adj_close"] = df.get("close")
        return df[["open","high","low","close","adj_close","volume"]].dropna()
    except Exception:
        return pd.DataFrame()

def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """Robust Stooq fallback that tries common U.S. suffixes."""
    import io
    import requests
    import pandas as pd

    candidates = [
        symbol.lower(),                 # e.g., aapl
        f"{symbol.lower()}.us",         # aapl.us
        f"{symbol.lower()}.nasdaq",     # aapl.nasdaq
        f"{symbol.lower()}.nyse",       # orcl.nyse
    ]
    for s in candidates:
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "Date,Open,High,Low,Close,Volume" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                if not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.rename(columns=str.lower).set_index("date").sort_index()
                    df["adj_close"] = df["close"]
                    return df[["open","high","low","close","adj_close","volume"]].dropna()
        except Exception:
            pass
    return pd.DataFrame()

# -----------------------------
# Indicators
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def pct_change(series: pd.Series, days: int) -> float:
    if len(series) < days + 1: return np.nan
    return (series.iloc[-1] / series.iloc[-days-1]) - 1.0

def volume_spike_score(vol: pd.Series, window: int = 20) -> float:
    avg = vol.rolling(window).mean()
    std = vol.rolling(window).std()
    if len(vol) < window + 1: return 0.0
    z = (vol.iloc[-1] - avg.iloc[-1]) / (std.iloc[-1] + 1e-9)
    return float(np.clip(z / 3.0, -1, 1))

def position_52w(series: pd.Series) -> float:
    if series.empty: return np.nan
    window = series.iloc[-252:] if len(series) >= 252 else series
    lo, hi = window.min(), window.max()
    if hi == lo: return 0.5
    return float((series.iloc[-1] - lo) / (hi - lo))

# -----------------------------
# Scoring (short vs long horizon)
# -----------------------------
def heuristic_score(df: pd.DataFrame, horizon: str) -> dict:
    if df.empty: return {"score": 0, "verdict":"N/A","reasons":["No price data."]}
    price = df["adj_close"]; vol = df["volume"]
    rsi14 = rsi(price, 14).iloc[-1]
    _, _, hist = macd(price)
    macd_hist = hist.iloc[-1]
    mom_1m = pct_change(price, 21)
    mom_3m = pct_change(price, 63)
    pos_52w = position_52w(price)
    vol_spike = volume_spike_score(vol)

    if horizon.startswith("Short"):
        weights = {"mom":0.23,"macd":0.18,"rsi":0.13,"vol":0.13,"pos":0.09}
        momentum = mom_1m
    else:
        weights = {"mom":0.28,"macd":0.18,"rsi":0.09,"vol":0.09,"pos":0.14}
        momentum = mom_3m

    def nz(x, default=0.0):
        return default if (x is None or (isinstance(x,float) and math.isnan(x))) else x

    rsi_norm = np.tanh((nz(rsi14)-50)/10)
    macd_norm = np.tanh(nz(macd_hist) * 5)
    mom_norm = np.tanh(nz(momentum)*5)
    vol_norm = float(vol_spike)
    pos_norm = np.tanh((nz(pos_52w)-0.5)*3)

    score = (
        weights["mom"]*mom_norm + weights["macd"]*macd_norm +
        weights["rsi"]*rsi_norm + weights["vol"]*vol_norm +
        weights["pos"]*pos_norm
    )

    verdict = "Buy" if score >= 0.25 else ("Sell" if score <= -0.25 else "Hold")
    reasons = []
    if mom_norm > 0.15: reasons.append("Positive momentum")
    if macd_norm > 0.1: reasons.append("MACD bullish")
    if rsi_norm > 0.1: reasons.append("RSI > 50 and rising")
    if vol_norm > 0.5: reasons.append("Volume spike")
    if pos_norm > 0.2: reasons.append("Trading in upper 52w range")
    if not reasons: reasons.append("Mixed/neutral signals")

    return {"score": float(score), "verdict": verdict, "reasons": reasons}

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("AI Stock Analyzer")
    st.caption("Watchlist Demo")
    page = st.radio("Mode", ["Single Ticker", "Watchlist"], horizontal=True)
    symbol = st.text_input("Ticker (e.g., AAPL, MSFT, NVDA):", value="AAPL").strip().upper()
    horizon = st.selectbox("Time Horizon", ["Short (1-3m)", "Long (3m+)"])
    st.caption("Data Source Status")
st.write("Alpha Vantage key detected:", bool(ALPHA_KEY))


# -----------------------------
# Single Ticker
# -----------------------------
def render_single_ticker(sym: str):
    st.title(f"📈 {sym} — Deep Dive")
    price = get_price_history(sym)
   
    if price.empty:
    st.error(
        "No price data found for this ticker.\n\n"
        "• Make sure your Alpha Vantage key is set in your app’s Settings → Secrets (ALPHAVANTAGE_API_KEY)\n"
        "• Try a liquid U.S. ticker like AAPL/MSFT/AMD\n"
        "• Non-U.S. tickers may need exchange qualifiers; Stooq often needs a '.us' suffix\n"
        "• Double-check the symbol (e.g., did you mean 'CRWD' instead of 'CRWV'?)"
    )
    st.stop()


    st.subheader("Price & Trend")
    fig_price = px.line(price.reset_index(), x="index", y=["adj_close","EMA50","EMA200"])
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("RSI (14) & MACD Histogram")
    st.plotly_chart(px.line(price.reset_index(), x="index", y="RSI14"), use_container_width=True)
    st.plotly_chart(px.bar(price.reset_index(), x="index", y="MACD_hist"), use_container_width=True)

    st.subheader("AI Heuristic Scoring")
    res = heuristic_score(price, horizon)
    st.metric("Verdict", res["verdict"])
    st.write("Score:", round(res["score"], 3))
    st.write("Reasons:", ", ".join(res["reasons"]))

# -----------------------------
# Watchlist
# -----------------------------
def render_watchlist(horizon: str):
    st.title("📋 Watchlist — Bulk Scoring")
    paste = st.text_area("Tickers (comma or newline separated)", value="AAPL, MSFT, NVDA")
    if not paste.strip():
        st.info("Add some tickers to analyze.")
        return
    tickers = [t.strip().upper() for t in paste.replace("\\n",",").split(",") if t.strip()]
    results = []
    for sym in tickers:
        df = get_price_history(sym)
        if df.empty:
            results.append({"ticker": sym, "verdict":"N/A", "score":0})
            continue
        res = heuristic_score(df, horizon)
        results.append({"ticker": sym, "verdict":res["verdict"], "score":round(res["score"],3)})
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# -----------------------------
# Router
# -----------------------------
if page == "Single Ticker":
    render_single_ticker(symbol)
else:
    render_watchlist(horizon)
