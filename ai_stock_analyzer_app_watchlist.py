# AI-Powered Stock Analyzer — Watchlist + News Tone
# -------------------------------------------------
# Features:
#  - Single-ticker deep dive w/ House+Senate trades, signals, and verdicts
#  - Watchlist mode: score many tickers at once (CSV upload or text list)
#  - Optional news sentiment factor via GDELT timeline tone
#
# Notes:
#  - Add your Alpha Vantage key in Streamlit Cloud: App ⋯ → Settings → Secrets
#      ALPHAVANTAGE_API_KEY = "YOUR_REAL_KEY"
#  - Add .streamlit/config.toml with:
#      [server]
#      fileWatcherType = "none"
#  - Optional runtime.txt (repo root): python-3.11
#
import os, io, math, requests, pandas as pd, numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Stock Analyzer — Watchlist + Tone", layout="wide")

# --- API key setup: Streamlit Secrets first, then env, then export to env ---
def get_alpha_key():
    key = ""
    try:
        # Streamlit Cloud stores secrets here
        key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    return (key or "").strip()

ALPHA_KEY = get_alpha_key()
if ALPHA_KEY:
    os.environ["ALPHAVANTAGE_API_KEY"] = ALPHA_KEY

# -----------------------------
# Data endpoints (politician trades)
# -----------------------------
HOUSE_TRADES_URL = "https://raw.githubusercontent.com/unitedstates/house-stock-watcher-data/master/data/all_transactions.csv"
SENATE_URL_CANDIDATES = [
    "https://raw.githubusercontent.com/unitedstates/senate-stock-watcher-data/master/data/all_transactions.csv",
    "https://raw.githubusercontent.com/smucclaw/senate-stock-watcher-data/master/data/all_transactions.csv",
    "https://raw.githubusercontent.com/unitedstates/senate-stock-watcher-data/master/data/all_transactions.json",
]

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
        if "Time Series (Daily)" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns=lambda c: c.split(". ")[1] if ". " in c else c)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_index()
        df = df.rename(columns={"adjusted close": "adj_close"})
        if "adj_close" not in df:
            df["adj_close"] = df.get("close")
        return df[["open", "high", "low", "close", "adj_close", "volume"]].dropna()
    except Exception:
        return pd.DataFrame()

def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """Robust Stooq fallback that tries common U.S. suffixes."""
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
                    return df[["open", "high", "low", "close", "adj_close", "volume"]].dropna()
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60*30)
def get_price_history(symbol: str) -> pd.DataFrame:
    df = fetch_alpha_vantage_daily(symbol)
    if df.empty:
        df = fetch_stooq_daily(symbol)
    return df

# -----------------------------
# Politician trade loaders
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def get_house_trades() -> pd.DataFrame:
    try:
        df = pd.read_csv(HOUSE_TRADES_URL)
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
        df["chamber"] = "House"
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60*60)
def get_senate_trades() -> pd.DataFrame:
    for url in SENATE_URL_CANDIDATES:
        try:
            if url.endswith(".csv"):
                df = pd.read_csv(url)
            else:
                js = requests.get(url, timeout=30).json()
                df = pd.DataFrame(js)
            cols = {c.lower(): c for c in df.columns}
            if "transactiondate" in cols: txc = cols["transactiondate"]
            elif "transaction_date" in cols: txc = cols["transaction_date"]
            elif "date" in cols: txc = cols["date"]
            else: txc = list(df.columns)[0]
            df["TransactionDate"] = pd.to_datetime(df[txc], errors="coerce")
            ticker_col = None
            for cand in ["Ticker", "ticker", "Symbol", "symbol", "AssetTicker"]:
                if cand in df.columns:
                    ticker_col = cand; break
            if ticker_col is None:
                df["Ticker"] = ""
                ticker_col = "Ticker"
            if "Transaction" not in df.columns:
                for cand in ["transaction", "Type", "type", "transaction_type"]:
                    if cand in df.columns:
                        df["Transaction"] = df[cand]; break
                else:
                    df["Transaction"] = ""
            if "Representative" not in df.columns:
                for cand in ["senator", "Senator", "Member"]:
                    if cand in df.columns:
                        df["Representative"] = df[cand]; break
                else:
                    df["Representative"] = ""
            if "Amount" not in df.columns:
                for cand in ["amount", "AmountRange", "amount_range"]:
                    if cand in df.columns:
                        df["Amount"] = df[cand]; break
                else:
                    df["Amount"] = ""
            df["Ticker"] = df[ticker_col]
            df["chamber"] = "Senate"
            return df
        except Exception:
            continue
    return pd.DataFrame()

def _dir_from_text(tx):
    s = str(tx).lower()
    if "purchase" in s or "buy" in s: return 1
    if "sale" in s or "sell" in s: return -1
    return 0

def summarize_politician_activity(df: pd.DataFrame, symbol: str, days: int = 90):
    if df.empty: return {"buys":0,"sells":0,"net":0,"recent_examples":[]}
    sym_col = None
    for col in ["Ticker","AssetTicker","TickerSymbol","Symbol","ticker","symbol"]:
        if col in df.columns: sym_col = col; break
    if sym_col is None: return {"buys":0,"sells":0,"net":0,"recent_examples":[]}
    symbol = symbol.upper()
    now = pd.Timestamp.utcnow().normalize()
    cutoff = now - pd.Timedelta(days=days)
    sub = df[(df[sym_col].astype(str).str.upper()==symbol) & (df["TransactionDate"]>=cutoff)].copy()
    if sub.empty: return {"buys":0,"sells":0,"net":0,"recent_examples":[]}
    if "dir" not in sub.columns:
        sub["dir"] = sub["Transaction"].apply(_dir_from_text) if "Transaction" in sub.columns else 0
    buys = int((sub["dir"]==1).sum()); sells = int((sub["dir"]==-1).sum())
    net = buys - sells
    cols = [c for c in ["chamber","Representative","TransactionDate","Transaction","Amount"] if c in sub.columns]
    examples = sub.sort_values("TransactionDate", ascending=False).head(10)[cols].to_dict(orient="records")
    return {"buys":buys,"sells":sells,"net":int(net),"recent_examples":examples}

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
# News tone (GDELT)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def gdelt_timeline_tone(query: str) -> pd.DataFrame:
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {"query": query, "mode": "timelinetone", "format": "json"}
        js = requests.get(url, params=params, timeout=30).json()
        if "timeline" not in js: return pd.DataFrame()
        df = pd.DataFrame(js["timeline"])
        df["date"] = pd.to_datetime(df["date"])
        df["tone"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date","tone"]].dropna()
    except Exception:
        return pd.DataFrame()

def tone_score(df: pd.DataFrame, days: int = 30) -> float:
    if df.empty: return 0.0
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days)
    recent = df[df["date"] >= cutoff]
    if recent.empty: return 0.0
    mean_tone = recent["tone"].mean()
    return float(np.tanh(mean_tone / 5.0))

# -----------------------------
# Scoring (short vs long horizon)
# -----------------------------
def heuristic_score(df: pd.DataFrame, horizon: str, pol_stats_combined: dict = None, news_component: float = 0.0) -> dict:
    if df.empty: return {"score": 0, "verdict":"N/A","reasons":["No price data."]}
    pol_stats_combined = pol_stats_combined or {"net": 0}

    price = df["adj_close"]; vol = df["volume"]
    rsi14 = rsi(price, 14).iloc[-1]
    _, _, hist = macd(price)
    macd_hist = hist.iloc[-1]
    mom_1m = pct_change(price, 21)
    mom_3m = pct_change(price, 63)
    pos_52w = position_52w(price)
    vol_spike = volume_spike_score(vol)

    pol_net = pol_stats_combined.get("net", 0)
    pol_component = np.tanh(pol_net / 6.0)

    if horizon.startswith("Short"):
        weights = {"mom":0.23,"macd":0.18,"rsi":0.13,"vol":0.13,"pos":0.09,"pol":0.14,"news":0.10}
        momentum = mom_1m
    else:
        weights = {"mom":0.28,"macd":0.18,"rsi":0.09,"vol":0.09,"pos":0.14,"pol":0.14,"news":0.08}
        momentum = mom_3m

    def nz(x, default=0.0):
        return default if (x is None or (isinstance(x,float) and math.isnan(x))) else x

    rsi_norm = np.tanh((nz(rsi14)-50)/10)
    macd_norm = np.tanh(nz(macd_hist) * 5)
    mom_norm = np.tanh(nz(momentum)*5)
    vol_norm = float(vol_spike)
    pos_norm = np.tanh((nz(pos_52w)-0.5)*3)
    news_norm = float(news_component)

    score = (
        weights["mom"]*mom_norm + weights["macd"]*macd_norm + weights["rsi"]*rsi_norm +
        weights["vol"]*vol_norm + weights["pos"]*pos_norm + weights["pol"]*pol_component +
        weights["news"]*news_norm
    )

    verdict = "Buy" if score >= 0.25 else ("Sell" if score <= -0.25 else "Hold")
    reasons = []
    if mom_norm > 0.15: reasons.append("Positive momentum")
    if macd_norm > 0.1: reasons.append("MACD bullish")
    if rsi_norm > 0.1: reasons.append("RSI > 50 and rising")
    if vol_norm > 0.5: reasons.append("Volume spike")
    if pos_norm > 0.2: reasons.append("Trading in upper 52w range")
    if pol_component > 0.1: reasons.append("Net politician buying")
    if news_norm > 0.1: reasons.append("Positive news tone")
    if not reasons: reasons.append("Mixed/neutral signals")

    return {
        "score": float(score),
        "verdict": verdict,
        "reasons": reasons,
        "raw": {
            "rsi14": float(nz(rsi14, np.nan)),
            "macd_hist": float(nz(macd_hist, np.nan)),
            "mom_1m": float(nz(mom_1m, np.nan)),
            "mom_3m": float(nz(mom_3m, np.nan)),
            "pos_52w": float(nz(pos_52w, np.nan)),
            "vol_spike": float(nz(vol_spike, np.nan)),
            "pol_net": int(pol_net),
            "news_tone_30d": float(news_norm),
        }
    }

# -----------------------------
# Sidebar + Pages
# -----------------------------
with st.sidebar:
    st.title("AI Stock Analyzer")
    st.caption("Watchlist + News Tone (Open data)")
    page = st.radio("Mode", ["Single Ticker", "Watchlist"], horizontal=True)
    symbol = st.text_input("Ticker (e.g., AAPL, MSFT, NVDA):", value="AAPL").strip().upper()
    horizon = st.selectbox("Time Horizon", ["Short (1-3m)", "Long (3m+)"])
    include_house = st.toggle("Include House trades", value=True)
    include_senate = st.toggle("Include Senate trades", value=True)
    include_news = st.toggle("Include News Tone (GDELT)", value=False)

    st.markdown("---")
    st.caption("Data Source Status")
    st.write("Alpha Vantage key detected:", bool(ALPHA_KEY))

# Politician dataframes preloaded (based on toggles)
house_df = get_house_trades() if include_house else pd.DataFrame()
senate_df = get_senate_trades() if include_senate else pd.DataFrame()

def combined_pol_stats(sym: str) -> dict:
    h = summarize_politician_activity(house_df, sym, days=90) if include_house else {"buys":0,"sells":0,"net":0,"recent_examples":[]}
    s = summarize_politician_activity(senate_df, sym, days=90) if include_senate else {"buys":0,"sells":0,"net":0,"recent_examples":[]}
    return {
        "buys": h["buys"] + s["buys"],
        "sells": h["sells"] + s["sells"],
        "net": h["net"] + s["net"],
        "recent_examples": (h["recent_examples"] + s["recent_examples"])[:10]
    }

# -----------------------------
# Single Ticker Page
# -----------------------------
def render_single_ticker(sym: str):
    st.title(f"📈 {sym} — Deep Dive")

    price = get_price_history(sym)

    # Handle missing data clearly
    if price.empty:
        st.error(
            "No price data found for this ticker.\n\n"
            "• Make sure your Alpha Vantage key is set in your app’s Settings → Secrets (ALPHAVANTAGE_API_KEY)\n"
            "• Try a liquid U.S. ticker like AAPL/MSFT/AMD\n"
            "• Non-U.S. tickers may need exchange qualifiers; Stooq often needs a '.us' suffix\n"
            "• Double-check the symbol (e.g., did you mean 'CRWD' instead of 'CRWV'?)"
        )
        st.stop()

    # Indicators
    price = price.copy()
    price["EMA50"] = ema(price["adj_close"], 50)
    price["EMA200"] = ema(price["adj_close"], 200)
    price["RSI14"] = rsi(price["adj_close"], 14)
    _, _, hist = macd(price["adj_close"])
    price["MACD_hist"] = hist

    # Charts
    st.subheader("Price & Trend")
    fig_price = px.line(price.reset_index(), x="index", y=["adj_close","EMA50","EMA200"], labels={"index":"Date","value":"Price","variable":"Series"})
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("RSI (14)")
    st.plotly_chart(px.line(price.reset_index(), x="index", y="RSI14"), use_container_width=True)

    st.subheader("MACD Histogram")
    st.plotly_chart(px.bar(price.reset_index(), x="index", y="MACD_hist"), use_container_width=True)

    # Politician stats
    st.subheader("Politician Trading Activity (last 90d)")
    pol = combined_pol_stats(sym)
    c1, c2, c3 = st.columns(3)
    c1.metric("Buys", pol["buys"]); c2.metric("Sells", pol["sells"]); c3.metric("Net", pol["net"])
    if pol["recent_examples"]:
        st.caption("Most recent examples:")
        st.dataframe(pd.DataFrame(pol["recent_examples"]))

    # News tone (optional)
    news_component = 0.0
    if include_news:
        st.subheader("News Tone (GDELT)")
        q = st.text_input("Query for GDELT", value=symbol, key=f"news_query_{sym}")
        tone_df = gdelt_timeline_tone(q)
        if tone_df.empty:
            st.info("No tone data found for this query.")
        else:
            st.plotly_chart(px.line(tone_df, x="date", y="tone", title="GDELT Tone Timeline"), use_container_width=True)
            news_component = tone_score(tone_df, days=30)
            st.caption(f"30-day mean tone (scaled): {news_component:.3f}")

    # Scoring
    st.subheader("AI Heuristic Scoring")
    res_short = heuristic_score(price, "Short (1-3m)", pol, news_component=news_component)
    res_long  = heuristic_score(price, "Long (3m+)", pol, news_component=news_component)
    t1, t2 = st.tabs(["Short (1-3m)", "Long (3m+)"])
    with t1:
        st.metric("Verdict", res_short["verdict"])
        st.write("Score:", round(res_short["score"], 3))
        st.write("Reasons:", ", ".join(res_short["reasons"]))
        st.json(res_short["raw"])
    with t2:
        st.metric("Verdict", res_long["verdict"])
        st.write("Score:", round(res_long["score"], 3))
        st.write("Reasons:", ", ".join(res_long["reasons"]))
        st.json(res_long["raw"])

# -----------------------------
# Watchlist Page
# -----------------------------
TEMPLATE = "ticker\nAAPL\nMSFT\nNVDA\nAMD\nGOOGL\n"

def analyze_ticker_for_watchlist(sym: str, horizon: str) -> dict:
    df = get_price_history(sym)
    if df.empty:
        return {"ticker": sym, "verdict": "N/A", "score": 0.0, "pol_net_90d": 0, "mom_1m": np.nan, "mom_3m": np.nan, "pos_52w": np.nan, "news_30d": 0.0, "status": "no_data"}
    pol = combined_pol_stats(sym)
    news_c = 0.0
    if include_news:
        tone_df = gdelt_timeline_tone(sym)
        news_c = tone_score(tone_df, days=30)
    res = heuristic_score(df, horizon, pol, news_component=news_c)
    price = df["adj_close"]
    return {
        "ticker": sym,
        "verdict": res["verdict"],
        "score": round(res["score"], 3),
        "pol_net_90d": pol["net"],
        "mom_1m": round(pct_change(price, 21), 3) if len(price) > 22 else np.nan,
        "mom_3m": round(pct_change(price, 63), 3) if len(price) > 64 else np.nan,
        "pos_52w": round(position_52w(price), 3),
        "news_30d": round(news_c, 3) if include_news else 0.0,
        "status": "ok",
    }

def render_watchlist(horizon: str):
    st.title("📋 Watchlist — Bulk Scoring")
    st.write("Upload a CSV with a `ticker` column **or** paste a comma/space/newline-separated list below.")

    # Template download
    st.download_button("Download CSV template", data=TEMPLATE, file_name="watchlist_template.csv", mime="text/csv")

    up = st.file_uploader("Upload CSV (column: ticker)", type=["csv"])
    tickers = []
    if up is not None:
        try:
            df = pd.read_csv(up)
            col = None
            for c in df.columns:
                if c.strip().lower() in ["ticker","tickers","symbol","symbols"]:
                    col = c; break
            if col is None and df.shape[1]>=1: col = df.columns[0]
            tickers = [str(t).strip().upper() for t in df[col].dropna().tolist()]
        except Exception:
            st.error("Could not read CSV. Please ensure it has a 'ticker' column.")

    st.write("Or paste your list:")
    paste = st.text_area("Tickers (AAPL, MSFT, NVDA)", value="AAPL, MSFT, NVDA")
    if paste.strip():
        more = [t.strip().upper() for t in paste.replace("\n",",").replace(" ",",").split(",") if t.strip()]
        tickers = list(dict.fromkeys((tickers or []) + more))

    if not tickers:
        st.info("Add some tickers to analyze.")
        return

    go = st.button("Analyze Watchlist")
    if not go:
        return

    results = []
    prog = st.progress(0)
    for i, sym in enumerate(tickers):
        results.append(analyze_ticker_for_watchlist(sym, horizon))
        prog.progress((i+1)/len(tickers))

    dfres = pd.DataFrame(results)
    # Basic sorting: strongest score first within verdict
    dfres = dfres.sort_values(["verdict","score"], ascending=[True, False])
    st.subheader("Results")
    st.dataframe(dfres, use_container_width=True)
    st.caption("Tip: click the column headers to sort.")

# -----------------------------
# Router
# -----------------------------
if page == "Single Ticker":
    render_single_ticker(symbol)
else:
    render_watchlist(horizon)

st.markdown("---")
st.markdown(
    "Data Sources: Alpha Vantage, Stooq, House Stock Watcher, Senate Stock Watcher (mirrors), "
    "GDELT Doc API (timelinetone). This is a research/education tool, not investment advice."
)
