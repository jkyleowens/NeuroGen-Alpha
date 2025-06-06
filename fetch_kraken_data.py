import os
import requests
import pandas as pd
import ta
from datetime import datetime

# Directory to store processed CSV files
DATA_DIR = "crypto_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping of common symbols to Kraken pairs
SYMBOLS = {
    "BTCUSD": "XXBTZUSD",
    "ETHUSD": "XETHZUSD",
    "LTCUSD": "XLTCZUSD",
    "ADAUSD": "ADAUSD",
    "DOTUSD": "DOTUSD",
}

def fetch_ohlc(pair: str, interval: int = 60, since: int | None = None) -> pd.DataFrame:
    """Fetch OHLC data from Kraken API and return as DataFrame."""
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")
    result = next(iter(data["result"].values()))
    df = pd.DataFrame(result, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few technical indicators using ta library."""
    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.macd(df["close"])
    df["macd"] = macd
    df["sma_fast"] = ta.trend.sma_indicator(df["close"], window=10)
    df["sma_slow"] = ta.trend.sma_indicator(df["close"], window=30)
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

def process_symbol(symbol: str):
    pair = SYMBOLS.get(symbol.upper())
    if not pair:
        print(f"Unknown symbol {symbol}")
        return
    print(f"Fetching {symbol} ({pair})...")
    df = fetch_ohlc(pair, interval=60)
    df = add_indicators(df)
    out_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(out_path)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    for sym in SYMBOLS:
        try:
            process_symbol(sym)
        except Exception as e:
            print(f"Failed {sym}: {e}")
