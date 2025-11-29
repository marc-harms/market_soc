"""
SOC Logic Module
Consolidated backend logic for Market SOC Application.
Includes Data Fetching, Metric Calculation, and Analysis Engine.
"""

import time
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf

# --- CONFIGURATION & CONSTANTS ---

# Binance API Configuration
BINANCE_BASE_URL: str = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT: str = "/api/v3/klines"

# Data Fetching Parameters
DEFAULT_INTERVAL: str = "1d"
DEFAULT_LOOKBACK_DAYS: int = 2000

# Analysis Parameters
SMA_PERIOD: int = 200
ROLLING_VOLATILITY_WINDOW: int = 30

# Volatility Thresholds (Dynamic Quantile-based)
VOLATILITY_LOW_PERCENTILE: float = 33.33
VOLATILITY_HIGH_PERCENTILE: float = 66.67

# Caching
CACHE_DIR: str = "data"
CACHE_FILENAME_TEMPLATE: str = "{symbol}_{interval}_cached.csv"

# API Settings
REQUEST_TIMEOUT: int = 10
MAX_RETRIES: int = 3
RETRY_DELAY: int = 2


# --- METRICS CALCULATOR ---

class SOCMetricsCalculator:
    """
    Calculates financial metrics for SOC analysis:
    - Daily returns and absolute magnitude
    - Simple Moving Average (trend detection)
    - Rolling volatility (criticality metric)
    - Volatility thresholds for traffic light system
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.vol_low_threshold = 0.0
        self.vol_high_threshold = 0.0
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        if "close" not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' column")

    def calculate_all_metrics(self) -> pd.DataFrame:
        # Calculate returns
        self.df["returns"] = self.df["close"].pct_change()
        self.df["abs_returns"] = self.df["returns"].abs()

        # Calculate moving averages
        self.df["sma_200"] = self.df["close"].rolling(window=SMA_PERIOD).mean()

        # Calculate rolling volatility
        self.df["volatility"] = self.df["returns"].rolling(window=ROLLING_VOLATILITY_WINDOW).std()

        # Calculate thresholds
        self.vol_low_threshold, self.vol_high_threshold = self._calculate_volatility_thresholds()
        
        # Assign zones
        self.df["vol_zone"] = self._assign_volatility_zones(
            self.vol_low_threshold, self.vol_high_threshold
        )

        # Drop NaN rows
        self.df.dropna(inplace=True)
        return self.df

    def _calculate_volatility_thresholds(self) -> Tuple[float, float]:
        volatility = self.df["volatility"].dropna()
        if volatility.empty:
            return 0.0, 0.0
        low = volatility.quantile(VOLATILITY_LOW_PERCENTILE / 100)
        high = volatility.quantile(VOLATILITY_HIGH_PERCENTILE / 100)
        return low, high

    def _assign_volatility_zones(self, low: float, high: float) -> pd.Series:
        volatility = self.df["volatility"]
        zones = pd.Series("medium", index=self.df.index)
        zones[volatility <= low] = "low"
        zones[volatility > high] = "high"
        return zones

    def get_summary_stats(self) -> dict:
        if self.df.empty:
            return {}
        return {
            "total_records": len(self.df),
            "date_range": f"{self.df.index.min().date()} to {self.df.index.max().date()}",
            "mean_return": self.df["returns"].mean(),
            "std_return": self.df["returns"].std(),
            "mean_volatility": self.df["volatility"].mean(),
            "max_volatility": self.df["volatility"].max(),
            "current_price": self.df["close"].iloc[-1],
            "current_sma_200": self.df["sma_200"].iloc[-1],
            "vol_low_threshold": self.vol_low_threshold,
            "vol_high_threshold": self.vol_high_threshold,
        }


# --- DATA FETCHING ---

class DataProvider(ABC):
    @abstractmethod
    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        pass

class BinanceProvider(DataProvider):
    def __init__(self):
        self.base_url = BINANCE_BASE_URL

    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        return {
            "name": symbol,
            "sector": "Cryptocurrency",
            "description": f"Cryptocurrency asset pair {symbol} from Binance exchange."
        }

    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        
        url = f"{self.base_url}{BINANCE_KLINES_ENDPOINT}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000,
        }
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            params["startTime"] = current_start
            try:
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                klines = response.json()
            except Exception as e:
                print(f"Binance Error: {e}")
                break

            if not klines:
                break
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            if len(klines) < 1000:
                break
                
        # Parse
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        if df.empty:
            return pd.DataFrame()
            
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

class YFinanceProvider(DataProvider):
    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            description = info.get("longBusinessSummary", "")
            if len(description) > 300:
                description = description[:297] + "..."
            return {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "description": description,
            }
        except Exception:
            return {"name": symbol, "description": ""}

    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        start_date = datetime.now() - timedelta(days=lookback_days)
        try:
            df = yf.download(symbol, start=start_date, progress=False, auto_adjust=True)
            if df.empty: return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
                
            df.columns = [str(c).lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = "timestamp"
            
            required = ["open", "high", "low", "close", "volume"]
            available = [c for c in required if c in df.columns]
            return df[available]
        except Exception as e:
            print(f"YFinance Error: {e}")
            return pd.DataFrame()

class DataFetcher:
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(CACHE_DIR)
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
        self.binance = BinanceProvider()
        self.yfinance = YFinanceProvider()

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        # Check cache
        cache_path = self._get_cache_path(symbol)
        if self.cache_enabled and cache_path.exists():
            try:
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            except: pass

        # Fetch
        if symbol.endswith("USDT") or symbol.endswith("BUSD"):
            df = self.binance.fetch_data(symbol, DEFAULT_INTERVAL, DEFAULT_LOOKBACK_DAYS)
        else:
            df = self.yfinance.fetch_data(symbol, DEFAULT_INTERVAL, DEFAULT_LOOKBACK_DAYS)

        # Save cache
        if self.cache_enabled and not df.empty:
            df.to_csv(cache_path)
        
        return df

    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        if symbol.endswith("USDT") or symbol.endswith("BUSD"):
            return self.binance.fetch_info(symbol)
        return self.yfinance.fetch_info(symbol)

    def _get_cache_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("^", "").replace(".", "_")
        return self.cache_dir / CACHE_FILENAME_TEMPLATE.format(symbol=safe_symbol, interval=DEFAULT_INTERVAL)


# --- ANALYZER ---

class SOCAnalyzer:
    """
    High-level analyzer for a single asset.
    Wraps metric calculation and determines market phase (Signal).
    """

    def __init__(self, df: pd.DataFrame, symbol: str, asset_info: Optional[Dict[str, Any]] = None):
        self.df = df
        self.symbol = symbol
        self.asset_info = asset_info or {}
        self.calculator = SOCMetricsCalculator(df)
        self.metrics_df = self.calculator.calculate_all_metrics()
        self.summary_stats = self.calculator.get_summary_stats()

    def get_market_phase(self) -> Dict[str, Any]:
        if self.metrics_df.empty:
            return {"signal": "NO_DATA", "color": "grey"}

        # Values
        current_price = self.summary_stats["current_price"]
        sma_200 = self.summary_stats["current_sma_200"]
        current_vol = self.metrics_df["volatility"].iloc[-1]
        
        # Thresholds
        vol_low = self.calculator.vol_low_threshold
        vol_high = self.calculator.vol_high_threshold
        
        # Logic
        is_bullish = current_price > sma_200
        is_high_stress = current_vol > vol_high
        is_low_stress = current_vol <= vol_low
        
        signal = "NEUTRAL"
        if is_bullish:
            if is_low_stress: signal = "🟢 BUY/ACCUMULATE"
            elif is_high_stress: signal = "🟠 OVERHEATED"
            else: signal = "⚪ UPTREND (Choppy)"
        else:
            if is_high_stress: signal = "🔴 CRASH RISK"
            elif is_low_stress: signal = "🟡 CAPITULATION/WAIT"
            else: signal = "⚪ DOWNTREND (Choppy)"

        return {
            "symbol": self.symbol,
            "price": current_price,
            "sma_200": sma_200,
            "dist_to_sma": (current_price - sma_200) / sma_200,
            "volatility": current_vol,
            "signal": signal,
            "trend": "BULL" if is_bullish else "BEAR",
            "stress": "HIGH" if is_high_stress else ("LOW" if is_low_stress else "MED")
        }

