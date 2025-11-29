"""
Data Fetcher Module
Handles fetching OHLCV data from multiple sources (Binance, Yahoo Finance)
Implements Strategy Pattern for extensible data sourcing
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
import yfinance as yf

from config.settings import (
    BINANCE_BASE_URL,
    BINANCE_KLINES_ENDPOINT,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK_DAYS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    CACHE_DIR,
    CACHE_FILENAME_TEMPLATE,
)


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """Fetch data and return standardized DataFrame"""
        pass

    @abstractmethod
    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch asset metadata"""
        pass


class BinanceProvider(DataProvider):
    """Fetches data from Binance Public API"""
    
    def __init__(self):
        self.base_url = BINANCE_BASE_URL

    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        return {
            "name": symbol,
            "sector": "Cryptocurrency",
            "description": f"Cryptocurrency asset pair {symbol} from Binance exchange."
        }

    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        print(f"Fetching {lookback_days} days of {symbol} from Binance...")
        
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
            klines = self._make_request(url, params)
            if not klines:
                break
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            if len(klines) < 1000:
                break
                
        return self._parse_klines(all_klines)

    def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[List]:
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"⚠ Binance API Error: {e}")
                    return None
                time.sleep(RETRY_DELAY)
        return None

    def _parse_klines(self, klines: List) -> pd.DataFrame:
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]


class YFinanceProvider(DataProvider):
    """Fetches data from Yahoo Finance"""

    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant fields
            description = info.get("longBusinessSummary", "")
            # Truncate description to keep it crisp (~300 chars)
            if len(description) > 300:
                description = description[:297] + "..."
                
            return {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "description": description,
                "country": info.get("country", "")
            }
        except Exception as e:
            print(f"⚠ YFinance Info Error for {symbol}: {e}")
            return {
                "name": symbol,
                "description": "No description available."
            }

    def fetch_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        print(f"Fetching {lookback_days} days of {symbol} from Yahoo Finance...")
        
        # Calculate start date
        start_date = datetime.now() - timedelta(days=lookback_days)
        
        # Download data
        try:
            df = yf.download(symbol, start=start_date, progress=False, auto_adjust=True)
            
            if df.empty:
                print(f"⚠ No data found for {symbol} on Yahoo Finance")
                return pd.DataFrame()

            # Handle MultiIndex columns (common in newer yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten or select just the price columns (Level 0)
                # Structure is typically (Price, Ticker) -> ('Close', 'NVDA')
                df.columns = df.columns.get_level_values(0)

            # Standardize columns (lowercase)
            # Ensure we are working with strings
            df.columns = [str(c).lower() for c in df.columns]
            
            # Ensure index is DatetimeIndex
            df.index = pd.to_datetime(df.index)
            df.index.name = "timestamp"
            
            # YFinance might return 'adj close', we prefer 'close' or 'adj close' if close missing
            # With auto_adjust=True, 'Close' is already adjusted.
            
            # Ensure required columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in required_cols if c in df.columns]
            
            return df[available_cols]

        except Exception as e:
            print(f"⚠ YFinance Error for {symbol}: {e}")
            return pd.DataFrame()


class DataFetcher:
    """
    Main Data Fetcher Class
    Selects the appropriate provider based on the symbol
    Handles Caching
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(CACHE_DIR)
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
            
        self.binance_provider = BinanceProvider()
        self.yfinance_provider = YFinanceProvider()

    def fetch_data(self, symbol: str, interval: str = DEFAULT_INTERVAL, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
        # Check cache
        if self.cache_enabled:
            cached = self._load_cache(symbol, interval)
            if cached is not None:
                return cached

        # Select Provider
        if symbol.endswith("USDT") or symbol.endswith("BUSD"):
            provider = self.binance_provider
        else:
            provider = self.yfinance_provider

        # Fetch
        df = provider.fetch_data(symbol, interval, lookback_days)

        # Save cache if data is valid
        if not df.empty and self.cache_enabled:
            self._save_cache(df, symbol, interval)

        return df

    def fetch_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch asset metadata from appropriate provider"""
        if symbol.endswith("USDT") or symbol.endswith("BUSD"):
            provider = self.binance_provider
        else:
            provider = self.yfinance_provider
            
        return provider.fetch_info(symbol)

    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        # Sanitize symbol for filename (e.g. ^GDAXI -> GDAXI)
        safe_symbol = symbol.replace("^", "").replace(".", "_")
        return self.cache_dir / CACHE_FILENAME_TEMPLATE.format(symbol=safe_symbol, interval=interval)

    def _save_cache(self, df: pd.DataFrame, symbol: str, interval: str):
        try:
            df.to_csv(self._get_cache_path(symbol, interval))
            print(f"✓ Saved to cache: {symbol}")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")

    def _load_cache(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        path = self._get_cache_path(symbol, interval)
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                print(f"✓ Loaded {symbol} from cache")
                return df
            except Exception:
                pass
        return None
