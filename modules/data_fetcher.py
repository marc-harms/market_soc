"""
Data Fetcher Module
Handles fetching OHLCV data from Binance API with caching support
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests

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


class BinanceDataFetcher:
    """
    Fetches cryptocurrency OHLCV data from Binance Public API
    Implements caching to avoid rate limits and improve performance
    """

    def __init__(self, cache_enabled: bool = True) -> None:
        """
        Initialize the data fetcher

        Args:
            cache_enabled: If True, will save/load data from CSV cache
        """
        self.base_url = BINANCE_BASE_URL
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(CACHE_DIR)

        # Create cache directory if it doesn't exist
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)

    def fetch_data(
        self,
        symbol: str,
        interval: str = DEFAULT_INTERVAL,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1d' for daily)
            lookback_days: Number of days of historical data to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            Exception: If API request fails after retries
        """
        # Check cache first
        if self.cache_enabled:
            cached_df = self._load_from_cache(symbol, interval)
            if cached_df is not None:
                print(f"✓ Loaded data from cache: {len(cached_df)} records")
                return cached_df

        print(f"Fetching {lookback_days} days of {symbol} data from Binance API...")

        # Calculate start time (milliseconds timestamp)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int(
            (datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000
        )

        # Build request parameters
        url = f"{self.base_url}{BINANCE_KLINES_ENDPOINT}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000,  # Binance max per request
        }

        # Fetch data with pagination
        all_klines = []
        current_start = start_time

        while current_start < end_time:
            params["startTime"] = current_start

            # Make request with retries
            klines = self._make_request_with_retry(url, params)

            if not klines:
                break

            all_klines.extend(klines)

            # Update start time for next batch
            current_start = klines[-1][0] + 1

            # If we got less than limit, we've reached the end
            if len(klines) < 1000:
                break

        print(f"✓ Fetched {len(all_klines)} records from API")

        # Convert to DataFrame
        df = self._parse_klines_to_dataframe(all_klines)

        # Save to cache
        if self.cache_enabled:
            self._save_to_cache(df, symbol, interval)

        return df

    def _make_request_with_retry(
        self, url: str, params: Dict[str, Any]
    ) -> Optional[list]:
        """
        Make API request with retry logic

        Args:
            url: API endpoint URL
            params: Request parameters

        Returns:
            List of kline data or None if failed

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                print(f"⚠ Timeout on attempt {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise Exception(f"API request timed out after {MAX_RETRIES} attempts")

            except requests.exceptions.HTTPError as e:
                print(f"⚠ HTTP Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise Exception(f"API request failed: {e}")

            except Exception as e:
                print(f"⚠ Unexpected error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise

        return None

    def _parse_klines_to_dataframe(self, klines: list) -> pd.DataFrame:
        """
        Convert raw Binance klines data to pandas DataFrame

        Args:
            klines: List of kline arrays from Binance API

        Returns:
            Cleaned DataFrame with OHLCV data
        """
        # Binance klines format:
        # [Open time, Open, High, Low, Close, Volume, Close time, ...]
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Select and convert relevant columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Convert price/volume to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        return df

    def _get_cache_filepath(self, symbol: str, interval: str) -> Path:
        """Generate cache file path"""
        filename = CACHE_FILENAME_TEMPLATE.format(symbol=symbol, interval=interval)
        return self.cache_dir / filename

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str) -> None:
        """Save DataFrame to CSV cache"""
        try:
            filepath = self._get_cache_filepath(symbol, interval)
            df.to_csv(filepath)
            print(f"✓ Saved to cache: {filepath}")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")

    def _load_from_cache(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from CSV cache if exists"""
        try:
            filepath = self._get_cache_filepath(symbol, interval)
            if filepath.exists():
                df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
                return df
        except Exception as e:
            print(f"⚠ Failed to load cache: {e}")

        return None

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data

        Args:
            symbol: If provided, only clear cache for this symbol
                   If None, clear all cache files
        """
        if symbol:
            filepath = self._get_cache_filepath(symbol, DEFAULT_INTERVAL)
            if filepath.exists():
                filepath.unlink()
                print(f"✓ Cleared cache for {symbol}")
        else:
            for cache_file in self.cache_dir.glob("*.csv"):
                cache_file.unlink()
            print("✓ Cleared all cache files")

