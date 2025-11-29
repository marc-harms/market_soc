"""
SOC Metrics Calculator Module
Calculates Self-Organized Criticality metrics for financial time series
"""

from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import SMA_PERIOD, ROLLING_VOLATILITY_WINDOW


class SOCMetricsCalculator:
    """
    Calculates financial metrics for SOC analysis:
    - Daily returns and absolute magnitude
    - Simple Moving Average (trend detection)
    - Rolling volatility (criticality metric)
    - Volatility thresholds for traffic light system
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize calculator with OHLCV data

        Args:
            df: DataFrame with at minimum a 'close' column
        """
        self.df = df.copy()
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        """Validate that required columns exist"""
        if "close" not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' column")

    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        Calculate all SOC metrics and return enriched DataFrame

        Returns:
            DataFrame with all calculated metrics
        """
        print("Calculating SOC metrics...")

        # Calculate returns
        self.df["returns"] = self._calculate_returns()
        self.df["abs_returns"] = self.df["returns"].abs()

        # Calculate moving averages
        self.df["sma_200"] = self._calculate_sma(SMA_PERIOD)

        # Calculate rolling volatility (criticality metric)
        self.df["volatility"] = self._calculate_rolling_volatility(
            ROLLING_VOLATILITY_WINDOW
        )

        # Calculate volatility thresholds and zones
        low_threshold, high_threshold = self._calculate_volatility_thresholds()
        self.df["vol_zone"] = self._assign_volatility_zones(
            low_threshold, high_threshold
        )

        # Store thresholds as attributes for visualization
        self.vol_low_threshold = low_threshold
        self.vol_high_threshold = high_threshold

        # Drop NaN rows (from rolling calculations)
        self.df.dropna(inplace=True)

        print(f"✓ Calculated metrics for {len(self.df)} records")
        print(f"  - Volatility thresholds: Low={low_threshold:.4f}, High={high_threshold:.4f}")

        return self.df

    def _calculate_returns(self) -> pd.Series:
        """
        Calculate daily percentage returns

        Returns:
            Series of daily returns (decimal, not percentage)
        """
        return self.df["close"].pct_change()

    def _calculate_sma(self, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average

        Args:
            period: Number of periods for moving average

        Returns:
            Series with SMA values
        """
        return self.df["close"].rolling(window=period).mean()

    def _calculate_rolling_volatility(self, window: int) -> pd.Series:
        """
        Calculate rolling standard deviation (volatility)
        This is our "Criticality" metric in SOC theory

        Args:
            window: Rolling window size

        Returns:
            Series with rolling volatility values
        """
        return self.df["returns"].rolling(window=window).std()

    def _calculate_volatility_thresholds(self) -> Tuple[float, float]:
        """
        Calculate dynamic volatility thresholds using quantiles (Option A)

        Returns:
            Tuple of (low_threshold, high_threshold)
        """
        from config.settings import VOLATILITY_LOW_PERCENTILE, VOLATILITY_HIGH_PERCENTILE

        volatility = self.df["volatility"].dropna()

        low_threshold = volatility.quantile(VOLATILITY_LOW_PERCENTILE / 100)
        high_threshold = volatility.quantile(VOLATILITY_HIGH_PERCENTILE / 100)

        return low_threshold, high_threshold

    def _assign_volatility_zones(
        self, low_threshold: float, high_threshold: float
    ) -> pd.Series:
        """
        Assign color zones to volatility values

        Args:
            low_threshold: Lower boundary (Green/Orange)
            high_threshold: Upper boundary (Orange/Red)

        Returns:
            Series with zone assignments ('low', 'medium', 'high')
        """
        volatility = self.df["volatility"]

        zones = pd.Series("medium", index=self.df.index)
        zones[volatility <= low_threshold] = "low"
        zones[volatility > high_threshold] = "high"

        return zones

    def get_power_law_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for power law (fat tail) analysis

        Returns:
            Tuple of (abs_returns_sorted, normal_distribution_samples, bins)
        """
        abs_returns = self.df["abs_returns"].dropna().values

        # Sort for log-log histogram
        abs_returns_sorted = np.sort(abs_returns)[::-1]  # Descending

        # Generate theoretical normal distribution with same mean/std (Option A)
        mean = abs_returns.mean()
        std = abs_returns.std()
        normal_samples = np.abs(np.random.normal(mean, std, len(abs_returns)))

        # Create log bins for histogram
        bins = np.logspace(
            np.log10(abs_returns.min() + 1e-8),
            np.log10(abs_returns.max()),
            50
        )

        return abs_returns_sorted, normal_samples, bins

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics for the analysis

        Returns:
            Dictionary with key metrics
        """
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

