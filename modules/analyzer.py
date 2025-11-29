"""
SOC Analyzer Engine
Orchestrates analysis for a single asset and determines market phase
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from modules.soc_metrics import SOCMetricsCalculator

class SOCAnalyzer:
    """
    High-level analyzer for a single asset.
    Wraps metric calculation and determines market phase (Signal).
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df
        self.symbol = symbol
        self.calculator = SOCMetricsCalculator(df)
        self.metrics_df = self.calculator.calculate_all_metrics()
        self.summary_stats = self.calculator.get_summary_stats()

    def get_market_phase(self) -> Dict[str, Any]:
        """
        Determines the current market phase (Traffic Light Logic)
        
        Returns:
            Dictionary containing signal, color, and key metrics
        """
        if self.metrics_df.empty:
            return {"signal": "NO_DATA", "color": "grey", "stress_ratio": 0}

        # Get latest values
        current_price = self.summary_stats["current_price"]
        sma_200 = self.summary_stats["current_sma_200"]
        current_vol = self.metrics_df["volatility"].iloc[-1]
        
        # Calculate Long-term Average Volatility (Stress Baseline)
        avg_vol = self.summary_stats["mean_volatility"]
        stress_ratio = current_vol / avg_vol if avg_vol > 0 else 0
        
        # Volatility Thresholds (from Calculator - Dynamic Quantiles)
        vol_low_thresh = self.calculator.vol_low_threshold
        vol_high_thresh = self.calculator.vol_high_threshold
        
        # Determine Stress Level
        is_high_stress = current_vol > vol_high_thresh
        is_low_stress = current_vol <= vol_low_thresh
        
        # Determine Trend
        is_bullish_trend = current_price > sma_200
        
        # --- SIGNAL LOGIC ---
        signal = "NEUTRAL"
        signal_color = "white"
        
        if is_bullish_trend:
            if is_low_stress:
                signal = "🟢 BUY/ACCUMULATE"
                signal_color = "green"
            elif is_high_stress:
                signal = "🟠 OVERHEATED"
                signal_color = "orange"
            else:
                signal = "⚪ UPTREND (Choppy)"
                signal_color = "white"
        else: # Bearish Trend (Price < SMA)
            if is_high_stress:
                signal = "🔴 CRASH RISK"
                signal_color = "red"
            elif is_low_stress:
                signal = "🟡 CAPITULATION/WAIT"
                signal_color = "yellow"
            else:
                signal = "⚪ DOWNTREND (Choppy)"
                signal_color = "grey"

        return {
            "symbol": self.symbol,
            "price": current_price,
            "sma_200": sma_200,
            "dist_to_sma": (current_price - sma_200) / sma_200,
            "volatility": current_vol,
            "stress_ratio": stress_ratio,
            "signal": signal,
            "color": signal_color,
            "trend": "BULL" if is_bullish_trend else "BEAR",
            "stress": "HIGH" if is_high_stress else ("LOW" if is_low_stress else "MED")
        }

    def get_dataframe(self) -> pd.DataFrame:
        return self.metrics_df
        
    def get_visualizer_data(self):
        """Returns data needed for plotting"""
        return {
            "df": self.metrics_df,
            "symbol": self.symbol,
            "vol_low": self.calculator.vol_low_threshold,
            "vol_high": self.calculator.vol_high_threshold
        }

