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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Volatility Thresholds (Dynamic Quantile-based) - 5-Tier System
VOLATILITY_LOW_PERCENTILE: float = 33.33   # Below = Low Stress
VOLATILITY_MEDIUM_PERCENTILE: float = 50.0  # Below = Normal Stress
VOLATILITY_HIGH_PERCENTILE: float = 80.0    # Above = High Stress (Orange zone)
VOLATILITY_EXTREME_PERCENTILE: float = 99.0 # Above = Extreme (instant Red)

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
    - 5-tier volatility thresholds for traffic light system
    """

    def __init__(self, df: pd.DataFrame, sma_window: int = SMA_PERIOD, vol_window: int = ROLLING_VOLATILITY_WINDOW) -> None:
        self.df = df.copy()
        self.sma_window = sma_window
        self.vol_window = vol_window
        # 5-tier thresholds
        self.vol_low_threshold = 0.0
        self.vol_medium_threshold = 0.0
        self.vol_high_threshold = 0.0
        self.vol_extreme_threshold = 0.0
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        if "close" not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' column")

    def calculate_all_metrics(self) -> pd.DataFrame:
        # Calculate returns
        self.df["returns"] = self.df["close"].pct_change()
        self.df["abs_returns"] = self.df["returns"].abs()

        # Calculate moving averages
        self.df["sma_200"] = self.df["close"].rolling(window=self.sma_window).mean()

        # Calculate rolling volatility (30-day)
        self.df["volatility"] = self.df["returns"].rolling(window=self.vol_window).std()
        
        # Calculate volatility percentile (for criticality score)
        # Rolling 2-year window (504 trading days) for percentile calculation
        lookback = min(504, len(self.df) - 1)
        self.df["vol_percentile"] = self.df["volatility"].rolling(window=lookback).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() * 100 if len(x) > 1 else 50, raw=False
        )

        # Calculate thresholds (5-tier system)
        self._calculate_volatility_thresholds()
        
        # Assign zones (5-tier)
        self.df["vol_zone"] = self._assign_volatility_zones()

        # Drop NaN rows
        self.df.dropna(inplace=True)
        return self.df

    def _calculate_volatility_thresholds(self) -> None:
        """Calculate 5-tier volatility thresholds."""
        volatility = self.df["volatility"].dropna()
        if volatility.empty:
            return
        self.vol_low_threshold = volatility.quantile(VOLATILITY_LOW_PERCENTILE / 100)
        self.vol_medium_threshold = volatility.quantile(VOLATILITY_MEDIUM_PERCENTILE / 100)
        self.vol_high_threshold = volatility.quantile(VOLATILITY_HIGH_PERCENTILE / 100)
        self.vol_extreme_threshold = volatility.quantile(VOLATILITY_EXTREME_PERCENTILE / 100)

    def _assign_volatility_zones(self) -> pd.Series:
        """Assign 5-tier volatility zones."""
        volatility = self.df["volatility"]
        zones = pd.Series("normal", index=self.df.index)
        zones[volatility <= self.vol_low_threshold] = "low"
        zones[(volatility > self.vol_low_threshold) & (volatility <= self.vol_medium_threshold)] = "normal"
        zones[(volatility > self.vol_medium_threshold) & (volatility <= self.vol_high_threshold)] = "medium"
        zones[(volatility > self.vol_high_threshold) & (volatility <= self.vol_extreme_threshold)] = "high"
        zones[volatility > self.vol_extreme_threshold] = "extreme"
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
            "current_vol_percentile": self.df["vol_percentile"].iloc[-1] if "vol_percentile" in self.df.columns else 50,
            "vol_low_threshold": self.vol_low_threshold,
            "vol_medium_threshold": self.vol_medium_threshold,
            "vol_high_threshold": self.vol_high_threshold,
            "vol_extreme_threshold": self.vol_extreme_threshold,
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
            if df.empty:
                return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception:
                    pass
                
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
            except Exception:
                pass

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

    def __init__(self, df: pd.DataFrame, symbol: str, asset_info: Optional[Dict[str, Any]] = None,
                 sma_window: int = SMA_PERIOD, vol_window: int = ROLLING_VOLATILITY_WINDOW, hysteresis: float = 0.0):
        self.df = df
        self.symbol = symbol
        self.asset_info = asset_info or {}
        self.hysteresis = hysteresis
        self.calculator = SOCMetricsCalculator(df, sma_window, vol_window)
        self.metrics_df = self.calculator.calculate_all_metrics()
        self.summary_stats = self.calculator.get_summary_stats()

    def get_market_phase(self) -> Dict[str, Any]:
        """
        5-Tier Traffic Light System:
        ⚪ GREY (Dormant): Trend DOWN + Low Stress = Dead money / Trap
        🟢 GREEN (Accumulation): Trend UP + Low/Normal Stress = Safe entry
        🟡 YELLOW (Active): Trend UP + Medium Stress = Trend maturing, hold
        🟠 ORANGE (Overheated): Trend UP + High Stress (>80th %ile) = FOMO phase, tighten stops
        🔴 RED (Critical): Trend DOWN + High Stress OR Extreme Vol (>99th %ile) = Crash/Liquidation risk
        """
        if self.metrics_df.empty:
            return {"signal": "NO_DATA", "color": "grey", "criticality_score": 0}

        # Current values
        current_price = self.summary_stats["current_price"]
        sma_200 = self.summary_stats["current_sma_200"]
        current_vol = self.metrics_df["volatility"].iloc[-1]
        vol_percentile = self.summary_stats.get("current_vol_percentile", 50)
        
        # Thresholds (5-tier)
        vol_low = self.calculator.vol_low_threshold
        vol_medium = self.calculator.vol_medium_threshold
        vol_high = self.calculator.vol_high_threshold
        vol_extreme = self.calculator.vol_extreme_threshold
        
        # Trend calculation with hysteresis
        upper_bound = sma_200 * (1.0 + self.hysteresis)
        lower_bound = sma_200 * (1.0 - self.hysteresis)
        
        is_uptrend = current_price > upper_bound
        is_downtrend = current_price < lower_bound
        
        # Distance from SMA (for parabolic detection)
        sma_distance_pct = ((current_price - sma_200) / sma_200) * 100 if sma_200 > 0 else 0
        is_parabolic = sma_distance_pct > 30  # >30% above SMA = parabolic
        
        # Stress levels (5-tier)
        is_low_stress = current_vol <= vol_low
        is_normal_stress = vol_low < current_vol <= vol_medium
        is_medium_stress = vol_medium < current_vol <= vol_high
        is_high_stress = vol_high < current_vol <= vol_extreme
        is_extreme_stress = current_vol > vol_extreme
        
        # --- Calculate Criticality Score (0-100) ---
        # Base: Volatility percentile (0-100)
        criticality_score = vol_percentile
        
        # Modifier: Downtrend adds +10 (risk is higher)
        if is_downtrend:
            criticality_score += 10
        
        # Modifier: Parabolic adds +10 (overextension risk)
        if is_parabolic:
            criticality_score += 10
        
        # Clamp to 0-100
        criticality_score = max(0, min(100, criticality_score))
        
        # --- Determine Regime (5-Tier Energy State System) ---
        # Compliance-safe: Describes market state, NOT investment advice
        signal = "⚪ DORMANT REGIME"
        tier = "DORMANT"
        trend_label = "NEUTRAL"
        stress_label = "LOW"
        
        # Extreme volatility always triggers CRITICAL (regardless of trend)
        if is_extreme_stress:
            signal = "🔴 CRITICAL REGIME"
            tier = "CRITICAL"
            trend_label = "VOLATILE"
            stress_label = "EXTREME"
        
        # Uptrend scenarios
        elif is_uptrend:
            trend_label = "UP"
            if is_low_stress or is_normal_stress:
                signal = "🟢 STABLE REGIME"
                tier = "STABLE"
                stress_label = "LOW"
            elif is_medium_stress:
                signal = "🟡 ACTIVE REGIME"
                tier = "ACTIVE"
                stress_label = "MEDIUM"
            elif is_high_stress:
                signal = "🟠 HIGH ENERGY REGIME"
                tier = "HIGH_ENERGY"
                stress_label = "HIGH"
        
        # Downtrend scenarios
        elif is_downtrend:
            trend_label = "DOWN"
            if is_high_stress:
                signal = "🔴 CRITICAL REGIME"
                tier = "CRITICAL"
                stress_label = "HIGH"
            else:
                # Low/Normal/Medium stress in downtrend = Dormant
                signal = "⚪ DORMANT REGIME"
                tier = "DORMANT"
                stress_label = "LOW" if is_low_stress else ("NORMAL" if is_normal_stress else "MEDIUM")
        
        # Neutral (between bounds)
        else:
            trend_label = "NEUTRAL"
            if is_high_stress:
                signal = "🟡 ACTIVE REGIME"
                tier = "ACTIVE"
                stress_label = "HIGH"
            elif is_medium_stress:
                signal = "🟡 ACTIVE REGIME"
                tier = "ACTIVE"
                stress_label = "MEDIUM"
            else:
                signal = "🟢 STABLE REGIME"
                tier = "STABLE"
                stress_label = "LOW" if is_low_stress else "NORMAL"

        return {
            "symbol": self.symbol,
            "price": current_price,
            "sma_200": sma_200,
            "dist_to_sma": sma_distance_pct / 100,  # As decimal
            "dist_to_sma_pct": sma_distance_pct,
            "volatility": current_vol,
            "vol_percentile": vol_percentile,
            "vol_threshold_high": vol_high,
            "criticality_score": round(criticality_score),
            "stress_score": current_vol / vol_high if vol_high > 0 else 0,
            "signal": signal,
            "tier": tier,
            "trend": trend_label,
            "stress": stress_label,
            "is_parabolic": is_parabolic
        }

    def get_plotly_figures(self, dark_mode: bool = True) -> Dict[str, go.Figure]:
        """
        Returns Plotly figures for Streamlit display.
        
        Args:
            dark_mode: If True, use dark theme. If False, use light theme.
            
        Returns:
            Dictionary containing the criticality chart (chart3).
        """
        figures = {}
        
        # Theme settings - explicit colors for full control
        if dark_mode:
            template = "plotly_dark"
            paper_bg = "#0E1117"
            plot_bg = "#0E1117"
            price_line_color = "white"
            text_color = "#FAFAFA"
            grid_color = "#333333"
            sma_color = "#FFD700"
        else:
            template = "plotly_white"
            paper_bg = "#FFFFFF"
            plot_bg = "#FFFFFF"
            price_line_color = "#1f77b4"  # Blue for better visibility on white
            text_color = "#212529"
            grid_color = "#E5E5E5"
            sma_color = "#DAA520"  # Darker gold for light mode
        
        # Get asset name for caption
        asset_name = self.asset_info.get('name', self.symbol) if self.asset_info else self.symbol
        
        # --- Chart 3: Criticality (Main SOC Chart) ---
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Map colors for 5-tier volatility zones
        color_map = {
            "low": "#00cc00",      # Green - safe
            "normal": "#88cc00",   # Light green - normal
            "medium": "#ffcc00",   # Yellow - caution
            "high": "#ff6600",     # Orange - elevated
            "extreme": "#ff0000"   # Red - critical
        }
        colors = [color_map.get(z, "#ffcc00") for z in self.metrics_df["vol_zone"]]
        
        # Volatility bars
        fig3.add_trace(go.Bar(
            x=self.metrics_df.index,
            y=self.metrics_df["volatility"],
            name="Volatility",
            marker_color=colors,
            marker_line_width=0
        ), secondary_y=False)
        
        # Price line
        fig3.add_trace(go.Scatter(
            x=self.metrics_df.index,
            y=self.metrics_df["close"],
            name="Price",
            line=dict(color=price_line_color, width=1.5)
        ), secondary_y=True)
        
        # SMA 200 line
        fig3.add_trace(go.Scatter(
            x=self.metrics_df.index,
            y=self.metrics_df["sma_200"],
            name="SMA 200",
            line=dict(color=sma_color, width=1)
        ), secondary_y=True)
        
        # Threshold lines
        if self.calculator.vol_low_threshold:
            fig3.add_hline(
                y=self.calculator.vol_low_threshold, 
                line_dash="dot", 
                line_color="green", 
                secondary_y=False,
                annotation_text="Low Vol",
                annotation_position="right"
            )
        if self.calculator.vol_high_threshold:
            fig3.add_hline(
                y=self.calculator.vol_high_threshold, 
                line_dash="dot", 
                line_color="red", 
                secondary_y=False,
                annotation_text="High Vol",
                annotation_position="right"
            )
        
        # Layout with explicit background colors and centered caption
        fig3.update_layout(
            template=template,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=text_color)
            ),
            margin=dict(b=80)
        )
        
        # Update axes separately for better compatibility
        fig3.update_yaxes(
            title_text="Price (Log)",
            type="log",
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color),
            gridcolor=grid_color,
            secondary_y=True
        )
        fig3.update_yaxes(
            title_text="Volatility",
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color),
            gridcolor=grid_color,
            secondary_y=False
        )
        fig3.update_xaxes(
            title_text=f"<b>{asset_name} - SOC Analysis</b>",
            title_font=dict(size=14, color=text_color),
            title_standoff=25,
            tickformat="%Y",
            tickfont=dict(color=text_color),
            gridcolor=grid_color
        )
        
        figures["chart3"] = fig3
        
        return figures

    def get_historical_signal_analysis(self) -> Dict[str, Any]:
        """
        Analyze historical signals using 5-tier traffic light system.
        Includes short-term (1d/3d/5d/10d), long-term (30d/60d/90d), and max drawdown analysis.
        
        Returns:
            Dictionary containing signal statistics and prose report.
        """
        try:
            if self.metrics_df is None or self.metrics_df.empty or len(self.metrics_df) < 50:
                return {"error": "Insufficient data for historical analysis"}
            
            df = self.metrics_df.copy()
            
            # Ensure required columns exist
            required_cols = ['close', 'sma_200', 'volatility']
            for col in required_cols:
                if col not in df.columns:
                    return {"error": f"Missing required column: {col}"}
            
            # Drop any remaining NaN values in critical columns
            df = df.dropna(subset=['close', 'sma_200', 'volatility'])
            
            if len(df) < 50:
                return {"error": "Insufficient data after cleaning"}
            
            # 5-tier thresholds
            vol_low = self.calculator.vol_low_threshold
            vol_medium = self.calculator.vol_medium_threshold
            vol_high = self.calculator.vol_high_threshold
            vol_extreme = self.calculator.vol_extreme_threshold
            
            # Handle None thresholds
            if vol_low is None or vol_high is None:
                return {"error": "Could not calculate volatility thresholds"}
            
            # Determine trend and stress for each row
            df['is_uptrend'] = df['close'] > df['sma_200']
            df['is_downtrend'] = df['close'] < df['sma_200']
            df['is_low_vol'] = df['volatility'] <= vol_low
            df['is_normal_vol'] = (df['volatility'] > vol_low) & (df['volatility'] <= vol_medium)
            df['is_medium_vol'] = (df['volatility'] > vol_medium) & (df['volatility'] <= vol_high)
            df['is_high_vol'] = (df['volatility'] > vol_high) & (df['volatility'] <= vol_extreme)
            df['is_extreme_vol'] = df['volatility'] > vol_extreme
            
            # 5-Tier regime assignment (Compliance-safe naming)
            def assign_signal(row):
                # Extreme volatility always = CRITICAL
                if row['is_extreme_vol']:
                    return 'CRITICAL'
                
                # Uptrend scenarios
                if row['is_uptrend']:
                    if row['is_low_vol'] or row['is_normal_vol']:
                        return 'STABLE'
                    elif row['is_medium_vol']:
                        return 'ACTIVE'
                    elif row['is_high_vol']:
                        return 'HIGH_ENERGY'
                
                # Downtrend scenarios
                elif row['is_downtrend']:
                    if row['is_high_vol']:
                        return 'CRITICAL'
                    else:
                        return 'DORMANT'
                
                # Neutral (near SMA)
                else:
                    if row['is_high_vol'] or row['is_medium_vol']:
                        return 'ACTIVE'
                    else:
                        return 'STABLE'
                
                return 'DORMANT'
            
            df['signal'] = df.apply(assign_signal, axis=1)
            
            # Calculate PRIOR returns - what happened BEFORE the signal (looking backward)
            df['prior_5d'] = df['close'].pct_change(5)
            df['prior_10d'] = df['close'].pct_change(10)
            df['prior_20d'] = df['close'].pct_change(20)
            df['prior_30d'] = df['close'].pct_change(30)
            
            # Calculate forward returns - SHORT TERM (from signal start)
            df['return_1d'] = df['close'].pct_change(1).shift(-1)
            df['return_3d'] = df['close'].pct_change(3).shift(-3)
            df['return_5d'] = df['close'].pct_change(5).shift(-5)
            df['return_10d'] = df['close'].pct_change(10).shift(-10)
            
            # Calculate forward returns - LONG TERM
            df['return_30d'] = df['close'].pct_change(30).shift(-30)
            df['return_60d'] = df['close'].pct_change(60).shift(-60)
            df['return_90d'] = df['close'].pct_change(90).shift(-90)
            
            # Calculate MAX DRAWDOWN over next 10 days (worst-case drop)
            def calc_max_drawdown_10d(idx):
                """Calculate max drawdown in 10 days following this index."""
                loc = df.index.get_loc(idx)
                if loc + 10 >= len(df):
                    return np.nan
                future_prices = df['close'].iloc[loc:loc+11]
                start_price = future_prices.iloc[0]
                min_price = future_prices.min()
                return ((min_price - start_price) / start_price) * 100
            
            df['max_dd_10d'] = df.index.map(calc_max_drawdown_10d)
            
            # Identify signal phases (consecutive periods of same signal)
            df['signal_change'] = (df['signal'] != df['signal'].shift()).cumsum()
            
            # Mark first day of each phase (signal START)
            df['is_phase_start'] = df['signal'] != df['signal'].shift()
            
            # 5-Tier regime types (Compliance-safe naming)
            signal_types = ['STABLE', 'ACTIVE', 'HIGH_ENERGY', 'CRITICAL', 'DORMANT']
            
            # Analyze each signal type
            signal_stats = {}
            for signal_type in signal_types:
                signal_df = df[df['signal'] == signal_type]
                
                if len(signal_df) == 0:
                    signal_stats[signal_type] = {
                        'total_days': 0,
                        'phase_count': 0,
                        # Duration statistics
                        'avg_duration': 0,
                        'median_duration': 0,
                        'p95_duration': 0,
                        'max_duration': 0,
                        'min_duration': 0,
                        'std_duration': 0,
                        'avg_price_change_during': 0,
                        'pct_of_time': 0,
                        # PRIOR returns
                        'prior_5d': 0,
                        'prior_10d': 0,
                        'prior_20d': 0,
                        'prior_30d': 0,
                        # Short-term returns (from signal START only)
                        'start_return_1d': 0,
                        'start_return_3d': 0,
                        'start_return_5d': 0,
                        'start_return_10d': 0,
                        # Long-term returns (any day in phase)
                        'avg_return_30d': 0,
                        'avg_return_60d': 0,
                        'avg_return_90d': 0,
                        # Max drawdown
                        'avg_max_dd_10d': 0,
                        'worst_max_dd_10d': 0
                    }
                    continue
                
                # Get only the FIRST DAY of each phase for short-term analysis
                phase_starts = signal_df[signal_df['is_phase_start']]
                
                # Count distinct phases
                phases = signal_df.groupby('signal_change').agg({
                    'close': ['first', 'last', 'count'],
                    'return_30d': 'first',  # Return from phase start
                    'return_60d': 'first',
                    'return_90d': 'first',
                    'max_dd_10d': 'first'   # Max drawdown from phase start
                })
                phases.columns = ['price_start', 'price_end', 'duration', 'ret_30', 'ret_60', 'ret_90', 'max_dd']
                phases['price_change_pct'] = (phases['price_end'] - phases['price_start']) / phases['price_start'] * 100
                
                # Calculate PRIOR returns (what happened before signal fired) - from phase START only
                prior_5d = phase_starts['prior_5d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['prior_5d'].isna().all() else 0
                prior_10d = phase_starts['prior_10d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['prior_10d'].isna().all() else 0
                prior_20d = phase_starts['prior_20d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['prior_20d'].isna().all() else 0
                prior_30d = phase_starts['prior_30d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['prior_30d'].isna().all() else 0
                
                # Calculate short-term FORWARD returns from phase START only
                start_1d = phase_starts['return_1d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['return_1d'].isna().all() else 0
                start_3d = phase_starts['return_3d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['return_3d'].isna().all() else 0
                start_5d = phase_starts['return_5d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['return_5d'].isna().all() else 0
                start_10d = phase_starts['return_10d'].mean() * 100 if len(phase_starts) > 0 and not phase_starts['return_10d'].isna().all() else 0
                
                # Max drawdown stats
                avg_max_dd = phases['max_dd'].mean() if len(phases) > 0 and not phases['max_dd'].isna().all() else 0
                worst_max_dd = phases['max_dd'].min() if len(phases) > 0 and not phases['max_dd'].isna().all() else 0
                
                # Duration statistics for regime persistence analysis
                durations = phases['duration']
                avg_dur = durations.mean() if len(phases) > 0 else 0
                median_dur = durations.median() if len(phases) > 0 else 0
                p95_dur = durations.quantile(0.95) if len(phases) > 0 else 0
                max_dur = durations.max() if len(phases) > 0 else 0
                min_dur = durations.min() if len(phases) > 0 else 0
                std_dur = durations.std() if len(phases) > 1 else 0
                
                signal_stats[signal_type] = {
                    'total_days': len(signal_df),
                    'phase_count': len(phases),
                    # Duration statistics (for regime persistence analysis)
                    'avg_duration': avg_dur,
                    'median_duration': median_dur,
                    'p95_duration': p95_dur,
                    'max_duration': max_dur,
                    'min_duration': min_dur,
                    'std_duration': std_dur,
                    'avg_price_change_during': phases['price_change_pct'].mean() if len(phases) > 0 else 0,
                    'pct_of_time': len(signal_df) / len(df) * 100,
                    # PRIOR returns (what led to this signal)
                    'prior_5d': prior_5d,
                    'prior_10d': prior_10d,
                    'prior_20d': prior_20d,
                    'prior_30d': prior_30d,
                    # Short-term FORWARD returns (from signal START only)
                    'start_return_1d': start_1d,
                    'start_return_3d': start_3d,
                    'start_return_5d': start_5d,
                    'start_return_10d': start_10d,
                    # Long-term FORWARD returns (from phase start)
                    'avg_return_30d': phases['ret_30'].mean() * 100 if len(phases) > 0 and not phases['ret_30'].isna().all() else 0,
                    'avg_return_60d': phases['ret_60'].mean() * 100 if len(phases) > 0 and not phases['ret_60'].isna().all() else 0,
                    'avg_return_90d': phases['ret_90'].mean() * 100 if len(phases) > 0 and not phases['ret_90'].isna().all() else 0,
                    # Max drawdown (worst-case)
                    'avg_max_dd_10d': avg_max_dd,
                    'worst_max_dd_10d': worst_max_dd
                }
            
            # Current signal streak
            current_signal = df['signal'].iloc[-1]
            current_streak = 1
            for i in range(len(df) - 2, -1, -1):
                if df['signal'].iloc[i] == current_signal:
                    current_streak += 1
                else:
                    break
            
            # Data range info
            years_of_data = (df.index[-1] - df.index[0]).days / 365.25
            
            # Generate prose report
            report = self._generate_prose_report(
                signal_stats, 
                current_signal, 
                current_streak, 
                years_of_data,
                df
            )
        
            # Calculate crash warning score
            crash_warning = self._calculate_crash_warning_score(df, signal_stats, current_signal, current_streak)
            
            return {
                'signal_stats': signal_stats,
                'current_signal': current_signal,
                'current_streak_days': current_streak,
                'years_of_data': years_of_data,
                'total_trading_days': len(df),
                'data_start': df.index[0].strftime('%Y-%m-%d'),
                'data_end': df.index[-1].strftime('%Y-%m-%d'),
                'prose_report': report,
                'crash_warning': crash_warning
            }
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}
    
    def _calculate_crash_warning_score(self, df: pd.DataFrame, signal_stats: Dict, 
                                        current_signal: str, current_streak: int) -> Dict[str, Any]:
        """
        Calculate Systemic Stress Level (0-100) based on current market conditions.
        Compliance-safe: Provides statistical analysis, NOT investment advice.
        
        Statistical Factors:
        - Current volatility percentile (25%)
        - Price deviation from SMA 200 (25%)
        - Duration in current regime (20%)
        - Volatility rate of change (15%)
        - Historical pattern correlation (15%)
        """
        asset_name = self.asset_info.get('name', self.symbol) if self.asset_info else self.symbol
        
        factors = {}
        score = 0
        risk_factors = []
        
        # Current values
        current_vol = df['volatility'].iloc[-1]
        current_price = df['close'].iloc[-1]
        current_sma = df['sma_200'].iloc[-1]
        
        # --- Factor 1: Volatility Percentile (25%) ---
        vol_percentile = (df['volatility'] < current_vol).mean() * 100
        vol_score = min(25, (vol_percentile / 100) * 25)
        factors['volatility_percentile'] = vol_percentile
        score += vol_score
        
        if vol_percentile >= 80:
            risk_factors.append(f"📊 Volatility at {vol_percentile:.0f}th percentile (statistically rare)")
        elif vol_percentile >= 60:
            risk_factors.append(f"📊 Volatility at {vol_percentile:.0f}th percentile (above median)")
        
        # --- Factor 2: Price Distance from SMA 200 (25%) ---
        if current_sma > 0:
            sma_distance = ((current_price - current_sma) / current_sma) * 100
            factors['sma_distance_pct'] = sma_distance
            
            # Statistical deviation analysis
            if sma_distance > 0:
                # Above SMA deviation
                distance_score = min(25, (sma_distance / 30) * 25)
                score += distance_score
                
                if sma_distance >= 30:
                    risk_factors.append(f"📊 Price deviation +{sma_distance:.1f}% from SMA200 (>2 std dev)")
                elif sma_distance >= 20:
                    risk_factors.append(f"📊 Price deviation +{sma_distance:.1f}% from SMA200 (statistically elevated)")
                elif sma_distance >= 10:
                    risk_factors.append(f"📊 Price deviation +{sma_distance:.1f}% from SMA200")
            else:
                # Below SMA deviation
                if sma_distance < -20:
                    score += 10
                    risk_factors.append(f"📊 Price deviation {sma_distance:.1f}% from SMA200 (below trend)")
        else:
            factors['sma_distance_pct'] = 0
        
        # --- Factor 3: Duration in Current Regime (20%) ---
        if current_signal in ['HIGH_ENERGY', '🟠 HIGH ENERGY REGIME']:
            streak_score = min(20, (current_streak / 30) * 20)
            score += streak_score
            factors['days_in_regime'] = current_streak
            
            if current_streak >= 20:
                risk_factors.append(f"📊 {current_streak} consecutive days in HIGH ENERGY regime")
            elif current_streak >= 10:
                risk_factors.append(f"📊 {current_streak} days in HIGH ENERGY regime")
        elif current_signal in ['CRITICAL', '🔴 CRITICAL REGIME']:
            score += 20
            factors['days_in_regime'] = current_streak
            risk_factors.append(f"📊 Currently in CRITICAL regime ({current_streak} days)")
        elif current_signal in ['DORMANT', '⚪ DORMANT REGIME']:
            factors['days_in_regime'] = current_streak
            if current_streak >= 30:
                score += 10
                risk_factors.append(f"📊 {current_streak} days in DORMANT regime (low activity)")
        else:
            factors['days_in_regime'] = 0
        
        # --- Factor 4: Volatility Acceleration (15%) ---
        if len(df) >= 20:
            vol_5d_ago = df['volatility'].iloc[-5] if len(df) >= 5 else current_vol
            vol_20d_ago = df['volatility'].iloc[-20] if len(df) >= 20 else current_vol
            
            vol_change_5d = ((current_vol - vol_5d_ago) / vol_5d_ago * 100) if vol_5d_ago > 0 else 0
            vol_change_20d = ((current_vol - vol_20d_ago) / vol_20d_ago * 100) if vol_20d_ago > 0 else 0
            
            factors['vol_acceleration_5d'] = vol_change_5d
            factors['vol_acceleration_20d'] = vol_change_20d
            
            # Volatility rate of change analysis
            if vol_change_5d > 50:
                score += 15
                risk_factors.append(f"📊 Volatility increased +{vol_change_5d:.0f}% over 5 days (rapid change)")
            elif vol_change_5d > 25:
                score += 10
                risk_factors.append(f"📊 Volatility increased +{vol_change_5d:.0f}% over 5 days")
            elif vol_change_20d > 50:
                score += 8
                risk_factors.append(f"📊 Volatility increased +{vol_change_20d:.0f}% over 20 days")
        
        # --- Factor 5: Historical Pattern Correlation (15%) ---
        high_energy_stats = signal_stats.get('HIGH_ENERGY', {})
        critical_stats = signal_stats.get('CRITICAL', {})
        
        # Check HIGH ENERGY regime historical statistics
        if high_energy_stats.get('phase_count', 0) > 0:
            worst_dd = high_energy_stats.get('worst_max_dd_10d', 0)
            if worst_dd < -10:
                score += 10
                risk_factors.append(f"📊 Historical: HIGH ENERGY regimes showed {worst_dd:.1f}% max 10d variance")
            factors['high_energy_worst_dd'] = worst_dd
        
        # Check CRITICAL regime historical statistics
        if critical_stats.get('phase_count', 0) > 0:
            worst_dd = critical_stats.get('worst_max_dd_10d', 0)
            if worst_dd < -15:
                score += 15
                risk_factors.append(f"📊 Historical: CRITICAL regimes showed {worst_dd:.1f}% max 10d variance")
            factors['critical_worst_dd'] = worst_dd
        
        # --- Calculate final score and level ---
        score = min(100, max(0, score))
        
        # Compliance-safe interpretations (statistical, not advisory)
        if score >= 75:
            level = "ELEVATED"
            level_color = "#FF0000"
            level_emoji = "📊"
            interpretation = "Current market phase shows statistical similarities to previous high-volatility periods with increased cross-asset correlation."
        elif score >= 50:
            level = "HEIGHTENED"
            level_color = "#FF6600"
            level_emoji = "📊"
            interpretation = "Statistical indicators suggest above-average market energy. Historically, such conditions precede regime transitions."
        elif score >= 25:
            level = "MODERATE"
            level_color = "#FFCC00"
            level_emoji = "📊"
            interpretation = "Market conditions within normal statistical parameters with some elevated metrics."
        else:
            level = "BASELINE"
            level_color = "#00CC00"
            level_emoji = "📊"
            interpretation = "Current market phase shows characteristics of a low-volatility trend with metrics near historical averages."
        
        return {
            'score': round(score),
            'level': level,
            'level_color': level_color,
            'level_emoji': level_emoji,
            'interpretation': interpretation,
            'risk_factors': risk_factors,
            'factors': factors,
            'asset_name': asset_name
        }
    
    def _generate_prose_report(self, stats: Dict, current_signal: str, streak: int, 
                                years: float, df: pd.DataFrame) -> str:
        """
        Generate a compliance-safe statistical report from regime data.
        
        COMPLIANCE: This report provides EDUCATIONAL market data analysis only.
        NO investment advice, recommendations, or imperatives (buy/sell/hold).
        All statements are historical/statistical observations.
        """
        
        asset_name = self.asset_info.get('name', self.symbol) if self.asset_info else self.symbol
        total_days = len(df)
        
        report_parts = []
        
        # Header
        report_parts.append(f"### 📊 Statistical Regime Analysis: {asset_name}")
        report_parts.append(f"*Based on {years:.1f} years of historical data ({total_days:,} trading days)*\n")
        
        # Current status with regime emoji
        signal_emoji = {
            'STABLE': '🟢', 'ACTIVE': '🟡', 'HIGH_ENERGY': '🟠', 'CRITICAL': '🔴', 'DORMANT': '⚪'
        }
        signal_name = {
            'STABLE': 'STABLE REGIME', 'ACTIVE': 'ACTIVE REGIME', 
            'HIGH_ENERGY': 'HIGH ENERGY REGIME', 'CRITICAL': 'CRITICAL REGIME', 'DORMANT': 'DORMANT REGIME'
        }
        emoji = signal_emoji.get(current_signal, '⚪')
        name = signal_name.get(current_signal, current_signal)
        report_parts.append(f"**Current Regime:** {emoji} **{name}** — *{streak} consecutive days in this state*\n")
        
        # --- REGIME PERSISTENCE ANALYSIS (Deep Statistical Context) ---
        current_stats = stats.get(current_signal, {})
        if current_stats.get('phase_count', 0) > 0:
            avg_dur = current_stats.get('avg_duration', 0)
            median_dur = current_stats.get('median_duration', 0)
            p95_dur = current_stats.get('p95_duration', 0)
            max_dur = current_stats.get('max_duration', 0)
            phase_count = current_stats.get('phase_count', 0)
            
            report_parts.append("---")
            report_parts.append("#### ⏱️ Regime Persistence Analysis")
            report_parts.append(
                f"**Current Duration:** **{streak} days** in {name}"
            )
            report_parts.append("")
            report_parts.append(f"**Historical Duration Statistics for {name}** (based on {phase_count} previous occurrences):")
            report_parts.append(
                f"| Metric | Value |"
            )
            report_parts.append(
                f"|:-------|------:|"
            )
            report_parts.append(
                f"| Ø Mean (Average) | **{avg_dur:.1f} days** |"
            )
            report_parts.append(
                f"| Median (Typical) | **{median_dur:.1f} days** |"
            )
            report_parts.append(
                f"| 95th Percentile (Extreme) | **{p95_dur:.1f} days** |"
            )
            report_parts.append(
                f"| Maximum Observed | **{max_dur:.0f} days** |"
            )
            
            # Contextual insight based on current duration vs historical
            report_parts.append("")
            if p95_dur > 0 and streak > p95_dur:
                report_parts.append(
                    f"⚠️ **Statistical Anomaly:** Current duration ({streak} days) exceeds the 95th percentile ({p95_dur:.0f} days). "
                    f"Historically, only 5% of {name} periods lasted this long. Mean reversion probability statistically elevated."
                )
            elif median_dur > 0 and streak < median_dur:
                report_parts.append(
                    f"ℹ️ **Early Phase:** Current duration ({streak} days) is below the median ({median_dur:.0f} days). "
                    f"This regime is still in its typical early-to-mid development phase."
                )
            elif avg_dur > 0 and streak >= avg_dur and streak <= p95_dur:
                report_parts.append(
                    f"📊 **Mature Phase:** Current duration ({streak} days) is at or above average ({avg_dur:.0f} days) "
                    f"but within normal statistical range."
                )
            else:
                report_parts.append(
                    f"📊 Current duration is within normal historical parameters for this regime."
                )
            report_parts.append("")
        
        # --- CRITICAL REGIME Analysis ---
        critical = stats.get('CRITICAL', {})
        if critical.get('phase_count', 0) > 0:
            report_parts.append("---")
            report_parts.append("#### 🔴 CRITICAL REGIME Periods")
            report_parts.append(
                f"Historically, **{asset_name}** has entered the CRITICAL regime **{critical['phase_count']} times**, "
                f"representing **{critical['pct_of_time']:.1f}%** of the analyzed period."
            )
            report_parts.append(
                f"- Average duration: **{critical['avg_duration']:.0f} days** (longest: {critical['max_duration']:.0f} days)"
            )
            report_parts.append(
                f"- Historical price variance during CRITICAL regime: **{critical['avg_price_change_during']:+.1f}%**"
            )
            
            # Statistical variance analysis
            report_parts.append("")
            report_parts.append("**📊 10-Day Price Variance (Historical):**")
            report_parts.append(
                f"- Mean max variance: **{critical.get('avg_max_dd_10d', 0):.1f}%**"
            )
            report_parts.append(
                f"- Extreme variance observed: **{critical.get('worst_max_dd_10d', 0):.1f}%**"
            )
            
            # Historical returns data
            report_parts.append("")
            report_parts.append("**📊 Historical Returns Following CRITICAL Regime:**")
            report_parts.append(
                f"| 1d | 3d | 5d | 10d | 30d |"
            )
            report_parts.append(
                f"|:--:|:--:|:--:|:---:|:---:|"
            )
            report_parts.append(
                f"| {critical.get('start_return_1d', 0):+.1f}% | {critical.get('start_return_3d', 0):+.1f}% | "
                f"{critical.get('start_return_5d', 0):+.1f}% | {critical.get('start_return_10d', 0):+.1f}% | "
                f"{critical.get('avg_return_30d', 0):+.1f}% |"
            )
        
        # --- HIGH ENERGY REGIME Analysis ---
        high_energy = stats.get('HIGH_ENERGY', {})
        if high_energy.get('phase_count', 0) > 0:
            report_parts.append("")
            report_parts.append("#### 🟠 HIGH ENERGY REGIME Periods")
            report_parts.append(
                f"**{high_energy['phase_count']} HIGH ENERGY periods** observed ({high_energy['pct_of_time']:.1f}% of time)."
            )
            report_parts.append(
                f"- Average duration: **{high_energy['avg_duration']:.0f} days** | "
                f"Price variance: **{high_energy['avg_price_change_during']:+.1f}%**"
            )
            
            # Variance stats
            report_parts.append("")
            report_parts.append("**📊 10-Day Price Variance (Historical):**")
            report_parts.append(
                f"- Mean: **{high_energy.get('avg_max_dd_10d', 0):.1f}%** | "
                f"Extreme: **{high_energy.get('worst_max_dd_10d', 0):.1f}%**"
            )
            
            # Historical returns
            report_parts.append("")
            report_parts.append("**📊 Historical Returns Following HIGH ENERGY Regime:**")
            report_parts.append(
                f"| 1d | 3d | 5d | 10d | 30d |"
            )
            report_parts.append(
                f"|:--:|:--:|:--:|:---:|:---:|"
            )
            report_parts.append(
                f"| {high_energy.get('start_return_1d', 0):+.1f}% | {high_energy.get('start_return_3d', 0):+.1f}% | "
                f"{high_energy.get('start_return_5d', 0):+.1f}% | {high_energy.get('start_return_10d', 0):+.1f}% | "
                f"{high_energy.get('avg_return_30d', 0):+.1f}% |"
            )
        
        # --- STABLE REGIME Analysis ---
        stable = stats.get('STABLE', {})
        if stable.get('phase_count', 0) > 0:
            report_parts.append("")
            report_parts.append("#### 🟢 STABLE REGIME Periods")
            report_parts.append(
                f"**{stable['phase_count']} STABLE periods** identified ({stable['pct_of_time']:.1f}% of time)."
            )
            report_parts.append(
                f"- Average duration: **{stable['avg_duration']:.0f} days** | "
                f"Historical price change: **{stable['avg_price_change_during']:+.1f}%**"
            )
            report_parts.append("")
            report_parts.append("**📊 Historical Returns Following STABLE Regime:**")
            report_parts.append(
                f"| 10d | 30d | 60d | 90d |"
            )
            report_parts.append(
                f"|:---:|:---:|:---:|:---:|"
            )
            report_parts.append(
                f"| {stable.get('start_return_10d', 0):+.1f}% | {stable.get('avg_return_30d', 0):+.1f}% | "
                f"{stable.get('avg_return_60d', 0):+.1f}% | {stable.get('avg_return_90d', 0):+.1f}% |"
            )
        
        # --- DORMANT REGIME Analysis ---
        dormant = stats.get('DORMANT', {})
        if dormant.get('phase_count', 0) > 0:
            report_parts.append("")
            report_parts.append("#### ⚪ DORMANT REGIME Periods")
            report_parts.append(
                f"**{dormant['phase_count']} DORMANT periods** observed ({dormant['pct_of_time']:.1f}% of time)."
            )
            report_parts.append(
                f"- Characterized by: below-trend price with low volatility (reduced market activity)"
            )
            report_parts.append(
                f"- Average duration: **{dormant['avg_duration']:.0f} days** | "
                f"Historical price change: **{dormant['avg_price_change_during']:+.1f}%**"
            )
        
        # --- STATISTICAL CONTEXT (Compliance-safe: NO advice/imperatives) ---
        report_parts.append("")
        report_parts.append("---")
        report_parts.append("#### 📊 Current Regime Context")
        
        # Generate statistical observations based on current regime
        if current_signal == 'STABLE':
            avg_30d = stable.get('avg_return_30d', 0)
            report_parts.append(
                f"**{asset_name}** is currently in a STABLE REGIME. "
                f"Historically, this phase is characterized by low volatility with price above trend. "
                f"Historical 30-day returns from similar regimes: **{avg_30d:+.1f}%** (mean)."
            )
        
        elif current_signal == 'ACTIVE':
            active = stats.get('ACTIVE', {})
            avg_30d = active.get('avg_return_30d', 0)
            report_parts.append(
                f"**{asset_name}** is in an ACTIVE REGIME. "
                f"This phase shows moderate volatility with positive trend momentum. "
                f"Historical 30-day returns from similar regimes: **{avg_30d:+.1f}%** (mean)."
            )
        
        elif current_signal == 'HIGH_ENERGY':
            avg_30d = high_energy.get('avg_return_30d', 0)
            worst_dd = high_energy.get('worst_max_dd_10d', 0)
            report_parts.append(
                f"**{asset_name}** is in a HIGH ENERGY REGIME. "
                f"Historically, this phase exhibits elevated volatility and increased price dispersion. "
                f"Historical 30-day returns: **{avg_30d:+.1f}%** (mean). "
                f"Maximum 10-day variance observed: **{worst_dd:.1f}%**."
            )
        
        elif current_signal == 'CRITICAL':
            avg_30d = critical.get('avg_return_30d', 0)
            worst_dd = critical.get('worst_max_dd_10d', 0)
            report_parts.append(
                f"**{asset_name}** is in a CRITICAL REGIME. "
                f"Current market phase shows statistical similarities to previous high-volatility correction periods. "
                f"Historical 30-day returns: **{avg_30d:+.1f}%** (mean). "
                f"Maximum 10-day variance observed: **{worst_dd:.1f}%**."
            )
        
        elif current_signal == 'DORMANT':
            avg_30d = dormant.get('avg_return_30d', 0)
            report_parts.append(
                f"**{asset_name}** is in a DORMANT REGIME. "
                f"This phase is characterized by below-trend price action with reduced volatility (low market activity). "
                f"Historical 30-day returns from similar regimes: **{avg_30d:+.1f}%** (mean)."
            )
        
        else:
            report_parts.append(
                f"**{asset_name}** is in a transitional phase between defined regimes."
            )
        
        # Compliance footer
        report_parts.append("")
        report_parts.append("---")
        report_parts.append(
            f"*Statistical analysis based on **{total_days:,} days** of historical data. "
            f"30-day volatility clustering measured against 2-year baseline. "
            f"**Past performance is not indicative of future results.***"
        )
        
        return "\n".join(report_parts)


# =============================================================================
# DYNAMIC POSITION SIZING SIMULATION MODULE
# =============================================================================

class DynamicExposureSimulator:
    """
    Simulates investment strategies with Dynamic Position Sizing based on Criticality Score.
    
    Strategy A (Benchmark): Buy & Hold - 100% invested always
    Strategy B (SOC Dynamic): Variable exposure based on volatility and trend
    
    EXPOSURE RULES:
    - Criticality > 80 (CRITICAL): 20% invested, 80% cash
    - Criticality > 60 (HIGH_ENERGY): 50% invested, 50% cash  
    - Criticality <= 60 AND Price > SMA200: 100% invested
    - Price < SMA200 (Bear Market): 0% invested, 100% cash
    
    Cash earns a risk-free rate (default 2% annual).
    
    COMPLIANCE: This is a BACKTESTING simulation for educational purposes only.
    Past performance is not indicative of future results.
    """
    
    ANNUAL_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate for cash
    
    def __init__(self, df: pd.DataFrame, symbol: str, initial_capital: float = 10000.0):
        self.df = df.copy()
        self.symbol = symbol
        self.initial_capital = initial_capital
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare dataframe with volatility metrics and criticality scores."""
        if 'close' not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Calculate metrics
        self.df['sma_200'] = self.df['close'].rolling(window=200).mean()
        self.df['returns'] = self.df['close'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(window=30).std()
        
        # Drop NaN rows first to ensure clean data
        self.df = self.df.dropna(subset=['close', 'sma_200', 'volatility'])
        
        if len(self.df) < 30:
            return
        
        # Calculate rolling volatility percentile (criticality score proxy)
        vol_window = min(504, len(self.df) - 1)  # ~2 years
        self.df['criticality_score'] = self.df['volatility'].rolling(
            window=vol_window, min_periods=30
        ).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100, raw=False
        )
        
        # Apply trend modifier to criticality score
        # If Price < SMA: boost criticality by +10 (downtrend = higher risk)
        # If Price > SMA by >30%: boost criticality by +10 (parabolic = higher risk)
        self.df['trend_modifier'] = 0
        self.df.loc[self.df['close'] < self.df['sma_200'], 'trend_modifier'] = 10
        
        price_deviation = (self.df['close'] - self.df['sma_200']) / self.df['sma_200'] * 100
        self.df.loc[price_deviation > 30, 'trend_modifier'] = 10
        
        self.df['criticality_score'] = (self.df['criticality_score'] + self.df['trend_modifier']).clip(0, 100)
        self.df['criticality_score'] = self.df['criticality_score'].fillna(50)
        
        # Determine if in uptrend (for exposure rules)
        self.df['is_uptrend'] = self.df['close'] > self.df['sma_200']
    
    def _calculate_exposure(self, criticality_score: float, is_uptrend: bool) -> float:
        """
        Calculate portfolio exposure based on criticality score and trend.
        
        Returns exposure as a decimal (0.0 to 1.0)
        """
        # STRICT RULE: If price below SMA200 (bear market), 0% exposure
        if not is_uptrend:
            return 0.0
        
        # HIGH CRITICALITY: Reduce exposure even in uptrend
        if criticality_score > 80:
            return 0.2  # 20% invested
        elif criticality_score > 60:
            return 0.5  # 50% invested
        else:
            return 1.0  # 100% invested
    
    def run_simulation(self, start_date: str = None, 
                       strategy_mode: str = "defensive",
                       trading_fee_pct: float = 0.005,
                       interest_rate_annual: float = 0.03) -> Dict[str, Any]:
        """
        Run both strategies and return results with realistic friction costs.
        
        Strategy A (Buy & Hold): 100% invested in asset always
        Strategy B (SOC Dynamic): Variable exposure based on criticality and strategy mode
        
        Args:
            start_date: Optional start date (YYYY-MM-DD)
            strategy_mode: "defensive" or "aggressive"
                - Defensive: Max safety, reduce exposure early
                - Aggressive: Max return, ride momentum longer
            trading_fee_pct: Trading fee as decimal (default 0.005 = 0.5%)
            interest_rate_annual: Annual interest on cash (default 0.03 = 3%)
            
        Returns:
            Dictionary with simulation results and equity curves.
        """
        df = self.df.copy()
        
        # Filter by start date if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        
        if len(df) < 30:
            return {"error": "Insufficient data for simulation"}
        
        # Daily interest rate for cash portion
        daily_interest_rate = interest_rate_annual / 365
        
        # Define exposure rules based on strategy mode
        is_aggressive = strategy_mode.lower() == "aggressive"
        
        # Thresholds (same for both modes)
        high_stress_threshold = 80  # Critical/Red
        medium_stress_threshold = 60  # High Energy/Orange
        
        # Exposure rules differ by mode
        if is_aggressive:
            # AGGRESSIVE MODE: Ride the bubble, reduce only at extremes
            high_stress_exposure = 0.5    # Red/Critical: 50%
            medium_stress_exposure = 1.0   # Orange/High Energy: 100% (ride momentum!)
            bear_market_exposure = 0.0     # Bear: 0% (hard exit)
        else:
            # DEFENSIVE MODE: Max safety, protect capital
            high_stress_exposure = 0.2     # Red/Critical: 20%
            medium_stress_exposure = 0.5   # Orange/High Energy: 50%
            bear_market_exposure = 0.0     # Bear: 0%
        
        # Calculate target exposure for each day
        def calc_exposure(row):
            if not row['is_uptrend']:
                return bear_market_exposure  # Bear market
            if row['criticality_score'] > high_stress_threshold:
                return high_stress_exposure  # Red/Critical
            elif row['criticality_score'] > medium_stress_threshold:
                return medium_stress_exposure  # Orange/High Energy
            else:
                return 1.0  # Green/Stable = 100%
        
        df['exposure'] = df.apply(calc_exposure, axis=1)
        
        # === ITERATIVE SIMULATION WITH FEES AND INTEREST ===
        # We need to iterate day-by-day to properly track fees and portfolio value
        
        # Buy & Hold: Simple vectorized calculation (no trades after initial buy)
        df['buyhold_return'] = df['returns']
        df['buyhold_equity'] = self.initial_capital * (1 + df['buyhold_return']).cumprod()
        
        # SOC Dynamic: Day-by-day simulation with fees and interest
        soc_equity = []
        soc_portfolio_value = self.initial_capital
        prev_exposure = 1.0  # Start fully invested
        
        total_fees_paid = 0.0
        total_interest_earned = 0.0
        trade_count = 0
        
        for idx, row in df.iterrows():
            current_exposure = row['exposure']
            daily_return = row['returns'] if not pd.isna(row['returns']) else 0.0
            
            # Step 1: Check for exposure change (TRADE)
            exposure_change = abs(current_exposure - prev_exposure)
            if exposure_change > 0.001:  # Threshold to avoid floating point issues
                # Calculate trade volume and fee
                trade_volume = exposure_change * soc_portfolio_value
                fee = trade_volume * trading_fee_pct
                soc_portfolio_value -= fee
                total_fees_paid += fee
                trade_count += 1
            
            # Step 2: Apply asset returns to invested portion
            invested_portion = soc_portfolio_value * current_exposure
            cash_portion = soc_portfolio_value * (1 - current_exposure)
            
            # Asset return on invested portion
            invested_portion *= (1 + daily_return)
            
            # Step 3: Apply daily interest on cash portion
            daily_interest = cash_portion * daily_interest_rate
            cash_portion += daily_interest
            total_interest_earned += daily_interest
            
            # Step 4: Update portfolio value
            soc_portfolio_value = invested_portion + cash_portion
            soc_equity.append(soc_portfolio_value)
            
            # Update previous exposure for next iteration
            prev_exposure = current_exposure
        
        df['soc_equity'] = soc_equity
        
        # Track exposure statistics
        days_full_invested = (df['exposure'] == 1.0).sum()
        days_partial = ((df['exposure'] > 0) & (df['exposure'] < 1.0)).sum()
        days_cash = (df['exposure'] == 0).sum()
        total_days = len(df)
        
        # Build equity curves (monthly snapshots for chart)
        df['year_month'] = df.index.to_period('M')
        monthly_snapshots = []
        
        for period in df['year_month'].unique():
            month_df = df[df['year_month'] == period]
            if not month_df.empty:
                last_day = month_df.iloc[-1]
                monthly_snapshots.append({
                    'date': last_day.name,
                    'buyhold_value': last_day['buyhold_equity'],
                    'soc_value': last_day['soc_equity'],
                    'exposure': last_day['exposure'] * 100,
                    'criticality': last_day['criticality_score']
                })
        
        # Final values
        final_buyhold = df['buyhold_equity'].iloc[-1] if len(df) > 0 else self.initial_capital
        final_soc = df['soc_equity'].iloc[-1] if len(df) > 0 else self.initial_capital
        
        # Calculate returns
        buyhold_return_pct = ((final_buyhold - self.initial_capital) / self.initial_capital) * 100
        soc_return_pct = ((final_soc - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate max drawdown for each strategy
        df['buyhold_peak'] = df['buyhold_equity'].cummax()
        df['buyhold_drawdown'] = (df['buyhold_equity'] - df['buyhold_peak']) / df['buyhold_peak'] * 100
        max_dd_buyhold = df['buyhold_drawdown'].min()
        
        df['soc_peak'] = df['soc_equity'].cummax()
        df['soc_drawdown'] = (df['soc_equity'] - df['soc_peak']) / df['soc_peak'] * 100
        max_dd_soc = df['soc_drawdown'].min()
        
        # Calculate daily returns from equity series for volatility calculation
        df['soc_daily_return'] = df['soc_equity'].pct_change()
        
        # Calculate Sharpe-like ratio (return / volatility)
        buyhold_vol = df['buyhold_return'].std() * np.sqrt(252) * 100  # Annualized
        soc_vol = df['soc_daily_return'].std() * np.sqrt(252) * 100  # Annualized
        
        buyhold_sharpe = buyhold_return_pct / buyhold_vol if buyhold_vol > 0 else 0
        soc_sharpe = soc_return_pct / soc_vol if soc_vol > 0 else 0
        
        # Average exposure over period
        avg_exposure = df['exposure'].mean() * 100
        
        # Build results
        results = {
            'initial_capital': self.initial_capital,
            'start_date': df.index[0] if len(df) > 0 else None,
            'end_date': df.index[-1] if len(df) > 0 else None,
            'total_days': total_days,
            'buyhold': {
                'final_value': final_buyhold,
                'total_return_pct': buyhold_return_pct,
                'max_drawdown_pct': max_dd_buyhold,
                'annualized_vol': buyhold_vol,
                'sharpe_ratio': buyhold_sharpe
            },
            'soc_dynamic': {
                'final_value': final_soc,
                'total_return_pct': soc_return_pct,
                'max_drawdown_pct': max_dd_soc,
                'annualized_vol': soc_vol,
                'sharpe_ratio': soc_sharpe,
                'avg_exposure_pct': avg_exposure,
                'days_full_invested': days_full_invested,
                'days_partial': days_partial,
                'days_cash': days_cash,
                'pct_full_invested': days_full_invested / total_days * 100 if total_days > 0 else 0,
                'pct_cash': days_cash / total_days * 100 if total_days > 0 else 0,
                'total_fees_paid': total_fees_paid,
                'total_interest_earned': total_interest_earned,
                'trade_count': trade_count
            },
            'outperformance_pct': soc_return_pct - buyhold_return_pct,
            'outperformance_abs': final_soc - final_buyhold,
            'drawdown_protection': max_dd_soc - max_dd_buyhold,  # Positive = less drawdown
            'equity_curve': pd.DataFrame(monthly_snapshots),
            'daily_data': df[['close', 'sma_200', 'criticality_score', 'exposure', 
                              'buyhold_equity', 'soc_equity', 'buyhold_drawdown', 'soc_drawdown']].copy(),
            'summary': {
                'symbol': self.symbol,
                'initial_capital': self.initial_capital,
                'total_days': total_days,
                'start_date': df.index[0] if len(df) > 0 else None,
                'end_date': df.index[-1] if len(df) > 0 else None,
                'buyhold_final': final_buyhold,
                'soc_final': final_soc,
                'buyhold_return_pct': buyhold_return_pct,
                'soc_return_pct': soc_return_pct,
                'outperformance_pct': soc_return_pct - buyhold_return_pct,
                'max_dd_buyhold': max_dd_buyhold,
                'max_dd_soc': max_dd_soc,
                'drawdown_reduction': max_dd_soc - max_dd_buyhold,
                'avg_exposure': avg_exposure,
                'days_in_cash': days_cash,
                'strategy_mode': 'Aggressive' if is_aggressive else 'Defensive',
                'high_stress_threshold': high_stress_threshold,
                'medium_stress_threshold': medium_stress_threshold,
                'high_stress_exposure': high_stress_exposure * 100,
                'medium_stress_exposure': medium_stress_exposure * 100,
                'trading_fee_pct': trading_fee_pct * 100,
                'interest_rate_annual': interest_rate_annual * 100,
                'total_fees_paid': total_fees_paid,
                'total_interest_earned': total_interest_earned,
                'trade_count': trade_count,
                'net_friction': total_interest_earned - total_fees_paid
            }
        }
        
        return results
    
    def run_savings_simulation(self, start_date: str = None,
                                 monthly_amount: float = 500.0,
                                 strategy_mode: str = "defensive",
                                 dca_strategy: str = "trend_follower",
                                 active_rebalancing: bool = False,
                                 sniper_mode: bool = False,
                                 hysteresis_pct: float = 0.03,
                                 min_exposure: float = 0.30,
                                 sell_threshold: float = 85.0,
                                 buy_threshold: float = 40.0,
                                 trading_fee_pct: float = 0.005,
                                 interest_rate_annual: float = 0.03) -> Dict[str, Any]:
        """
        Run Monthly Savings Plan simulation: Blind DCA vs Smart SOC DCA.
        
        Strategy A (Blind DCA): Buy on 1st of each month, never sell.
        Strategy B (Smart SOC DCA): 
            - Without Active Rebalancing: Only manages monthly inflow
            - With Active Rebalancing: Manages ENTIRE portfolio with hysteresis
            - With Sniper Mode: Pro-active profit-taking & deep dip buying
        
        Args:
            start_date: Optional start date (YYYY-MM-DD)
            monthly_amount: Amount to invest each month (default: 500)
            strategy_mode: "defensive" or "aggressive" (affects exposure levels)
            dca_strategy: "trend_follower" or "smart_value" (only used when not active_rebalancing)
            active_rebalancing: If True, actively sell/buy to match target exposure
            sniper_mode: If True, use pro-active profit taking & deep dip buying
            hysteresis_pct: Buffer zone around SMA to prevent whipsaws (default: 3%)
            min_exposure: Minimum exposure even in crashes (default: 30% "core position")
            sell_threshold: Criticality score above which to take profits (default: 85)
            buy_threshold: Criticality score below which to go all-in (default: 40)
            trading_fee_pct: Trading fee as decimal (default 0.005 = 0.5%)
            interest_rate_annual: Annual interest on cash (default 0.03 = 3%)
            
        Returns:
            Dictionary with simulation results for both strategies.
        """
        df = self.df.copy()
        
        # Filter by start date if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        
        if len(df) < 30:
            return {"error": "Insufficient data for simulation"}
        
        # Daily interest rate
        daily_interest_rate = interest_rate_annual / 365
        
        # Thresholds
        high_stress_threshold = 80  # Critical/Red
        medium_stress_threshold = 60  # High Energy/Orange
        is_smart_value = dca_strategy.lower() == "smart_value"
        is_aggressive = strategy_mode.lower() == "aggressive"
        
        # Calculate hysteresis bands
        # Upper band: SMA * (1 + hysteresis) - price must cross above to go BULL
        # Lower band: SMA * (1 - hysteresis) - price must cross below to go BEAR
        df['sma_upper_band'] = df['sma_200'] * (1 + hysteresis_pct)
        df['sma_lower_band'] = df['sma_200'] * (1 - hysteresis_pct)
        
        # Calculate hysteresis state (Bull/Bear with sticky logic)
        # Start in Bull if price > SMA, else Bear
        hysteresis_state = []
        current_state = 'bull' if df['close'].iloc[0] > df['sma_200'].iloc[0] else 'bear'
        
        for idx, row in df.iterrows():
            price = row['close']
            upper = row['sma_upper_band']
            lower = row['sma_lower_band']
            
            if current_state == 'bear':
                # In Bear mode: only switch to Bull if price crosses ABOVE upper band
                if price > upper:
                    current_state = 'bull'
            else:  # current_state == 'bull'
                # In Bull mode: only switch to Bear if price crosses BELOW lower band
                if price < lower:
                    current_state = 'bear'
            
            hysteresis_state.append(current_state)
        
        df['hysteresis_state'] = hysteresis_state
        
        # Calculate deep crash level (20% below SMA)
        df['deep_crash_level'] = df['sma_200'] * 0.80
        
        # Determine target exposure based on mode
        def get_target_exposure(row):
            """
            Calculate target exposure for Active Rebalancing.
            
            Standard Mode (hysteresis-based):
            - Bull (hysteresis state): 100%
            - Bear (hysteresis state): min_exposure
            
            Sniper Mode (pro-active, criticality-based):
            - Criticality > sell_threshold: 10% (SELL THE TOP!)
            - Criticality < buy_threshold AND Price > SMA: 100% (ALL IN!)
            - Price < SMA * 0.80 (deep crash): Deploy cash aggressively
            """
            crit_score = row['criticality_score']
            price = row['close']
            sma = row['sma_200']
            state = row['hysteresis_state']
            deep_crash = row['deep_crash_level']
            
            if sniper_mode:
                # === SMART SNIPER MODE (Pro-Active) ===
                
                # SELL THE TOP: Criticality is extreme -> Take profits!
                if crit_score > sell_threshold:
                    return 0.10  # Sell 90% into cash
                
                # DEEP CRASH BUY: Price crashed 20%+ below trend -> Deploy cash
                if price < deep_crash:
                    return 0.50  # Keep 50% in, deploy 50% of cash
                
                # RE-ENTRY: Storm is over, trend is up -> GO ALL IN
                if crit_score < buy_threshold and price > sma:
                    return 1.0  # 100% exposure
                
                # NEUTRAL ZONE: Between thresholds -> Use hysteresis
                if state == 'bear':
                    return min_exposure
                else:
                    return 1.0
            else:
                # === STANDARD MODE (Hysteresis-based) ===
                if state == 'bear':
                    return min_exposure
                else:
                    return 1.0
        
        def get_market_condition(row):
            """Determine market condition for display/logging."""
            crit_score = row['criticality_score']
            is_uptrend = row['is_uptrend']
            
            if crit_score > high_stress_threshold:
                if not is_uptrend:
                    return 'crash'
                else:
                    return 'overheated'
            elif crit_score > medium_stress_threshold:
                return 'active'
            elif not is_uptrend:
                return 'bear'
            else:
                return 'stable'
        
        df['target_exposure'] = df.apply(get_target_exposure, axis=1)
        df['market_condition'] = df.apply(get_market_condition, axis=1)
        df['year_month'] = df.index.to_period('M')
        df['is_month_start'] = df['year_month'] != df['year_month'].shift()
        
        # Calculate exposure for non-active strategies (Trend Follower / Smart Value)
        def calc_exposure_passive(row):
            """Exposure for passive DCA (only controls inflow, not selling)"""
            if not row['is_uptrend']:
                return 0.0  # Bear market: don't buy
            if row['criticality_score'] > high_stress_threshold:
                return 0.5 if is_aggressive else 0.0
            elif row['criticality_score'] > medium_stress_threshold:
                return 1.0 if is_aggressive else 0.5
            return 1.0
        
        df['passive_exposure'] = df.apply(calc_exposure_passive, axis=1)
        
        # Initialize strategies
        blind_dca = {
            'shares': 0, 'total_invested': 0, 'fees_paid': 0, 'trades': 0,
            'equity_curve': [], 'buy_dates': []
        }
        
        smart_soc = {
            'shares': 0, 'cash_wallet': 0, 'total_invested': 0, 
            'fees_paid': 0, 'interest_earned': 0, 'trades': 0,
            'equity_curve': [], 'buy_dates': [], 'sell_dates': [], 'bulk_buy_dates': [],
            'skipped_months': 0, 'prev_exposure': 1.0, 'dip_buys': 0, 'sell_count': 0,
            'profit_takes': 0, 'deep_dip_buys': 0, 'prev_target_exp': 1.0
        }
        
        # Process each day
        for date, row in df.iterrows():
            price = row['close']
            target_exp = row['target_exposure']
            passive_exp = row['passive_exposure']
            market_cond = row['market_condition']
            is_month_start = row['is_month_start']
            
            # === BLIND DCA: Buy on 1st of each month ===
            if is_month_start:
                blind_dca['total_invested'] += monthly_amount
                fee = monthly_amount * trading_fee_pct
                blind_dca['fees_paid'] += fee
                blind_dca['trades'] += 1
                net_amount = monthly_amount - fee
                blind_dca['shares'] += net_amount / price
                blind_dca['buy_dates'].append(date)
            
            # === SMART SOC DCA ===
            # Apply daily interest to cash wallet
            if smart_soc['cash_wallet'] > 0:
                daily_interest = smart_soc['cash_wallet'] * daily_interest_rate
                smart_soc['cash_wallet'] += daily_interest
                smart_soc['interest_earned'] += daily_interest
            
            if active_rebalancing:
                # ============ ACTIVE PORTFOLIO MANAGEMENT ============
                # This is the key logic that can beat Blind DCA!
                
                if is_month_start:
                    # Step 1: Add monthly contribution to cash wallet
                    smart_soc['total_invested'] += monthly_amount
                    smart_soc['cash_wallet'] += monthly_amount
                    
                    # Step 2: Calculate current portfolio state
                    current_asset_value = smart_soc['shares'] * price
                    total_wealth = current_asset_value + smart_soc['cash_wallet']
                    
                    # Step 3: Calculate target allocation
                    target_asset_value = total_wealth * target_exp
                    
                    # Step 4: Rebalance
                    delta = target_asset_value - current_asset_value
                    
                    # Track sniper-specific events
                    crit_score = row['criticality_score']
                    deep_crash = row['deep_crash_level']
                    is_profit_take = (sniper_mode and crit_score > sell_threshold and 
                                     smart_soc['prev_target_exp'] > target_exp)
                    is_deep_dip = (sniper_mode and price < deep_crash)
                    
                    if delta > 1:  # Need to BUY (tolerance of €1)
                        # Buy assets using available cash
                        buy_amount = min(delta, smart_soc['cash_wallet'])
                        if buy_amount > 0:
                            fee = buy_amount * trading_fee_pct
                            smart_soc['fees_paid'] += fee
                            smart_soc['trades'] += 1
                            net_amount = buy_amount - fee
                            smart_soc['shares'] += net_amount / price
                            smart_soc['cash_wallet'] -= buy_amount
                            smart_soc['buy_dates'].append(date)
                            
                            # Mark as bulk buy if deploying significant cash
                            if buy_amount > monthly_amount * 1.5:
                                smart_soc['bulk_buy_dates'].append({'date': date, 'amount': buy_amount})
                                smart_soc['dip_buys'] += 1
                            
                            # Track deep dip buys (sniper mode)
                            if is_deep_dip:
                                smart_soc['deep_dip_buys'] += 1
                                
                    elif delta < -1:  # Need to SELL (tolerance of €1)
                        # SELL assets to protect capital!
                        sell_value = abs(delta)
                        shares_to_sell = sell_value / price
                        
                        if shares_to_sell > 0 and smart_soc['shares'] > 0:
                            shares_to_sell = min(shares_to_sell, smart_soc['shares'])
                            sell_proceeds = shares_to_sell * price
                            fee = sell_proceeds * trading_fee_pct
                            smart_soc['fees_paid'] += fee
                            smart_soc['trades'] += 1
                            smart_soc['shares'] -= shares_to_sell
                            smart_soc['cash_wallet'] += (sell_proceeds - fee)
                            smart_soc['sell_dates'].append(date)
                            smart_soc['sell_count'] += 1
                            
                            # Track profit takes (sniper mode)
                            if is_profit_take:
                                smart_soc['profit_takes'] += 1
                    
                    # Track if we're in cash mode (not buying)
                    if target_exp < 0.5:
                        smart_soc['skipped_months'] += 1
                    
                    # Store for next iteration
                    smart_soc['prev_target_exp'] = target_exp
                        
            else:
                # ============ PASSIVE DCA (Original Logic) ============
                # Only manages inflow, never sells existing positions
                
                if is_month_start:
                    smart_soc['total_invested'] += monthly_amount
                    
                    if is_smart_value:
                        # Smart Value: Skip when overheated, buy MORE during crashes
                        if market_cond == 'overheated':
                            smart_soc['cash_wallet'] += monthly_amount
                            smart_soc['skipped_months'] += 1
                        elif market_cond in ['crash', 'bear']:
                            # Buy aggressively during crashes
                            dip_rate = 0.25 if is_aggressive else 0.20
                            cash_to_deploy = smart_soc['cash_wallet'] * dip_rate
                            total_to_invest = monthly_amount + cash_to_deploy
                            
                            fee = total_to_invest * trading_fee_pct
                            smart_soc['fees_paid'] += fee
                            smart_soc['trades'] += 1
                            net_amount = total_to_invest - fee
                            smart_soc['shares'] += net_amount / price
                            
                            if cash_to_deploy > 0:
                                smart_soc['bulk_buy_dates'].append({'date': date, 'amount': total_to_invest})
                                smart_soc['dip_buys'] += 1
                            
                            smart_soc['cash_wallet'] -= cash_to_deploy
                            smart_soc['buy_dates'].append(date)
                        else:
                            # Normal market - buy normally
                            fee = monthly_amount * trading_fee_pct
                            smart_soc['fees_paid'] += fee
                            smart_soc['trades'] += 1
                            net_amount = monthly_amount - fee
                            smart_soc['shares'] += net_amount / price
                            smart_soc['buy_dates'].append(date)
                    else:
                        # Trend Follower: Skip during crashes, deploy on recovery
                        if passive_exp > 0:
                            total_to_invest = monthly_amount + smart_soc['cash_wallet']
                            fee = total_to_invest * trading_fee_pct
                            smart_soc['fees_paid'] += fee
                            smart_soc['trades'] += 1
                            net_amount = total_to_invest - fee
                            smart_soc['shares'] += net_amount / price
                            
                            if smart_soc['cash_wallet'] > 0:
                                smart_soc['bulk_buy_dates'].append({'date': date, 'amount': total_to_invest})
                            
                            smart_soc['cash_wallet'] = 0
                            smart_soc['buy_dates'].append(date)
                        else:
                            smart_soc['cash_wallet'] += monthly_amount
                            smart_soc['skipped_months'] += 1
                
                # Re-entry trigger (only for Trend Follower, non-active)
                elif not is_smart_value and passive_exp > 0 and smart_soc['prev_exposure'] == 0 and smart_soc['cash_wallet'] > 0:
                    total_to_invest = smart_soc['cash_wallet']
                    fee = total_to_invest * trading_fee_pct
                    smart_soc['fees_paid'] += fee
                    smart_soc['trades'] += 1
                    net_amount = total_to_invest - fee
                    smart_soc['shares'] += net_amount / price
                    smart_soc['bulk_buy_dates'].append({'date': date, 'amount': total_to_invest})
                    smart_soc['cash_wallet'] = 0
                
                smart_soc['prev_exposure'] = passive_exp
            
            # Record daily equity (for chart)
            if is_month_start or date == df.index[-1]:
                blind_value = blind_dca['shares'] * price
                smart_value = smart_soc['shares'] * price + smart_soc['cash_wallet']
                
                blind_dca['equity_curve'].append({
                    'date': date, 
                    'value': blind_value,
                    'invested': blind_dca['total_invested']
                })
                smart_soc['equity_curve'].append({
                    'date': date, 
                    'value': smart_value,
                    'invested': smart_soc['total_invested']
                })
        
        # Final calculations
        final_price = df['close'].iloc[-1]
        total_months = df['year_month'].nunique()
        total_invested = monthly_amount * total_months
        
        # Blind DCA results
        blind_final = blind_dca['shares'] * final_price
        blind_return_pct = ((blind_final - total_invested) / total_invested * 100) if total_invested > 0 else 0
        
        blind_equity_df = pd.DataFrame(blind_dca['equity_curve'])
        if not blind_equity_df.empty:
            blind_equity_df['peak'] = blind_equity_df['value'].cummax()
            blind_equity_df['drawdown'] = (blind_equity_df['value'] - blind_equity_df['peak']) / blind_equity_df['peak'] * 100
            blind_max_dd = blind_equity_df['drawdown'].min()
        else:
            blind_max_dd = 0
        
        # Smart SOC results
        smart_final = smart_soc['shares'] * final_price + smart_soc['cash_wallet']
        smart_return_pct = ((smart_final - total_invested) / total_invested * 100) if total_invested > 0 else 0
        
        smart_equity_df = pd.DataFrame(smart_soc['equity_curve'])
        if not smart_equity_df.empty:
            smart_equity_df['peak'] = smart_equity_df['value'].cummax()
            smart_equity_df['drawdown'] = (smart_equity_df['value'] - smart_equity_df['peak']) / smart_equity_df['peak'] * 100
            smart_max_dd = smart_equity_df['drawdown'].min()
        else:
            smart_max_dd = 0
        
        # Determine strategy label
        if sniper_mode:
            strategy_label = 'Smart Sniper'
        elif active_rebalancing:
            strategy_label = 'Active Rebalancing'
        elif is_smart_value:
            strategy_label = 'Smart Value'
        else:
            strategy_label = 'Trend Follower'
        
        # Count state transitions (for hysteresis effectiveness)
        state_changes = (df['hysteresis_state'] != df['hysteresis_state'].shift()).sum() - 1
        
        results = {
            'symbol': self.symbol,
            'monthly_amount': monthly_amount,
            'total_months': total_months,
            'total_invested': total_invested,
            'start_date': df.index[0] if len(df) > 0 else None,
            'end_date': df.index[-1] if len(df) > 0 else None,
            'strategy_mode': 'Aggressive' if is_aggressive else 'Defensive',
            'dca_strategy': strategy_label,
            'active_rebalancing': active_rebalancing,
            'sniper_mode': sniper_mode,
            'hysteresis_pct': hysteresis_pct,
            'min_exposure': min_exposure,
            'sell_threshold': sell_threshold,
            'buy_threshold': buy_threshold,
            'state_transitions': max(0, state_changes),
            
            'blind_dca': {
                'final_value': blind_final,
                'return_pct': blind_return_pct,
                'max_drawdown': blind_max_dd,
                'fees_paid': blind_dca['fees_paid'],
                'trades': blind_dca['trades'],
                'equity_df': blind_equity_df,
                'final_shares': blind_dca['shares']
            },
            
            'smart_soc': {
                'final_value': smart_final,
                'return_pct': smart_return_pct,
                'max_drawdown': smart_max_dd,
                'fees_paid': smart_soc['fees_paid'],
                'interest_earned': smart_soc['interest_earned'],
                'trades': smart_soc['trades'],
                'equity_df': smart_equity_df,
                'final_shares': smart_soc['shares'],
                'final_cash': smart_soc['cash_wallet'],
                'skipped_months': smart_soc['skipped_months'],
                'bulk_buy_dates': smart_soc['bulk_buy_dates'],
                'dip_buys': smart_soc.get('dip_buys', 0),
                'sell_count': smart_soc.get('sell_count', 0),
                'sell_dates': smart_soc.get('sell_dates', []),
                'profit_takes': smart_soc.get('profit_takes', 0),
                'deep_dip_buys': smart_soc.get('deep_dip_buys', 0)
            },
            
            'outperformance_pct': smart_return_pct - blind_return_pct,
            'outperformance_abs': smart_final - blind_final,
            'drawdown_improvement': smart_max_dd - blind_max_dd
        }
        
        return results


def run_dca_simulation(symbol: str, initial_capital: float = 10000.0, 
                       start_date: str = None, years_back: int = 10,
                       strategy_mode: str = "defensive",
                       trading_fee_pct: float = 0.005,
                       interest_rate_annual: float = 0.03) -> Dict[str, Any]:
    """
    Convenience function to run Dynamic Position Sizing (Lump Sum) simulation.
    """
    try:
        lookback_days = years_back * 365 + 365
        start = datetime.now() - timedelta(days=lookback_days)
        
        df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
        
        if df.empty:
            return {"error": f"Could not fetch data for {symbol}"}
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        
        required = ["open", "high", "low", "close", "volume"]
        available = [c for c in required if c in df.columns]
        df = df[available]
        
    except Exception as e:
        return {"error": f"Error fetching data: {str(e)}"}
    
    simulator = DynamicExposureSimulator(df, symbol, initial_capital)
    return simulator.run_simulation(start_date, strategy_mode, trading_fee_pct, interest_rate_annual)


def run_savings_simulation(symbol: str, monthly_amount: float = 500.0,
                            start_date: str = None, years_back: int = 10,
                            strategy_mode: str = "defensive",
                            dca_strategy: str = "trend_follower",
                            active_rebalancing: bool = False,
                            sniper_mode: bool = False,
                            hysteresis_pct: float = 0.03,
                            min_exposure: float = 0.30,
                            sell_threshold: float = 85.0,
                            buy_threshold: float = 40.0,
                            trading_fee_pct: float = 0.005,
                            interest_rate_annual: float = 0.03) -> Dict[str, Any]:
    """
    Convenience function to run Monthly Savings Plan simulation.
    
    Args:
        symbol: Ticker symbol
        monthly_amount: Amount to invest each month (default: 500)
        start_date: Optional start date
        years_back: Number of years (default: 10)
        strategy_mode: "defensive" or "aggressive"
        dca_strategy: "trend_follower" (safety) or "smart_value" (buy the dip)
        active_rebalancing: If True, actively sell positions during crashes
        sniper_mode: If True, use pro-active profit taking & deep dip buying
        hysteresis_pct: Buffer zone around SMA to prevent whipsaws (default: 3%)
        min_exposure: Minimum exposure even in crashes (default: 30%)
        sell_threshold: Criticality score to trigger profit taking (default: 85)
        buy_threshold: Criticality score to trigger all-in buying (default: 40)
        trading_fee_pct: Trading fee (default: 0.5%)
        interest_rate_annual: Interest on cash (default: 3%)
    """
    try:
        lookback_days = years_back * 365 + 365
        start = datetime.now() - timedelta(days=lookback_days)
        
        df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
        
        if df.empty:
            return {"error": f"Could not fetch data for {symbol}"}
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        
        required = ["open", "high", "low", "close", "volume"]
        available = [c for c in required if c in df.columns]
        df = df[available]
        
    except Exception as e:
        return {"error": f"Error fetching data: {str(e)}"}
    
    simulator = DynamicExposureSimulator(df, symbol, 0)
    return simulator.run_savings_simulation(start_date, monthly_amount, strategy_mode,
                                            dca_strategy, active_rebalancing, sniper_mode,
                                            hysteresis_pct, min_exposure,
                                            sell_threshold, buy_threshold,
                                            trading_fee_pct, interest_rate_annual)


