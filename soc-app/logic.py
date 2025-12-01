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

    def __init__(self, df: pd.DataFrame, sma_window: int = SMA_PERIOD, vol_window: int = ROLLING_VOLATILITY_WINDOW) -> None:
        self.df = df.copy()
        self.sma_window = sma_window
        self.vol_window = vol_window
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
        self.df["sma_200"] = self.df["close"].rolling(window=self.sma_window).mean()

        # Calculate rolling volatility
        self.df["volatility"] = self.df["returns"].rolling(window=self.vol_window).std()

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
        if self.metrics_df.empty:
            return {"signal": "NO_DATA", "color": "grey"}

        # Values
        current_price = self.summary_stats["current_price"]
        sma_200 = self.summary_stats["current_sma_200"]
        current_vol = self.metrics_df["volatility"].iloc[-1]
        
        # Thresholds
        vol_low = self.calculator.vol_low_threshold
        vol_high = self.calculator.vol_high_threshold
        
        # Trend Logic with Hysteresis
        # Standard: price > sma
        # Hysteresis: price > sma * (1 + h) for Bull, price < sma * (1 - h) for Bear
        
        upper_bound = sma_200 * (1.0 + self.hysteresis)
        lower_bound = sma_200 * (1.0 - self.hysteresis)
        
        is_bullish = current_price > upper_bound
        is_bearish = current_price < lower_bound
        # if neither, it is neutral
        
        is_high_stress = current_vol > vol_high
        is_low_stress = current_vol <= vol_low
        
        signal = "NEUTRAL"
        trend_label = "NEUTRAL"
        
        if is_bullish:
            trend_label = "BULL"
            if is_low_stress: signal = "🟢 ACCUMULATE"
            elif is_high_stress: signal = "🟠 OVERHEATED"
            else: signal = "⚪ UPTREND (Choppy)"
        elif is_bearish:
            trend_label = "BEAR"
            if is_high_stress: signal = "🔴 CRASH RISK"
            elif is_low_stress: signal = "🟡 CAPITULATION/WAIT"
            else: signal = "⚪ DOWNTREND (Choppy)"
        else:
            # Neutral / Hysteresis Zone
            trend_label = "NEUTRAL"
            signal = "⚪ NEUTRAL (Range)"

        return {
            "symbol": self.symbol,
            "price": current_price,
            "sma_200": sma_200,
            "dist_to_sma": (current_price - sma_200) / sma_200,
            "volatility": current_vol,
            "vol_threshold_high": vol_high,
            "stress_score": current_vol / vol_high if vol_high > 0 else 0,
            "signal": signal,
            "trend": trend_label,
            "stress": "HIGH" if is_high_stress else ("LOW" if is_low_stress else "MED")
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
        
        # Map colors for volatility zones
        color_map = {"low": "#00cc00", "medium": "#ff9900", "high": "#ff0000"}
        colors = [color_map.get(z, "#ff9900") for z in self.metrics_df["vol_zone"]]
        
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
        Analyze historical signals and generate comprehensive statistics.
        Includes both short-term (1d/3d/5d/10d) and long-term (30d/60d/90d) analysis.
        
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
            
            # Calculate signals for each day in history
            vol_low = self.calculator.vol_low_threshold
            vol_high = self.calculator.vol_high_threshold
            
            # Handle None thresholds
            if vol_low is None or vol_high is None:
                return {"error": "Could not calculate volatility thresholds"}
            
            # Determine trend and stress for each row
            df['is_bullish'] = df['close'] > df['sma_200']
            df['is_bearish'] = df['close'] < df['sma_200']
            df['is_high_vol'] = df['volatility'] > vol_high
            df['is_low_vol'] = df['volatility'] <= vol_low
            
            # Assign signals
            def assign_signal(row):
                if row['is_bullish']:
                    if row['is_low_vol']:
                        return 'ACCUMULATE'
                    elif row['is_high_vol']:
                        return 'OVERHEATED'
                    else:
                        return 'NEUTRAL'
                elif row['is_bearish']:
                    if row['is_high_vol']:
                        return 'CRASH_RISK'
                    elif row['is_low_vol']:
                        return 'CAPITULATION'
                    else:
                        return 'NEUTRAL'
                return 'NEUTRAL'
            
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
            
            # Identify signal phases (consecutive periods of same signal)
            df['signal_change'] = (df['signal'] != df['signal'].shift()).cumsum()
            
            # Mark first day of each phase (signal START)
            df['is_phase_start'] = df['signal'] != df['signal'].shift()
            
            # Analyze each signal type
            signal_stats = {}
            for signal_type in ['ACCUMULATE', 'CRASH_RISK', 'OVERHEATED', 'CAPITULATION', 'NEUTRAL']:
                signal_df = df[df['signal'] == signal_type]
                
                if len(signal_df) == 0:
                    signal_stats[signal_type] = {
                        'total_days': 0,
                        'phase_count': 0,
                        'avg_duration': 0,
                        'max_duration': 0,
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
                        'avg_return_90d': 0
                    }
                    continue
                
                # Get only the FIRST DAY of each phase for short-term analysis
                phase_starts = signal_df[signal_df['is_phase_start']]
                
                # Count distinct phases
                phases = signal_df.groupby('signal_change').agg({
                    'close': ['first', 'last', 'count'],
                    'return_30d': 'first',  # Return from phase start
                    'return_60d': 'first',
                    'return_90d': 'first'
                })
                phases.columns = ['price_start', 'price_end', 'duration', 'ret_30', 'ret_60', 'ret_90']
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
                
                signal_stats[signal_type] = {
                    'total_days': len(signal_df),
                    'phase_count': len(phases),
                    'avg_duration': phases['duration'].mean() if len(phases) > 0 else 0,
                    'max_duration': phases['duration'].max() if len(phases) > 0 else 0,
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
                    'avg_return_90d': phases['ret_90'].mean() * 100 if len(phases) > 0 and not phases['ret_90'].isna().all() else 0
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
        Calculate a crash warning score (0-100) based on current market conditions.
        
        Factors:
        - Current volatility percentile (25%)
        - Price distance from SMA 200 (25%)
        - Days in high-risk phase (20%)
        - Volatility acceleration (15%)
        - Historical pattern matching (15%)
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
            risk_factors.append(f"⚠️ Volatility at {vol_percentile:.0f}th percentile (extreme)")
        elif vol_percentile >= 60:
            risk_factors.append(f"📊 Volatility at {vol_percentile:.0f}th percentile (elevated)")
        
        # --- Factor 2: Price Distance from SMA 200 (25%) ---
        if current_sma > 0:
            sma_distance = ((current_price - current_sma) / current_sma) * 100
            factors['sma_distance_pct'] = sma_distance
            
            # Only add to crash score if price is ABOVE SMA (overextended)
            if sma_distance > 0:
                # Scale: 0% = 0 points, 30%+ = 25 points
                distance_score = min(25, (sma_distance / 30) * 25)
                score += distance_score
                
                if sma_distance >= 30:
                    risk_factors.append(f"⚠️ Price {sma_distance:.1f}% above SMA 200 (severely overextended)")
                elif sma_distance >= 20:
                    risk_factors.append(f"📈 Price {sma_distance:.1f}% above SMA 200 (overextended)")
                elif sma_distance >= 10:
                    risk_factors.append(f"📊 Price {sma_distance:.1f}% above SMA 200 (extended)")
        else:
            factors['sma_distance_pct'] = 0
        
        # --- Factor 3: Days in High-Risk Phase (20%) ---
        high_risk_signals = ['OVERHEATED', 'CRASH_RISK']
        if current_signal in ['🟠 OVERHEATED', 'OVERHEATED']:
            # Days in overheated is concerning - longer = more risk
            streak_score = min(20, (current_streak / 30) * 20)
            score += streak_score
            factors['days_in_risk_phase'] = current_streak
            
            if current_streak >= 20:
                risk_factors.append(f"⚠️ {current_streak} days in OVERHEATED phase (extended stress)")
            elif current_streak >= 10:
                risk_factors.append(f"🔥 {current_streak} days in OVERHEATED phase")
        elif current_signal in ['🔴 CRASH RISK', 'CRASH_RISK']:
            # Already in crash - high risk continues
            score += 15
            factors['days_in_risk_phase'] = current_streak
            risk_factors.append(f"🔴 Currently in CRASH RISK phase ({current_streak} days)")
        else:
            factors['days_in_risk_phase'] = 0
        
        # --- Factor 4: Volatility Acceleration (15%) ---
        if len(df) >= 20:
            vol_5d_ago = df['volatility'].iloc[-5] if len(df) >= 5 else current_vol
            vol_20d_ago = df['volatility'].iloc[-20] if len(df) >= 20 else current_vol
            
            vol_change_5d = ((current_vol - vol_5d_ago) / vol_5d_ago * 100) if vol_5d_ago > 0 else 0
            vol_change_20d = ((current_vol - vol_20d_ago) / vol_20d_ago * 100) if vol_20d_ago > 0 else 0
            
            factors['vol_acceleration_5d'] = vol_change_5d
            factors['vol_acceleration_20d'] = vol_change_20d
            
            # Rapidly rising volatility is dangerous
            if vol_change_5d > 50:
                score += 15
                risk_factors.append(f"⚠️ Volatility surging +{vol_change_5d:.0f}% in 5 days")
            elif vol_change_5d > 25:
                score += 10
                risk_factors.append(f"📈 Volatility rising +{vol_change_5d:.0f}% in 5 days")
            elif vol_change_20d > 50:
                score += 8
                risk_factors.append(f"📊 Volatility up +{vol_change_20d:.0f}% over 20 days")
        
        # --- Factor 5: Historical Pattern Matching (15%) ---
        overheat_stats = signal_stats.get('OVERHEATED', {})
        if overheat_stats.get('phase_count', 0) > 0:
            # How often did overheated lead to negative returns?
            avg_30d_after_overheat = overheat_stats.get('avg_return_30d', 0)
            
            if avg_30d_after_overheat < -5:
                score += 15
                risk_factors.append(f"📉 Historical: OVERHEATED → {avg_30d_after_overheat:.1f}% avg 30d return")
            elif avg_30d_after_overheat < 0:
                score += 10
                risk_factors.append(f"📊 Historical: OVERHEATED phases show weak follow-through")
        
        # Check crash risk historical pattern
        crash_stats = signal_stats.get('CRASH_RISK', {})
        if crash_stats.get('phase_count', 0) > 0:
            avg_prior_30d = crash_stats.get('prior_30d', 0)
            if avg_prior_30d < -15:
                factors['typical_crash_drawdown'] = avg_prior_30d
        
        # --- Calculate final score and level ---
        score = min(100, max(0, score))
        
        if score >= 75:
            level = "CRITICAL"
            level_color = "#FF0000"
            level_emoji = "🚨"
            interpretation = "Extreme risk conditions. Historical patterns suggest high probability of significant decline."
        elif score >= 50:
            level = "ELEVATED"
            level_color = "#FF6600"
            level_emoji = "⚠️"
            interpretation = "Risk conditions building. Monitor closely and consider reducing exposure."
        elif score >= 25:
            level = "MODERATE"
            level_color = "#FFCC00"
            level_emoji = "📊"
            interpretation = "Some risk factors present but not at concerning levels."
        else:
            level = "LOW"
            level_color = "#00CC00"
            level_emoji = "✅"
            interpretation = "Risk conditions favorable. No immediate warning signs detected."
        
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
        """Generate a human-readable prose report from signal statistics."""
        
        asset_name = self.asset_info.get('name', self.symbol) if self.asset_info else self.symbol
        
        report_parts = []
        
        # Header
        report_parts.append(f"### 📊 Historical Signal Analysis: {asset_name}")
        report_parts.append(f"*Based on {years:.1f} years of data ({len(df):,} trading days)*\n")
        
        # Current status
        signal_emoji = {
            'ACCUMULATE': '🟢', 'CRASH_RISK': '🔴', 'OVERHEATED': '🟠',
            'CAPITULATION': '🟡', 'NEUTRAL': '⚪'
        }
        emoji = signal_emoji.get(current_signal, '⚪')
        report_parts.append(f"**Current Status:** {emoji} {current_signal} for **{streak} consecutive days**\n")
        
        # Crash Risk Analysis
        crash = stats.get('CRASH_RISK', {})
        if crash.get('phase_count', 0) > 0:
            report_parts.append("---")
            report_parts.append("#### 🔴 Crash Risk Phases")
            report_parts.append(
                f"Over the analyzed period, **{asset_name}** has experienced **{crash['phase_count']} crash risk phases**, "
                f"accounting for **{crash['pct_of_time']:.1f}%** of total trading time."
            )
            report_parts.append(
                f"- Average phase duration: **{crash['avg_duration']:.0f} days** (longest: {crash['max_duration']:.0f} days)"
            )
            report_parts.append(
                f"- Average price change during crash phases: **{crash['avg_price_change_during']:+.1f}%**"
            )
            
            # PRIOR returns - what led to the crash signal (THE ACTUAL CRASH)
            report_parts.append("")
            report_parts.append("**📉 What triggered this signal (price decline BEFORE signal):**")
            report_parts.append(
                f"| Prior 5d | Prior 10d | Prior 20d | Prior 30d |"
            )
            report_parts.append(
                f"|:--------:|:---------:|:---------:|:---------:|"
            )
            report_parts.append(
                f"| {crash.get('prior_5d', 0):+.1f}% | {crash.get('prior_10d', 0):+.1f}% | "
                f"{crash.get('prior_20d', 0):+.1f}% | {crash.get('prior_30d', 0):+.1f}% |"
            )
            report_parts.append("")
            report_parts.append(
                f"*The crash signal fires AFTER the decline. Average 30-day drawdown before signal: "
                f"**{crash.get('prior_30d', 0):+.1f}%***"
            )
            
            # FORWARD returns - what happens after crash signal starts (RECOVERY)
            report_parts.append("")
            report_parts.append("**📈 What happens AFTER the signal (recovery potential):**")
            report_parts.append(
                f"| 1 Day | 3 Days | 5 Days | 10 Days |"
            )
            report_parts.append(
                f"|:-----:|:------:|:------:|:-------:|"
            )
            report_parts.append(
                f"| {crash.get('start_return_1d', 0):+.2f}% | {crash.get('start_return_3d', 0):+.2f}% | "
                f"{crash.get('start_return_5d', 0):+.2f}% | {crash.get('start_return_10d', 0):+.2f}% |"
            )
            
            # Long-term recovery
            report_parts.append("")
            report_parts.append("**📅 Long-term Recovery (from signal start):**")
            report_parts.append(
                f"| 30 Days | 60 Days | 90 Days |"
            )
            report_parts.append(
                f"|:-------:|:-------:|:-------:|"
            )
            report_parts.append(
                f"| {crash.get('avg_return_30d', 0):+.1f}% | {crash.get('avg_return_60d', 0):+.1f}% | "
                f"{crash.get('avg_return_90d', 0):+.1f}% |"
            )
        else:
            report_parts.append("---")
            report_parts.append("#### 🔴 Crash Risk Phases")
            report_parts.append(f"No crash risk phases detected in the analyzed period for {asset_name}.")
        
        # Accumulation Analysis  
        accum = stats.get('ACCUMULATE', {})
        if accum.get('phase_count', 0) > 0:
            report_parts.append("")
            report_parts.append("#### 🟢 Accumulation Phases")
            report_parts.append(
                f"**{accum['phase_count']} accumulation phases** identified, representing "
                f"**{accum['pct_of_time']:.1f}%** of the time."
            )
            report_parts.append(
                f"- Average phase duration: **{accum['avg_duration']:.0f} days** (longest: {accum['max_duration']:.0f} days)"
            )
            report_parts.append(
                f"- Average price gain during accumulation: **{accum['avg_price_change_during']:+.1f}%**"
            )
            
            # PRIOR returns - what conditions led to accumulation signal
            report_parts.append("")
            report_parts.append("**📊 Market conditions BEFORE accumulation signal:**")
            report_parts.append(
                f"| Prior 5d | Prior 10d | Prior 20d | Prior 30d |"
            )
            report_parts.append(
                f"|:--------:|:---------:|:---------:|:---------:|"
            )
            report_parts.append(
                f"| {accum.get('prior_5d', 0):+.1f}% | {accum.get('prior_10d', 0):+.1f}% | "
                f"{accum.get('prior_20d', 0):+.1f}% | {accum.get('prior_30d', 0):+.1f}% |"
            )
            
            # FORWARD returns - what happens after accumulation signal starts
            report_parts.append("")
            report_parts.append("**📈 Performance AFTER accumulation signal:**")
            report_parts.append(
                f"| 1 Day | 3 Days | 5 Days | 10 Days |"
            )
            report_parts.append(
                f"|:-----:|:------:|:------:|:-------:|"
            )
            report_parts.append(
                f"| {accum.get('start_return_1d', 0):+.2f}% | {accum.get('start_return_3d', 0):+.2f}% | "
                f"{accum.get('start_return_5d', 0):+.2f}% | {accum.get('start_return_10d', 0):+.2f}% |"
            )
            
            # Long-term analysis
            report_parts.append("")
            report_parts.append("**📅 Long-term Trajectory (from signal start):**")
            report_parts.append(
                f"| 30 Days | 60 Days | 90 Days |"
            )
            report_parts.append(
                f"|:-------:|:-------:|:-------:|"
            )
            report_parts.append(
                f"| {accum.get('avg_return_30d', 0):+.1f}% | {accum.get('avg_return_60d', 0):+.1f}% | "
                f"{accum.get('avg_return_90d', 0):+.1f}% |"
            )
        else:
            report_parts.append("")
            report_parts.append("#### 🟢 Accumulation Phases")
            report_parts.append(f"No accumulation phases detected in the analyzed period.")
        
        # Overheated Analysis
        overheat = stats.get('OVERHEATED', {})
        if overheat.get('phase_count', 0) > 0:
            report_parts.append("")
            report_parts.append("#### 🟠 Overheated Phases")
            report_parts.append(
                f"**{overheat['phase_count']} overheated phases** detected "
                f"({overheat['pct_of_time']:.1f}% of time)."
            )
            report_parts.append(
                f"- These phases averaged **{overheat['avg_duration']:.0f} days** with "
                f"**{overheat['avg_price_change_during']:+.1f}%** price movement."
            )
            if overheat['avg_return_30d'] != 0:
                report_parts.append(
                    f"- Forward 30-day return after overheated signal: **{overheat['avg_return_30d']:+.1f}%** "
                    f"(caution: high volatility expected)"
                )
        
        # Key Insight
        report_parts.append("")
        report_parts.append("---")
        report_parts.append("#### 💡 Key Insight")
        
        # Generate insight based on current signal and historical performance
        if current_signal == 'ACCUMULATE' and accum.get('avg_return_30d', 0) > 0:
            report_parts.append(
                f"Historically, when {asset_name} shows an ACCUMULATE signal, "
                f"it has delivered positive returns over the following months. "
                f"The current {streak}-day accumulation phase aligns with favorable conditions."
            )
        elif current_signal == 'CRASH_RISK':
            report_parts.append(
                f"⚠️ **Elevated Risk Warning:** {asset_name} is currently in a CRASH RISK phase. "
                f"Historical data shows crash phases last an average of {crash.get('avg_duration', 0):.0f} days "
                f"with significant drawdowns. Consider risk management."
            )
        elif current_signal == 'OVERHEATED':
            report_parts.append(
                f"⚠️ {asset_name} is showing signs of overheating. While the trend is up, "
                f"high volatility suggests caution. Historically, overheated phases precede "
                f"either corrections or continuation after consolidation."
            )
        else:
            report_parts.append(
                f"{asset_name} is currently in a neutral/transitional phase. "
                f"Wait for a clearer signal before making directional decisions."
            )
        
        return "\n".join(report_parts)

