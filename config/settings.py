"""
Configuration settings for Market SOC application
API endpoints, default parameters, and constants
"""

from typing import Dict, Any

# Binance API Configuration
BINANCE_BASE_URL: str = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT: str = "/api/v3/klines"

# Data Fetching Parameters
DEFAULT_SYMBOL: str = "BTCUSDT"
DEFAULT_INTERVAL: str = "1d"  # Daily data
DEFAULT_LOOKBACK_DAYS: int = 2000  # ~5.5 years of data

# Analysis Parameters
SMA_PERIOD: int = 200  # Simple Moving Average period
ROLLING_VOLATILITY_WINDOW: int = 30  # Rolling standard deviation window

# Volatility Thresholds (Dynamic Quantile-based - Option A)
VOLATILITY_LOW_PERCENTILE: float = 33.33  # Green zone
VOLATILITY_HIGH_PERCENTILE: float = 66.67  # Red zone above this

# Visualization Settings
CHART_HEIGHT: int = 400  # Height per subplot
CHART_TEMPLATE: str = "plotly_dark"  # Dark theme for financial charts

# Caching Configuration
CACHE_DIR: str = "data"
CACHE_FILENAME_TEMPLATE: str = "{symbol}_{interval}_cached.csv"

# API Request Settings
REQUEST_TIMEOUT: int = 10  # seconds
MAX_RETRIES: int = 3
RETRY_DELAY: int = 2  # seconds

# Color Schemes
COLORS: Dict[str, str] = {
    "low_volatility": "#00FF00",      # Green
    "medium_volatility": "#FFA500",   # Orange
    "high_volatility": "#FF0000",     # Red
    "sma_line": "#FFD700",            # Gold/Yellow
    "price_line": "#FFFFFF",          # White
    "stable": "#0000FF",              # Blue (for heatmap)
}

