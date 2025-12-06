"""
SOC Market Seismograph - Configuration
=======================================

Centralized configuration file containing all constants, thresholds,
colors, and default values used across the application.

Author: Market Analysis Team
Version: 7.0 (Modularized)
"""

# =============================================================================
# AUTHENTICATION
# =============================================================================
ACCESS_CODE = "BETA2025"

# =============================================================================
# DATA ANALYSIS PARAMETERS
# =============================================================================
DEFAULT_SMA_WINDOW = 200
DEFAULT_VOL_WINDOW = 30
DEFAULT_HYSTERESIS = 0.0
MIN_DATA_POINTS = 200

# =============================================================================
# ASSET FILTERING
# =============================================================================
# Precious metals excluded from main risk scan - they act as hedges (inverse correlation)
# and distort market risk scoring. Available separately in "Hedge Assets" category.
PRECIOUS_METALS = {'GC=F', 'SI=F', 'PL=F', 'PA=F', 'GLD', 'SLV'}

# =============================================================================
# POPULAR TICKERS FOR QUICK ACCESS
# =============================================================================
POPULAR_TICKERS = {
    "US Tech": ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META'],
    "Crypto": ['BTC-USD', 'ETH-USD', 'SOL-USD'],
    "ETFs": ['SPY', 'QQQ', 'IWM', 'VTI'],
}

# =============================================================================
# TICKER NAME FIXES
# =============================================================================
# Hardcoded fixes for known problematic ticker names (German stocks, etc.)
TICKER_NAME_FIXES = {
    "SIEMENS                    N": "Siemens",
    "Allianz                    v": "Allianz",
    "DEUTSCHE TELEKOM           N": "Deutsche Telekom",
    "Airbus                     A": "Airbus",
    "BAYERISCHE MOTOREN WERKE   S": "BMW",
    "VOLKSWAGEN                 V": "Volkswagen",
    "BASF                       N": "BASF",
    "MUENCHENER RUECKVERS.-GES. N": "Munich Re",
    "SAP                       ": "SAP"
}

SPECIAL_TICKER_NAMES = {
    "^GDAXI": "DAX 40 Index"
}

# =============================================================================
# REGIME CLASSIFICATION COLORS
# =============================================================================
REGIME_COLORS = {
    'STABLE': '#00C864',      # Green
    'ACTIVE': '#FFCC00',      # Yellow
    'HIGH_ENERGY': '#FF6600', # Orange
    'CRITICAL': '#FF4040',    # Red
    'DORMANT': '#888888'      # Grey
}

REGIME_COLORS_TRANSPARENT = {
    'STABLE': 'rgba(0, 200, 100, 0.15)',
    'ACTIVE': 'rgba(255, 204, 0, 0.15)',
    'HIGH_ENERGY': 'rgba(255, 102, 0, 0.15)',
    'CRITICAL': 'rgba(255, 64, 64, 0.15)',
    'DORMANT': 'rgba(136, 136, 136, 0.2)'
}

REGIME_EMOJIS = {
    'STABLE': '🟢',
    'ACTIVE': '🟡',
    'HIGH_ENERGY': '🟠',
    'CRITICAL': '🔴',
    'DORMANT': '⚪'
}

# =============================================================================
# STRESS LEVEL COLORS
# =============================================================================
STRESS_LEVEL_COLORS = {
    'BASELINE': '#00CC00',
    'MODERATE': '#FFCC00',
    'HEIGHTENED': '#FF6600',
    'ELEVATED': '#FF0000'
}

STRESS_LEVEL_EMOJIS = {
    'BASELINE': '🟢',
    'MODERATE': '🟡',
    'HEIGHTENED': '🟠',
    'ELEVATED': '🔴'
}

# =============================================================================
# THEME COLORS
# =============================================================================
DARK_THEME = {
    "bg": "#0E1117",
    "bg2": "#262730",
    "card": "#1E1E1E",
    "border": "#333",
    "text": "#FAFAFA",
    "muted": "#888",
    "input": "#262730"
}

LIGHT_THEME = {
    "bg": "#FFFFFF",
    "bg2": "#F0F2F6",
    "card": "#F8F9FA",
    "border": "#DEE2E6",
    "text": "#212529",
    "muted": "#6C757D",
    "input": "#FFFFFF"
}

# =============================================================================
# LEGAL DISCLAIMER TEXT
# =============================================================================
LEGAL_DISCLAIMER = """
<div class="disclaimer-box">
    <h2>⚠️ Important Legal Disclaimer</h2>
    
    <h3>📊 For Educational & Research Purposes Only</h3>
    <p>
        This tool is designed for <strong>educational and informational purposes only</strong>. 
        It is <strong>NOT</strong> intended to provide financial, investment, or trading advice.
    </p>
    
    <h3>🔬 Experimental Analysis Tool</h3>
    <p>
        The Self-Organized Criticality (SOC) analysis presented here is based on academic research 
        and statistical models. Past performance and historical patterns <strong>do not guarantee 
        future results</strong>.
    </p>
    
    <h3>⚠️ No Investment Recommendations</h3>
    <p>
        <strong>Nothing on this platform constitutes financial advice.</strong> We do not recommend 
        buying, selling, or holding any financial instrument. All investment decisions are your own 
        responsibility.
    </p>
    
    <h3>📉 Risk Warning</h3>
    <ul>
        <li>Trading and investing involve <strong>substantial risk of loss</strong></li>
        <li>You may lose some or all of your invested capital</li>
        <li>Never invest money you cannot afford to lose</li>
        <li>Market volatility can result in rapid and significant losses</li>
    </ul>
    
    <h3>🛡️ No Warranty</h3>
    <p>
        This tool is provided <strong>"as is"</strong> without any warranty, express or implied. 
        We make no guarantees about the accuracy, reliability, or completeness of any data or analysis.
    </p>
    
    <h3>💼 Consult a Professional</h3>
    <p>
        Before making any investment decision, consult with a <strong>licensed financial advisor</strong> 
        who understands your individual financial situation, goals, and risk tolerance.
    </p>
    
    <h3>📝 Your Responsibility</h3>
    <p>
        By using this tool, you acknowledge that:
    </p>
    <ul>
        <li>You are using it for educational purposes only</li>
        <li>You understand the risks involved in financial markets</li>
        <li>You will not hold the developers liable for any losses</li>
        <li>You agree to conduct your own due diligence</li>
    </ul>
    
    <hr style="border-color: #444; margin: 1.5rem 0;">
    
    <p style="text-align: center; font-size: 0.9rem; color: #888;">
        <strong>If you do not agree to these terms, please do not use this tool.</strong>
    </p>
</div>
"""

# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================
DEFAULT_INITIAL_CAPITAL = 10000
DEFAULT_SIMULATION_YEARS = 7
DEFAULT_TRADING_FEE_PCT = 0.1
DEFAULT_INTEREST_RATE_ANNUAL = 4.0

# =============================================================================
# CHART COLORS FOR PLOTLY
# =============================================================================
CHART_COLORS = {
    'price': '#667eea',
    'sma': '#FFD700',
    'volatility': '#FF6B6B',
    'positive': '#00C864',
    'negative': '#FF4040',
    'neutral': '#CCCCCC'
}

