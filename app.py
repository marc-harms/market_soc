"""
SOC Market Seismograph - Streamlit Application
==============================================

Interactive web dashboard for Self-Organized Criticality (SOC) market analysis.

Features:
    - Multi-asset scanning with 5-Tier regime classification
    - Deep dive analysis with historical signal performance
    - Systemic Stress Level indicator (0-100)
    - Lump Sum investment simulation with Dynamic Position Sizing
    - Dark/Light theme support

Theory:
    Markets exhibit Self-Organized Criticality - they naturally evolve toward
    critical states where small inputs can trigger events of any size.
    This app visualizes market "energy states" through volatility clustering.

Author: Market Analysis Team
Version: 6.0 (Cleaned & Documented)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import time
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from logic import DataFetcher, SOCAnalyzer, run_dca_simulation

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SOC Market Seismograph",
    page_icon="assets/logo-soc.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CONSTANTS
# =============================================================================
ACCESS_CODE = "BETA2025"
DEFAULT_SMA_WINDOW = 200
DEFAULT_VOL_WINDOW = 30
DEFAULT_HYSTERESIS = 0.0
MIN_DATA_POINTS = 200

FOOTER_TICKERS = {"Bitcoin": "BTC-USD", "S&P 500": "^GSPC", "Gold": "GC=F"}

# Precious metals excluded from main risk scan - they act as hedges (inverse correlation)
# and distort market risk scoring. Available separately in "Hedge Assets" category.
PRECIOUS_METALS = {'GC=F', 'SI=F', 'PL=F', 'PA=F', 'GLD', 'SLV'}

MARKET_SETS = {
    "US Big Tech": ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'AMD', 'NFLX'],
    "DAX Top 10": ['^GDAXI', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'VOW3.DE', 'BAS.DE', 'MUV2.DE'],
    "Crypto Assets": ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD']
}

TICKER_NAME_FIXES = {
    "SIEMENS                    N": "Siemens", "Allianz                    v": "Allianz",
    "DEUTSCHE TELEKOM           N": "Deutsche Telekom", "Airbus                     A": "Airbus",
    "BAYERISCHE MOTOREN WERKE   S": "BMW", "VOLKSWAGEN                 V": "Volkswagen",
    "BASF                       N": "BASF", "MUENCHENER RUECKVERS.-GES. N": "Munich Re",
    "SAP                       ": "SAP"
}

SPECIAL_TICKER_NAMES = {"^GDAXI": "DAX 40 Index"}

# =============================================================================
# STYLING
# =============================================================================
def get_theme_css(is_dark: bool) -> str:
    """
    Generate comprehensive CSS styles for theme (dark/light mode).
    
    Handles styling for: app background, text colors, inputs, tables,
    buttons, cards, asset list items, and footer elements.
    """
    c = {
        "bg": "#0E1117" if is_dark else "#FFFFFF",
        "bg2": "#262730" if is_dark else "#F0F2F6",
        "card": "#1E1E1E" if is_dark else "#F8F9FA",
        "border": "#333" if is_dark else "#DEE2E6",
        "text": "#FAFAFA" if is_dark else "#212529",
        "muted": "#888" if is_dark else "#6C757D",
        "input": "#262730" if is_dark else "#FFFFFF"
    }
    
    return f"""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {c['bg']} !important;
    }}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stMarkdown, .stMarkdown p {{
        color: {c['text']} !important;
    }}
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"], [data-baseweb="select"] > div {{
        background-color: {c['input']} !important;
        color: {c['text']} !important;
        border-color: {c['border']} !important;
    }}
    .streamlit-expanderHeader {{ background-color: {c['card']} !important; }}
    .streamlit-expanderContent {{ background-color: {c['bg2']} !important; }}
    .stDataFrame, [data-testid="stDataFrame"], .stDataFrame div, .stDataFrame table,
    .stDataFrame th, .stDataFrame td, [data-testid="glideDataEditor"], .dvn-scroller {{
        background-color: {c['card']} !important;
        color: {c['text']} !important;
    }}
    .stDataFrame th {{ background-color: {c['bg2']} !important; }}
    .stButton > button {{
        background-color: {c['card']} !important;
        color: {c['text']} !important;
        border-color: {c['border']} !important;
        font-weight: bold;
        border-radius: 8px;
    }}
    .stButton > button:hover {{ background-color: {c['bg2']} !important; }}
    .stButton > button[kind="primary"] {{
        background-color: #667eea !important;
        color: white !important;
        border-color: #667eea !important;
    }}
    .stRadio label {{ color: {c['text']} !important; }}
    .stRadio [role="radiogroup"] label {{ background-color: {c['card']} !important; border-color: {c['border']} !important; }}
    [data-testid="stMetricValue"] {{ color: {c['text']} !important; }}
    [data-testid="stMetricLabel"] {{ color: {c['muted']} !important; }}
    [data-testid="stSidebar"] {{ display: none; }}
    .stDeployButton {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; max-width: 1400px; margin: 0 auto; }}
    hr {{ border-color: {c['border']} !important; }}
    .app-header {{
        display: flex; align-items: center; gap: 1rem;
        padding: 0.75rem 0; border-bottom: 2px solid {c['border']}; margin-bottom: 1rem;
    }}
    .logo {{ width: 45px; height: 45px; background: linear-gradient(135deg, #667eea, #764ba2);
             border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; }}
    .app-title {{ font-size: 1.6rem; font-weight: 700;
                  background: linear-gradient(90deg, #667eea, #764ba2);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }}
    .app-subtitle {{ font-size: 0.8rem; color: {c['muted']} !important; margin: 0; }}
    .asset-item {{
        padding: 0.6rem 0.8rem; border-radius: 6px; margin-bottom: 4px;
        cursor: pointer; transition: all 0.15s ease;
        border: 1px solid transparent;
    }}
    .asset-item:hover {{ background-color: {c['bg2']}; }}
    .asset-item.selected {{ background-color: {c['bg2']}; border-color: #667eea; }}
    .asset-symbol {{ font-weight: 600; font-size: 0.95rem; }}
    .asset-price {{ color: {c['muted']}; font-size: 0.85rem; }}
    .asset-signal {{ font-size: 0.8rem; }}
    .detail-header {{ padding: 1rem; background: {c['card']}; border-radius: 8px; margin-bottom: 1rem; }}
    .signal-badge {{
        display: inline-block; padding: 0.4rem 0.8rem; border-radius: 6px;
        font-weight: 600; font-size: 0.9rem;
    }}
    .footer {{ border-top: 1px solid {c['border']}; padding-top: 0.75rem; margin-top: 1.5rem; }}
    .footer-item {{ display: inline-block; margin-right: 1.5rem; font-size: 0.8rem; color: {c['muted']}; }}
</style>
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_name(name: str) -> str:
    """
    Clean and normalize ticker names from Yahoo Finance.
    
    Handles German stock suffixes (SE, AG), fixes common formatting issues,
    and applies hardcoded fixes for known problematic names (DAX stocks).
    """
    name = name.replace(" SE", "").replace(" AG", "").strip()
    if name in TICKER_NAME_FIXES:
        return TICKER_NAME_FIXES[name]
    name = " ".join(name.split())
    return name[:-2] if len(name) > 2 and name[-2] == " " else name


def get_signal_color(signal: str) -> str:
    """
    Get display color for 5-tier regime classification.
    
    Mapping:
        STABLE → Green (#00FF00)
        ACTIVE → Yellow (#FFCC00)
        HIGH_ENERGY → Orange (#FF6600)
        CRITICAL → Red (#FF0000)
        DORMANT → Grey (#888888)
    """
    signal_upper = signal.upper()
    if "STABLE" in signal_upper:
        return "#00FF00"
    if "CRITICAL" in signal_upper:
        return "#FF0000"
    if "HIGH_ENERGY" in signal_upper or "HIGH ENERGY" in signal_upper:
        return "#FF6600"
    if "ACTIVE" in signal_upper:
        return "#FFCC00"
    if "DORMANT" in signal_upper:
        return "#888888"
    return "#888888"


def get_signal_bg(signal: str) -> str:
    """Get semi-transparent background color for regime badge display."""
    signal_upper = signal.upper()
    if "STABLE" in signal_upper:
        return "rgba(0, 255, 0, 0.15)"
    if "CRITICAL" in signal_upper:
        return "rgba(255, 0, 0, 0.15)"
    if "HIGH_ENERGY" in signal_upper or "HIGH ENERGY" in signal_upper:
        return "rgba(255, 102, 0, 0.15)"
    if "ACTIVE" in signal_upper:
        return "rgba(255, 204, 0, 0.15)"
    if "DORMANT" in signal_upper:
        return "rgba(136, 136, 136, 0.2)"
    return "rgba(136, 136, 136, 0.15)"


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_footer_data() -> List[Dict[str, Any]]:
    """
    Fetch current prices for footer market pulse indicators.
    
    Cached for 1 hour. Returns Bitcoin, S&P 500, and Gold prices
    with daily percentage change.
    """
    results = []
    fetcher = DataFetcher(cache_enabled=True)
    for name, symbol in FOOTER_TICKERS.items():
        try:
            df = fetcher.fetch_data(symbol)
            if len(df) > 1:
                curr, prev = df["close"].iloc[-1], df["close"].iloc[-2]
                results.append({"name": name, "price": curr, "change": ((curr - prev) / prev) * 100})
        except Exception:
            pass
    return results


def run_analysis(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Run SOC analysis on multiple tickers with progress indicator.
    
    For each ticker: fetches data, calculates metrics, determines market
    phase (5-tier classification), and returns analysis results.
    
    Includes robust API error handling with user-friendly messages.
    
    Returns:
        List of dictionaries containing symbol, price, signal, trend,
        criticality_score, and other phase metrics.
    """
    fetcher = DataFetcher(cache_enabled=True)
    results = []
    failed_tickers = []
    api_error_count = 0
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, symbol in enumerate(tickers):
        status.caption(f"Analyzing {symbol}...")
        try:
            df = fetcher.fetch_data(symbol)
            info = fetcher.fetch_info(symbol)
            if not df.empty and len(df) > MIN_DATA_POINTS:
                analyzer = SOCAnalyzer(df, symbol, info, DEFAULT_SMA_WINDOW, DEFAULT_VOL_WINDOW, DEFAULT_HYSTERESIS)
                phase = analyzer.get_market_phase()
                phase['info'] = info
                phase['name'] = clean_name(info.get('name', symbol))
                results.append(phase)
            else:
                failed_tickers.append(symbol)
                api_error_count += 1
        except ConnectionError:
            failed_tickers.append(symbol)
            api_error_count += 1
        except TimeoutError:
            failed_tickers.append(symbol)
            api_error_count += 1
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                api_error_count += 1
            elif 'connection' in error_msg or 'timeout' in error_msg or 'network' in error_msg:
                api_error_count += 1
            failed_tickers.append(symbol)
        
        progress.progress((i + 1) / len(tickers))
    
    status.empty()
    progress.empty()
    
    # Show API error warning if significant failures
    if api_error_count >= len(tickers) * 0.5:  # More than 50% failed
        st.error("""
        ⚠️ **Data Provider Unavailable**
        
        The market data API (Yahoo Finance) appears to be experiencing issues.
        This could be due to:
        - Rate limiting (too many requests)
        - Temporary service outage
        - Network connectivity issues
        
        **Please try again in 5-10 minutes.**
        
        If the problem persists, check [Yahoo Finance Status](https://finance.yahoo.com).
        """)
    elif failed_tickers:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers[:5])}{'...' if len(failed_tickers) > 5 else ''}")
    
    return results


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render app header with logo, title, subtitle, and theme toggle button."""
    is_dark = st.session_state.get('dark_mode', True)
    
    col_logo, col_title, col_theme = st.columns([1, 6, 1])
    
    with col_logo:
        try:
            st.image("assets/logo-soc.png", width=160)
        except Exception:
            st.markdown('<div class="logo">⚡</div>', unsafe_allow_html=True)
    
    with col_title:
        st.markdown("""
        <div style="padding-top: 10px;">
            <h1 class="app-title">Market Seismograph</h1>
            <p class="app-subtitle">Self-Organized Criticality Analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col_theme:
        if st.button("🌙" if is_dark else "☀️", key="theme"):
            st.session_state.dark_mode = not is_dark
            st.rerun()
    
    st.markdown('<hr style="margin: 0.5rem 0 1rem 0;">', unsafe_allow_html=True)


def render_theory():
    """
    Render expandable theory section explaining SOC methodology.
    
    Content: Origins, transfer to finance, why it works, 5-tier classification
    system, systemic stress level explanation, and academic references.
    """
    with st.expander("📖 How this theory works"):
        st.markdown("""
## Self-Organized Criticality (SOC)

### Origins

Self-Organized Criticality was introduced by physicists **Per Bak, Chao Tang, and Kurt Wiesenfeld** 
in 1987. They discovered that certain complex systems naturally evolve toward a "critical state" 
where small perturbations can trigger chain reactions of all sizes—from minor fluctuations to 
catastrophic avalanches.

The famous **sandpile model** illustrates this: as you slowly add grains of sand to a pile, 
it eventually reaches a critical slope. At this point, adding just one more grain can cause 
anything from a tiny slide to a massive avalanche. The system organizes itself to this 
critical state without any external tuning.

### Transfer to Financial Markets

In the 1990s, researchers recognized that financial markets exhibit strikingly similar behavior:

- **Benoit Mandelbrot** demonstrated that market returns follow "fat-tailed" distributions—extreme 
  events occur far more frequently than traditional models predict.

- **Didier Sornette** applied SOC principles to predict market crashes, showing that bubbles 
  exhibit characteristic patterns of accelerating oscillations before collapse.

- Markets, like sandpiles, accumulate stress (through leverage, speculation, herding behavior) 
  until they reach a critical state where a small trigger can cause disproportionate moves.

### Why It Works

Traditional finance assumes markets are **efficient** and returns are **normally distributed**. 
Reality shows otherwise:

1. **Volatility Clustering**: Large price changes tend to follow large changes, and small 
   changes follow small changes (GARCH effects). This is the market "remembering" recent stress.

2. **Power Laws**: The distribution of returns follows a power law, not a bell curve. This means 
   "once in a century" events happen every few years.

3. **Feedback Loops**: Markets are reflexive—prices affect fundamentals which affect prices. 
   This creates self-reinforcing cycles that drive the system toward criticality.

4. **Phase Transitions**: Markets shift between stable and unstable regimes. By monitoring 
   volatility relative to trend, we can identify when the system approaches a critical state.

---

### 5-Tier Regime Classification (Energy States)

Markets exhibit characteristics similar to physical systems. Each regime represents a 
distinct **energy state** — not investment advice, but observable market conditions:

| Regime | Color | Physical State | Statistical Characteristics |
|--------|-------|----------------|----------------------------|
| ⚪ **DORMANT** | Grey | Low Energy, Below Equilibrium | Price < SMA200, Low volatility (reduced activity) |
| 🟢 **STABLE** | Green | Low Energy, Above Equilibrium | Price > SMA200, Low/normal volatility (ordered state) |
| 🟡 **ACTIVE** | Yellow | Medium Energy | Price > SMA200, Medium volatility (increased activity) |
| 🟠 **HIGH ENERGY** | Orange | High Energy | Price > SMA200, High volatility >80th percentile (excited state) |
| 🔴 **CRITICAL** | Red | Critical Energy | High stress with downtrend OR Extreme vol >99th percentile |

---

### Systemic Stress Level (0-100)

The **Systemic Stress Level** is a statistical measure of how far the market deviates 
from its baseline equilibrium. This is purely a measurement, not a prediction.

**Components:**
- **Volatility Percentile** (0-100): Current 30-day volatility vs. 2-year historical range
- **Trend Deviation**: +10 if price significantly below SMA200
- **Extension Deviation**: +10 if price >30% above SMA200 (statistically rare)

**Statistical Interpretation:**
- **0-25 (Baseline)**: Metrics near historical averages
- **26-50 (Moderate)**: Some elevated metrics
- **51-75 (Heightened)**: Above-average energy state
- **76-100 (Elevated)**: Statistical similarities to previous high-volatility periods

---

### Understanding DORMANT Regime

The **⚪ DORMANT** regime represents a low-energy market state:
- Price is **below** the 200-day moving average 
- Volatility is **low** (reduced market activity)

This state often indicates reduced participation and can persist for extended periods. 
It represents a statistical observation of market conditions, not a trading signal.

---

*References: Bak, Tang & Wiesenfeld (1987) "Self-organized criticality"; Mandelbrot (1963) 
"The variation of certain speculative prices"; Sornette (2003) "Why Stock Markets Crash"*
        """)


def render_market_selection() -> List[str]:
    """
    Render market universe selection UI with clickable panel cards.
    
    Features centered panels for US Tech, DAX, Crypto with icons.
    Custom option is greyed out (coming soon).
    
    Returns:
        List of ticker symbols to analyze.
    """
    # Initialize selected universe in session state
    if 'selected_universe' not in st.session_state:
        st.session_state.selected_universe = "US Big Tech"
    
    # Market panel definitions with icons and descriptions
    market_panels = {
        "US Big Tech": {
            "icon": "🇺🇸",
            "description": "NVIDIA, Apple, Microsoft, Amazon, Google, Tesla, Meta, AMD, Netflix",
            "count": len(MARKET_SETS["US Big Tech"])
        },
        "DAX Top 10": {
            "icon": "🇩🇪",
            "description": "SAP, Siemens, Allianz, Deutsche Telekom, Airbus, BMW, VW, BASF, Munich Re",
            "count": len(MARKET_SETS["DAX Top 10"])
        },
        "Crypto Assets": {
            "icon": "₿",
            "description": "Bitcoin, Ethereum, Solana, BNB, XRP, Dogecoin, Cardano",
            "count": len(MARKET_SETS["Crypto Assets"])
        }
    }
    
    # CSS for market panels
    st.markdown("""
    <style>
        .market-panel {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #333;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .market-panel:hover {
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        .market-panel.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, #1e2a4a 0%, #1a2744 100%);
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
        }
        .market-panel-icon {
            font-size: 2.5rem;
            margin-bottom: 8px;
        }
        .market-panel-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: #FAFAFA;
        }
        .market-panel-desc {
            font-size: 0.75rem;
            color: #888;
            line-height: 1.3;
        }
        .market-panel-count {
            font-size: 0.8rem;
            color: #667eea;
            margin-top: 8px;
        }
        .market-panel-disabled {
            opacity: 0.4;
            cursor: not-allowed;
            background: #1a1a1a;
        }
        .market-panel-disabled:hover {
            transform: none;
            box-shadow: none;
            border-color: #333;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create 4 columns for panels (3 active + 1 disabled)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        panel = market_panels["US Big Tech"]
        is_selected = st.session_state.selected_universe == "US Big Tech"
        if st.button(
            f"{panel['icon']}\n\n**US Big Tech**\n\n{panel['count']} assets",
            key="panel_us",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            st.session_state.selected_universe = "US Big Tech"
            st.rerun()
    
    with col2:
        panel = market_panels["DAX Top 10"]
        is_selected = st.session_state.selected_universe == "DAX Top 10"
        if st.button(
            f"{panel['icon']}\n\n**DAX Top 10**\n\n{panel['count']} assets",
            key="panel_dax",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            st.session_state.selected_universe = "DAX Top 10"
            st.rerun()
    
    with col3:
        panel = market_panels["Crypto Assets"]
        is_selected = st.session_state.selected_universe == "Crypto Assets"
        if st.button(
            f"{panel['icon']}\n\n**Crypto**\n\n{panel['count']} assets",
            key="panel_crypto",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            st.session_state.selected_universe = "Crypto Assets"
            st.rerun()
    
    with col4:
        # Disabled Custom panel
        st.markdown("""
        <div class="market-panel market-panel-disabled">
            <div class="market-panel-icon">⚙️</div>
            <div class="market-panel-title">Custom</div>
            <div class="market-panel-desc">Coming soon</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show selected universe description
    selected = st.session_state.selected_universe
    panel_info = market_panels[selected]
    st.caption(f"📋 **{selected}**: {panel_info['description']}")
    
    return MARKET_SETS[selected]


def render_detail_panel(result: Dict[str, Any]):
    """
    Render detailed analysis panel for a selected asset.
    
    Displays: Header with regime badge, key metrics (price, criticality,
    vol percentile, trend), SOC chart, historical signal analysis report
    with Systemic Stress Level box, and regime distribution statistics.
    """
    is_dark = st.session_state.get('dark_mode', True)
    symbol = result['symbol']
    signal = result['signal']
    color = get_signal_color(signal)
    bg = get_signal_bg(signal)
    
    # Header with signal
    st.markdown(f"""
    <div class="detail-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0;">{result.get('name', symbol)}</h3>
                <span style="color: #888;">{symbol}</span>
            </div>
            <div class="signal-badge" style="background: {bg}; color: {color};">
                {signal}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row - including new Criticality Score
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${result['price']:,.2f}")
    col2.metric("Criticality", f"{result.get('criticality_score', 0)}/100")
    col3.metric("Vol %ile", f"{result.get('vol_percentile', 50):.0f}th")
    col4.metric("Trend", result['trend'])
    
    # Chart
    fetcher = DataFetcher(cache_enabled=True)
    df = fetcher.fetch_data(symbol)
    
    if not df.empty:
        analyzer = SOCAnalyzer(df, symbol, result.get('info'))
        figs = analyzer.get_plotly_figures(dark_mode=is_dark)
        st.plotly_chart(figs['chart3'], width="stretch")
        
        # Historical Signal Analysis Report
        with st.expander("📈 Historical Signal Analysis & Performance Report"):
            with st.spinner("Analyzing historical signals..."):
                analysis = analyzer.get_historical_signal_analysis()
            
            if 'error' in analysis:
                st.warning(analysis['error'])
            else:
                # === SYSTEMIC STRESS LEVEL (Compliance-safe) ===
                stress_data = analysis.get('crash_warning', {})
                if stress_data:
                    score = stress_data.get('score', 0)
                    level = stress_data.get('level', 'BASELINE')
                    level_color = stress_data.get('level_color', '#00CC00')
                    level_emoji = stress_data.get('level_emoji', '📊')
                    interpretation = stress_data.get('interpretation', '')
                    statistical_factors = stress_data.get('risk_factors', [])
                    
                    # Determine background color based on level
                    if level == "ELEVATED":
                        bg_color = "rgba(255, 0, 0, 0.15)"
                        border_color = "#FF0000"
                    elif level == "HEIGHTENED":
                        bg_color = "rgba(255, 102, 0, 0.15)"
                        border_color = "#FF6600"
                    elif level == "MODERATE":
                        bg_color = "rgba(255, 204, 0, 0.15)"
                        border_color = "#FFCC00"
                    else:
                        bg_color = "rgba(0, 204, 0, 0.1)"
                        border_color = "#00CC00"
                    
                    # Build statistical factors HTML
                    factors_html = ""
                    if statistical_factors:
                        factors_html = "<div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.2);'>"
                        factors_html += "<strong>Statistical Indicators:</strong><ul style='margin: 8px 0 0 0; padding-left: 20px;'>"
                        for factor in statistical_factors:
                            factors_html += f"<li style='margin: 4px 0;'>{factor}</li>"
                        factors_html += "</ul></div>"
                    
                    st.markdown(f"""
                    <div style="
                        background: {bg_color};
                        border: 2px solid {border_color};
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 24px;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8; margin-bottom: 4px;">
                                    📊 Systemic Stress Level
                                </div>
                                <div style="font-size: 42px; font-weight: bold; color: {level_color};">
                                    {score}<span style="font-size: 20px; opacity: 0.7;">/100</span>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="
                                    background: {level_color};
                                    color: {'#000' if level in ['BASELINE', 'MODERATE'] else '#FFF'};
                                    padding: 8px 16px;
                                    border-radius: 20px;
                                    font-weight: bold;
                                    font-size: 14px;
                                ">
                                    {level_emoji} {level}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 12px; font-size: 14px; opacity: 0.9;">
                            {interpretation}
                        </div>
                        {factors_html}
                        <div style="margin-top: 12px; font-size: 10px; opacity: 0.6; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">
                            ⚠️ Purely statistical analysis. Past performance is not indicative of future results.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display the prose report
                st.markdown(analysis['prose_report'])
                
                # Additional statistics in columns - Regime Distribution
                st.markdown("---")
                st.markdown("#### 📊 Historical Regime Distribution")
                
                stats = analysis['signal_stats']
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    stable = stats.get('STABLE', {})
                    st.metric("🟢 STABLE", f"{stable.get('phase_count', 0)} periods", f"{stable.get('pct_of_time', 0):.1f}%")
                
                with col2:
                    active = stats.get('ACTIVE', {})
                    st.metric("🟡 ACTIVE", f"{active.get('phase_count', 0)} periods", f"{active.get('pct_of_time', 0):.1f}%")
                
                with col3:
                    high_energy = stats.get('HIGH_ENERGY', {})
                    st.metric("🟠 HIGH ENERGY", f"{high_energy.get('phase_count', 0)} periods", f"{high_energy.get('pct_of_time', 0):.1f}%")
                
                with col4:
                    critical = stats.get('CRITICAL', {})
                    st.metric("🔴 CRITICAL", f"{critical.get('phase_count', 0)} periods", f"{critical.get('pct_of_time', 0):.1f}%")
                
                with col5:
                    dormant = stats.get('DORMANT', {})
                    st.metric("⚪ DORMANT", f"{dormant.get('phase_count', 0)} periods", f"{dormant.get('pct_of_time', 0):.1f}%")
                
                # Performance tables - Regime Statistics
                st.markdown("#### 📊 Historical Returns by Regime")
                st.caption("Statistical analysis of price movements following each regime classification")
                
                signal_order = ['STABLE', 'ACTIVE', 'HIGH_ENERGY', 'CRITICAL', 'DORMANT']
                signal_names = {
                    'STABLE': 'Stable', 'ACTIVE': 'Active', 
                    'HIGH_ENERGY': 'High Energy', 'CRITICAL': 'Critical', 'DORMANT': 'Dormant'
                }
                signal_emojis = {'STABLE': '🟢', 'ACTIVE': '🟡', 'HIGH_ENERGY': '🟠', 'CRITICAL': '🔴', 'DORMANT': '⚪'}
                
                forward_rows = []
                for sig in signal_order:
                    data = stats.get(sig, {})
                    phase_count = data.get('phase_count', 0)
                    if phase_count > 0:
                        emoji = signal_emojis[sig]
                        forward_rows.append({
                            'Regime': f"{emoji} {signal_names[sig]}",
                            'Periods': str(phase_count),
                            '10d': f"{data.get('start_return_10d', 0):+.1f}%",
                            '30d': f"{data.get('avg_return_30d', 0):+.1f}%",
                            '90d': f"{data.get('avg_return_90d', 0):+.1f}%",
                            'Max Var (10d)': f"{data.get('worst_max_dd_10d', 0):.1f}%"
                        })
                
                if forward_rows:
                    st.table(pd.DataFrame(forward_rows))
                
                # Prior conditions
                st.markdown("#### 📊 Pre-Regime Market Conditions")
                st.caption("Historical price movements BEFORE each regime was classified")
                
                prior_rows = []
                for sig in signal_order:
                    data = stats.get(sig, {})
                    phase_count = data.get('phase_count', 0)
                    if phase_count > 0:
                        emoji = signal_emojis[sig]
                        prior_rows.append({
                            'Regime': f"{emoji} {signal_names[sig]}",
                            'Prior 5d': f"{data.get('prior_5d', 0):+.1f}%",
                            'Prior 10d': f"{data.get('prior_10d', 0):+.1f}%",
                            'Prior 30d': f"{data.get('prior_30d', 0):+.1f}%"
                        })
                
                if prior_rows:
                    st.table(pd.DataFrame(prior_rows))


def render_footer():
    """Render footer with market pulse indicators (BTC, S&P 500, Gold)."""
    data = fetch_footer_data()
    if data:
        st.markdown('<div class="footer">', unsafe_allow_html=True)
        cols = st.columns(len(data))
        for i, d in enumerate(data):
            color = "#00FF00" if d['change'] >= 0 else "#FF0000"
            cols[i].markdown(
                f"<span class='footer-item'>{d['name']}: ${d['price']:,.0f} "
                f"<span style='color:{color}'>{d['change']:+.1f}%</span></span>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# INVESTMENT SIMULATION UI
# =============================================================================

def render_dca_simulation(tickers: List[str]):
    """
    Render Lump Sum Investment Simulation tab.
    
    Allows users to:
        - Select asset and simulation parameters
        - Choose strategy mode (Defensive/Aggressive)
        - Configure friction costs (fees and interest)
        - Run simulation comparing Buy & Hold vs SOC Dynamic
        - View results: returns, drawdowns, risk metrics, equity curves
    """
    is_dark = st.session_state.get('dark_mode', True)
    
    st.markdown("### 📊 Portfolio Simulation (Lump Sum)")
    st.caption("Compare Buy & Hold vs. SOC Dynamic Position Sizing")
    
    # Basic parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sim_ticker = st.selectbox(
            "Select Asset:",
            options=tickers if tickers else ['BTC-USD', 'AAPL', 'MSFT'],
            index=0
        )
    
    with col2:
        initial_capital = st.number_input(
            "Initial Capital (€):",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
    
    with col3:
        years_back = st.selectbox(
            "Simulation Period:",
            options=[3, 5, 7, 10, 15],
            index=2,
            format_func=lambda x: f"{x} Years"
        )
    
    # Strategy Mode Selection
    st.markdown("#### 🎯 SOC Strategy Mode")
    strategy_mode = st.radio(
        "Choose SOC risk profile:",
        options=["defensive", "aggressive"],
        index=0,
        format_func=lambda x: "🛡️ Defensive (Max Safety)" if x == "defensive" else "🚀 Aggressive (Max Return)",
        horizontal=True
    )
    
    # Strategy explanation
    if strategy_mode == "defensive":
        st.info("""
        **🛡️ DEFENSIVE** - Max safety, reduce exposure early
        - Bear Market: 0% | Critical: 0% | High Energy: 50% | Stable: 100%
        """)
    else:
        st.warning("""
        **🚀 AGGRESSIVE** - Max return, ride momentum longer
        - Bear Market: 0% | Critical: 50% | High Energy: 100% | Stable: 100%
        """)
    
    # Reality Settings (Fees & Interest)
    with st.expander("⚙️ Reality Settings (Fees & Interest)"):
        col_fee, col_interest = st.columns(2)
        
        with col_fee:
            trading_fee_pct = st.slider(
                "Trading Fee & Slippage (%):",
                min_value=0.0, max_value=2.0, value=0.5, step=0.1,
                format="%.1f%%"
            )
        
        with col_interest:
            interest_rate_annual = st.slider(
                "Interest on Cash (% p.a.):",
                min_value=0.0, max_value=5.0, value=3.0, step=0.5,
                format="%.1f%%"
            )
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        from datetime import datetime, timedelta
        
        start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
        
        # === LUMP SUM MODE ===
        with st.spinner(f"Simulating {years_back} years for {sim_ticker} ({strategy_mode.upper()} mode)..."):
            results = run_dca_simulation(
                sim_ticker, 
                initial_capital=initial_capital, 
                start_date=start_date, 
                years_back=years_back,
                strategy_mode=strategy_mode,
                trading_fee_pct=trading_fee_pct / 100,
                interest_rate_annual=interest_rate_annual / 100
            )
        
        if 'error' in results:
            st.error(results['error'])
            return
        
        summary = results.get('summary', {})
        
        # Results header
        st.markdown("---")
        st.markdown("#### 📊 Lump Sum Simulation Results")
        
        # Key metrics - Top row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Initial Capital",
                f"€{summary.get('initial_capital', 0):,.0f}",
                f"{summary.get('total_days', 0):,} days"
            )
        
        with col2:
            st.metric(
                "Buy & Hold Final",
                f"€{summary.get('buyhold_final', 0):,.0f}",
                f"{summary.get('buyhold_return_pct', 0):+.1f}%"
            )
        
        with col3:
            st.metric(
                "SOC Dynamic Final",
                f"€{summary.get('soc_final', 0):,.0f}",
                f"{summary.get('soc_return_pct', 0):+.1f}%"
            )
        
        with col4:
            outperformance = summary.get('outperformance_pct', 0)
            st.metric(
                "SOC Outperformance",
                f"{outperformance:+.1f}%",
                f"€{results.get('outperformance_abs', 0):+,.0f}",
                delta_color="normal" if outperformance >= 0 else "inverse"
            )
        
        # Risk metrics - Second row
        st.markdown("#### 🛡️ Risk Metrics")
        col_dd1, col_dd2, col_dd3, col_exp = st.columns(4)
        
        with col_dd1:
            max_dd_bh = summary.get('max_dd_buyhold', 0)
            st.metric(
                "Max Drawdown (B&H)",
                f"{max_dd_bh:.1f}%",
                delta=None
            )
        
        with col_dd2:
            max_dd_soc = summary.get('max_dd_soc', 0)
            st.metric(
                "Max Drawdown (SOC)",
                f"{max_dd_soc:.1f}%",
                delta=None
            )
        
        with col_dd3:
            dd_reduction = summary.get('drawdown_reduction', 0)
            st.metric(
                "Drawdown Reduction",
                f"{dd_reduction:+.1f}%",
                "Less risk" if dd_reduction > 0 else "More risk",
                delta_color="normal" if dd_reduction > 0 else "inverse"
            )
        
        with col_exp:
            avg_exp = summary.get('avg_exposure', 100)
            days_cash = summary.get('days_in_cash', 0)
            st.metric(
                "Avg. Exposure",
                f"{avg_exp:.0f}%",
                f"{days_cash} days in cash"
            )
        
        # Friction costs - Third row
        st.markdown("#### 💸 Friction Costs (Reality Check)")
        col_trades, col_fees, col_interest, col_net = st.columns(4)
        
        with col_trades:
            trade_count = summary.get('trade_count', 0)
            st.metric(
                "Total Trades",
                f"{trade_count}",
                f"~{trade_count / (years_back * 12):.1f}/month" if years_back > 0 else ""
            )
        
        with col_fees:
            total_fees = summary.get('total_fees_paid', 0)
            st.metric(
                "Fees Paid",
                f"€{total_fees:,.0f}",
                f"-{(total_fees / initial_capital) * 100:.1f}% of capital",
                delta_color="inverse"
            )
        
        with col_interest:
            total_interest = summary.get('total_interest_earned', 0)
            st.metric(
                "Interest Earned",
                f"€{total_interest:,.0f}",
                f"+{(total_interest / initial_capital) * 100:.1f}% of capital",
                delta_color="normal"
            )
        
        with col_net:
            net_friction = summary.get('net_friction', 0)
            st.metric(
                "Net Friction",
                f"€{net_friction:+,.0f}",
                "Interest > Fees" if net_friction > 0 else "Fees > Interest",
                delta_color="normal" if net_friction > 0 else "inverse"
            )
        
        # Equity curves chart
        st.markdown("#### 📈 Equity Curves Comparison")
        
        equity_df = results.get('equity_curve', pd.DataFrame())
        
        if not equity_df.empty:
            fig = go.Figure()
            
            # Buy & Hold line
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=equity_df['buyhold_value'],
                name='Buy & Hold',
                line=dict(color='#888888', width=2),
                mode='lines'
            ))
            
            # SOC Dynamic line
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=equity_df['soc_value'],
                name='SOC Dynamic',
                line=dict(color='#667eea', width=2),
                mode='lines'
            ))
            
            # Initial capital line
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=[initial_capital] * len(equity_df),
                name='Initial Capital',
                line=dict(color='#FF6600', width=1, dash='dash'),
                mode='lines'
            ))
            
            fig.update_layout(
                template="plotly_dark" if is_dark else "plotly_white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(t=20, b=40, l=40, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                xaxis_title="Date",
                yaxis_title="Portfolio Value (€)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown comparison chart
        daily_data = results.get('daily_data', pd.DataFrame())
        if not daily_data.empty and 'buyhold_drawdown' in daily_data.columns:
            st.markdown("#### 📉 Drawdown Comparison")
            
            fig_dd = go.Figure()
            
            fig_dd.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data['buyhold_drawdown'],
                name='Buy & Hold Drawdown',
                fill='tozeroy',
                line=dict(color='rgba(255,100,100,0.8)', width=1),
                fillcolor='rgba(255,100,100,0.3)'
            ))
            
            fig_dd.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data['soc_drawdown'],
                name='SOC Dynamic Drawdown',
                fill='tozeroy',
                line=dict(color='rgba(102,126,234,0.8)', width=1),
                fillcolor='rgba(102,126,234,0.3)'
            ))
            
            fig_dd.update_layout(
                template="plotly_dark" if is_dark else "plotly_white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                margin=dict(t=20, b=40, l=40, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # Strategy explanation
        with st.expander("ℹ️ Strategy Explanation"):
            soc_data = results.get('soc_dynamic', {})
            buyhold_data = results.get('buyhold', {})
            mode_name = summary.get('strategy_mode', 'Defensive')
            high_exp = summary.get('high_stress_exposure', 20)
            med_exp = summary.get('medium_stress_exposure', 50)
            
            fee_pct = summary.get('trading_fee_pct', 0.5)
            int_pct = summary.get('interest_rate_annual', 3.0)
            
            st.markdown(f"""
            **Buy & Hold (Benchmark):**
            - 100% invested in {sim_ticker} at all times
            - Simple, passive strategy
            - Final Return: **{buyhold_data.get('total_return_pct', 0):+.1f}%**
            - Max Drawdown: **{buyhold_data.get('max_drawdown_pct', 0):.1f}%**
            
            **SOC Dynamic Exposure ({mode_name} Mode):**
            - Adjusts portfolio exposure DAILY based on market conditions
            - Reduces exposure during high volatility (criticality) periods
            - Moves to 0% in bear markets (Price < SMA200)
            
            **Exposure Rules ({mode_name}):**
            - Bear Market (Price < SMA200): **0%** invested
            - Critical/Red (Criticality > 80): **{high_exp:.0f}%** invested
            - High Energy/Orange (Criticality > 60): **{med_exp:.0f}%** invested
            - Stable/Green (Uptrend, low stress): **100%** invested
            
            **Exposure Statistics:**
            - Days fully invested (100%): **{soc_data.get('days_full_invested', 0):,}** ({soc_data.get('pct_full_invested', 0):.1f}%)
            - Days partial exposure: **{soc_data.get('days_partial', 0):,}**
            - Days in cash (0%): **{soc_data.get('days_cash', 0):,}** ({soc_data.get('pct_cash', 0):.1f}%)
            - Average exposure: **{soc_data.get('avg_exposure_pct', 100):.1f}%**
            
            **Friction Costs (included in results):**
            - Trading fee: **{fee_pct:.1f}%** per trade
            - Cash interest: **{int_pct:.1f}%** p.a.
            - Total trades: **{soc_data.get('trade_count', 0)}**
            - Fees paid: **€{soc_data.get('total_fees_paid', 0):,.0f}**
            - Interest earned: **€{soc_data.get('total_interest_earned', 0):,.0f}**
            
            **Risk-Adjusted Performance:**
            - Buy & Hold Sharpe: **{buyhold_data.get('sharpe_ratio', 0):.2f}**
            - SOC Dynamic Sharpe: **{soc_data.get('sharpe_ratio', 0):.2f}**
            
            *⚠️ This is a historical backtest simulation for educational purposes only. 
            Past performance is not indicative of future results.*
            """)


# =============================================================================
# LEGAL DISCLAIMER
# =============================================================================

LEGAL_DISCLAIMER = """
## ⚖️ Legal Disclaimer & Terms of Use

**IMPORTANT: Please read this disclaimer carefully before using this application.**

---

### 1. Educational & Informational Purpose Only

This application ("SOC Market Seismograph") is provided **exclusively for educational 
and informational purposes**. The analysis, data, charts, signals, and any other 
information displayed are intended solely to help users understand market dynamics 
through the lens of Self-Organized Criticality (SOC) theory.

### 2. No Financial Advice

**THIS APPLICATION DOES NOT PROVIDE FINANCIAL, INVESTMENT, TAX, LEGAL, OR 
PROFESSIONAL ADVICE OF ANY KIND.**

- The content is not a recommendation to buy, sell, or hold any security, 
  cryptocurrency, or financial instrument.
- No fiduciary relationship is established between you and the creators of this application.
- The "signals," "regimes," and "scores" are purely statistical observations based on 
  historical data and mathematical models. They are NOT predictions of future performance.

### 3. No Guarantees or Warranties

- Past performance is **NOT indicative of future results**.
- All investment involves risk, including the potential loss of principal.
- The accuracy, completeness, or reliability of the data and analysis is NOT guaranteed.
- The application is provided "AS IS" without warranty of any kind, express or implied.

### 4. Limitation of Liability

To the fullest extent permitted by applicable law:

- The creators, developers, and operators of this application shall NOT be liable for 
  any direct, indirect, incidental, special, consequential, or punitive damages arising 
  from your use of or reliance on this application.
- This includes, but is not limited to, any losses, damages, or claims arising from 
  investment decisions made based on information displayed in this application.

### 5. Independent Verification Required

Before making any financial decision, you should:

- Consult with a qualified financial advisor, broker, or other professional.
- Conduct your own independent research and due diligence.
- Consider your personal financial situation, risk tolerance, and investment objectives.

### 6. Data Sources

Market data is sourced from third-party providers (Yahoo Finance). We do not control 
or guarantee the accuracy, timeliness, or availability of this data.

### 7. Jurisdiction

This disclaimer is governed by applicable laws. If any provision is found unenforceable, 
the remaining provisions shall continue in full force and effect.

---

**By clicking "I Understand & Accept" below, you acknowledge that:**

✓ You have read and understood this disclaimer in its entirety.

✓ You agree that this application provides NO financial advice.

✓ You accept full responsibility for any decisions you make.

✓ You waive any claims against the creators arising from your use of this application.
"""


def render_disclaimer():
    """
    Render legal disclaimer page that must be accepted before using the app.
    Uses session state to track acceptance.
    """
    st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        .disclaimer-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 24px;
            margin: 20px 0;
        }
    </style>
                    """, unsafe_allow_html=True)
                
    st.title("🔬 SOC Market Seismograph")
    st.caption("Self-Organized Criticality Analysis Tool")
    
    st.markdown("---")
    
    # Display disclaimer in scrollable container
    with st.container():
        st.markdown(LEGAL_DISCLAIMER)
    
    st.markdown("---")
    
    # Acceptance checkbox and button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        accepted = st.checkbox(
            "I have read, understood, and agree to the terms above",
            key="disclaimer_checkbox"
        )
        
        if st.button(
            "✅ I Understand & Accept",
            type="primary",
            disabled=not accepted,
            use_container_width=True
        ):
            st.session_state.disclaimer_accepted = True
            st.rerun()
        
        if not accepted:
            st.caption("⚠️ You must check the box above to continue")
    
    st.stop()


# =============================================================================
# AUTHENTICATION
# =============================================================================

def check_auth():
    """Validate access code from session state against ACCESS_CODE constant."""
    if st.session_state.get("pwd") == ACCESS_CODE:
        st.session_state.authenticated = True
        del st.session_state.pwd
    else:
        st.error("Incorrect password")


def login_page():
    """Render login page and block access until authenticated."""
    st.title("🔒 Login Required")
    st.text_input("Access Code", type="password", key="pwd", on_change=check_auth)
    st.stop()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application entry point.
    
    Flow:
        1. Show legal disclaimer (must accept to continue)
        2. Initialize session state (auth, theme, selected asset)
        3. Check authentication
        4. Apply theme CSS
        5. Render header, theory, market selection
        6. Run SOC analysis on button click
        7. Display results in tabbed layout (Deep Dive, Simulation)
        8. Render footer
    """
    # Session state initialization
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state.disclaimer_accepted = False
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    if 'selected_asset' not in st.session_state:
        st.session_state.selected_asset = 0
    
    # Legal disclaimer gate (must accept before anything else)
    if not st.session_state.disclaimer_accepted:
        render_disclaimer()
        return
    
    # Auth gate
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Apply theme
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Header
    render_header()
    render_theory()
    
    # Market selection - centered title and subtitle
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem;">
        <h3 style="margin-bottom: 0.3rem;">Market Selection</h3>
        <p style="color: #888; font-size: 0.95rem; margin-bottom: 1.5rem;">Select your Market Universe</p>
    </div>
    """, unsafe_allow_html=True)
    
    tickers = render_market_selection()
    
    # Centered Fetch Data button
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("🔍  Fetch Data", type="primary", use_container_width=True):
            st.session_state.scan_results = run_analysis(tickers)
            st.session_state.selected_asset = 0
    
    # Results - Tabbed Layout
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        results = st.session_state.scan_results
        
        st.divider()
        
        # Centered "Analysis of Results" title
        st.markdown("""
        <div style="text-align: center;">
            <h3>Analysis of Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different views (reordered as requested)
        tab_detail, tab_simulation = st.tabs([
            "🔍 Asset Deep Dive", 
            "📊 Position Sizing Simulation"
        ])
        
        with tab_detail:
            # Master-Detail Layout
            col_list, col_detail = st.columns([1, 2])
            
            with col_list:
                st.markdown("**Select an asset:**")
                
                for i, r in enumerate(results):
                    is_selected = i == st.session_state.selected_asset
                    btn_label = f"{r['symbol']} | ${r['price']:,.0f} | {r['signal'].split()[0]}"
                    if st.button(btn_label, key=f"asset_{i}", width="stretch",
                                type="primary" if is_selected else "secondary"):
                        st.session_state.selected_asset = i
                        st.rerun()
            
            with col_detail:
                if results:
                    selected = results[st.session_state.selected_asset]
                    render_detail_panel(selected)
        
        with tab_simulation:
            # Get ticker list from results
            result_tickers = [r['symbol'] for r in results]
            render_dca_simulation(result_tickers)
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()
