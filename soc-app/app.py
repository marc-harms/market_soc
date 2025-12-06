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
import requests
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from logic import DataFetcher, SOCAnalyzer, run_dca_simulation, calculate_audit_metrics
from ui_simulation import render_dca_simulation

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

# Precious metals excluded from main risk scan - they act as hedges (inverse correlation)
# and distort market risk scoring. Available separately in "Hedge Assets" category.
PRECIOUS_METALS = {'GC=F', 'SI=F', 'PL=F', 'PA=F', 'GLD', 'SLV'}

# Popular tickers for quick suggestions
POPULAR_TICKERS = {
    "US Tech": ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META'],
    "Crypto": ['BTC-USD', 'ETH-USD', 'SOL-USD'],
    "ETFs": ['SPY', 'QQQ', 'IWM', 'VTI'],
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
# STYLING & THEME
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
    .streamlit-expanderHeader {{ 
        background-color: {c['card']} !important; 
        color: {c['text']} !important;
    }}
    .streamlit-expanderHeader p, .streamlit-expanderHeader span,
    .streamlit-expanderHeader svg {{ 
        color: {c['text']} !important; 
        fill: {c['text']} !important;
    }}
    .streamlit-expanderContent {{ 
        background-color: {c['bg2']} !important; 
        color: {c['text']} !important;
    }}
    .streamlit-expanderContent p, .streamlit-expanderContent span,
    .streamlit-expanderContent label, .streamlit-expanderContent div {{
        color: {c['text']} !important;
    }}
    [data-testid="stExpander"] {{
        background-color: {c['card']} !important;
        border-color: {c['border']} !important;
    }}
    [data-testid="stExpander"] details {{
        background-color: {c['card']} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {c['text']} !important;
    }}
    .stDataFrame, [data-testid="stDataFrame"], .stDataFrame div, .stDataFrame table,
    .stDataFrame th, .stDataFrame td, [data-testid="glideDataEditor"], .dvn-scroller {{
        background-color: {c['card']} !important;
        color: {c['text']} !important;
    }}
    .stDataFrame th {{ background-color: {c['bg2']} !important; }}
    .stButton > button,
    [data-testid="baseButton-secondary"],
    [data-testid="stBaseButton-secondary"],
    button[kind="secondary"] {{
        background-color: {c['card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        font-weight: 600;
        border-radius: 8px;
    }}
    .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {{ 
        background-color: {c['bg2']} !important; 
        color: {c['text']} !important;
        border-color: #667eea !important;
    }}
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stBaseButton-primary"],
    button[kind="primary"] {{
        background-color: #667eea !important;
        color: white !important;
        border-color: #667eea !important;
    }}
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover {{
        background-color: #5568d9 !important;
    }}
    /* Ensure all button text is visible */
    .stButton button p,
    .stButton button span,
    [data-testid="baseButton-secondary"] p,
    [data-testid="baseButton-secondary"] span {{
        color: {c['text']} !important;
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
# UTILITY FUNCTIONS
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
# TICKER SEARCH & VALIDATION FUNCTIONS
# =============================================================================

def search_ticker(query: str) -> List[Dict[str, Any]]:
    """
    Search for tickers by company name or symbol using Yahoo Finance.
    
    Args:
        query: Search term (company name or ticker symbol)
    
    Returns:
        List of matching results with ticker, name, type, exchange
    """
    if not query or len(query) < 2:
        return []
    
    try:
        # Use Yahoo Finance search API
        url = f"https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': 8,
            'newsCount': 0,
            'enableFuzzyQuery': True,
            'quotesQueryId': 'tss_match_phrase_query'
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for quote in data.get('quotes', []):
            # Filter for stocks, ETFs, and crypto only
            quote_type = quote.get('quoteType', '').upper()
            if quote_type in ['EQUITY', 'ETF', 'CRYPTOCURRENCY', 'INDEX', 'MUTUALFUND']:
                results.append({
                    'ticker': quote.get('symbol', ''),
                    'name': quote.get('shortname') or quote.get('longname') or quote.get('symbol', ''),
                    'type': quote_type,
                    'exchange': quote.get('exchange', '')
                })
        
        return results
        
    except requests.exceptions.RequestException:
        return []
    except Exception:
        return []


def validate_ticker(ticker: str) -> Dict[str, Any]:
    """
    Validate a ticker symbol using yfinance.
    
    Args:
        ticker: Stock/crypto ticker symbol (e.g., 'AAPL', 'BTC-USD')
    
    Returns:
        Dict with 'valid' (bool), 'name' (str), 'error' (str if invalid)
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return {'valid': False, 'error': 'Empty ticker'}
        
        stock = yf.Ticker(ticker)
        
        # Try to get recent price history first (most reliable check)
        hist = stock.history(period='5d')
        
        if hist.empty:
            return {'valid': False, 'error': f'Ticker "{ticker}" not found'}
        
        # Get latest price from history
        latest_price = hist['Close'].iloc[-1] if not hist.empty else 0
        
        # Try to get info for name (but don't fail if unavailable)
        name = ticker
        try:
            info = stock.info
            if info:
                # Try multiple name fields
                name = (info.get('shortName') or 
                       info.get('longName') or 
                       info.get('name') or 
                       ticker)
                # Try multiple price fields if history price is 0
                if latest_price == 0:
                    latest_price = (info.get('regularMarketPrice') or 
                                  info.get('previousClose') or 
                                  info.get('currentPrice') or 
                                  0)
        except:
            # If info fails, just use ticker as name
            pass
        
        return {
            'valid': True,
            'ticker': ticker,
            'name': name,
            'price': latest_price
        }
        
    except requests.exceptions.RequestException:
        return {'valid': False, 'error': 'API not responding. Please try again in 5 minutes.'}
    except Exception as e:
        error_msg = str(e).lower()
        if 'connection' in error_msg or 'timeout' in error_msg or 'network' in error_msg:
            return {'valid': False, 'error': 'API not responding. Please try again in 5 minutes.'}
        return {'valid': False, 'error': f'Ticker "{ticker}" not found'}


# =============================================================================
# DETAIL PANEL UI COMPONENTS
# =============================================================================

def render_regime_persistence_chart(current_regime: str, current_duration: int, regime_stats: Dict[str, Any], is_dark: bool = False):
    """
    Render a horizontal bar chart showing current regime duration vs historical average.
    
    Args:
        current_regime: Name of current regime (e.g., 'STABLE')
        current_duration: Days in current regime
        regime_stats: Historical statistics for this regime
        is_dark: Dark mode flag
    """
    # Get historical stats (using correct keys from logic.py)
    mean_duration = regime_stats.get('avg_duration', 0)  # avg_duration, not mean_duration
    median_duration = regime_stats.get('median_duration', 0)
    max_duration = regime_stats.get('max_duration', 0)
    p95_duration = regime_stats.get('p95_duration', 0)
    
    # Regime colors
    regime_colors = {
        'STABLE': '#00C864',
        'ACTIVE': '#FFCC00',
        'HIGH_ENERGY': '#FF6600',
        'CRITICAL': '#FF4040',
        'DORMANT': '#888888'
    }
    
    regime_color = regime_colors.get(current_regime, '#667eea')
    
    # Handle edge cases
    if max_duration == 0:
        max_duration = max(current_duration * 2, 30)  # Fallback
    if mean_duration == 0:
        mean_duration = current_duration  # Use current as reference
    
    # Theme-aware colors
    text_color = '#FFFFFF' if is_dark else '#1a1a1a'
    axis_color = '#CCCCCC' if is_dark else '#333333'
    grid_color = '#444444' if is_dark else '#E0E0E0'
    bg_color = 'rgba(0,0,0,0)' if is_dark else 'rgba(248,248,248,1)'
    annotation_color = '#FFFFFF' if is_dark else '#333333'
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Background range (0 to max)
    fig.add_trace(go.Bar(
        y=['Duration'],
        x=[max_duration],
        orientation='h',
        marker=dict(color='rgba(200,200,200,0.3)' if is_dark else 'rgba(200,200,200,0.5)'),
        name='Max Observed',
        showlegend=False
    ))
    
    # Current duration bar
    fig.add_trace(go.Bar(
        y=['Duration'],
        x=[current_duration],
        orientation='h',
        marker=dict(color=regime_color),
        name='Current',
        showlegend=False
    ))
    
    # Add vertical lines for mean and P95
    fig.add_vline(x=mean_duration, line_dash="dash", line_color="#667eea", line_width=3,
                  annotation_text=f"Avg: {mean_duration:.0f}d", annotation_position="top",
                  annotation=dict(font=dict(color=annotation_color, size=13)))
    
    if p95_duration > 0:
        fig.add_vline(x=p95_duration, line_dash="dot", line_color="#FF6600", line_width=3,
                      annotation_text=f"95th: {p95_duration:.0f}d", annotation_position="bottom",
                      annotation=dict(font=dict(color=annotation_color, size=13)))
    
    # Update layout with explicit colors
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,0)',
        plot_bgcolor=bg_color,
        height=180,
        margin=dict(l=80, r=30, t=50, b=50),
        showlegend=False,
        font=dict(color=text_color, size=13)
    )
    
    # Update axes separately (correct Plotly API)
    fig.update_xaxes(
        range=[0, max_duration * 1.1],
        title_text="Days",
        title_font=dict(color=axis_color, size=14),
        tickfont=dict(color=axis_color, size=12),
        gridcolor=grid_color
    )
    
    fig.update_yaxes(
        title_text="",
        tickfont=dict(color=axis_color, size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    if current_duration > p95_duration:
        interpretation = f"⚠️ **Statistical Anomaly:** This regime has lasted {current_duration} days, which is unusually long (above 95th percentile of {p95_duration:.0f} days). Mean reversion probability is elevated."
        st.warning(interpretation)
    elif current_duration > mean_duration:
        interpretation = f"📊 This regime has lasted {current_duration} days, which is **above** the historical average of {mean_duration:.0f} days. Median duration: {median_duration:.0f} days."
        st.info(interpretation)
    else:
        interpretation = f"📊 This regime is still relatively young at {current_duration} days, **below** the historical average of {mean_duration:.0f} days. Median duration: {median_duration:.0f} days."
        st.info(interpretation)


def render_current_regime_outlook(current_regime: str, regime_data: Dict[str, Any]):
    """
    Render a table showing the historical outlook for the CURRENT regime only.
    
    Args:
        current_regime: Name of current regime
        regime_data: Statistical data for this regime
    """
    regime_display = current_regime.replace('_', ' ').title()
    regime_emojis = {'STABLE': '🟢', 'ACTIVE': '🟡', 'HIGH_ENERGY': '🟠', 'CRITICAL': '🔴', 'DORMANT': '⚪'}
    emoji = regime_emojis.get(current_regime, '📊')
    
    st.markdown(f"##### 🎯 Historical Outlook: {emoji} {regime_display} Regime")
    st.caption("📊 **How to read this:** Shows average price movements following the start of this regime in the past. Use this to understand typical behavior patterns for the current market state.")
    
    # Check if we have data
    phase_count = regime_data.get('phase_count', 0)
    if phase_count == 0:
        st.info("No historical data available for this regime.")
        return
    
    st.markdown(f"*Based on **{phase_count}** historical occurrences of this regime*")
    
    # Build outlook table using available data
    ret_10d = regime_data.get('start_return_10d', 0)
    ret_30d = regime_data.get('avg_return_30d', 0)
    ret_90d = regime_data.get('avg_return_90d', 0)
    dd_10d = regime_data.get('worst_max_dd_10d', 0)
    avg_price_change = regime_data.get('avg_price_change_during', 0)
    
    # Create metrics in columns for better display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("10-Day Avg Return", f"{ret_10d:+.1f}%")
    with col2:
        st.metric("30-Day Avg Return", f"{ret_30d:+.1f}%")
    with col3:
        st.metric("90-Day Avg Return", f"{ret_90d:+.1f}%")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("Avg During Phase", f"{avg_price_change:+.1f}%")
    with col5:
        st.metric("Worst 10d Drawdown", f"{dd_10d:.1f}%" if dd_10d != 0 else "N/A")
    with col6:
        st.metric("Phase Count", f"{phase_count}")
    
    # Add interpretation
    if ret_30d > 5:
        st.success(f"📈 Historically, this regime has shown **positive momentum** with an average 30-day return of {ret_30d:+.1f}%.")
    elif ret_30d < -5:
        st.error(f"📉 Historically, this regime has shown **negative momentum** with an average 30-day return of {ret_30d:+.1f}%.")
    elif ret_30d != 0:
        st.info(f"📊 Historically, this regime has shown **neutral momentum** with an average 30-day return of {ret_30d:+.1f}%.")
    else:
        st.info("📊 Insufficient historical data for return analysis.")


def render_detail_panel(result: Dict[str, Any]):
    """
    Render detailed analysis panel for a selected asset.
    
    Displays: Header with regime badge, key metrics (price, criticality,
    vol percentile, trend), SOC chart, and VISUAL analysis with:
    - Regime Persistence Visualizer (bar chart)
    - Current Regime Outlook (focused table)
    - Historical data in expander with donut chart
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
    
    # Explanation of Regime
    st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.1); border-left: 3px solid #667eea; padding: 12px; margin: 12px 0; border-radius: 4px;">
        <strong>📖 What is a "Regime"?</strong><br>
        <span style="font-size: 0.9rem; opacity: 0.9;">
        A <strong>regime</strong> is the asset's current statistical behavior pattern based on price volatility and trend direction. 
        Think of it as the market's "mood" for this asset: <span style="color: #00C864;">🟢 Stable</span> (low volatility, clear trend), 
        <span style="color: #FFB800;">🟡 Transitioning</span> (moderate volatility, changing direction), or 
        <span style="color: #FF4040;">🔴 Volatile</span> (high volatility, unclear direction).
        </span>
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
        
        # Historical Signal Analysis
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
                    
                    # Explanation of relationship between Regime and Stress Level
                    st.markdown("""
                    <div style="background: rgba(102, 126, 234, 0.1); border-left: 3px solid #667eea; padding: 12px; margin: 12px 0; border-radius: 4px;">
                        <strong>🎯 Regime vs. Stress Level – What's the Difference?</strong><br>
                        <span style="font-size: 0.9rem; opacity: 0.9;">
                        • <strong>Regime</strong> (shown at top) = This asset's individual price behavior pattern<br>
                        • <strong>Systemic Stress Level</strong> (shown above) = Overall market-wide risk across volatility, correlations, and trends<br><br>
                        <strong>Why can they differ?</strong> An asset can be in a 🟢 Stable regime (behaving normally) while the broader market shows 🟠 Heightened stress (system-wide instability). 
                        The asset might be insulated now, but elevated systemic stress suggests potential future spillover risk.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Get current regime from analysis
                stats = analysis['signal_stats']
                current_regime_key = analysis.get('current_signal', 'STABLE')  # From historical analysis
                current_regime_data = stats.get(current_regime_key, {})
                current_duration = analysis.get('current_streak_days', 0)  # Days in current regime
                
                st.markdown("---")
                
                # === SECTION A: REGIME PERSISTENCE VISUALIZER ===
                st.markdown("#### ⏱️ Regime Persistence Analysis")
                st.caption("📊 **How to read this:** The colored bar shows how long the asset has been in the current regime. The dashed line shows the historical average duration. If the bar extends beyond the 95th percentile line, the regime may be nearing exhaustion.")
                render_regime_persistence_chart(current_regime_key, current_duration, current_regime_data, is_dark)
                
                st.markdown("---")
                
                # === SECTION B: CURRENT REGIME OUTLOOK ===
                render_current_regime_outlook(current_regime_key, current_regime_data)
                
                st.markdown("---")
                
                # === SECTION C: FULL HISTORICAL DATA (EXPANDER) ===
                with st.expander("📚 View All Historical Regime Data"):
                    # Regime Distribution Donut Chart
                    st.markdown("##### Historical Regime Distribution")
                    
                    signal_order = ['STABLE', 'ACTIVE', 'HIGH_ENERGY', 'CRITICAL', 'DORMANT']
                    signal_names = {
                        'STABLE': 'Stable', 'ACTIVE': 'Active', 
                        'HIGH_ENERGY': 'High Energy', 'CRITICAL': 'Critical', 'DORMANT': 'Dormant'
                    }
                    signal_emojis = {'STABLE': '🟢', 'ACTIVE': '🟡', 'HIGH_ENERGY': '🟠', 'CRITICAL': '🔴', 'DORMANT': '⚪'}
                    signal_colors_map = {
                        'STABLE': '#00C864', 'ACTIVE': '#FFCC00', 
                        'HIGH_ENERGY': '#FF6600', 'CRITICAL': '#FF4040', 'DORMANT': '#888888'
                    }
                    
                    # Build donut chart data
                    labels = []
                    values = []
                    colors = []
                    
                    for sig in signal_order:
                        data = stats.get(sig, {})
                        pct = data.get('pct_of_time', 0)
                        if pct > 0:
                            emoji = signal_emojis.get(sig, '')
                            name = signal_names.get(sig, sig)
                            labels.append(f"{emoji} {name}")
                            values.append(pct)
                            colors.append(signal_colors_map.get(sig, '#888888'))
                    
                    # Create donut chart
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        marker=dict(colors=colors),
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    
                    bg_color_chart = 'rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,0)'
                    text_color_chart = '#FFFFFF' if is_dark else '#333333'
                    
                    fig_donut.update_layout(
                        template="plotly_dark" if is_dark else "plotly_white",
                        paper_bgcolor=bg_color_chart,
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=True,
                        legend=dict(font=dict(color=text_color_chart)),
                        font=dict(color=text_color_chart)
                    )
                    
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Full Returns Table (All Regimes)
                    st.markdown("##### Complete Historical Returns by Regime")
                    st.caption("Statistical analysis of price movements following each regime classification")
                    
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
                                'Max DD (10d)': f"{data.get('worst_max_dd_10d', 0):.1f}%"
                            })
                    
                    if forward_rows:
                        st.table(pd.DataFrame(forward_rows))
                    else:
                        st.info("No historical regime data available.")
                    
                    st.markdown("---")
                    
                    # Pre-Regime Conditions Table
                    st.markdown("##### Pre-Regime Market Conditions")
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
                    else:
                        st.info("No historical regime data available.")


# =============================================================================
# INVESTMENT SIMULATION UI
# =============================================================================

# NOTE: Simulation UI moved to ui_simulation.py
# Import: from ui_simulation import render_dca_simulation

# =============================================================================
# LEGAL DISCLAIMER & AUTHENTICATION
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

def render_sticky_cockpit_header():
    """
    Render the persistent "Cockpit" header with search and status.
    
    Layout:
        Row 1: Logo | Search Field | Status Badge
        Row 2: Daily Summary Metrics (planned feature - placeholder for now)
    """
    is_dark = st.session_state.get('dark_mode', False)
    
    with st.container(border=True):
        # === ROW 1: Logo | Search | Status ===
        col_logo, col_search, col_status = st.columns([2, 4, 2])
        
        with col_logo:
            st.markdown('<h4 style="margin: 0;">📉 SOC Seismograph</h4>', unsafe_allow_html=True)
        
        with col_search:
            # Search with inline button
            search_col, btn_col = st.columns([4, 1])
            
            with search_col:
                search_query = st.text_input(
                    "Analyze Asset...",
                    placeholder="Type ticker (e.g., AAPL, BTC-USD)",
                    label_visibility="collapsed",
                    key="cockpit_search"
                )
            
            with btn_col:
                if st.button("🔍", key="search_btn", help="Analyze", use_container_width=True):
                    if search_query and len(search_query) > 0:
                        ticker_input = search_query.strip().upper()
                        
                        # Try to find ticker if user typed company name
                        with st.spinner(f"Searching for {ticker_input}..."):
                            try:
                                # First try direct ticker
                                validation = validate_ticker(ticker_input)
                                
                                if validation.get('valid'):
                                    # Valid ticker - analyze it
                                    results = run_analysis([ticker_input])
                                    if results and len(results) > 0:
                                        st.session_state.current_ticker = ticker_input
                                        st.session_state.scan_results = results
                                        st.session_state.selected_asset = 0
                                        st.session_state.analysis_mode = "deep_dive"
                                        st.rerun()
                                else:
                                    # Not a valid ticker - try searching for it
                                    search_results = search_ticker(ticker_input)
                                    if search_results:
                                        # Found matches - store for selection
                                        st.session_state.ticker_suggestions = search_results
                                    else:
                                        st.error(f"Could not find '{ticker_input}'. Try entering the exact ticker symbol (e.g., AAPL, SIE.DE, BTC-USD).")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please enter a ticker symbol")
        
        with col_status:
            # Show status badge if asset is selected
            if 'scan_results' in st.session_state and st.session_state.scan_results:
                results = st.session_state.scan_results
                if results:
                    selected = results[st.session_state.selected_asset]
                    regime_emoji = selected['signal'].split()[0] if selected.get('signal') else "⚪"
                    score = int(selected.get('criticality_score', 0))
                    
                    # Color code based on score
                    if score > 80:
                        badge_color = "#FF4040"
                    elif score > 60:
                        badge_color = "#FF6600"
                    else:
                        badge_color = "#00C864"
                    
                    badge_html = f'<div style="text-align: center; padding: 8px; background: rgba(102, 126, 234, 0.1); border-radius: 8px; border: 1px solid #667eea;"><span style="font-size: 0.85rem; color: #888;">Active</span><br><span style="font-size: 1.1rem; font-weight: 600;">{selected["symbol"]}</span> <span style="font-size: 1.3rem;">{regime_emoji}</span><br><span style="color: {badge_color}; font-weight: 700;">{score}</span></div>'
                    st.markdown(badge_html, unsafe_allow_html=True)


def render_education_landing():
    """
    Render the landing page when no asset is selected.
    
    Shows:
        - How SOC works (brief)
        - Top Movers / Risk List (teaser)
    """
    st.markdown("### Welcome to the SOC Market Seismograph")
    
    st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        <h4>🔬 How It Works</h4>
        <p style="font-size: 0.95rem; line-height: 1.6;">
            Self-Organized Criticality (SOC) is a physics concept applied to financial markets. 
            This app analyzes volatility patterns and trend deviations to classify assets into 
            <b>5 regime states</b> (Dormant → Stable → Active → High Energy → Critical).
        </p>
        <p style="font-size: 0.95rem; line-height: 1.6;">
            The <b>Criticality Score (0-100)</b> indicates systemic stress levels. Higher scores 
            suggest increased volatility and potential instability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    <div style="font-size: 0.95rem; line-height: 1.8; margin-left: 1rem;">
        1. <b>Type a ticker</b> in the search box above (e.g., AAPL, TSLA, BTC-USD)<br>
        2. Analysis runs automatically<br>
        3. Explore <b>Deep Dive</b> (signals, charts) or <b>Simulation</b> (backtesting)<br>
        4. Switch assets instantly using the search bar
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Quick add popular assets
    st.markdown("### 📊 Popular Assets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    popular_tickers = [
        ("AAPL", "Apple"),
        ("NVDA", "NVIDIA"),
        ("BTC-USD", "Bitcoin"),
        ("^GSPC", "S&P 500")
    ]
    
    for i, (ticker, name) in enumerate(popular_tickers):
        with [col1, col2, col3, col4][i]:
            if st.button(f"{ticker}\n{name}", key=f"quick_{ticker}", use_container_width=True):
                with st.spinner(f"Analyzing {name}..."):
                    try:
                        results = run_analysis([ticker])
                        if results and len(results) > 0:
                            st.session_state.current_ticker = ticker
                            st.session_state.scan_results = results
                            st.session_state.selected_asset = 0
                            st.session_state.analysis_mode = "deep_dive"
                            st.rerun()
                        else:
                            st.error(f"Could not analyze {ticker}")
                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {str(e)}")


def main():
    """
    Main application entry point with Sticky Cockpit Header.
    
    Flow:
        1. Show legal disclaimer (must accept to continue)
        2. Initialize session state
        3. Check authentication
        4. Apply theme CSS
        5. Render sticky cockpit header (always visible)
        6. Main content area:
           - No asset: Education landing + quick picks
           - Asset selected: Deep Dive or Simulation tabs
    """
    # Session state initialization
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state.disclaimer_accepted = False
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False  # Light mode default
    if 'selected_asset' not in st.session_state:
        st.session_state.selected_asset = 0
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "deep_dive"
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
    
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
    
    # === STICKY COCKPIT HEADER (Always Visible) ===
    render_sticky_cockpit_header()
    
    # === TICKER SUGGESTIONS (if user searched by company name) ===
    if 'ticker_suggestions' in st.session_state and st.session_state.ticker_suggestions:
        # Centered info box
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.info("🔍 Not a ticker symbol. Did you mean one of these?")
        
        st.markdown("#### Select a ticker:")
        suggestions = st.session_state.ticker_suggestions[:6]  # Max 6 suggestions
        
        num_cols = min(3, len(suggestions))
        cols = st.columns(num_cols)
        
        for i, suggestion in enumerate(suggestions):
            col_idx = i % num_cols
            ticker = suggestion.get('ticker', '') or suggestion.get('symbol', '')  # Handle both keys
            
            # Skip empty tickers
            if not ticker:
                continue
                
            name = suggestion.get('name', ticker)[:25]
            exchange = suggestion.get('exchange', '')
            
            with cols[col_idx]:
                btn_label = f"{ticker}\n{name}"
                if exchange:
                    btn_label += f"\n({exchange})"
                
                if st.button(btn_label, key=f"suggest_{ticker}_{i}", use_container_width=True):
                    # Clear suggestions and analyze this ticker
                    st.session_state.ticker_suggestions = []
                    st.session_state.current_ticker = ticker
                    
                    # Run analysis
                    with st.spinner(f"Analyzing {ticker}..."):
                        try:
                            results = run_analysis([ticker])
                            if results and len(results) > 0:
                                st.session_state.scan_results = results
                                st.session_state.selected_asset = 0
                                st.session_state.analysis_mode = "deep_dive"
                                st.rerun()
                            else:
                                st.error(f"No data available for {ticker}. Try a different exchange variant.")
                        except Exception as e:
                            st.error(f"Error analyzing {ticker}: {str(e)}")
        
        # Clear button
        if st.button("✕ Clear suggestions", key="clear_suggestions"):
            st.session_state.ticker_suggestions = []
            st.rerun()
        
        st.markdown("---")
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # === MAIN CONTENT AREA (Dynamic) ===
    if 'scan_results' not in st.session_state or not st.session_state.scan_results:
        # CONDITION A: No Asset Selected - Show Education Landing
        render_education_landing()
    else:
        # CONDITION B: Asset Selected - Show Analysis
        results = st.session_state.scan_results
        
        # === ANALYSIS MODE TABS ===
        col_spacer1, col_tab1, col_tab2, col_spacer2 = st.columns([1, 2, 2, 1])
        
        with col_tab1:
            if st.button(
                "📊 Asset Deep Dive",
                key="btn_deep_dive",
                use_container_width=True,
                type="primary" if st.session_state.analysis_mode == "deep_dive" else "secondary"
            ):
                st.session_state.analysis_mode = "deep_dive"
                st.rerun()
        
        with col_tab2:
            if st.button(
                "🎯 Portfolio Simulation",
                key="btn_simulation",
                use_container_width=True,
                type="primary" if st.session_state.analysis_mode == "simulation" else "secondary"
            ):
                st.session_state.analysis_mode = "simulation"
                st.rerun()
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        # === RENDER SELECTED ANALYSIS ===
        if st.session_state.analysis_mode == "deep_dive":
            # Deep Dive Analysis
            if results:
                selected = results[st.session_state.selected_asset]
                render_detail_panel(selected)
        else:
            # Portfolio Simulation
            result_tickers = [r['symbol'] for r in results]
            render_dca_simulation(result_tickers)


if __name__ == "__main__":
    main()
