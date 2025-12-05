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
import requests
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
        info = stock.info
        
        # Check if we got valid data
        if not info or info.get('regularMarketPrice') is None:
            # Try to get history as fallback
            hist = stock.history(period='5d')
            if hist.empty:
                return {'valid': False, 'error': f'Ticker "{ticker}" not found'}
            return {
                'valid': True,
                'ticker': ticker,
                'name': ticker,
                'price': hist['Close'].iloc[-1] if not hist.empty else 0
            }
        
        name = info.get('shortName') or info.get('longName') or ticker
        price = info.get('regularMarketPrice') or info.get('previousClose', 0)
        
        return {
            'valid': True,
            'ticker': ticker,
            'name': name,
            'price': price
        }
        
    except requests.exceptions.RequestException:
        return {'valid': False, 'error': 'API not responding. Please try again in 5 minutes.'}
    except Exception as e:
        error_msg = str(e).lower()
        if 'connection' in error_msg or 'timeout' in error_msg or 'network' in error_msg:
            return {'valid': False, 'error': 'API not responding. Please try again in 5 minutes.'}
        return {'valid': False, 'error': f'Ticker "{ticker}" not found'}


def render_ticker_search() -> List[str]:
    """
    Render universal ticker search UI with autocomplete.
    
    Features:
        - Search by company name OR ticker symbol
        - Autocomplete dropdown with matching results
        - Real-time validation via yfinance
        - Error handling for API issues
    
    Returns:
        List of validated ticker symbols to analyze.
    """
    # Initialize session state
    if 'ticker_list' not in st.session_state:
        st.session_state.ticker_list = []
    if 'validated_tickers' not in st.session_state:
        st.session_state.validated_tickers = {}
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""
    if 'clear_search' not in st.session_state:
        st.session_state.clear_search = False
    
    # Clear search input if flagged
    if st.session_state.clear_search:
        st.session_state.clear_search = False
        st.session_state.search_results = []
        st.session_state.last_search = ""
    
    # CSS for styling
    st.markdown("""
    <style>
        .ticker-chip {
            display: inline-block;
            background: rgba(102, 126, 234, 0.2);
            border: 1px solid #667eea;
            border-radius: 20px;
            padding: 6px 14px;
            margin: 4px;
            font-size: 0.9rem;
        }
        .ticker-chip-valid {
            background: rgba(0, 200, 0, 0.15);
            border-color: #00C800;
        }
        .search-result {
            padding: 8px 12px;
            border-bottom: 1px solid #333;
            cursor: pointer;
        }
        .search-result:hover {
            background: rgba(102, 126, 234, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Search input with autocomplete
    st.markdown("#### Search Assets")
    st.caption("Type a company name or ticker symbol (e.g., Tesla, AAPL, Bitcoin)")
    
    # Search input - use dynamic key to allow clearing
    search_key = f"ticker_search_{len(st.session_state.ticker_list)}"
    search_query = st.text_input(
        "Search:",
        placeholder="Type to search... (e.g., Tesla, Apple, Bitcoin, NVDA)",
        key=search_key,
        label_visibility="collapsed"
    )
    
    # Search and show results as dropdown
    if search_query and len(search_query) >= 2:
        # Only search if query changed
        if search_query != st.session_state.last_search:
            st.session_state.last_search = search_query
            with st.spinner("Searching..."):
                st.session_state.search_results = search_ticker(search_query)
        
        # Display search results as selectable options
        if st.session_state.search_results:
            st.markdown("""
            <div style="background: rgba(30,30,40,0.9); border: 1px solid #444; 
                        border-radius: 8px; margin-top: -10px; max-height: 250px; overflow-y: auto;">
            """, unsafe_allow_html=True)
            
            for i, result in enumerate(st.session_state.search_results[:8]):
                ticker = result['ticker']
                name = result['name']
                asset_type = result['type']
                exchange = result['exchange']
                
                # Type emoji
                type_emoji = "📈" if asset_type == "EQUITY" else "📊" if asset_type == "ETF" else "₿" if asset_type == "CRYPTOCURRENCY" else "📉"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{ticker}** — {name[:35]}{'...' if len(name) > 35 else ''}")
                    st.caption(f"{type_emoji} {asset_type} · {exchange}")
                with col2:
                    if st.button("Add", key=f"add_{ticker}_{i}", use_container_width=True):
                        if ticker not in st.session_state.ticker_list:
                            st.session_state.ticker_list.append(ticker)
                            # Auto-validate when adding
                            validation = validate_ticker(ticker)
                            st.session_state.validated_tickers[ticker] = validation
                        # Clear search state - the dynamic key will reset the input
                        st.session_state.search_results = []
                        st.session_state.last_search = ""
                        st.session_state.clear_search = True
                        st.rerun()
                
                st.markdown("<hr style='margin: 4px 0; border-color: #333;'>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif st.session_state.last_search == search_query:
            st.warning(f"No results found for '{search_query}'. Try a different search term.")
    
    # Quick add buttons
    st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
    st.caption("Quick add popular:")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    quick_tickers = [('AAPL', 'Apple'), ('NVDA', 'NVIDIA'), ('MSFT', 'Microsoft'), 
                     ('BTC-USD', 'Bitcoin'), ('TSLA', 'Tesla'), ('SPY', 'S&P 500')]
    
    for col, (ticker, name) in zip([col1, col2, col3, col4, col5, col6], quick_tickers):
        with col:
            if st.button(ticker, key=f"quick_{ticker}", help=name, use_container_width=True):
                if ticker not in st.session_state.ticker_list:
                    st.session_state.ticker_list.append(ticker)
                    validation = validate_ticker(ticker)
                    st.session_state.validated_tickers[ticker] = validation
                    st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display selected tickers
    if st.session_state.ticker_list:
        st.markdown("---")
        st.markdown("#### Selected Assets")
        
        # Display as chips
        valid_tickers = []
        chips_html = []
        
        for ticker in st.session_state.ticker_list:
            validation = st.session_state.validated_tickers.get(ticker, {})
            
            if validation.get('valid'):
                name = validation.get('name', ticker)
                price = validation.get('price', 0)
                display_name = name[:18] + '...' if len(name) > 18 else name
                chips_html.append(
                    f'<span class="ticker-chip ticker-chip-valid">✓ <b>{ticker}</b> ({display_name}) ${price:,.2f}</span>'
                )
                valid_tickers.append(ticker)
            else:
                error = validation.get('error', 'Validating...')
                chips_html.append(
                    f'<span class="ticker-chip" style="background: rgba(255,100,100,0.15); border-color: #FF6464;">✗ {ticker} - {error}</span>'
                )
        
        ticker_html = '<div style="margin: 8px 0;">' + ' '.join(chips_html) + '</div>'
        st.markdown(ticker_html, unsafe_allow_html=True)
        
        # Clear and remove buttons
        col_clear, col_space = st.columns([1, 3])
        with col_clear:
            if st.button("Clear All", use_container_width=True):
                st.session_state.ticker_list = []
                st.session_state.validated_tickers = {}
                st.session_state.pop('scan_results', None)
                st.rerun()
        
        # Remove individual tickers
        if len(st.session_state.ticker_list) > 0:
            st.caption("Click to remove:")
            num_cols = min(len(st.session_state.ticker_list), 6)
            cols = st.columns(num_cols)
            for i, ticker in enumerate(st.session_state.ticker_list[:6]):
                with cols[i]:
                    if st.button(f"✗ {ticker}", key=f"remove_{ticker}"):
                        st.session_state.ticker_list.remove(ticker)
                        st.session_state.validated_tickers.pop(ticker, None)
                        st.session_state.pop('scan_results', None)
                        st.rerun()
        
        # Summary
        if valid_tickers:
            st.success(f"Ready to analyze {len(valid_tickers)} asset{'s' if len(valid_tickers) > 1 else ''}")
        
        return valid_tickers
    
    else:
        st.info("Search for assets above or use quick add buttons to begin.")
        return []


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
    Render Lump Sum Investment Simulation comparing all three strategies:
    Buy & Hold, Defensive SOC, and Aggressive SOC.
    
    Shows combined equity curves, drawdown comparison, and metrics table.
    """
    is_dark = st.session_state.get('dark_mode', True)
    
    st.markdown("### Portfolio Simulation")
    st.caption("Compare Buy & Hold vs. Defensive vs. Aggressive SOC Strategies")
    
    # Get selected asset from Analysis Results section
    selected_idx = st.session_state.get('selected_asset', 0)
    if tickers and selected_idx < len(tickers):
        sim_ticker = tickers[selected_idx]
    else:
        sim_ticker = tickers[0] if tickers else 'BTC-USD'
    
    # Show currently selected asset
    st.markdown(f"""
    <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea; 
                border-radius: 8px; padding: 12px; margin-bottom: 1rem; text-align: center;">
        <span style="color: #888; font-size: 0.85rem;">Simulating for:</span>
        <span style="color: #667eea; font-weight: 600; font-size: 1.1rem; margin-left: 8px;">{sim_ticker}</span>
        <span style="color: #666; font-size: 0.8rem; margin-left: 8px;">(change in Asset Selection above)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters row
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Capital (€):",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
    
    with col2:
        years_back = st.selectbox(
            "Simulation Period:",
            options=[3, 5, 7, 10, 15],
            index=2,
            format_func=lambda x: f"{x} Years"
        )
    
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
    
    # Run simulation button - centered and smaller
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col_left, col_btn, col_right = st.columns([1, 1, 1])
    with col_btn:
        run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)
    
    if run_clicked:
        from datetime import datetime, timedelta
        
        start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
        
        # === RUN BOTH SIMULATIONS ===
        with st.spinner(f"Simulating {years_back} years for {sim_ticker} (Defensive + Aggressive)..."):
            # Run Defensive simulation
            results_def = run_dca_simulation(
                sim_ticker, 
                initial_capital=initial_capital, 
                start_date=start_date, 
                years_back=years_back,
                strategy_mode="defensive",
                trading_fee_pct=trading_fee_pct / 100,
                interest_rate_annual=interest_rate_annual / 100
            )
            
            # Run Aggressive simulation
            results_agg = run_dca_simulation(
                sim_ticker, 
                initial_capital=initial_capital, 
                start_date=start_date, 
                years_back=years_back,
                strategy_mode="aggressive",
                trading_fee_pct=trading_fee_pct / 100,
                interest_rate_annual=interest_rate_annual / 100
            )
        
        # Check for errors
        if 'error' in results_def:
            st.error(f"Defensive simulation error: {results_def['error']}")
            return
        if 'error' in results_agg:
            st.error(f"Aggressive simulation error: {results_agg['error']}")
            return
        
        # Store in session state for theme persistence
        st.session_state.sim_results_def = results_def
        st.session_state.sim_results_agg = results_agg
        st.session_state.sim_initial_capital = initial_capital
        st.session_state.sim_years_back = years_back
        st.session_state.sim_ticker = sim_ticker
        st.session_state.sim_trading_fee_pct = trading_fee_pct
        st.session_state.sim_interest_rate_annual = interest_rate_annual
    
    # Display results if available in session state
    if 'sim_results_def' not in st.session_state:
        return
    
    results_def = st.session_state.sim_results_def
    results_agg = st.session_state.sim_results_agg
    initial_capital = st.session_state.sim_initial_capital
    years_back = st.session_state.sim_years_back
    trading_fee_pct = st.session_state.sim_trading_fee_pct
    interest_rate_annual = st.session_state.sim_interest_rate_annual
    
    sum_def = results_def.get('summary', {})
    sum_agg = results_agg.get('summary', {})
    
    # Results header
    st.markdown("---")
    st.markdown("#### Strategy Comparison Results")
    
    # === COMPARISON TABLE ===
    st.markdown("##### Performance Overview")
    
    # Build comparison data
    bh_final = sum_def.get('buyhold_final', 0)
    bh_return = sum_def.get('buyhold_return_pct', 0)
    bh_dd = sum_def.get('max_dd_buyhold', 0)
    
    def_final = sum_def.get('soc_final', 0)
    def_return = sum_def.get('soc_return_pct', 0)
    def_dd = sum_def.get('max_dd_soc', 0)
    def_trades = sum_def.get('trade_count', 0)
    def_exposure = sum_def.get('avg_exposure', 100)
    
    agg_final = sum_agg.get('soc_final', 0)
    agg_return = sum_agg.get('soc_return_pct', 0)
    agg_dd = sum_agg.get('max_dd_soc', 0)
    agg_trades = sum_agg.get('trade_count', 0)
    agg_exposure = sum_agg.get('avg_exposure', 100)
    
    # Calculate trades per month
    def_trades_per_month = def_trades / (years_back * 12) if years_back > 0 else 0
    agg_trades_per_month = agg_trades / (years_back * 12) if years_back > 0 else 0
    
    # Determine if trade frequency is high (warning indicator)
    def_high_trades = def_trades_per_month > 2
    agg_high_trades = agg_trades_per_month > 2
    
    # Build styled HTML table
    def_trades_bg = "background: rgba(255, 100, 100, 0.2);" if def_high_trades else ""
    agg_trades_bg = "background: rgba(255, 100, 100, 0.2);" if agg_high_trades else ""
    
    table_html = f"""
    <style>
        .perf-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        .perf-table th {{
            background: rgba(102, 126, 234, 0.3);
            padding: 12px 10px;
            text-align: center;
            font-weight: 700;
            border-bottom: 2px solid #667eea;
        }}
        .perf-table td {{
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid #444;
        }}
        .perf-table tr:hover {{
            background: rgba(102, 126, 234, 0.05);
        }}
        .row-highlight {{
            background: rgba(102, 126, 234, 0.15);
            font-weight: 600;
        }}
        .row-bold {{
            font-weight: 600;
        }}
        .metric-label {{
            text-align: left;
            padding-left: 15px;
        }}
    </style>
    <table class="perf-table">
        <thead>
            <tr>
                <th class="metric-label">Metric</th>
                <th>Buy & Hold</th>
                <th>Defensive</th>
                <th>Aggressive</th>
            </tr>
        </thead>
        <tbody>
            <tr class="row-highlight">
                <td class="metric-label"><b>Final Value</b></td>
                <td><b>€{bh_final:,.0f}</b></td>
                <td><b>€{def_final:,.0f}</b></td>
                <td><b>€{agg_final:,.0f}</b></td>
            </tr>
            <tr>
                <td class="metric-label">Total Return</td>
                <td>{bh_return:+.1f}%</td>
                <td>{def_return:+.1f}%</td>
                <td>{agg_return:+.1f}%</td>
            </tr>
            <tr>
                <td class="metric-label">Max Drawdown</td>
                <td>{bh_dd:.1f}%</td>
                <td>{def_dd:.1f}%</td>
                <td>{agg_dd:.1f}%</td>
            </tr>
            <tr>
                <td class="metric-label">Avg. Exposure</td>
                <td>100%</td>
                <td>{def_exposure:.0f}%</td>
                <td>{agg_exposure:.0f}%</td>
            </tr>
            <tr>
                <td class="metric-label">Total Trades</td>
                <td>0</td>
                <td style="{def_trades_bg}">{def_trades}</td>
                <td style="{agg_trades_bg}">{agg_trades}</td>
            </tr>
            <tr>
                <td class="metric-label">Trades/Month (Ø)</td>
                <td>0</td>
                <td style="{def_trades_bg}">{def_trades_per_month:.1f}</td>
                <td style="{agg_trades_bg}">{agg_trades_per_month:.1f}</td>
            </tr>
            <tr class="row-highlight">
                <td class="metric-label"><b>vs. Buy & Hold</b></td>
                <td><b>—</b></td>
                <td><b>{def_return - bh_return:+.1f}%</b></td>
                <td><b>{agg_return - bh_return:+.1f}%</b></td>
            </tr>
        </tbody>
    </table>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Key insight callout
    best_return = max(bh_return, def_return, agg_return)
    best_dd = max(bh_dd, def_dd, agg_dd)  # Less negative is better
    
    if def_return == best_return:
        winner_return = "Defensive"
    elif agg_return == best_return:
        winner_return = "Aggressive"
    else:
        winner_return = "Buy & Hold"
    
    if def_dd == best_dd:
        winner_dd = "Defensive"
    elif agg_dd == best_dd:
        winner_dd = "Aggressive"
    else:
        winner_dd = "Buy & Hold"
    
    col_insight1, col_insight2 = st.columns(2)
    with col_insight1:
        st.success(f"**Best Return:** {winner_return} ({best_return:+.1f}%)")
    with col_insight2:
        st.info(f"**Lowest Drawdown:** {winner_dd} ({best_dd:.1f}%)")
    
    # === EQUITY CURVES (3 lines) ===
    st.markdown("#### Equity Curves Comparison")
    
    equity_def = results_def.get('equity_curve', pd.DataFrame())
    equity_agg = results_agg.get('equity_curve', pd.DataFrame())
    
    if not equity_def.empty:
        fig = go.Figure()
        
        # Determine legend text color based on theme
        legend_color = '#333333' if not is_dark else '#FFFFFF'
        
        # Buy & Hold (grey)
        fig.add_trace(go.Scatter(
            x=equity_def['date'],
            y=equity_def['buyhold_value'],
            name='Buy & Hold',
            line=dict(color='#888888', width=2),
            mode='lines'
        ))
        
        # Defensive SOC (blue)
        fig.add_trace(go.Scatter(
            x=equity_def['date'],
            y=equity_def['soc_value'],
            name='Defensive',
            line=dict(color='#667eea', width=2),
            mode='lines'
        ))
        
        # Aggressive SOC (orange)
        if not equity_agg.empty:
            fig.add_trace(go.Scatter(
                x=equity_agg['date'],
                y=equity_agg['soc_value'],
                name='Aggressive',
                line=dict(color='#FF6600', width=2),
                mode='lines'
            ))
        
        # Initial capital line (dashed)
        fig.add_trace(go.Scatter(
            x=equity_def['date'],
            y=[initial_capital] * len(equity_def),
            name='Initial Capital',
            line=dict(color='#888888', width=1, dash='dash'),
            mode='lines'
        ))
        
        # Theme-aware colors
        axis_color = '#333333' if not is_dark else '#CCCCCC'
        grid_color = '#E0E0E0' if not is_dark else '#333333'
        
        fig.update_layout(
            template="plotly_white" if not is_dark else "plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,0)',
            plot_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(248,248,248,1)',
            height=400,
            margin=dict(t=20, b=50, l=60, r=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color=legend_color, size=13)
            ),
            hovermode='x unified'
        )
        fig.update_xaxes(
            title_text="Date",
            title_font=dict(color=axis_color, size=12),
            tickfont=dict(color=axis_color, size=11),
            gridcolor=grid_color
        )
        fig.update_yaxes(
            title_text="Portfolio Value (€)",
            title_font=dict(color=axis_color, size=12),
            tickfont=dict(color=axis_color, size=11),
            gridcolor=grid_color
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # === DRAWDOWN COMPARISON (3 areas) ===
    daily_def = results_def.get('daily_data', pd.DataFrame())
    daily_agg = results_agg.get('daily_data', pd.DataFrame())
    
    if not daily_def.empty and 'buyhold_drawdown' in daily_def.columns:
        st.markdown("#### Drawdown Comparison")
        
        # Legend color based on theme
        legend_color_dd = '#333333' if not is_dark else '#FFFFFF'
        
        fig_dd = go.Figure()
        
        # Buy & Hold (red)
        fig_dd.add_trace(go.Scatter(
            x=daily_def.index,
            y=daily_def['buyhold_drawdown'],
            name='Buy & Hold',
            fill='tozeroy',
            line=dict(color='rgba(255,100,100,0.8)', width=1),
            fillcolor='rgba(255,100,100,0.3)'
        ))
        
        # Defensive (blue)
        fig_dd.add_trace(go.Scatter(
            x=daily_def.index,
            y=daily_def['soc_drawdown'],
            name='Defensive',
            fill='tozeroy',
            line=dict(color='rgba(102,126,234,0.8)', width=1),
            fillcolor='rgba(102,126,234,0.3)'
        ))
        
        # Aggressive (orange)
        if not daily_agg.empty and 'soc_drawdown' in daily_agg.columns:
            fig_dd.add_trace(go.Scatter(
                x=daily_agg.index,
                y=daily_agg['soc_drawdown'],
                name='Aggressive',
                fill='tozeroy',
                line=dict(color='rgba(255,140,0,0.8)', width=1),
                fillcolor='rgba(255,140,0,0.3)'
            ))
        
        # Theme-aware colors for drawdown
        axis_color_dd = '#333333' if not is_dark else '#CCCCCC'
        grid_color_dd = '#E0E0E0' if not is_dark else '#333333'
        
        fig_dd.update_layout(
            template="plotly_white" if not is_dark else "plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,0)',
            plot_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(248,248,248,1)',
            height=280,
            margin=dict(t=20, b=50, l=60, r=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color=legend_color_dd, size=13)
            ),
            hovermode='x unified'
        )
        fig_dd.update_xaxes(
            title_text="Date",
            title_font=dict(color=axis_color_dd, size=12),
            tickfont=dict(color=axis_color_dd, size=11),
            gridcolor=grid_color_dd
        )
        fig_dd.update_yaxes(
            title_text="Drawdown (%)",
            title_font=dict(color=axis_color_dd, size=12),
            tickfont=dict(color=axis_color_dd, size=11),
            gridcolor=grid_color_dd
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)


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
        st.session_state.dark_mode = False  # Light mode default
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
    
    # Market selection - centered title and subtitle
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem;">
        <h3 style="margin-bottom: 0.3rem;">Asset Search</h3>
        <p style="color: #888; font-size: 0.95rem; margin-bottom: 1.5rem;">Search any stock, ETF, or crypto by ticker symbol</p>
    </div>
    """, unsafe_allow_html=True)
    
    tickers = render_ticker_search()
    
    # Analyze button (only show if we have valid tickers)
    if tickers:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("Run SOC Analysis", type="primary", use_container_width=True):
                st.session_state.scan_results = run_analysis(tickers)
                st.session_state.selected_asset = 0
    
    # Results - Redesigned Layout
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        results = st.session_state.scan_results
        
        # Initialize analysis mode in session state
        if 'analysis_mode' not in st.session_state:
            st.session_state.analysis_mode = "deep_dive"
        
        st.divider()
        
        # Centered "Analysis of Results" title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3>Analysis of Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # === ASSET SELECTION GRID (max 3 columns) ===
        st.markdown("""
        <div style="text-align: center; margin-bottom: 0.5rem;">
            <span style="color: #888; font-size: 0.9rem;">Select an asset to analyze</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate number of columns (max 3)
        num_assets = len(results)
        num_cols = min(3, num_assets)
        
        # Create asset button grid
        cols = st.columns(num_cols)
        for i, r in enumerate(results):
            col_idx = i % num_cols
            with cols[col_idx]:
                is_selected = i == st.session_state.selected_asset
                # Compact label with emoji for regime
                regime_emoji = r['signal'].split()[0] if r['signal'] else "⚪"
                btn_label = f"{regime_emoji} {r['symbol']}\n${r['price']:,.0f}"
                
                if st.button(
                    btn_label, 
                    key=f"asset_{i}", 
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state.selected_asset = i
                    st.rerun()
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # === ANALYSIS MODE SELECTION (centered) ===
        st.markdown("""
        <div style="text-align: center; margin-bottom: 0.5rem;">
            <span style="color: #888; font-size: 0.9rem;">Choose analysis type</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered analysis option buttons
        col_spacer1, col_opt1, col_opt2, col_spacer2 = st.columns([1, 2, 2, 1])
        
        with col_opt1:
            if st.button(
                "Asset Deep Dive",
                key="btn_deep_dive",
                use_container_width=True,
                type="primary" if st.session_state.analysis_mode == "deep_dive" else "secondary"
            ):
                st.session_state.analysis_mode = "deep_dive"
                st.rerun()
        
        with col_opt2:
            if st.button(
                "Portfolio Simulation",
                key="btn_simulation",
                use_container_width=True,
                type="primary" if st.session_state.analysis_mode == "simulation" else "secondary"
            ):
                st.session_state.analysis_mode = "simulation"
                st.rerun()
        
        st.divider()
        
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
