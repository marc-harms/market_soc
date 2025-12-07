"""
SOC Market Seismograph - Streamlit Application
==============================================

Interactive web dashboard for Self-Organized Criticality (SOC) market analysis.

Features:
    - Multi-asset scanning with 5-Tier regime classification
    - Deep dive analysis with historical signal performance
    - Instability Score indicator (0-100)
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
from ui_detail import render_detail_panel, render_regime_persistence_chart, render_current_regime_outlook
from ui_auth import render_disclaimer, render_auth_page, render_sticky_cockpit_header, render_education_landing
from auth_manager import (
    is_authenticated, logout, get_current_user_id, get_current_user_email,
    get_user_portfolio, can_access_simulation, show_upgrade_prompt,
    can_run_simulation, increment_simulation_count
)
from config import get_scientific_heritage_css, HERITAGE_THEME, REGIME_COLORS

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TECTONIQ - Market Analysis Platform",
    page_icon="assets/logo-soc.png",
    layout="wide",
    initial_sidebar_state="collapsed"  # No sidebar used - user menu in header
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
# NOTE: Detail panel UI moved to ui_detail.py
# Import: from ui_detail import render_detail_panel, render_regime_persistence_chart, render_current_regime_outlook



# =============================================================================
# INVESTMENT SIMULATION UI
# =============================================================================

# =============================================================================
# AUTHENTICATION & NAVIGATION UI
# =============================================================================
# NOTE: Auth & navigation UI moved to ui_auth.py
# Import: from ui_auth import render_disclaimer, check_auth, login_page, render_sticky_cockpit_header, render_education_landing


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application entry point - Multi-User SaaS Edition.
    
    Flow:
        1. Show legal disclaimer (must accept to continue)
        2. Check user authentication (Supabase)
        3. Initialize session state
        4. Apply theme CSS
        5. Render sidebar with logout + user info
        6. Render sticky cockpit header (always visible)
        7. Main content area:
           - No asset: Education landing + quick picks
           - Asset selected: Deep Dive or Simulation tabs (tier-gated)
    """
    # Session state initialization
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state.disclaimer_accepted = False
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
    
    # === AUTHENTICATION GATE (THE GATEKEEPER) ===
    # Check if user is authenticated via Supabase
    if not is_authenticated():
        render_auth_page()
        return
    
    # Apply Scientific Heritage CSS theme
    st.markdown(get_scientific_heritage_css(), unsafe_allow_html=True)
    
    # === STICKY COCKPIT HEADER (with User Menu) ===
    render_sticky_cockpit_header(validate_ticker, search_ticker, run_analysis)
    
    # === PORTFOLIO VIEW (if toggled on) ===
    if st.session_state.get('show_portfolio', False):
        with st.container(border=True):
            col_header, col_close = st.columns([4, 1])
            with col_header:
                st.markdown("### 📁 My Portfolio")
            with col_close:
                if st.button("Close portfolio window", key="close_portfolio", use_container_width=True):
                    st.session_state.show_portfolio = False
                    st.rerun()
            
            user_id = get_current_user_id()
            user_tier = st.session_state.get('tier', 'free')
            
            if user_id:
                portfolio = get_user_portfolio(user_id)
                if portfolio:
                    st.caption(f"**{len(portfolio)}** assets tracked")
                    st.markdown("---")
                    
                    # Fetch full analysis for all portfolio assets (includes crash_warning)
                    with st.spinner("Loading portfolio data..."):
                        import time
                        portfolio_analysis = []
                        fetcher = DataFetcher(cache_enabled=True)
                        failed_tickers = []
                        
                        for i, ticker in enumerate(portfolio):
                            try:
                                # Add small delay between requests to avoid rate limiting
                                if i > 0:
                                    time.sleep(0.5)  # 500ms delay between tickers
                                
                                df = fetcher.fetch_data(ticker)
                                info = fetcher.fetch_info(ticker)
                                if not df.empty and len(df) > MIN_DATA_POINTS:
                                    analyzer = SOCAnalyzer(df, ticker, info, DEFAULT_SMA_WINDOW, DEFAULT_VOL_WINDOW, DEFAULT_HYSTERESIS)
                                    phase = analyzer.get_market_phase()
                                    
                                    # Get full analysis including crash_warning for accurate stress level
                                    try:
                                        full_analysis = analyzer.get_full_analysis()
                                        crash_warning = full_analysis.get('crash_warning', {})
                                        
                                        # Ensure crash_warning has a score
                                        if crash_warning and 'score' in crash_warning:
                                            phase['crash_warning'] = crash_warning
                                        else:
                                            print(f"Warning: {ticker} crash_warning missing score, recalculating...")
                                            # Force recalculation if missing
                                            phase['crash_warning'] = full_analysis.get('crash_warning', {'score': 0})
                                    except Exception as analysis_error:
                                        # Fallback: calculate basic stress if full analysis completely fails
                                        print(f"Full analysis failed for {ticker}: {str(analysis_error)}")
                                        phase['crash_warning'] = {'score': 0}
                                    
                                    phase['name'] = clean_name(info.get('name', ticker))
                                    portfolio_analysis.append(phase)
                                else:
                                    failed_tickers.append(ticker)
                            except Exception as e:
                                failed_tickers.append(ticker)
                                print(f"Error loading {ticker}: {str(e)}")  # Debug log
                        
                        # Show warning/error based on results
                        if failed_tickers and portfolio_analysis:
                            # Some tickers failed, but others loaded successfully
                            st.warning(f"⚠️ Could not load data for: {', '.join(failed_tickers)}")
                        elif failed_tickers and not portfolio_analysis:
                            # All tickers failed
                            st.error(f"""
                            ❌ **Could not load portfolio data**
                            
                            Failed to fetch: {', '.join(failed_tickers)}
                            
                            **Possible reasons:**
                            - Yahoo Finance API rate limiting (try again in 1-2 minutes)
                            - Network connectivity issues
                            - Invalid ticker symbols
                            
                            💡 **Tip:** Try removing and re-adding the assets, or search for them individually first.
                            """)
                    
                    if portfolio_analysis:
                        # Create table data
                        table_data = []
                        for result in portfolio_analysis:
                            table_data.append({
                                "Ticker": result['symbol'],
                                "Asset Name": result.get('name', result['symbol']),
                                "Criticality": int(result.get('criticality_score', 0)),
                                "Regime": result.get('signal', 'Unknown'),
                                "_result": result  # Store full result for actions
                            })
                        
                        # Sort by criticality (highest first)
                        table_data.sort(key=lambda x: x['Criticality'], reverse=True)
                        
                        # Display table
                        for i, row in enumerate(table_data):
                            # Color code criticality
                            crit = row['Criticality']
                            if crit > 80:
                                crit_color = "#FF4040"
                            elif crit > 60:
                                crit_color = "#FF6600"
                            else:
                                crit_color = "#00C864"
                            
                            # Get regime emoji
                            regime_text = row['Regime']
                            regime_emoji = regime_text.split()[0] if regime_text else "⚪"
                            
                            # Row container
                            with st.container():
                                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
                                
                                with col1:
                                    st.markdown(f"**{row['Ticker']}**")
                                
                                with col2:
                                    st.markdown(f"{row['Asset Name']}")
                                
                                with col3:
                                    st.markdown(f"{regime_emoji} <span style='color: {crit_color}; font-weight: 600;'>Criticality: {crit}</span>", unsafe_allow_html=True)
                                
                                with col4:
                                    if st.button("→ Deep Dive", key=f"deepdive_{row['Ticker']}", use_container_width=True, type="primary"):
                                        # Load this asset
                                        st.session_state.current_ticker = row['Ticker']
                                        st.session_state.scan_results = [row['_result']]
                                        st.session_state.selected_asset = 0
                                        st.session_state.analysis_mode = "deep_dive"
                                        st.session_state.show_portfolio = False  # Close portfolio
                                        st.rerun()
                                
                                with col5:
                                    if st.button("🗑️", key=f"remove_{row['Ticker']}", help="Remove from portfolio", use_container_width=True):
                                        from auth_manager import remove_asset_from_portfolio
                                        success, error = remove_asset_from_portfolio(user_id, row['Ticker'])
                                        if success:
                                            st.rerun()
                                        else:
                                            st.error(error)
                                
                                st.markdown("<hr style='margin: 8px 0; opacity: 0.2;'>", unsafe_allow_html=True)
                    # If portfolio_analysis is empty, error message was already shown above
                else:
                    st.info("📌 No assets yet. Search for a ticker and click '⭐ Add to Portfolio'")
        
        st.markdown("---")
    
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
        render_education_landing(run_analysis)
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
            # Simulation is now free for all (with daily limits)
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
                render_detail_panel(selected, get_signal_color, get_signal_bg)
        else:
            # Portfolio Simulation (unlimited for all users)
            st.markdown("### 💰 DCA Simulation")
            st.markdown("---")
            
            result_tickers = [r['symbol'] for r in results]
            render_dca_simulation(result_tickers)
    
    # === FOOTER WITH LEGAL LINKS ===
    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; font-family: 'Merriweather', serif; font-size: 0.85rem; color: #666;">
            <p style="margin: 0 0 8px 0;">© 2025 TECTONIQ. All rights reserved.</p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <a href="?page=disclaimer" style="color: #2C3E50; text-decoration: none; font-weight: 600;">Disclaimer</a>
                <span style="color: #BDC3C7;">|</span>
                <a href="?page=data-protection" style="color: #2C3E50; text-decoration: none; font-weight: 600;">Data Protection</a>
                <span style="color: #BDC3C7;">|</span>
                <a href="?page=imprint" style="color: #2C3E50; text-decoration: none; font-weight: 600;">Imprint</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle legal page clicks via query params
        query_params = st.query_params
        if 'page' in query_params:
            page = query_params['page']
            if page == 'disclaimer':
                render_legal_page_disclaimer()
            elif page == 'data-protection':
                render_legal_page_data_protection()
            elif page == 'imprint':
                render_legal_page_imprint()


def render_legal_page_disclaimer():
    """Render disclaimer legal page."""
    st.markdown("### ⚖️ Legal Disclaimer")
    st.markdown("""
    This application is provided for educational and informational purposes only.
    Nothing on this platform constitutes financial, investment, or trading advice.
    
    **No Investment Recommendations:**
    - We do not recommend buying, selling, or holding any financial instruments
    - All analysis is purely statistical observation
    - Past performance is not indicative of future results
    
    **Limitation of Liability:**
    - The creators shall not be liable for any damages arising from use
    - Users accept full responsibility for their investment decisions
    
    **Independent Verification Required:**
    - Consult with qualified financial advisors before making decisions
    - Conduct your own research and due diligence
    """)
    if st.button("← Back to App", key="back_disclaimer"):
        st.query_params.clear()
        st.rerun()


def render_legal_page_data_protection():
    """Render data protection legal page."""
    st.markdown("### 🔒 Data Protection Policy")
    st.markdown("""
    **Data Controller:** TECTONIQ Platform
    
    **Data We Collect:**
    - Email address (for authentication)
    - Portfolio preferences (ticker symbols you save)
    - Usage analytics (anonymous)
    
    **How We Use Your Data:**
    - To provide authentication services
    - To save your portfolio preferences
    - To improve the application
    
    **Data Storage:**
    - Stored securely via Supabase (EU servers)
    - Encrypted in transit and at rest
    - Not shared with third parties
    
    **Your Rights:**
    - Right to access your data
    - Right to delete your account
    - Right to data portability
    
    **Contact:** For data protection inquiries, email privacy@tectoniq.app
    """)
    if st.button("← Back to App", key="back_data"):
        st.query_params.clear()
        st.rerun()


def render_legal_page_imprint():
    """Render imprint legal page."""
    st.markdown("### 📄 Imprint / Legal Notice")
    st.markdown("""
    **Service Provider:**  
    TECTONIQ Platform  
    [Your Address]  
    [City, Postal Code]  
    [Country]
    
    **Contact:**  
    Email: info@tectoniq.app  
    Web: tectoniq.app
    
    **Responsible for Content:**  
    [Your Name / Company Name]
    
    **Disclaimer:**  
    This platform provides educational content only. We assume no liability for the 
    accuracy, completeness, or timeliness of the information provided.
    
    **Copyright:**  
    © 2025 TECTONIQ. All rights reserved. Unauthorized reproduction or distribution 
    of this application or its content is prohibited.
    
    **Third-Party Data:**  
    Market data provided by Yahoo Finance. We do not control or guarantee the 
    accuracy of third-party data sources.
    """)
    if st.button("← Back to App", key="back_imprint"):
        st.query_params.clear()
        st.rerun()


if __name__ == "__main__":
    main()
