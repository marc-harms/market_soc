"""
SOC Market Seismograph - Streamlit Application
==============================================
A market analysis dashboard based on Self-Organized Criticality (SOC) theory.
Master-Detail layout for seamless asset analysis workflow.

Author: Market Analysis Team
Version: 3.0 (Beta)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import time
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import yfinance as yf

from logic import DataFetcher, SOCAnalyzer

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

MARKET_SETS = {
    "US Big Tech": ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'AMD', 'NFLX'],
    "DAX Top 10": ['^GDAXI', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'VOW3.DE', 'BAS.DE', 'MUV2.DE'],
    "Crypto Assets": ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD'],
    "Precious Metals": ['GC=F', 'SI=F', 'PL=F', 'PA=F', 'GLD', 'SLV']
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
    """Generate theme-aware CSS."""
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
    /* Global Theme */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {c['bg']} !important;
    }}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stMarkdown, .stMarkdown p {{
        color: {c['text']} !important;
    }}
    
    /* Inputs & Controls */
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"], [data-baseweb="select"] > div {{
        background-color: {c['input']} !important;
        color: {c['text']} !important;
        border-color: {c['border']} !important;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{ background-color: {c['card']} !important; }}
    .streamlit-expanderContent {{ background-color: {c['bg2']} !important; }}
    
    /* Tables */
    .stDataFrame, [data-testid="stDataFrame"], .stDataFrame div, .stDataFrame table,
    .stDataFrame th, .stDataFrame td, [data-testid="glideDataEditor"], .dvn-scroller {{
        background-color: {c['card']} !important;
        color: {c['text']} !important;
    }}
    .stDataFrame th {{ background-color: {c['bg2']} !important; }}
    
    /* Buttons */
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
    
    /* Radio */
    .stRadio label {{ color: {c['text']} !important; }}
    .stRadio [role="radiogroup"] label {{ background-color: {c['card']} !important; border-color: {c['border']} !important; }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{ color: {c['text']} !important; }}
    [data-testid="stMetricLabel"] {{ color: {c['muted']} !important; }}
    
    /* Layout */
    [data-testid="stSidebar"] {{ display: none; }}
    .stDeployButton {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; max-width: 1400px; margin: 0 auto; }}
    hr {{ border-color: {c['border']} !important; }}
    
    /* Custom Components */
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
    
    /* Asset List Item */
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
    
    /* Detail Panel */
    .detail-header {{ padding: 1rem; background: {c['card']}; border-radius: 8px; margin-bottom: 1rem; }}
    .signal-badge {{
        display: inline-block; padding: 0.4rem 0.8rem; border-radius: 6px;
        font-weight: 600; font-size: 0.9rem;
    }}
    
    /* Footer */
    .footer {{ border-top: 1px solid {c['border']}; padding-top: 0.75rem; margin-top: 1.5rem; }}
    .footer-item {{ display: inline-block; margin-right: 1.5rem; font-size: 0.8rem; color: {c['muted']}; }}
</style>
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def clean_name(name: str) -> str:
    """Clean ticker names from Yahoo Finance."""
    name = name.replace(" SE", "").replace(" AG", "").strip()
    if name in TICKER_NAME_FIXES:
        return TICKER_NAME_FIXES[name]
    name = " ".join(name.split())
    return name[:-2] if len(name) > 2 and name[-2] == " " else name


def get_signal_color(signal: str) -> str:
    """Get color for SOC signal."""
    if "ACCUMULATE" in signal: return "#00FF00"
    if "CRASH" in signal: return "#FF0000"
    if "OVERHEATED" in signal: return "#FFA500"
    return "#888888"


def get_signal_bg(signal: str) -> str:
    """Get background color for signal badge."""
    if "ACCUMULATE" in signal: return "rgba(0, 255, 0, 0.15)"
    if "CRASH" in signal: return "rgba(255, 0, 0, 0.15)"
    if "OVERHEATED" in signal: return "rgba(255, 165, 0, 0.15)"
    return "rgba(136, 136, 136, 0.15)"


# =============================================================================
# DATA FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_footer_data() -> List[Dict[str, Any]]:
    """Fetch footer market indicators."""
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
    """Run SOC analysis on tickers."""
    fetcher = DataFetcher(cache_enabled=True)
    results = []
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
        except Exception:
            pass
        progress.progress((i + 1) / len(tickers))
    
    status.empty()
    progress.empty()
    return results


# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_header():
    """Render app header with logo and theme toggle."""
    is_dark = st.session_state.get('dark_mode', True)
    
    col_logo, col_title, col_theme = st.columns([1, 6, 1])
    
    with col_logo:
        # Logo image
        try:
            st.image("assets/logo-soc.png", width=80)
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
    """Render theory expander."""
    with st.expander("📖 How this theory works"):
        st.markdown("""
        **Self-Organized Criticality (SOC)** - Markets are complex adaptive systems, not efficient.
        
        | Signal | Condition | Meaning |
        |--------|-----------|---------|
        | 🟢 ACCUMULATE | Low Vol + Uptrend | Safe growth |
        | 🔴 CRASH RISK | High Vol + Downtrend | Danger zone |
        | 🟠 OVERHEATED | High Vol + Uptrend | Correction risk |
        | ⚪ NEUTRAL | Mixed | Range-bound |
        """)


def render_market_selection() -> List[str]:
    """Render market selection and return tickers."""
    universe = st.radio("Asset Universe:", list(MARKET_SETS.keys()) + ["Custom"], horizontal=True)
    
    if universe == "Custom":
        raw = st.text_input("Tickers (comma-separated):", "NVDA, BTC-USD, GLD")
        return [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]
    return MARKET_SETS[universe]


def render_asset_list(results: List[Dict[str, Any]], selected_idx: int) -> int:
    """Render clickable asset list and return selected index."""
    for i, r in enumerate(results):
        is_selected = i == selected_idx
        color = get_signal_color(r['signal'])
        
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f"**{r['symbol']}**")
        with col2:
            st.markdown(f"${r['price']:,.2f}")
        with col3:
            st.markdown(f"<span style='color:{color}'>{r['signal']}</span>", unsafe_allow_html=True)
        
        if st.button("Select", key=f"select_{i}", use_container_width=True, 
                     type="primary" if is_selected else "secondary"):
            return i
    
    return selected_idx


def render_detail_panel(result: Dict[str, Any]):
    """Render detail panel for selected asset."""
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
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${result['price']:,.2f}")
    col2.metric("Stress Score", f"{result['stress_score']:.2f}")
    col3.metric("Trend", result['trend'])
    
    # Chart
    fetcher = DataFetcher(cache_enabled=True)
    df = fetcher.fetch_data(symbol)
    
    if not df.empty:
        analyzer = SOCAnalyzer(df, symbol, result.get('info'))
        figs = analyzer.get_plotly_figures(dark_mode=is_dark)
        st.plotly_chart(figs['chart3'], use_container_width=True)


def render_footer():
    """Render footer with market pulse."""
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
# AUTHENTICATION
# =============================================================================
def check_auth():
    if st.session_state.get("pwd") == ACCESS_CODE:
        st.session_state.authenticated = True
        del st.session_state.pwd
    else:
        st.error("Incorrect password")


def login_page():
    st.title("🔒 Login Required")
    st.text_input("Access Code", type="password", key="pwd", on_change=check_auth)
    st.stop()


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Session state initialization
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    if 'selected_asset' not in st.session_state:
        st.session_state.selected_asset = 0
    
    # Auth gate
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Apply theme
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Header
    render_header()
    render_theory()
    
    # Market selection
    st.markdown("### Market Selection")
    tickers = render_market_selection()
    
    # Run button
    col1, col2 = st.columns([5, 1])
    with col1:
        if st.button("RUN SOC ANALYSIS", type="primary", use_container_width=True):
            st.session_state.scan_results = run_analysis(tickers)
            st.session_state.selected_asset = 0
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop('scan_results', None)
            st.rerun()
    
    # Results - Master-Detail Layout
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        results = st.session_state.scan_results
        
        st.divider()
        st.markdown("### Analysis Results")
        
        # Master-Detail Layout
        col_list, col_detail = st.columns([1, 2])
        
        with col_list:
            st.markdown("**Select an asset:**")
            
            # Asset list as selectable items
            for i, r in enumerate(results):
                is_selected = i == st.session_state.selected_asset
                
                # Create a button for each asset
                btn_label = f"{r['symbol']} | ${r['price']:,.0f} | {r['signal'].split()[0]}"
                if st.button(
                    btn_label,
                    key=f"asset_{i}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state.selected_asset = i
                    st.rerun()
        
        with col_detail:
            if results:
                selected = results[st.session_state.selected_asset]
                render_detail_panel(selected)
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()
