import streamlit as st
import pandas as pd
import time
from logic import DataFetcher, SOCAnalyzer

# 1. Page Config
st.set_page_config(
    page_title="SOC Market Seismograph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS / STYLING ---
st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #262730;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00ff00, #ffff00, #ff0000);
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & DATA ---

KPI_UNIVERSE = {
    "Gold": {"symbol": "GC=F", "name": "Gold"},
    "S&P 500": {"symbol": "^GSPC", "name": "S&P 500"},
    "Bitcoin": {"symbol": "BTC-USD", "name": "Bitcoin"},
    "Dow Jones": {"symbol": "^DJI", "name": "Dow Jones"},
    "NASDAQ 100": {"symbol": "^NDX", "name": "NASDAQ 100"},
    "MSCI China": {"symbol": "MCHI", "name": "MSCI China (ETF)"},
    "TSX Composite": {"symbol": "^GSPTSE", "name": "TSX Composite"},
    "FTSE 100": {"symbol": "^FTSE", "name": "FTSE 100"},
    "CAC 40": {"symbol": "^FCHI", "name": "CAC 40"},
    "Nikkei 225": {"symbol": "^N225", "name": "Nikkei 225 (Japan)"},
    "MSCI Emerging": {"symbol": "EEM", "name": "MSCI Emerging (ETF)"},
    "MSCI World": {"symbol": "URTH", "name": "MSCI World (ETF)"}
}

MARKET_SETS = {
    "🇺🇸 US BigTech": ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'AMD', 'NFLX'],
    "🇩🇪 DAX Top 10": ['^GDAXI', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'VOW3.DE', 'BAS.DE', 'MUV2.DE'],
    "₿ Crypto": ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD'],
    "🥇 Precious Metals": ['GC=F', 'SI=F', 'PL=F', 'PA=F', 'GLD', 'SLV']
}

# --- HELPER FUNCTIONS ---

if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Scanner"
if 'kpi_selection' not in st.session_state:
    # Default selection
    st.session_state.kpi_selection = ["Bitcoin", "S&P 500", "Gold"]

@st.cache_data(ttl=3600)
def fetch_market_snapshot(selected_keys):
    """Fetches key assets for the top dashboard."""
    fetcher = DataFetcher(cache_enabled=True)
    snapshot = []
    
    # Build list of asset dicts from keys
    target_assets = []
    for k in selected_keys:
        if k in KPI_UNIVERSE:
            target_assets.append(KPI_UNIVERSE[k])
            
    for item in target_assets:
        sym = item["symbol"]
        try:
            df = fetcher.fetch_data(sym)
            if not df.empty:
                analyzer = SOCAnalyzer(df, sym)
                phase = analyzer.get_market_phase()
                snapshot.append({
                    "name": item["name"],
                    "symbol": sym,
                    "price": phase["price"],
                    "change": df["close"].pct_change().iloc[-1] * 100, # Approx last daily change
                    "signal": phase["signal"]
                })
        except Exception as e:
            pass
            
    return snapshot

@st.cache_data(ttl=3600)
def run_full_scan(ticker_list, sma_w, vol_w, hyst):
    fetcher = DataFetcher(cache_enabled=True)
    results = []
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(ticker_list)
    for i, symbol in enumerate(ticker_list):
        symbol = symbol.strip()
        if not symbol: continue
        
        status_text.caption(f"Analyzing {symbol}...")
        try:
            df = fetcher.fetch_data(symbol)
            info = fetcher.fetch_info(symbol)
            
            if not df.empty and len(df) > 200:
                analyzer = SOCAnalyzer(
                    df, symbol, asset_info=info,
                    sma_window=sma_w, vol_window=vol_w, hysteresis=hyst
                )
                phase = analyzer.get_market_phase()
                phase['info'] = info
                results.append(phase)
        except Exception as e:
            print(f"Error {symbol}: {e}")
            
        progress_bar.progress((i + 1) / total)
        
    status_text.empty()
    progress_bar.empty()
    return results

# --- SIDEBAR NAVIGATION ---

with st.sidebar:
    st.logo("https://img.icons8.com/color/48/bullish.png")
    st.title("📡 SOC Scanner")
    st.markdown("---")
    
    # 2. Smart Sidebar
    st.subheader("Market Universe")
    
    # Build options dynamically
    radio_options = list(MARKET_SETS.keys())
    
    # Custom Portfolio option
    radio_options.append("✏️ Custom Portfolio")
    
    universe_mode = st.radio(
        "Select Asset Class",
        radio_options,
        horizontal=False
    )
    
    target_tickers = []
    
    if universe_mode == "✏️ Custom Portfolio":
        input_tickers = st.text_area(
            "Paste tickers (comma or new line)",
            value="BTC-USD, NVDA, TSLA, SAP.DE",
            height=150,
            help="Paste from Excel or list. One ticker per line or comma separated."
        )
        # Smart cleaning: Replace newlines with commas, split, strip
        cleaned_input = input_tickers.replace("\n", ",").replace(";", ",")
        target_tickers = [t.strip().upper() for t in cleaned_input.split(',') if t.strip()]
    else:
        target_tickers = MARKET_SETS[universe_mode]
        st.info(f"Loaded {len(target_tickers)} assets.")
        with st.expander("View Assets"):
            st.code(", ".join(target_tickers))

    st.markdown("### Actions")
    
    # Simple button logic (reverted from rate limiting)
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("v3.0 | Powered by SOC Logic")

    # 3. Quant Settings
    with st.expander("⚙️ Quant Settings", expanded=True):
        st.caption("Adjust model parameters to stress-test the thesis.")
        
        sma_window = st.slider(
            "Trend Window (SMA)", 
            min_value=50, max_value=365, value=200, step=10,
            help="Days to look back for the Simple Moving Average. Shorter = faster reaction, more false signals."
        )
        
        vol_window = st.slider(
            "Volatility Window", 
            min_value=10, max_value=90, value=30, step=5,
            help="Days to measure volatility. Lower = sensitive to recent shocks."
        )
        
        hysteresis = st.slider(
            "Whipsaw Filter (Hysteresis) %", 
            min_value=0.0, max_value=5.0, value=0.0, step=0.1,
            help="Requires price to break SMA by this % to trigger a signal. Reduces false alarms in sideways markets."
        ) / 100.0

# --- MAIN DASHBOARD ---

st.title("⚡ Market Seismograph")
st.markdown("**Self-Organized Criticality (SOC) Risk Analysis**")

# 1. KPI Dashboard (The "Cockpit")
with st.container():
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 🌍 Market Health Check")
    with c2:
        selected_kpis = st.multiselect(
            "Select Indicators (Max 6)",
            options=list(KPI_UNIVERSE.keys()),
            default=st.session_state.kpi_selection,
            max_selections=6,
            label_visibility="collapsed",
            key="kpi_multiselect"
        )
        # Update session state if changed
        st.session_state.kpi_selection = selected_kpis
        
    # Fetch snapshot immediately
    if selected_kpis:
        snapshot_data = fetch_market_snapshot(selected_kpis)
    else:
        snapshot_data = []
        st.info("Select indicators above to view market health.")
    
    # Grid Layout: 3 columns max per row
    cols_per_row = 3
    for i in range(0, len(snapshot_data), cols_per_row):
        row_items = snapshot_data[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        
        for idx, data in enumerate(row_items):
            with cols[idx]:
                is_risk = "CRASH" in data['signal'] or "OVERHEATED" in data['signal']
                label_display = f"{data['name']} | {data['signal']}"
                st.metric(
                    label=label_display,
                    value=f"${data['price']:,.2f}",
                    delta=f"{data['change']:.2f}%",
                    delta_color="inverse" if is_risk else "normal"
                )

st.divider()

# 5. Instructions
with st.expander("📖 How to read this dashboard"):
    st.markdown("""
    **The SOC Traffic Light System:**
    * 🔴 **CRASH RISK:** High Volatility + Downtrend. High probability of cascading sell-offs.
    * 🟠 **OVERHEATED:** High Volatility + Uptrend. Price is rising but instability is high. Risk of correction.
    * 🟢 **ACCUMULATION:** Low Volatility + Uptrend. Stable growth phase.
    * 🟡 **CAPITULATION:** Low Volatility + Downtrend. Market is quiet but bearish. Watch for reversals.
    * ⚪ **NEUTRAL / CHOPPY:** Market is in a sideways range or strictly following the SMA. Hysteresis active.
    
    **Stress Score:** Measures how close the asset is to a critical phase transition (values > 1.0 indicate high stress).
    """)

# --- SCANNER LOGIC ---

if run_btn:
    st.session_state.scan_results = run_full_scan(target_tickers, sma_window, vol_window, hysteresis)

results = st.session_state.get('scan_results', [])

# Tabs for View
# Use session state to force tab selection when "Analyze" is clicked
# Manual tab selection updates state
def on_tab_change():
    pass

tab_options = ["📊 Market Scanner", "🔍 Deep Dive Analysis", "📈 Portfolio Simulation", "📘 Model Theory & Risks"]

# Ensure valid state
if st.session_state.active_tab not in tab_options:
    st.session_state.active_tab = tab_options[0]

# Determine index
current_tab_index = tab_options.index(st.session_state.active_tab)

# Logic to handle "Switch to Deep Dive" trigger from elsewhere
# Must be handled BEFORE st.radio renders to ensure index matches
if st.session_state.get("switch_to_deep_dive", False):
    st.session_state.active_tab = "🔍 Deep Dive Analysis"
    st.session_state.nav_radio = "🔍 Deep Dive Analysis" # Sync widget state
    st.session_state.switch_to_deep_dive = False
    current_tab_index = tab_options.index(st.session_state.active_tab)

st.markdown("---")
selected_tab = st.radio(
    "Navigation", 
    tab_options, 
    index=current_tab_index, 
    horizontal=True, 
    label_visibility="collapsed",
    key="nav_radio"
)

# Update state if changed by user
if selected_tab != st.session_state.active_tab:
    st.session_state.active_tab = selected_tab
    st.rerun()

if selected_tab == "📊 Market Scanner":
    # Tab 1 Content
    if not results:
        st.info("👈 Select a universe and click 'Run Analysis' to see the matrix.")
    else:
        # Prepare Data
        df_res = pd.DataFrame(results)
        
        # Sort logic
        signal_order = {
            "🔴 CRASH RISK": 0, "🟠 OVERHEATED": 1,
            "🟢 BUY/ACCUMULATE": 2, "🟡 CAPITULATION/WAIT": 3,
            "⚪ UPTREND (Choppy)": 4, "⚪ DOWNTREND (Choppy)": 5,
            "NEUTRAL": 6, "NO_DATA": 7
        }
        df_res['sort_key'] = df_res['signal'].map(signal_order)
        df_res = df_res.sort_values('sort_key').drop(columns=['sort_key'])
        
        # 3. Enhanced Styling
        # Background coloring function
        def highlight_signal(row):
            val = row['signal']
            color = ''
            if "BUY" in val: color = 'background-color: rgba(0, 255, 0, 0.1)'
            elif "CRASH" in val: color = 'background-color: rgba(255, 0, 0, 0.1)'
            elif "OVERHEATED" in val: color = 'background-color: rgba(255, 165, 0, 0.1)'
            elif "NEUTRAL (Range)" in val: color = 'background-color: rgba(128, 128, 128, 0.1)'
            return [color] * len(row)

        # Columns to display
        display_df = df_res[["symbol", "price", "signal", "stress_score", "dist_to_sma", "trend"]]
        
        st.dataframe(
            display_df.style.apply(highlight_signal, axis=1),
            use_container_width=True,
            column_config={
                "symbol": "Asset",
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "signal": st.column_config.TextColumn("SOC Phase", help="Current Market Phase based on Volatility & Trend"),
                "stress_score": st.column_config.ProgressColumn(
                    "Stress Level",
                    help="Criticality Score (Vol / Threshold). >1.0 is High Stress.",
                    format="%.2f",
                    min_value=0,
                    max_value=2.0,
                ),
                "dist_to_sma": st.column_config.NumberColumn("SMA Deviance", format="%.2f%%"),
                "trend": st.column_config.TextColumn("Trend")
            },
            height=600
        )
        
        # Export
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scan Results (CSV)",
            data=csv,
            file_name=f"soc_market_scan_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

elif selected_tab == "🔍 Deep Dive Analysis":
    # Tab 2 Content (Deep Dive)
    if not results:
        st.warning("Please run a scan first to populate the deep dive list.")
    else:
        # Create a mapping for the selectbox
        symbol_list = [r['symbol'] for r in results]
        
        col_sel, _ = st.columns([1, 2])
        
        with col_sel:
            ix = 0
            if st.session_state.selected_asset in symbol_list:
                ix = symbol_list.index(st.session_state.selected_asset)
            
            def update_selected():
                st.session_state.selected_asset = st.session_state.box_selected_asset
            
            selected_asset = st.selectbox(
                "Select Asset to Analyze", 
                symbol_list, 
                index=ix,
                key="box_selected_asset",
                on_change=update_selected
            )
        
        if selected_asset:
            # Re-fetch specific asset data for charts (utilizing cache)
            fetcher = DataFetcher(cache_enabled=True)
            df_asset = fetcher.fetch_data(selected_asset)
            
            # Retrieve cached analysis result
            asset_res = next(r for r in results if r['symbol'] == selected_asset)
            
            analyzer = SOCAnalyzer(
                df_asset, selected_asset, asset_info=asset_res.get('info'),
                sma_window=sma_window, vol_window=vol_window, hysteresis=hysteresis
            )
            
            # Asset Header
            st.markdown("---")
            h1, h2, h3 = st.columns([1.5, 2.5, 1])
            with h1:
                st.header(f"{selected_asset}")
                info = asset_res.get('info', {})
                st.caption(f"{info.get('name', '')} | {info.get('sector', '')}")
            with h2:
                c_m, c_i = st.columns([1, 0.15])
                with c_m:
                    st.metric("Phase", asset_res['signal'])
                with c_i:
                    st.write("")
                    with st.popover("ℹ️"):
                        st.markdown("""
                        **Strategy Legend:**
                        
                        * 🟢 **Green + Trend:** Accumulation Phase. Low risk, high potential.
                        * 🔴 **Red + Downtrend:** Critical Instability. High crash probability.
                        
                        _Disclaimer: Not financial advice._
                        """)
            with h3:
                st.metric("Stress Score", f"{asset_res['stress_score']:.2f}")

            # Charts
            figs = analyzer.get_plotly_figures()
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Volatility Clustering")
                st.plotly_chart(figs['chart1'], use_container_width=True)
            with c2:
                st.subheader("Criticality")
                st.plotly_chart(figs['chart3'], use_container_width=True)
                
            st.subheader("Power Law Distribution")
            st.plotly_chart(figs['chart2'], use_container_width=True)

elif selected_tab == "📈 Portfolio Simulation":
    # Tab 3 Content (Portfolio Simulation)
    st.header("Portfolio Simulation")
    
    # 1. Quant Settings (One line)
    with st.expander("🔧 Simulation Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sim_years = st.slider("Simulation Years", 1, 30, 10)
            st.caption("Duration of the backtest. Longer periods test robustness across different market cycles (Bull/Bear).")
        with c2:
            rebalance = st.slider("Rebalance Frequency (Months)", 1, 12, 3)
            st.caption("How often the portfolio aligns with the SOC strategy signals. Frequent rebalancing captures trends faster but increases transaction costs.")
        with c3:
            alloc_risk = st.slider("Risk Allocation %", 0, 100, 70)
            st.caption("Percentage of capital allocated to 'Green/Accumulate' assets. The remainder is held in cash/safe assets to reduce drawdown.")

    st.divider()

    # 2. Inputs
    c_in1, c_in2 = st.columns(2)
    with c_in1:
        start_year = st.number_input("Starting Year", min_value=2000, max_value=2024, value=2015)
    with c_in2:
        start_capital = st.number_input("Starting Capital ($)", min_value=1000, value=10000, step=1000)

    # Display selected universe
    st.info(f"Selected Asset Class: **{universe_mode if 'universe_mode' in locals() else 'Current Selection'}**")

    st.divider()

    # 3. Action
    st.button("🚀 Run Simulation", type="primary", disabled=True, help="Feature coming soon")

elif selected_tab == "📘 Model Theory & Risks":
    # Tab 3 Content
    st.markdown("## 📘 Why this model is robust, but not bulletproof.")
    st.markdown("---")
    
    st.markdown("### 1. The Whipsaw Trap (Sideways Markets)")
    st.markdown("""
    **The Problem:** Simple Moving Average (SMA) models excel in trending markets but fail miserably in sideways/choppy markets. 
    In a flat market, price constantly crosses the SMA, generating false "Buy" and "Sell" signals (whipsaws) that bleed capital.
    
    **The Solution (Hysteresis):** We introduced a **"Whipsaw Filter"** (Hysteresis). 
    * Instead of triggering a signal the moment price touches the SMA, we require it to break the SMA by a specific percentage (e.g., 1.0% or 3.0%).
    * This creates a "Neutral Zone" where no new signals are generated, protecting you from chop.
    """)
    
    st.markdown("### 2. The Lag Effect")
    st.markdown("""
    **Reality Check:** SOC and Trend Following signals are **reactive**, not predictive. 
    * We do not catch the absolute bottom or top.
    * We aim to capture the "meat" of the move (the middle 60-80%).
    * **Risk:** In a "V-shaped" recovery or crash, the model will be late. This is the cost of doing business to avoid false alarms.
    """)
    
    st.markdown("### 3. The Minsky Paradox (The Coiled Spring)")
    st.markdown("""
    **"Stability leads to Instability."** — Hyman Minsky.
    
    * **Green (Low Volatility)** usually means safe accumulation. 
    * **However:** Extremely low volatility for extended periods often precedes a violent breakout (volatility compression).
    * If you see **Volatility near zero**, be ready for a massive move in *either* direction, even if the current signal is Green.
    """)
    
    st.markdown("### 4. Stationarity & Parameter Drift")
    st.markdown("""
    Markets change. A **200-day SMA** is a standard institutional benchmark, but in faster crypto markets, a **100-day** or **50-day** might be more relevant.
    * Use the **"Quant Settings"** in the sidebar to stress-test your thesis. 
    * If a signal disappears when you change the window slightly, it might not be robust.
    """)
