import streamlit as st
import pandas as pd
import time
import yfinance as yf
from logic import DataFetcher, SOCAnalyzer

# 1. Global Setup
st.set_page_config(
    page_title="SOC Market Seismograph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS / STYLING ---
st.markdown("""
<style>
    /* Clean, Minimalist Dashboard Styling */
    
    /* Remove Sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Minimize Top Padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px; /* Center content max width */
        margin: 0 auto;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    
    /* Custom Card for Metrics */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Hide Deploy Button */
    .stDeployButton {
        visibility: hidden;
    }
    
    /* Button Styling */
    div.stButton > button {
        font-weight: bold;
        border-radius: 8px;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --- CONSTANTS & DATA ---
GLOBAL_TICKERS = {
    "Bitcoin": "BTC-USD",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "Gold": "GC=F",
    "MSCI China": "MCHI",
    "Nikkei 225": "^N225"
}

MARKET_SETS = {
    "US Big Tech": ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'AMD', 'NFLX'],
    "DAX Top 10": ['^GDAXI', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'VOW3.DE', 'BAS.DE', 'MUV2.DE'],
    "Crypto Assets": ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD'],
    "Precious Metals": ['GC=F', 'SI=F', 'PL=F', 'PA=F', 'GLD', 'SLV']
}

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_global_ticker_tape():
    """Fetches simple price/change data for the header."""
    results = []
    fetcher = DataFetcher(cache_enabled=True)
    for name, symbol in GLOBAL_TICKERS.items():
        try:
            df = fetcher.fetch_data(symbol)
            if not df.empty:
                current = df["close"].iloc[-1]
                prev = df["close"].iloc[-2] if len(df) > 1 else current
                change = ((current - prev) / prev) * 100
                results.append({
                    "Name": name,
                    "Price": current,
                    "Change": change
                })
        except Exception:
            pass
    return results

@st.cache_data(ttl=86400)
def fetch_asset_names(ticker_list):
    """Fetches names for the preview table. Cached for 24h."""
    data = []
    for t in ticker_list:
        try:
            # Special handling for Indices to look cleaner
            if t == "^GDAXI":
                data.append({"Ticker": t, "Name": "DAX 40 Index"})
                continue
                
            # Quick fetch name only
            ticker = yf.Ticker(t)
            # Use fast info if available, else standard info
            name = ticker.info.get('shortName') or ticker.info.get('longName') or t
            
            # Clean up messy Yahoo Finance names
            name = name.replace(" SE", "").replace(" AG", "").strip()
            # Remove trailing ' I' or similar artifacts if common
            if name.endswith(" I"): name = name[:-2]
            
            data.append({"Ticker": t, "Name": name})
        except:
            data.append({"Ticker": t, "Name": "Unknown"})
    return pd.DataFrame(data)

def run_analysis_logic(tickers):
    """Runs the full SOC analysis."""
    fetcher = DataFetcher(cache_enabled=True)
    results = []
    
    # Defaults for now (could be exposed in 'Advanced' later)
    sma_w, vol_w, hyst = 200, 30, 0.0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers)
    for i, symbol in enumerate(tickers):
        status_text.caption(f"Processing {symbol}...")
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
            pass
        progress_bar.progress((i + 1) / total)
        
    status_text.empty()
    progress_bar.empty()
    return results

# --- MAIN LAYOUT ---

# 1. Global Market Header (The Ticker Tape)
with st.container():
    st.markdown("### 🌍 Current Market Health")
    global_data = fetch_global_ticker_tape()
    
    cols = st.columns(6) # Try 6 cols for wide mode
    
    # If fetch failed or partial, handle gracefully
    if not global_data:
        st.warning("Unable to fetch global market data.")
    else:
        for idx, item in enumerate(global_data):
            # Wrap if needed or just limit to available
            if idx < 6:
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #888;">{item['Name']}</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">${item['Price']:,.2f}</div>
                        <div style="color: {'#00FF00' if item['Change'] >= 0 else '#FF0000'};">
                            {item['Change']:+.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    st.divider()

# 2. Education
with st.expander("📖 How to read this dashboard"):
    st.markdown("""
    **SOC Traffic Light System:**
    * 🟢 **ACCUMULATE:** Low Volatility + Uptrend. Safe growth.
    * 🔴 **RISK:** High Volatility + Downtrend. Crash probability.
    * 🟠 **OVERHEATED:** High Volatility + Uptrend. Correction risk.
    * ⚪ **NEUTRAL:** Choppy / Range-bound market.
    """)

# 3. Workflow - Market Selection
st.header("Market Selection")

# Step A: Universe Selection
universe_options = ["US Big Tech", "DAX Top 10", "Crypto Assets", "Precious Metals", "Custom Portfolio"]
selected_universe = st.radio("Choose Asset Universe:", universe_options, horizontal=True)

# Logic for Tick List
active_tickers = []

if selected_universe == "Custom Portfolio":
    raw_input = st.text_area("Enter Tickers (comma or newline separated):", "NVDA, BTC-USD, GLD")
    cleaned = raw_input.replace("\n", ",").replace(";", ",")
    active_tickers = [t.strip().upper() for t in cleaned.split(",") if t.strip()]
else:
    active_tickers = MARKET_SETS[selected_universe]

# Step B: Preview (Asset Names)
if active_tickers:
    with st.expander(f"Preview: {len(active_tickers)} Assets", expanded=False):
        # Fetch names (cached)
        df_preview = fetch_asset_names(active_tickers)
        st.dataframe(df_preview, hide_index=True, width="stretch")

# Step C: Action
if st.button("RUN SOC ANALYSIS", type="primary", width="stretch"):
    with st.spinner("Analyzing Market Criticality..."):
        scan_results = run_analysis_logic(active_tickers)
        st.session_state['scan_results'] = scan_results
        st.session_state['run_timestamp'] = time.time()

# 4. Results Area
if 'scan_results' in st.session_state and st.session_state['scan_results']:
    results = st.session_state['scan_results']
    
    st.divider()
    st.header("Analysis Results")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Table Overview", "Deep Dive Analysis", "Portfolio Simulation", "Model Theory"])
    
    # --- TAB 1: Table Overview ---
    with tab1:
        if not results:
            st.warning("No valid data found for the selected assets.")
        else:
            df_res = pd.DataFrame(results)
            
            # Styling
            def highlight_signal(row):
                val = row['signal']
                color = ''
                if "BUY" in val: color = 'background-color: rgba(0, 255, 0, 0.1)'
                elif "CRASH" in val: color = 'background-color: rgba(255, 0, 0, 0.1)'
                elif "OVERHEATED" in val: color = 'background-color: rgba(255, 165, 0, 0.1)'
                return [color] * len(row)
            
            # Reset index to start at 1
            df_res.index = df_res.index + 1

            # Event handling for row click
            selection = st.dataframe(
                df_res[["symbol", "price", "signal", "stress_score", "trend"]].style.apply(highlight_signal, axis=1),
                width="stretch",
                column_config={
                    "symbol": "Asset",
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "signal": st.column_config.TextColumn("SOC Signal"),
                    "stress_score": st.column_config.ProgressColumn("Stress Level", min_value=0, max_value=2.0, format="%.2f"),
                    "trend": "Trend Status"
                },
                height=500,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            if selection.selection.rows:
                selected_row_index = selection.selection.rows[0]
                # Adjust for 1-based index if needed, but session state stores 0-based list usually
                # Actually, df_res.iloc uses 0-based position even if index is changed
                selected_symbol = df_res.iloc[selected_row_index]["symbol"]
                st.session_state['selected_asset_deep_dive'] = selected_symbol
                # Switch tab logic isn't native easily in basic Streamlit without extra component or reruns
                # We'll handle pre-selection in Tab 2

    # --- TAB 2: Deep Dive ---
    with tab2:
        if not results:
            st.info("Run analysis first.")
        else:
            symbol_list = [r['symbol'] for r in results]
            
            # Check if we have a selection from Tab 1
            default_ix = 0
            if 'selected_asset_deep_dive' in st.session_state:
                if st.session_state['selected_asset_deep_dive'] in symbol_list:
                    default_ix = symbol_list.index(st.session_state['selected_asset_deep_dive'])
            
            # Use cols to keep dropdown small
            c_sel, _ = st.columns([1, 3])
            with c_sel:
                selected_asset = st.selectbox("Select Asset to Inspect:", symbol_list, index=default_ix)
            
            # Fetch Deep Data
            asset_res = next(r for r in results if r['symbol'] == selected_asset)
            
            # Fetch full DF for charts
            fetcher = DataFetcher(cache_enabled=True)
            df_asset = fetcher.fetch_data(selected_asset)
            
            if not df_asset.empty:
                # Re-run analyzer for charts
                analyzer = SOCAnalyzer(df_asset, selected_asset, asset_info=asset_res.get('info'))
                figs = analyzer.get_plotly_figures()
                
                # --- Header with Traffic Light ---
                h_c1, h_c2 = st.columns([2, 1])
                with h_c1:
                    st.subheader(f"{selected_asset} - Criticality & Trend")
                with h_c2:
                    # Traffic Light Logic
                    sig = asset_res['signal']
                    color = "#00FF00" if "BUY" in sig or "ACCUMULATE" in sig else \
                            "#FF0000" if "CRASH" in sig else \
                            "#FFA500" if "OVERHEATED" in sig else "#CCCCCC"
                    
                    st.markdown(f"""
                    <div style="border: 2px solid {color}; padding: 10px; border-radius: 8px; text-align: center;">
                        <span style="font-size: 1.1rem; font-weight: bold; color: {color};">
                             {sig}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # --- Timeline Slider ---
                min_date = df_asset.index.min().date()
                max_date = df_asset.index.max().date()
                
                date_range = st.slider(
                    "Analysis Timeline",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),
                    format="YYYY-MM-DD"
                )
                
                # Filter data for chart based on slider
                filtered_df = df_asset.loc[str(date_range[0]):str(date_range[1])]
                
                # Re-generate ONLY the main chart with filtered data to be responsive
                # Note: For full correctness we should re-run analyzer on filtered data 
                # OR just zoom the chart. Re-running analyzer might change signals which is confusing.
                # Easier: Just update chart x-axis range.
                
                figs['chart3'].update_layout(xaxis_range=[date_range[0], date_range[1]], title="")
                
                # Hero Element: Criticality Chart (Chart 3)
                st.plotly_chart(figs['chart3'], width="stretch")
                
                # Context
                with st.expander("Show Price History & Power Law Details"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(figs['chart1'], width="stretch")
                    with c2:
                        st.plotly_chart(figs['chart2'], width="stretch")
            else:
                st.error("Could not load chart data.")

    # --- TAB 3: Portfolio Simulation ---
    with tab3:
        st.subheader("Portfolio Simulation")
        st.info("🚧 Module under construction. Use the 'Deep Dive' to analyze individual assets first.")
        
        c_sim1, c_sim2 = st.columns(2)
        with c_sim1:
            st.slider("Years", 1, 30, 10)
            st.number_input("Capital", value=10000)
        with c_sim2:
            st.slider("Risk Allocation", 0, 100, 70)
        
        st.button("Run Backtest (Simulated)", disabled=True)

    # --- TAB 4: Model Theory ---
    with tab4:
        st.markdown("""
        ### Self-Organized Criticality (SOC)
        Markets are not efficient; they are complex adaptive systems.
        
        1.  **Volatility Clustering:** Large price changes tend to be followed by large changes (Mandelbrot).
        2.  **Power Laws:** Market returns follow a Fat-Tailed distribution, not a Bell Curve. Standard models underestimate risk.
        3.  **Phase Transitions:** We look for the "Edge of Chaos" — when a system becomes unstable (High Volatility) and prone to collapse (Downtrend).
        """)
