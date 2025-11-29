import streamlit as st
import pandas as pd
from logic import DataFetcher, SOCAnalyzer

# 1. Config
st.set_page_config(
    page_title="SOC Market Seismograph",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ SOC Market Seismograph")
st.markdown("### Professional Risk Analytics")

# --- SIDEBAR ---
st.sidebar.header("Scanner Settings")

# Presets
presets = {
    "Crypto Top 5": "BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, ADAUSDT",
    "US Tech": "NVDA, TSLA, AAPL, MSFT, GOOGL, AMZN, META",
    "German DAX": "^GDAXI, SAP.DE, SIE.DE, ALV.DE, DTE.DE",
    "Global Indices": "^GDAXI, ^GSPC, ^IXIC, ^DJI, GC=F"
}

# Add a key for the selectbox to persist state properly or handle changes
selected_preset = st.sidebar.selectbox("Load Preset", ["Custom"] + list(presets.keys()))

# Initialize tickers in session state if not present
if "tickers" not in st.session_state:
    st.session_state.tickers = "BTCUSDT, ETHUSDT, SOLUSDT, NVDA, TSLA, ^GDAXI"

# Update tickers if preset changes
if selected_preset != "Custom":
    st.session_state.tickers = presets[selected_preset]

# Text Area (bound to session state manually or via key, here simplistic approach)
# We use key='tickers_input' and update session_state.tickers on change if needed, 
# but simpler is just to let the text area drive the input.
ticker_input = st.sidebar.text_area(
    "Watchlist (comma separated)", 
    value=st.session_state.tickers, 
    height=150
)

# Parsing
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Run Button
run_scan = st.sidebar.button("🚀 Run Scan", type="primary")

# --- LOGIC ---

@st.cache_data(ttl=3600)
def perform_scan(ticker_list):
    fetcher = DataFetcher(cache_enabled=True)
    results = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(ticker_list):
        status_text.text(f"Scanning {symbol}...")
        try:
            df = fetcher.fetch_data(symbol)
            info = fetcher.fetch_info(symbol)
            
            if not df.empty and len(df) > 200:
                analyzer = SOCAnalyzer(df, symbol, asset_info=info)
                phase = analyzer.get_market_phase()
                phase['info'] = info # Store info for display
                results.append(phase)
        except Exception as e:
            print(f"Error {symbol}: {e}")
            
        progress_bar.progress((i + 1) / len(ticker_list))
        
    status_text.empty()
    progress_bar.empty()
    
    return results

# Store results in session state
if run_scan:
    with st.spinner("Scanning markets..."):
        st.session_state.scan_results = perform_scan(tickers)

results = st.session_state.get('scan_results', [])

# --- MAIN AREA ---

tab1, tab2 = st.tabs(["📊 Market Scanner", "🔍 Deep Dive Analysis"])

with tab1:
    if not results:
        st.info("Click 'Run Scan' to start analysis.")
    else:
        # Create DF
        df_res = pd.DataFrame(results)
        
        # Sort by Signal Priority
        signal_order = {
            "🔴 CRASH RISK": 0,
            "🟢 BUY/ACCUMULATE": 1,
            "🟠 OVERHEATED": 2,
            "🟡 CAPITULATION/WAIT": 3,
            "⚪ UPTREND (Choppy)": 4,
            "⚪ DOWNTREND (Choppy)": 5,
            "NEUTRAL": 6,
            "NO_DATA": 7
        }
        df_res['sort_key'] = df_res['signal'].map(signal_order)
        df_res = df_res.sort_values('sort_key').drop(columns=['sort_key'])
        
        # Display Table with Styler
        def style_signal(val):
            color = 'transparent'
            text_color = 'white'
            if "BUY" in val: 
                color = 'rgba(0, 255, 0, 0.2)' 
                text_color = '#90EE90'
            elif "CRASH" in val: 
                color = 'rgba(255, 0, 0, 0.2)'
                text_color = '#FF7F7F'
            elif "OVERHEATED" in val: 
                color = 'rgba(255, 165, 0, 0.2)'
                text_color = '#FFA500'
            elif "CAPITULATION" in val: 
                color = 'rgba(255, 255, 0, 0.1)'
                text_color = '#FFFFE0'
            return f'background-color: {color}; color: {text_color}'

        # Select columns for display
        display_cols = ["symbol", "price", "signal", "stress", "trend", "dist_to_sma"]
        
        st.dataframe(
            df_res[display_cols].style.applymap(style_signal, subset=['signal']),
            use_container_width=True,
            column_config={
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "dist_to_sma": st.column_config.NumberColumn("SMA Dist", format="%.2f%%"),
            },
            height=600
        )

with tab2:
    if not results:
        st.warning("Please run a scan first.")
    else:
        # Asset Selector
        symbols = [r['symbol'] for r in results]
        selected_symbol = st.selectbox("Select Asset", symbols)
        
        if selected_symbol:
            # Fetch data again (cached)
            fetcher = DataFetcher(cache_enabled=True)
            df_asset = fetcher.fetch_data(selected_symbol)
            
            # Find the result dict for info
            res_item = next(r for r in results if r['symbol'] == selected_symbol)
            
            analyzer = SOCAnalyzer(df_asset, selected_symbol, asset_info=res_item.get('info'))
            
            # Header
            st.markdown("---")
            col_head1, col_head2 = st.columns([3, 1])
            with col_head1:
                st.header(f"{selected_symbol}")
                if res_item.get('info'):
                    st.caption(f"**{res_item['info'].get('name', '')}** | {res_item['info'].get('sector', '')}")
                    st.write(res_item['info'].get('description', ''))
            with col_head2:
                st.metric("Signal", res_item['signal'])
                st.metric("Stress Level", res_item['stress'])

            st.markdown("---")

            # Charts
            figs = analyzer.get_plotly_figures()
            
            st.subheader("1. Volatility Clustering")
            st.plotly_chart(figs['chart1'], use_container_width=True)
            
            st.subheader("2. Power Law Analysis")
            st.plotly_chart(figs['chart2'], use_container_width=True)
            
            st.subheader("3. Criticality & Trend")
            st.plotly_chart(figs['chart3'], use_container_width=True)

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** Educational purposes only. No financial advice. Market data provided by Binance and Yahoo Finance.")
