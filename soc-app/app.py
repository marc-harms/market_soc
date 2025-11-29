import streamlit as st
import pandas as pd
from logic import DataFetcher, SOCAnalyzer

st.set_page_config(page_title="SOC Market Scanner", layout="wide")

st.title("⚡ SOC Market Scanner")
st.markdown("### Self-Organized Criticality Analysis")

# Sidebar
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Enter Symbol", value="BTCUSDT").upper()
days = st.sidebar.slider("Lookback Days", 365, 3000, 2000)

if st.sidebar.button("Analyze Asset"):
    with st.spinner(f"Analyzing {symbol}..."):
        # Fetch
        fetcher = DataFetcher()
        df = fetcher.fetch_data(symbol)
        info = fetcher.fetch_info(symbol)
        
        if df.empty:
            st.error(f"No data found for {symbol}")
        else:
            # Analyze
            analyzer = SOCAnalyzer(df, symbol, asset_info=info)
            phase = analyzer.get_market_phase()
            
            # Display
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"{phase['price']:,.2f}")
            col2.metric("SMA 200", f"{phase['sma_200']:,.2f}", f"{phase['dist_to_sma']:.1%}")
            col3.metric("Signal", phase['signal'])
            
            # Info
            if info.get('description'):
                st.caption(f"**{info.get('name')}**: {info.get('description')}")
            
            # TODO: Integrate Plotly charts here
            st.success("Analysis Complete")

