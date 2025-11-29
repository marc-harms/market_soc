#!/usr/bin/env python3
"""
SOC Scanner - Phase 1.5
Multi-Asset Financial Self-Organized Criticality Scanner
"""

import sys
import pandas as pd
from tabulate import tabulate
from typing import List

from modules.data_fetcher import DataFetcher
from modules.analyzer import SOCAnalyzer
from modules.visualizer import PlotlyVisualizer
import webbrowser
import os

# Configuration
WATCHLIST = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'NVDA', 'TSLA', '^GDAXI', 'SAP.DE', 'MSFT', 'AAPL', 'GC=F']

def run_scanner():
    """
    Main Scanner Loop
    """
    print("=" * 60)
    print("  SOC MARKET SCANNER (Crypto + Stocks)")
    print("=" * 60)

    fetcher = DataFetcher(cache_enabled=True)
    results = []
    
    analyzers = {} # Store analyzer instances for plotting later

    print(f"\nScanning {len(WATCHLIST)} assets...\n")

    for symbol in WATCHLIST:
        try:
            print(f"Analyzing {symbol}...", end=" ", flush=True)
            
            # 1. Fetch Data
            df = fetcher.fetch_data(symbol)
            
            if df.empty or len(df) < 200:
                print("❌ Insufficient Data")
                continue

            # 1.5 Fetch Info
            asset_info = fetcher.fetch_info(symbol)

            # 2. Analyze
            analyzer = SOCAnalyzer(df, symbol, asset_info=asset_info)
            phase = analyzer.get_market_phase()
            
            results.append(phase)
            analyzers[symbol] = analyzer
            print(f"✓ {phase['signal']}")

        except Exception as e:
            print(f"❌ Error: {e}")

    # 3. Create DataFrame for Sorting
    if not results:
        print("No results found.")
        return

    results_df = pd.DataFrame(results)
    
    # Custom Sort Order: CRASH RISK > BUY > OVERHEATED > CAPITULATION > NEUTRAL
    signal_priority = {
        "🔴 CRASH RISK": 0,
        "🟢 BUY/ACCUMULATE": 1,
        "🟠 OVERHEATED": 2,
        "🟡 CAPITULATION/WAIT": 3,
        "⚪ UPTREND (Choppy)": 4,
        "⚪ DOWNTREND (Choppy)": 5,
        "NEUTRAL": 6
    }
    
    results_df["sort_key"] = results_df["signal"].map(signal_priority)
    results_df = results_df.sort_values("sort_key")
    
    # Prepare Table
    table_data = results_df[[
        "symbol", "price", "sma_200", "dist_to_sma", 
        "stress", "signal"
    ]].copy()
    
    # Format columns
    table_data["price"] = table_data["price"].apply(lambda x: f"{x:,.2f}")
    table_data["sma_200"] = table_data["sma_200"].apply(lambda x: f"{x:,.2f}")
    table_data["dist_to_sma"] = table_data["dist_to_sma"].apply(lambda x: f"{x:+.1%}")
    
    print("\n" + "=" * 80)
    print("  MARKET PHASE REPORT")
    print("=" * 80)
    print(tabulate(table_data, headers="keys", tablefmt="simple", showindex=False))
    print("=" * 80)
    
    # Interactive Plotting Option
    while True:
        choice = input("\nEnter symbol to visualize (or 'q' to quit): ").strip().upper()
        if choice == 'Q':
            break
            
        # Handle case variations (e.g. btcusdt -> BTCUSDT)
        # But watchlist has specific casing. Let's try to match.
        target = next((s for s in analyzers.keys() if s.upper() == choice), None)
        
        if target:
            plot_asset(analyzers[target])
        else:
            print(f"Symbol {choice} not in scanned results.")

def plot_asset(analyzer: SOCAnalyzer):
    """
    Generate and open plot for a specific asset
    """
    data = analyzer.get_visualizer_data()
    visualizer = PlotlyVisualizer(
        df=data["df"],
        symbol=data["symbol"],
        vol_low_threshold=data["vol_low"],
        vol_high_threshold=data["vol_high"],
        asset_info=data.get("asset_info")
    )
    
    html = visualizer.create_interactive_dashboard()
    filename = f"soc_analysis_{data['symbol']}.html"
    visualizer.save_html(html, filename)
    
    print(f"Opening {filename}...")
    webbrowser.open('file://' + os.path.realpath(filename))

if __name__ == "__main__":
    run_scanner()

