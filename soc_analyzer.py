#!/usr/bin/env python3
"""
SOC Analyzer - Financial Self-Organized Criticality Analysis Tool
Analyzes cryptocurrency markets for phase transitions, volatility clustering, and fat tail risks

Usage:
    python soc_analyzer.py                    # Fetch fresh data from API
    python soc_analyzer.py --cached           # Use cached data if available
    python soc_analyzer.py --symbol ETHUSDT   # Analyze different symbol
"""

import argparse
import sys
from typing import Optional

from modules.data_fetcher import BinanceDataFetcher
from modules.soc_metrics import SOCMetricsCalculator
from modules.visualizer import PlotlyVisualizer
from config.settings import DEFAULT_SYMBOL, DEFAULT_LOOKBACK_DAYS


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Financial SOC Analysis Tool - Analyze crypto markets for criticality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Trading symbol to analyze (default: {DEFAULT_SYMBOL})",
    )

    parser.add_argument(
        "--cached",
        action="store_true",
        help="Use cached data if available (avoids API rate limits)",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Number of days to fetch (default: {DEFAULT_LOOKBACK_DAYS})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="soc_analysis.html",
        help="Output HTML filename (default: soc_analysis.html)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't open browser automatically",
    )

    return parser.parse_args()


def run_soc_analysis(
    symbol: str,
    cached: bool = False,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    output_file: str = "soc_analysis.html",
    show_browser: bool = True,
) -> None:
    """
    Main analysis pipeline

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        cached: Whether to use cached data
        lookback_days: Number of days to analyze
        output_file: Output HTML filename
        show_browser: Whether to open in browser
    """
    print("=" * 60)
    print(f"  SOC ANALYSIS - {symbol}")
    print("=" * 60)

    try:
        # Step 1: Fetch Data
        print("\n[1/4] Fetching Data...")
        fetcher = BinanceDataFetcher(cache_enabled=cached)
        df = fetcher.fetch_data(
            symbol=symbol,
            lookback_days=lookback_days,
        )

        if df.empty:
            print("❌ No data retrieved. Exiting.")
            sys.exit(1)

        print(f"✓ Data loaded: {len(df)} records from {df.index.min().date()} to {df.index.max().date()}")

        # Step 2: Calculate SOC Metrics
        print("\n[2/4] Calculating SOC Metrics...")
        calculator = SOCMetricsCalculator(df)
        df_enriched = calculator.calculate_all_metrics()

        # Display summary stats
        stats = calculator.get_summary_stats()
        print(f"\n📊 Summary Statistics:")
        print(f"  - Price Range: ${df_enriched['close'].min():,.2f} - ${df_enriched['close'].max():,.2f}")
        print(f"  - Current Price: ${stats['current_price']:,.2f}")
        print(f"  - SMA 200: ${stats['current_sma_200']:,.2f}")
        print(f"  - Mean Return: {stats['mean_return']:.4%}")
        print(f"  - Return Volatility: {stats['std_return']:.4%}")
        print(f"  - Mean Rolling Vol: {stats['mean_volatility']:.6f}")

        # Traffic Light Status
        current_vol = df_enriched["volatility"].iloc[-1]
        current_zone = df_enriched["vol_zone"].iloc[-1]
        print(f"\n🚦 Current Status:")
        print(f"  - Volatility Zone: {current_zone.upper()}")
        print(f"  - Current Volatility: {current_vol:.6f}")

        if stats['current_price'] < stats['current_sma_200'] and current_zone == "high":
            print("  - ⚠️  WARNING: Price below SMA + High Volatility = CRASH RISK")
        elif current_zone == "high":
            print("  - ⚠️  CAUTION: High volatility detected")
        elif current_zone == "low":
            print("  - ✓ STABLE: Low volatility environment")

        # Step 3: Create Visualizations
        print("\n[3/4] Creating Visualizations...")
        visualizer = PlotlyVisualizer(
            df=df_enriched,
            symbol=symbol,
            vol_low_threshold=calculator.vol_low_threshold,
            vol_high_threshold=calculator.vol_high_threshold,
        )
        # Generates a standalone HTML string with embedded data and custom CSS
        fig = visualizer.create_interactive_dashboard()

        # Step 4: Save and Display
        print("\n[4/4] Saving Output...")
        visualizer.save_html(fig, filename=output_file)

        if show_browser:
            print(f"\n🌐 Opening visualization in browser...")
            import webbrowser
            import os
            webbrowser.open('file://' + os.path.realpath(output_file))
        else:
            print(f"\n✓ Visualization saved to: {output_file}")

        print("\n" + "=" * 60)
        print("  ANALYSIS COMPLETE")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point"""
    args = parse_arguments()

    run_soc_analysis(
        symbol=args.symbol,
        cached=args.cached,
        lookback_days=args.lookback,
        output_file=args.output,
        show_browser=not args.no_show,
    )


if __name__ == "__main__":
    # Default execution: Analyze BTC with fresh data
    main()

