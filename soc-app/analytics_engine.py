"""
TECTONIQ Analytics Engine - Event-Based Market Forensics
========================================================

Pure logic module for regime statistics and crash detection.
Uses run-length encoding to properly group consecutive days into events.

Key Principle: Calculate statistics on BLOCKS (events), not daily rows.

Author: Market Analysis Team
Version: 1.0 (Clean Rebuild)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MarketForensics:
    """
    Event-based analytics engine for regime and crash analysis.
    
    Correctly handles temporal grouping to avoid daily-count inflation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price dataframe.
        
        Args:
            df: DataFrame with DatetimeIndex and 'Regime' column
        """
        self.df = df.copy()
        if 'Regime' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Regime' column")
    
    def get_regime_stats(self) -> pd.DataFrame:
        """
        Calculate regime statistics using run-length encoding.
        
        Algorithm:
        1. Identify consecutive blocks of same regime
        2. Calculate duration of each block
        3. Aggregate statistics FROM BLOCKS (not daily rows)
        
        Returns:
            DataFrame with columns:
            - Frequency_Pct: % of total days in this regime
            - Count_Events: Number of times regime occurred
            - Avg_Duration_Days: Mean block duration
            - Median_Duration_Days: Median block duration
            - Min_Duration_Days: Shortest block
            - Max_Duration_Days: Longest block
        """
        df = self.df
        
        # STEP 1: CREATE BLOCK IDS
        # block_id increments whenever Regime changes
        df['block_id'] = (df['Regime'] != df['Regime'].shift(1)).cumsum()
        
        # STEP 2: AGGREGATE BY BLOCK
        # Get duration of each individual block
        # Result: Series like [Stable: 45 days, Critical: 12 days, Stable: 4 days]
        regime_blocks = df.groupby(['Regime', 'block_id']).size().reset_index(name='duration_days')
        
        # STEP 3: CALCULATE STATS FROM BLOCKS (NOT DAILY ROWS)
        # This is the critical fix - we aggregate the block durations, not the daily data
        regime_stats = regime_blocks.groupby('Regime')['duration_days'].agg([
            ('Count_Events', 'count'),      # How many times did this regime happen?
            ('Avg_Duration_Days', 'mean'),   # Average length of a phase
            ('Median_Duration_Days', 'median'),  # Median length
            ('Min_Duration_Days', 'min'),    # Shortest phase
            ('Max_Duration_Days', 'max')     # Longest phase
        ]).reset_index()
        
        # Calculate Frequency % based on total DAYS (not blocks)
        total_days = len(df)
        days_per_regime = df['Regime'].value_counts()
        regime_stats['Frequency_Pct'] = regime_stats['Regime'].map(
            lambda x: (days_per_regime.get(x, 0) / total_days) * 100
        )
        
        # Set Regime as index
        regime_stats = regime_stats.set_index('Regime')
        
        return regime_stats
    
    def get_crash_metrics(self, price_column: str = 'Close') -> Dict[str, Any]:
        """
        Identify significant crashes and evaluate signal detection performance.
        
        Strict Crash Definition:
        - Drawdown from 90-day rolling high
        - Threshold: -20% (filters noise)
        - De-bounced: Consecutive crash days = 1 event
        
        Returns:
            Dictionary containing:
            - total_crashes: Number of unique crash events
            - detected_count: Crashes with prior warning (RED/ORANGE)
            - missed_count: Crashes without prior warning
            - false_alarm_rate: % of warning signals without subsequent crash
            - avg_lead_time: Average days from warning to crash
            - lead_times: List of individual lead times
        """
        df = self.df
        
        if price_column not in df.columns:
            return {
                'total_crashes': 0,
                'detected_count': 0,
                'missed_count': 0,
                'false_alarm_rate': 0,
                'avg_lead_time': 0,
                'lead_times': []
            }
        
        prices = df[price_column]
        
        # GROUND TRUTH: Calculate drawdown from 90-day rolling high
        rolling_peak = prices.rolling(window=90, min_periods=1).max()
        drawdown = (prices - rolling_peak) / rolling_peak
        
        # STRICT THRESHOLD: Only -20% or deeper counts as crash
        crash_days = drawdown < -0.20
        
        # DE-BOUNCE: Group consecutive crash days into single events
        crash_block_id = (crash_days != crash_days.shift(1)).cumsum()
        
        # Get unique crash events (start dates)
        if crash_days.sum() == 0:
            return {
                'total_crashes': 0,
                'detected_count': 0,
                'missed_count': 0,
                'false_alarm_rate': 0,
                'avg_lead_time': 0,
                'lead_times': [],
                'total_signals': 0,
                'justified_signals': 0,
                'false_alarms': 0
            }
        
        crash_events_grouped = df[crash_days].groupby(crash_block_id)
        crash_start_dates = []
        for block_id, group in crash_events_grouped:
            crash_start_dates.append(group.index[0])
        
        total_crashes = len(crash_start_dates)
        true_crash_dates = crash_start_dates
        
        if total_crashes == 0:
            return {
                'total_crashes': 0,
                'detected_count': 0,
                'missed_count': 0,
                'false_alarm_rate': 0,
                'avg_lead_time': 0,
                'lead_times': []
            }
        
        # DETECTION EVALUATION (Recall)
        detected_crashes = 0
        missed_crashes = 0
        lead_times = []
        
        for crash_date in true_crash_dates:
            try:
                crash_idx = df.index.get_loc(crash_date)
            except:
                continue
            
            # Look 14 days before crash for warning (CRITICAL or HIGH_ENERGY)
            lookback_idx = max(0, crash_idx - 14)
            warning_window = df.iloc[lookback_idx:crash_idx + 1]
            
            had_warning = any(warning_window['Regime'].isin(['CRITICAL', 'HIGH_ENERGY']))
            
            if had_warning:
                detected_crashes += 1
                # Calculate lead time
                warning_days = warning_window[warning_window['Regime'].isin(['CRITICAL', 'HIGH_ENERGY'])]
                if len(warning_days) > 0:
                    first_warning_date = warning_days.index[0]
                    lead_days = (crash_date - first_warning_date).days
                    lead_times.append(max(0, lead_days))
            else:
                missed_crashes += 1
        
        # FALSE ALARM EVALUATION (Precision)
        # Identify warning signal blocks (RED/ORANGE lasting >= 5 days)
        df['block_id'] = (df['Regime'] != df['Regime'].shift(1)).cumsum()
        blocks = df.groupby('block_id').agg({
            'Regime': 'first'
        })
        blocks['Duration'] = df.groupby('block_id').size()
        blocks['Start_Date'] = df.groupby('block_id').apply(lambda x: x.index[0])
        
        warning_blocks = blocks[
            (blocks['Regime'].isin(['CRITICAL', 'HIGH_ENERGY'])) &
            (blocks['Duration'] >= 5)
        ]
        
        total_signals = len(warning_blocks)
        justified_signals = 0
        false_alarms = 0
        
        for idx, signal in warning_blocks.iterrows():
            signal_start = signal['Start_Date']
            lookahead_date = signal_start + pd.Timedelta(days=30)
            
            # Check if any crash started within 30 days
            crash_found = any(
                signal_start <= crash_date <= lookahead_date 
                for crash_date in true_crash_dates
            )
            
            if crash_found:
                justified_signals += 1
            else:
                false_alarms += 1
        
        false_alarm_rate = (false_alarms / total_signals * 100) if total_signals > 0 else 0
        avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0
        
        return {
            'total_crashes': total_crashes,
            'detected_count': detected_crashes,
            'missed_count': missed_crashes,
            'false_alarm_rate': false_alarm_rate,
            'avg_lead_time': avg_lead_time,
            'lead_times': lead_times,
            'total_signals': total_signals,
            'justified_signals': justified_signals,
            'false_alarms': false_alarms
        }


# =============================================================================
# INTEGRATION TEST
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("TECTONIQ Analytics Engine - Integration Test")
    print("="*60)
    
    # Create test data with known patterns
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Pattern: 30 days Stable, 10 days Critical, 20 days Active, 15 days Stable, 25 days Critical
    regimes = (
        ['STABLE'] * 30 + 
        ['CRITICAL'] * 10 + 
        ['ACTIVE'] * 20 + 
        ['STABLE'] * 15 + 
        ['CRITICAL'] * 25
    )
    
    # Create price data with crashes during CRITICAL periods
    prices = [100] * 30  # Stable: flat
    prices += [100 - i*2 for i in range(10)]  # Critical: drop 2% per day (20% total)
    prices += [80 + i*0.5 for i in range(20)]  # Active: recover
    prices += [90] * 15  # Stable: flat
    prices += [90 - i*1.5 for i in range(25)]  # Critical: drop 37.5%
    
    test_df = pd.DataFrame({
        'Close': prices,
        'Regime': regimes
    }, index=dates)
    
    print("\nTest DataFrame:")
    print(f"Total days: {len(test_df)}")
    print(f"Regime counts: {test_df['Regime'].value_counts().to_dict()}")
    
    # Initialize engine
    engine = MarketForensics(test_df)
    
    # Test Method A: Regime Stats
    print("\n" + "="*60)
    print("METHOD A: get_regime_stats()")
    print("="*60)
    
    regime_stats = engine.get_regime_stats()
    print("\nRegime Statistics (Block-Based):")
    print(regime_stats.to_string())
    
    print("\n✅ VALIDATION:")
    print(f"   Stable Median: {regime_stats.loc['STABLE', 'Median_Duration_Days']:.1f} days")
    print(f"   Expected: ~22.5 days (average of 30 and 15)")
    print(f"   Critical Median: {regime_stats.loc['CRITICAL', 'Median_Duration_Days']:.1f} days")
    print(f"   Expected: ~17.5 days (average of 10 and 25)")
    
    if regime_stats.loc['STABLE', 'Median_Duration_Days'] > 1:
        print("\n   ✅ PASS: Median > 1 (block-based logic working!)")
    else:
        print("\n   ❌ FAIL: Median = 1 (still counting daily rows!)")
    
    # Test Method B: Crash Metrics
    print("\n" + "="*60)
    print("METHOD B: get_crash_metrics()")
    print("="*60)
    
    crash_metrics = engine.get_crash_metrics(price_column='Close')
    print("\nCrash Detection Metrics:")
    for key, value in crash_metrics.items():
        if isinstance(value, list):
            print(f"   {key}: {value}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ VALIDATION:")
    print(f"   Total Crashes: {crash_metrics['total_crashes']}")
    print(f"   Expected: 2 (one at day 30, one at day 65)")
    print(f"   Detected: {crash_metrics['detected_count']}")
    print(f"   False Alarms: {crash_metrics['false_alarms']}")
    
    if crash_metrics['total_crashes'] == 2:
        print("\n   ✅ PASS: Correctly identified 2 unique crash events!")
    else:
        print(f"\n   ❌ FAIL: Found {crash_metrics['total_crashes']} crashes (expected 2)")
    
    print("\n" + "="*60)
    print("Integration Test Complete")
    print("="*60)

