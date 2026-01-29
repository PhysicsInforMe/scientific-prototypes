#!/usr/bin/env python
"""
Backtest Analysis Example.

Demonstrates how to use the Backtester to validate
MarketIntelligence on historical data.

Usage:
    python examples/backtest_analysis.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import pandas as pd

from timeseries_toolkit.data_sources.equities import EquityDataLoader
from timeseries_toolkit.intelligence.backtester import Backtester


def main():
    """Run backtest on SPY."""
    print("=" * 60)
    print("  BACKTEST ANALYSIS: SPY")
    print("=" * 60)

    # Fetch 3 years of SPY data.
    print("\n  Fetching data...")
    loader = EquityDataLoader()
    prices_df = loader.get_prices(["SPY"], period="3y")

    # Extract Close prices.
    close_col = [c for c in prices_df.columns if "close" in str(c).lower()][0]
    prices = prices_df[close_col]
    prices.name = "SPY"

    print(f"  Data: {len(prices)} observations")
    print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Run backtest.
    print("\n  Running walk-forward backtest (7-day horizon, 14-day step)...\n")
    bt = Backtester()
    result = bt.run_backtest(
        prices,
        start_date="2024-06-01",
        end_date="2025-12-01",
        horizon=7,
        step=14,
        verbose=True,
    )

    # Print summary.
    print("\n" + result.summary())

    # Validate regime detection.
    print("\n  --- Regime Detection Validation ---")
    regime_result = bt.validate_regime_detection(
        prices, start_date="2024-01-01", end_date="2025-12-01"
    )
    print(f"  Volatility aligned: {regime_result.volatility_aligned}")
    print(f"  Drawdown aligned:   {regime_result.drawdown_aligned}")
    if regime_result.details:
        for k, v in regime_result.details.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
