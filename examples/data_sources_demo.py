#!/usr/bin/env python
"""
Data Sources Demo.

Demonstrates how to use the DataHub and individual data loaders
to fetch financial and economic data from free public APIs.

Usage:
    python examples/data_sources_demo.py

Requirements:
    pip install yfinance fredapi requests

    For FRED data, set FRED_API_KEY environment variable:
    https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from timeseries_toolkit.data_sources import (
    CryptoDataLoader,
    EquityDataLoader,
    VolatilityDataLoader,
    DataHub,
)


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_result(name: str, data, success: bool = True):
    """Print a formatted result row."""
    status = "OK" if success else "FAIL"
    if isinstance(data, pd.DataFrame):
        info = f"{data.shape[0]} rows x {data.shape[1]} cols"
    elif isinstance(data, pd.Series):
        info = f"{len(data)} values"
    else:
        info = str(type(data).__name__)
    print(f"  [{status:^4}] {name:<35} {info}")


def demo_yahoo_sources():
    """Demo Yahoo Finance-based loaders (no API key required)."""
    print_header("YAHOO FINANCE DATA (no API key needed)")

    # Crypto
    print("\n  --- Crypto ---")
    try:
        loader = CryptoDataLoader(source="yahoo")
        btc = loader.get_prices(["BTC-USD"], period="1mo")
        print_result("BTC-USD (1 month)", btc)
    except Exception as e:
        print(f"  [FAIL] Crypto: {e}")

    # Fear & Greed Index
    try:
        loader = CryptoDataLoader()
        fgi = loader.get_fear_greed_index(limit=30)
        print_result("Fear & Greed Index", fgi)
    except Exception as e:
        print(f"  [FAIL] Fear & Greed: {e}")

    # Equities
    print("\n  --- Equities ---")
    try:
        loader = EquityDataLoader()
        spy = loader.get_prices(["SPY"], period="1mo")
        print_result("SPY (1 month)", spy)
    except Exception as e:
        print(f"  [FAIL] SPY: {e}")

    # Volatility
    print("\n  --- Volatility ---")
    try:
        loader = VolatilityDataLoader()
        vix = loader.get_vix(period="1mo")
        print_result("VIX (1 month)", vix)
    except Exception as e:
        print(f"  [FAIL] VIX: {e}")

    try:
        loader = VolatilityDataLoader()
        vol_indices = loader.get_volatility_indices(period="1mo")
        print_result("Volatility indices", vol_indices)
    except Exception as e:
        print(f"  [FAIL] Vol indices: {e}")


def demo_fred_sources():
    """Demo FRED-based loaders (API key required)."""
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print_header("FRED DATA (skipped - no FRED_API_KEY set)")
        print("  Set FRED_API_KEY env var to test FRED sources.")
        print("  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    print_header("FRED DATA")

    from timeseries_toolkit.data_sources import (
        RatesDataLoader,
        LiquidityDataLoader,
        MacroDataLoader,
        EuropeanDataLoader,
    )

    # Rates
    print("\n  --- Interest Rates ---")
    try:
        loader = RatesDataLoader()
        yields = loader.get_treasury_yields(["3M", "2Y", "10Y", "30Y"])
        print_result("Treasury yields", yields)
    except Exception as e:
        print(f"  [FAIL] Treasury yields: {e}")

    try:
        loader = RatesDataLoader()
        slope = loader.get_yield_curve_slope()
        print_result("Yield curve slope (10Y-2Y)", slope)
    except Exception as e:
        print(f"  [FAIL] Yield curve: {e}")

    # Liquidity
    print("\n  --- Liquidity ---")
    try:
        loader = LiquidityDataLoader()
        m2 = loader.get_money_supply(["M2"])
        print_result("M2 money supply", m2)
    except Exception as e:
        print(f"  [FAIL] M2: {e}")

    try:
        loader = LiquidityDataLoader()
        fed_bs = loader.get_fed_balance_sheet()
        print_result("Fed balance sheet", fed_bs)
    except Exception as e:
        print(f"  [FAIL] Fed BS: {e}")

    try:
        loader = LiquidityDataLoader()
        spreads = loader.get_credit_spreads()
        print_result("Credit spreads (IG/HY)", spreads)
    except Exception as e:
        print(f"  [FAIL] Credit spreads: {e}")

    # Macro
    print("\n  --- Macro ---")
    try:
        loader = MacroDataLoader()
        gdp = loader.get_gdp()
        print_result("Real GDP", gdp)
    except Exception as e:
        print(f"  [FAIL] GDP: {e}")

    try:
        loader = MacroDataLoader()
        cpi = loader.get_inflation("CPI")
        print_result("CPI", cpi)
    except Exception as e:
        print(f"  [FAIL] CPI: {e}")

    try:
        loader = MacroDataLoader()
        emp = loader.get_employment()
        print_result("Employment indicators", emp)
    except Exception as e:
        print(f"  [FAIL] Employment: {e}")

    # European
    print("\n  --- European ---")
    try:
        loader = EuropeanDataLoader()
        ecb = loader.get_ecb_rates()
        print_result("ECB rates", ecb)
    except Exception as e:
        print(f"  [FAIL] ECB rates: {e}")

    try:
        loader = EuropeanDataLoader()
        spread = loader.get_btp_bund_spread()
        print_result("BTP-Bund spread", spread)
    except Exception as e:
        print(f"  [FAIL] BTP-Bund: {e}")


def demo_bundles():
    """Demo DataHub bundles."""
    print_header("DATAHUB BUNDLES")

    hub = DataHub()

    # Crypto bundle (no FRED key needed)
    try:
        crypto = hub.load_bundle("crypto_ecosystem", period="1mo")
        for name, data in crypto.items():
            print_result(f"crypto_ecosystem/{name}", data)
    except Exception as e:
        print(f"  [FAIL] crypto_ecosystem: {e}")

    # FRED bundles (need key)
    if os.environ.get("FRED_API_KEY"):
        for bundle_name in ["risk_indicators", "macro_us"]:
            try:
                bundle = hub.load_bundle(bundle_name)
                for name, data in bundle.items():
                    print_result(f"{bundle_name}/{name}", data)
            except Exception as e:
                print(f"  [FAIL] {bundle_name}: {e}")


def main():
    """Run all demos."""
    warnings.filterwarnings("ignore")

    print_header("DATA SOURCES DEMO")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  FRED_API_KEY: {'set' if os.environ.get('FRED_API_KEY') else 'not set'}")

    demo_yahoo_sources()
    demo_fred_sources()
    demo_bundles()

    print_header("DEMO COMPLETE")


if __name__ == "__main__":
    main()
