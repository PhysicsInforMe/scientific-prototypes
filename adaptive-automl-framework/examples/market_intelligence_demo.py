#!/usr/bin/env python
"""
Market Intelligence Demo.

Demonstrates the full MarketIntelligence workflow:
    1. Single asset analysis with forecast
    2. Regime detection
    3. Multi-asset comparison
    4. Markdown report export

Usage:
    python examples/market_intelligence_demo.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from timeseries_toolkit.intelligence import MarketIntelligence


def demo_single_asset():
    """Run full analysis on a single asset."""
    print("=" * 60)
    print("  SINGLE ASSET ANALYSIS: BTC-USD")
    print("=" * 60)

    mi = MarketIntelligence()
    report = mi.analyze(["BTC-USD"], horizon="7D", verbose=True)

    print(f"\n{report.summary}")
    print(f"\nPipeline: {report.pipeline_used}")
    print(f"Reason:   {report.pipeline_reason[:80]}...")
    print(f"Quality:  {report.quality_score:.2f}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.recommendations:
        print("\nRecommendations:")
        for r in report.recommendations:
            print(f"  - {r}")

    print("\nForecast:")
    print(report.forecast.to_string())

    return report


def demo_regime():
    """Detect market regime without forecasting."""
    print("\n" + "=" * 60)
    print("  REGIME DETECTION: SPY")
    print("=" * 60)

    mi = MarketIntelligence()
    regime = mi.get_regime(["SPY"])

    print(f"\n  Current regime:    {regime.current_regime}")
    print(f"  Confidence:        {regime.confidence:.0%}")
    print(f"  Days in regime:    {regime.days_in_regime}")
    print(f"  Transition risk:   {regime.transition_risk:.0%}")

    print("\n  Regime probabilities:")
    for name, prob in sorted(regime.regime_probabilities.items(),
                              key=lambda x: -x[1]):
        print(f"    {name:<12} {prob:.0%}")


def demo_quick_forecast():
    """Quick forecast without full analysis."""
    print("\n" + "=" * 60)
    print("  QUICK FORECAST: ETH-USD")
    print("=" * 60)

    mi = MarketIntelligence()
    fc = mi.quick_forecast("ETH-USD", horizon="7D")

    print("\n  Forecast:")
    print(fc.to_string())


def demo_markdown_export(report):
    """Export report to Markdown."""
    print("\n" + "=" * 60)
    print("  MARKDOWN EXPORT")
    print("=" * 60)

    md = report.to_markdown()
    print(f"\n  Markdown length: {len(md)} characters")
    print(f"  First 200 chars:")
    print(f"  {md[:200]}...")

    # Save to file.
    out_path = os.path.join(os.path.dirname(__file__), "sample_report.md")
    report.save_markdown(out_path)
    print(f"\n  Saved to: {out_path}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  MARKET INTELLIGENCE DEMO")
    print("=" * 60)

    report = demo_single_asset()
    demo_regime()
    demo_quick_forecast()
    demo_markdown_export(report)

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
