"""
Integration tests for MarketIntelligence with real data.

These tests hit live Yahoo Finance APIs.
Mark: @pytest.mark.integration

Validates:
    1. Full analysis runs on real BTC-USD data.
    2. Full analysis runs on real SPY data.
    3. Multi-asset comparison works.
    4. Output sensibility checks (price range, regime coherence).
    5. Markdown export produces valid output.
"""

import os

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.intelligence import MarketIntelligence


# ---------------------------------------------------------------------------
# Sensibility validators (from MARKET_INTELLIGENCE.md Task 5)
# ---------------------------------------------------------------------------

def validate_forecast_sensibility(
    forecast: pd.DataFrame, current_price: float
) -> bool:
    """
    Check forecast is sensible, not garbage.

    - Forecast within 50% of current price (for 7-day horizon).
    - Confidence bands have positive width.
    - Bands not wider than 50% of current price.
    """
    if forecast is None or len(forecast) == 0:
        return False

    forecast_vals = forecast["forecast"].values
    max_change = 0.5

    if any(forecast_vals > current_price * (1 + max_change)):
        return False
    if any(forecast_vals < current_price * (1 - max_change)):
        return False

    widths = forecast["upper"] - forecast["lower"]
    if any(widths <= 0):
        return False
    if any(widths > current_price * 0.5):
        return False

    return True


def validate_regime_sensibility(regime, vix: float, recent_return: float) -> bool:
    """
    Check regime makes sense given observable indicators.

    - If VIX > 30 and recent return < -10%, should not be 'bull'.
    - If VIX < 15 and recent return > 5%, should not be 'crisis'.
    """
    if vix > 30 and recent_return < -0.1:
        if regime.current_regime == "bull":
            return False
    if vix < 15 and recent_return > 0.05:
        if regime.current_regime == "crisis":
            return False
    return True


# ===========================================================================
# Integration Tests
# ===========================================================================

@pytest.mark.integration
class TestMarketIntelligenceLive:
    """Live integration tests for MarketIntelligence."""

    def test_analyze_btc(self):
        """Full analysis on real BTC-USD data."""
        mi = MarketIntelligence()
        report = mi.analyze(["BTC-USD"], horizon="7D", verbose=True)

        assert report is not None
        assert len(report.summary) > 20
        assert report.quality_score > 0
        assert report.pipeline_used != ""
        assert report.forecast is not None
        assert len(report.forecast) == 7
        assert report.regime is not None
        assert report.regime.current_regime in ("bull", "bear", "crisis", "sideways")

    def test_analyze_spy(self):
        """Full analysis on real SPY data."""
        mi = MarketIntelligence()
        report = mi.analyze(["SPY"], horizon="7D", verbose=True)

        assert report is not None
        assert report.forecast is not None
        assert len(report.forecast) == 7
        assert report.regime is not None

    def test_forecast_sensibility_btc(self):
        """BTC forecast should be within sensible bounds."""
        mi = MarketIntelligence()
        report = mi.analyze(["BTC-USD"], horizon="7D")

        current = report.current_prices.get("BTC-USD", 0)
        if current > 0:
            assert validate_forecast_sensibility(report.forecast, current)

    def test_forecast_sensibility_spy(self):
        """SPY forecast should be within sensible bounds."""
        mi = MarketIntelligence()
        report = mi.analyze(["SPY"], horizon="7D")

        current = report.current_prices.get("SPY", 0)
        if current > 0:
            assert validate_forecast_sensibility(report.forecast, current)

    def test_quick_forecast(self):
        """Quick forecast returns valid DataFrame."""
        mi = MarketIntelligence()
        fc = mi.quick_forecast("SPY", horizon="7D")

        assert isinstance(fc, pd.DataFrame)
        assert "forecast" in fc.columns
        assert len(fc) == 7
        assert not fc.isna().any().any()

    def test_get_regime(self):
        """Regime-only analysis works."""
        mi = MarketIntelligence()
        regime = mi.get_regime(["SPY"])

        assert regime.current_regime in ("bull", "bear", "crisis", "sideways")
        assert 0.0 <= regime.confidence <= 1.0

    def test_compare_assets(self):
        """Multi-asset comparison works."""
        mi = MarketIntelligence()
        comparison = mi.compare_assets(["SPY", "QQQ"], horizon="7D")

        assert comparison.assets == ["SPY", "QQQ"]
        if comparison.correlations is not None:
            assert comparison.correlations.shape == (2, 2)

    def test_markdown_export(self):
        """Markdown export produces valid content."""
        mi = MarketIntelligence()
        report = mi.analyze(["SPY"], horizon="7D")
        md = report.to_markdown()

        assert "# Market Intelligence Report" in md
        assert "## Executive Summary" in md
        assert "SPY" in md

    def test_dict_export(self):
        """Dict export works with real data."""
        mi = MarketIntelligence()
        report = mi.analyze(["SPY"], horizon="7D")
        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["regime"]["current"] in ("bull", "bear", "crisis", "sideways")
        assert d["quality_score"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
