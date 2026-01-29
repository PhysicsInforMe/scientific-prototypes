"""
MarketIntelligence - Unified market analysis system.

Combines regime detection, automatic pipeline selection,
and explainable forecasting into a single orchestrated system.

Architecture:
    Layer 1: RegimeAnalyzer  - Context detection (bull/bear/crisis/sideways)
    Layer 2: AutoPilot       - Automatic pipeline selection
    Layer 3: Explainer       - Human-readable report generation
"""

from timeseries_toolkit.intelligence.market_intelligence import MarketIntelligence
from timeseries_toolkit.intelligence.regime_analyzer import RegimeAnalyzer, RegimeResult
from timeseries_toolkit.intelligence.autopilot import (
    AutoPilot,
    Pipeline,
    PipelineRegistry,
    DataCharacteristics,
)
from timeseries_toolkit.intelligence.explainer import (
    Explainer,
    IntelligenceReport,
    ComparisonReport,
)
from timeseries_toolkit.intelligence.backtester import Backtester, BacktestResult

__all__ = [
    "MarketIntelligence",
    "RegimeAnalyzer",
    "RegimeResult",
    "AutoPilot",
    "Pipeline",
    "PipelineRegistry",
    "DataCharacteristics",
    "Explainer",
    "IntelligenceReport",
    "ComparisonReport",
    "Backtester",
    "BacktestResult",
]
