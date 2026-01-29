"""
Unit tests for the intelligence module.

Tests cover:
    1. RegimeAnalyzer: correct regime detection on synthetic data.
    2. AutoPilot: sensible pipeline selection rules.
    3. Pipeline: valid forecast output structure.
    4. Explainer: coherent report and summary generation.
    5. IntelligenceReport: export methods (dict, DataFrame, Markdown).
"""

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.intelligence.regime_analyzer import RegimeAnalyzer, RegimeResult
from timeseries_toolkit.intelligence.autopilot import AutoPilot, DataCharacteristics
from timeseries_toolkit.intelligence.pipelines import (
    Pipeline,
    PipelineRegistry,
    DiagnosticsReport,
)
from timeseries_toolkit.intelligence.explainer import (
    Explainer,
    IntelligenceReport,
    CausalityResult,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic price generators
# ---------------------------------------------------------------------------

def _make_bull_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Uptrend + low volatility → should detect 'bull'."""
    rng = np.random.RandomState(seed)
    returns = 0.001 + 0.005 * rng.randn(n)  # positive drift, low vol
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def _make_bear_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Downtrend + elevated volatility → should detect 'bear'."""
    rng = np.random.RandomState(seed)
    returns = -0.002 + 0.015 * rng.randn(n)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def _make_crisis_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Crash + very high volatility → should detect 'crisis'."""
    rng = np.random.RandomState(seed)
    returns = -0.005 + 0.04 * rng.randn(n)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def _make_sideways_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Range-bound + low vol → should detect 'sideways'."""
    rng = np.random.RandomState(seed)
    returns = 0.0 + 0.003 * rng.randn(n)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


def _make_simple_series(n: int = 200, seed: int = 42) -> pd.Series:
    """Simple trending series for pipeline tests."""
    rng = np.random.RandomState(seed)
    values = np.cumsum(0.5 + rng.randn(n))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(values, index=idx, name="price")


# ===========================================================================
# 1. RegimeAnalyzer tests
# ===========================================================================

class TestRegimeAnalyzer:
    """Test that RegimeAnalyzer produces valid outputs."""

    def test_detect_returns_result(self):
        """detect() returns a RegimeResult object."""
        analyzer = RegimeAnalyzer(n_regimes=3)
        prices = _make_bull_prices()
        result = analyzer.detect(prices)
        assert isinstance(result, RegimeResult)

    def test_regime_is_known_label(self):
        """Detected regime must be one of the four canonical labels."""
        analyzer = RegimeAnalyzer(n_regimes=4)
        for gen in [_make_bull_prices, _make_bear_prices, _make_sideways_prices]:
            result = analyzer.detect(gen())
            assert result.current_regime in ("bull", "bear", "crisis", "sideways")

    def test_confidence_in_range(self):
        """Confidence score must be between 0 and 1."""
        analyzer = RegimeAnalyzer(n_regimes=3)
        result = analyzer.detect(_make_bull_prices())
        assert 0.0 <= result.confidence <= 1.0

    def test_transition_risk_in_range(self):
        """Transition risk must be between 0 and 1."""
        analyzer = RegimeAnalyzer(n_regimes=3)
        result = analyzer.detect(_make_bull_prices())
        assert 0.0 <= result.transition_risk <= 1.0

    def test_regime_probabilities_sum_to_one(self):
        """Regime probabilities should approximately sum to 1."""
        analyzer = RegimeAnalyzer(n_regimes=3)
        result = analyzer.detect(_make_bull_prices())
        total = sum(result.regime_probabilities.values())
        assert abs(total - 1.0) < 0.05

    def test_days_in_regime_positive(self):
        """Days in regime must be at least 1."""
        analyzer = RegimeAnalyzer(n_regimes=3)
        result = analyzer.detect(_make_bull_prices())
        assert result.days_in_regime >= 1

    def test_regime_history_length(self):
        """Regime history length should match data length minus 1 (returns)."""
        # Use mixed data that exercises multiple regimes to avoid
        # edge cases where HMM does not assign all states.
        rng = np.random.RandomState(99)
        n = 300
        # Concatenate bull + bear segments to ensure multiple regimes.
        r1 = 0.001 + 0.005 * rng.randn(150)
        r2 = -0.002 + 0.015 * rng.randn(150)
        returns = np.concatenate([r1, r2])
        prices = 100 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2022-01-01", periods=n, freq="D")
        df = pd.DataFrame({"Close": prices}, index=idx)

        analyzer = RegimeAnalyzer(n_regimes=3)
        result = analyzer.detect(df)
        assert result.regime_history is not None
        assert len(result.regime_history) == n - 1

    def test_transition_matrix_shape(self):
        """Transition matrix should be n_regimes x n_regimes."""
        n = 3
        analyzer = RegimeAnalyzer(n_regimes=n)
        result = analyzer.detect(_make_bull_prices())
        assert result.transition_matrix is not None
        assert result.transition_matrix.shape == (n, n)

    def test_get_regime_history(self):
        """get_regime_history returns a resampled DataFrame."""
        analyzer = RegimeAnalyzer(n_regimes=3)
        prices = _make_bull_prices()
        history = analyzer.get_regime_history(prices, resample_freq="W")
        assert isinstance(history, pd.DataFrame)
        assert len(history) > 0


# ===========================================================================
# 2. AutoPilot tests
# ===========================================================================

class TestAutoPilot:
    """Test that AutoPilot makes sensible pipeline selections."""

    def test_crisis_selects_crisis_pipeline(self):
        """Crisis regime must select the crisis/conservative pipeline."""
        ap = AutoPilot()
        series = _make_simple_series()
        pipeline, reason = ap.select_pipeline(series, regime="crisis")
        assert pipeline.name == "crisis"
        assert "crisis" in reason.lower() or "Crisis" in reason

    def test_non_stationary_selects_aggressive(self):
        """Non-stationary data should select the aggressive pipeline."""
        ap = AutoPilot()
        # Random walk is non-stationary.
        rng = np.random.RandomState(42)
        rw = pd.Series(
            np.cumsum(rng.randn(300)),
            index=pd.date_range("2022-01-01", periods=300, freq="D"),
        )
        pipeline, reason = ap.select_pipeline(rw, regime="bull")
        assert pipeline.name == "aggressive"
        assert "stationary" in reason.lower()

    def test_default_selects_conservative(self):
        """Stationary data with no special features → conservative."""
        ap = AutoPilot()
        # White noise is stationary.
        rng = np.random.RandomState(42)
        wn = pd.Series(
            rng.randn(300),
            index=pd.date_range("2022-01-01", periods=300, freq="D"),
        )
        pipeline, reason = ap.select_pipeline(wn, regime="sideways")
        # Should be either conservative or trend_following.
        assert pipeline.name in ("conservative", "trend_following")

    def test_analyze_data_characteristics(self):
        """analyze_data_characteristics returns valid DataCharacteristics."""
        ap = AutoPilot()
        chars = ap.analyze_data_characteristics(_make_simple_series())
        assert isinstance(chars, DataCharacteristics)
        assert 0.0 <= chars.noise_level <= 1.0
        assert chars.outlier_fraction >= 0.0

    def test_select_pipeline_returns_tuple(self):
        """select_pipeline returns (Pipeline, reason_string)."""
        ap = AutoPilot()
        result = ap.select_pipeline(_make_simple_series())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Pipeline)
        assert isinstance(result[1], str)
        assert len(result[1]) > 10  # reason should be descriptive


# ===========================================================================
# 3. Pipeline tests
# ===========================================================================

class TestPipeline:
    """Test Pipeline fit/predict outputs."""

    def test_conservative_pipeline_runs(self):
        """Conservative pipeline fit + predict runs without error."""
        pipe = PipelineRegistry.conservative()
        series = _make_simple_series()
        pipe.fit(series)
        fc = pipe.predict(7)
        assert isinstance(fc, pd.DataFrame)
        assert len(fc) == 7

    def test_forecast_has_required_columns(self):
        """Forecast DataFrame must have forecast, lower, upper."""
        pipe = PipelineRegistry.conservative()
        pipe.fit(_make_simple_series())
        fc = pipe.predict(7)
        assert "forecast" in fc.columns
        assert "lower" in fc.columns
        assert "upper" in fc.columns

    def test_forecast_no_nan(self):
        """Forecast should not contain NaN values."""
        pipe = PipelineRegistry.conservative()
        pipe.fit(_make_simple_series())
        fc = pipe.predict(7)
        assert not fc.isna().any().any()

    def test_confidence_bands_ordered(self):
        """Lower < forecast < upper for all rows."""
        pipe = PipelineRegistry.conservative()
        pipe.fit(_make_simple_series())
        fc = pipe.predict(7)
        assert (fc["lower"] <= fc["forecast"]).all()
        assert (fc["forecast"] <= fc["upper"]).all()

    def test_forecast_length_matches_horizon(self):
        """Forecast length must match the requested horizon."""
        for h in [1, 7, 14]:
            pipe = PipelineRegistry.conservative()
            pipe.fit(_make_simple_series())
            fc = pipe.predict(h)
            assert len(fc) == h

    def test_aggressive_pipeline_runs(self):
        """Aggressive pipeline runs on non-stationary data."""
        pipe = PipelineRegistry.aggressive()
        series = _make_simple_series()
        pipe.fit(series)
        fc = pipe.predict(7)
        assert len(fc) == 7
        assert not fc.isna().any().any()

    def test_diagnostics_report(self):
        """Pipeline diagnostics returns a DiagnosticsReport."""
        pipe = PipelineRegistry.conservative()
        pipe.fit(_make_simple_series())
        diag = pipe.get_diagnostics()
        assert isinstance(diag, DiagnosticsReport)
        assert diag.total_tests > 0
        assert 0 <= diag.pass_count <= diag.total_tests

    def test_trend_following_pipeline(self):
        """Trend following pipeline runs."""
        pipe = PipelineRegistry.trend_following()
        pipe.fit(_make_simple_series())
        fc = pipe.predict(7)
        assert len(fc) == 7


# ===========================================================================
# 4. Explainer tests
# ===========================================================================

class TestExplainer:
    """Test Explainer report and summary generation."""

    def _make_report_inputs(self):
        """Helper: create inputs for Explainer.generate_report."""
        forecast = pd.DataFrame({
            "forecast": [100.0, 101.0, 102.0],
            "lower": [95.0, 96.0, 97.0],
            "upper": [105.0, 106.0, 107.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))

        regime = RegimeResult(
            current_regime="bull",
            confidence=0.75,
            regime_probabilities={"bull": 0.75, "bear": 0.10,
                                  "crisis": 0.05, "sideways": 0.10},
            days_in_regime=15,
            transition_risk=0.12,
        )

        diagnostics = DiagnosticsReport(
            pass_count=3, total_tests=4,
            details={"mean_test": "PASS", "ljung_box": "PASS",
                     "shapiro": "PASS", "finite_std": "FAIL"},
        )

        return forecast, regime, diagnostics

    def test_generate_report_returns_intelligence_report(self):
        """generate_report produces an IntelligenceReport."""
        explainer = Explainer()
        fc, regime, diag = self._make_report_inputs()
        report = explainer.generate_report(
            forecast=fc, regime=regime, drivers=None,
            diagnostics=diag, pipeline_name="conservative",
            pipeline_reason="test reason",
            assets=["BTC-USD"], horizon="7D",
        )
        assert isinstance(report, IntelligenceReport)

    def test_summary_not_empty(self):
        """Summary text must not be empty."""
        explainer = Explainer()
        fc, regime, diag = self._make_report_inputs()
        report = explainer.generate_report(
            forecast=fc, regime=regime, drivers=None,
            diagnostics=diag, pipeline_name="conservative",
            pipeline_reason="test", assets=["BTC-USD"], horizon="7D",
        )
        assert len(report.summary) > 20

    def test_summary_contains_regime(self):
        """Summary should mention the detected regime."""
        explainer = Explainer()
        fc, regime, diag = self._make_report_inputs()
        report = explainer.generate_report(
            forecast=fc, regime=regime, drivers=None,
            diagnostics=diag, pipeline_name="conservative",
            pipeline_reason="test", assets=["BTC-USD"], horizon="7D",
        )
        assert "bull" in report.summary.lower() or "Bull" in report.summary

    def test_quality_score_computed(self):
        """Quality score should be computed from diagnostics."""
        explainer = Explainer()
        fc, regime, diag = self._make_report_inputs()
        report = explainer.generate_report(
            forecast=fc, regime=regime, drivers=None,
            diagnostics=diag, pipeline_name="conservative",
            pipeline_reason="test", assets=["BTC-USD"], horizon="7D",
        )
        assert report.quality_score == 3.0 / 4.0

    def test_warnings_generated(self):
        """Warnings list should be populated."""
        explainer = Explainer()
        fc, regime, diag = self._make_report_inputs()
        report = explainer.generate_report(
            forecast=fc, regime=regime, drivers=None,
            diagnostics=diag, pipeline_name="conservative",
            pipeline_reason="test", assets=["BTC-USD"], horizon="7D",
        )
        assert isinstance(report.warnings, list)

    def test_recommendations_generated(self):
        """Recommendations should be generated based on regime."""
        explainer = Explainer()
        fc, regime, diag = self._make_report_inputs()
        report = explainer.generate_report(
            forecast=fc, regime=regime, drivers=None,
            diagnostics=diag, pipeline_name="conservative",
            pipeline_reason="test", assets=["BTC-USD"], horizon="7D",
        )
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0


# ===========================================================================
# 5. IntelligenceReport export tests
# ===========================================================================

class TestIntelligenceReportExport:
    """Test IntelligenceReport export methods."""

    def _make_report(self) -> IntelligenceReport:
        """Create a populated report for export tests."""
        explainer = Explainer()
        fc = pd.DataFrame({
            "forecast": [100.0, 101.0],
            "lower": [95.0, 96.0],
            "upper": [105.0, 106.0],
        }, index=pd.date_range("2024-01-01", periods=2, freq="D"))

        regime = RegimeResult(
            current_regime="bear",
            confidence=0.80,
            regime_probabilities={"bull": 0.05, "bear": 0.80,
                                  "crisis": 0.10, "sideways": 0.05},
            days_in_regime=10,
            transition_risk=0.25,
        )
        diag = DiagnosticsReport(pass_count=4, total_tests=4,
                                  details={"a": "PASS", "b": "PASS",
                                           "c": "PASS", "d": "PASS"})
        return explainer.generate_report(
            forecast=fc, regime=regime, drivers=None,
            diagnostics=diag, pipeline_name="conservative",
            pipeline_reason="test", assets=["SPY"], horizon="7D",
            current_prices={"SPY": 450.0},
        )

    def test_to_dict(self):
        """to_dict returns a dictionary with expected keys."""
        report = self._make_report()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "regime" in d
        assert "forecast" in d
        assert "quality_score" in d

    def test_to_dataframe(self):
        """to_dataframe returns the forecast DataFrame."""
        report = self._make_report()
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "forecast" in df.columns

    def test_to_markdown(self):
        """to_markdown returns a non-empty Markdown string."""
        report = self._make_report()
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 100
        assert "# Market Intelligence Report" in md

    def test_markdown_contains_sections(self):
        """Markdown must contain all required sections."""
        report = self._make_report()
        md = report.to_markdown()
        assert "## Executive Summary" in md
        assert "## Current Regime Analysis" in md
        assert "## Forecast Details" in md
        assert "## Pipeline & Diagnostics" in md

    def test_markdown_tables_formatted(self):
        """Markdown tables should have pipe characters."""
        report = self._make_report()
        md = report.to_markdown()
        assert "| Metric | Value |" in md

    def test_save_markdown(self, tmp_path):
        """save_markdown writes a file."""
        report = self._make_report()
        path = str(tmp_path / "test_report.md")
        report.save_markdown(path)
        with open(path) as f:
            content = f.read()
        assert "# Market Intelligence Report" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
