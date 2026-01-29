"""
MarketIntelligence - Main Orchestrator.

This is the top-level entry point that coordinates the three layers:
    1. RegimeAnalyzer  → detect current market context
    2. AutoPilot       → select the best analytical pipeline
    3. Explainer       → compile results into an actionable report

The orchestrator fetches data through DataHub, runs each layer in order,
validates outputs, and returns a fully-populated IntelligenceReport.

Usage:
    mi = MarketIntelligence()
    report = mi.analyze(["BTC-USD"], horizon="7D")
    print(report.summary)
    print(report.to_markdown())
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from timeseries_toolkit.data_sources.hub import DataHub
from timeseries_toolkit.intelligence.autopilot import AutoPilot
from timeseries_toolkit.intelligence.explainer import (
    CausalityResult,
    ComparisonReport,
    Explainer,
    IntelligenceReport,
)
from timeseries_toolkit.intelligence.regime_analyzer import RegimeAnalyzer, RegimeResult


class MarketIntelligence:
    """
    Unified market analysis system combining regime detection,
    automatic pipeline selection, and explainable forecasting.

    This system runs entirely locally using free APIs (Yahoo Finance,
    FRED, alternative.me).

    Example::

        mi = MarketIntelligence()

        # Full analysis
        report = mi.analyze(["BTC-USD", "ETH-USD"], horizon="7D")
        print(report.summary)

        # Quick forecast only
        fc = mi.quick_forecast("SPY", horizon="7D")

        # Regime only
        regime = mi.get_regime()
    """

    # Map horizon strings to integer days.
    _HORIZON_MAP = {"1D": 1, "7D": 7, "14D": 14, "30D": 30}

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize MarketIntelligence.

        Args:
            fred_api_key: Optional FRED API key for macro data.
                         If not provided, reads from FRED_API_KEY env var.
                         Macro features are disabled if unavailable.
        """
        self.data_hub = DataHub(fred_api_key=fred_api_key)
        self.regime_analyzer = RegimeAnalyzer()
        self.autopilot = AutoPilot()
        self.explainer = Explainer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        assets: List[str],
        horizon: str = "7D",
        include_drivers: bool = True,
        include_regime: bool = True,
        confidence_level: float = 0.95,
        verbose: bool = False,
    ) -> IntelligenceReport:
        """
        Run complete market intelligence analysis.

        Steps:
        1. Fetch price data via DataHub.
        2. Detect market regime (if requested).
        3. Select the best pipeline via AutoPilot.
        4. Fit pipeline and generate forecast.
        5. Optionally run causal analysis.
        6. Compile into IntelligenceReport.

        Args:
            assets: Asset symbols (e.g. ``["BTC-USD", "SPY"]``).
            horizon: Forecast horizon (``"1D"``, ``"7D"``, ``"14D"``, ``"30D"``).
            include_drivers: Run causal analysis for key drivers.
            include_regime: Detect market regime.
            confidence_level: For prediction intervals (default 95%).
            verbose: Print progress.

        Returns:
            Fully populated IntelligenceReport.
        """
        horizon_days = self._HORIZON_MAP.get(horizon, 7)

        # --- Step 1: Fetch data -------------------------------------------
        if verbose:
            print(f"[1/5] Fetching data for {assets}...")

        prices_df = self._fetch_prices(assets)
        if prices_df.empty:
            # Return minimal report if data unavailable.
            report = IntelligenceReport(assets=assets, horizon=horizon)
            report.summary = "Unable to fetch price data."
            report.warnings = ["Data fetch failed for all assets."]
            return report

        # Use the first asset as the primary series for analysis.
        primary = prices_df.iloc[:, 0]
        current_prices = {
            col: float(prices_df[col].dropna().iloc[-1])
            for col in prices_df.columns
        }

        # --- Step 2: Regime detection -------------------------------------
        regime_result: Optional[RegimeResult] = None
        regime_label = "unknown"

        if include_regime:
            if verbose:
                print("[2/5] Detecting market regime...")
            try:
                regime_result = self.regime_analyzer.detect(prices_df)
                regime_label = regime_result.current_regime
            except Exception as e:
                warnings.warn(f"Regime detection failed: {e}")

        # --- Step 3: Pipeline selection -----------------------------------
        if verbose:
            print("[3/5] Selecting pipeline...")

        pipeline, pipeline_reason = self.autopilot.select_pipeline(
            data=primary, regime=regime_label, horizon=horizon_days
        )

        # --- Step 4: Fit and forecast -------------------------------------
        if verbose:
            print(f"[4/5] Running {pipeline.name} pipeline...")

        try:
            pipeline.fit(primary)
            forecast_df = pipeline.predict(horizon_days, confidence_level)
        except Exception as e:
            warnings.warn(f"Pipeline fit/predict failed: {e}")
            forecast_df = pd.DataFrame(columns=["forecast", "lower", "upper"])

        diagnostics = pipeline.get_diagnostics()

        # --- Step 5: Causal drivers (optional) ----------------------------
        drivers: Optional[CausalityResult] = None
        if include_drivers and len(assets) > 1:
            if verbose:
                print("[5/5] Analysing causal drivers...")
            try:
                drivers = self._run_causal_analysis(prices_df)
            except Exception:
                pass
        elif verbose:
            print("[5/5] Skipping driver analysis (single asset).")

        # --- Step 6: Compile report ---------------------------------------
        data_range = ""
        if not prices_df.empty:
            data_range = (
                f"{prices_df.index[0].strftime('%Y-%m-%d')} to "
                f"{prices_df.index[-1].strftime('%Y-%m-%d')}"
            )

        report = self.explainer.generate_report(
            forecast=forecast_df,
            regime=regime_result,
            drivers=drivers,
            diagnostics=diagnostics,
            pipeline_name=pipeline.name,
            pipeline_reason=pipeline_reason,
            assets=assets,
            horizon=horizon,
            current_prices=current_prices,
            data_range=data_range,
            observations_used=len(prices_df),
        )

        return report

    def quick_forecast(
        self,
        asset: str,
        horizon: str = "7D",
    ) -> pd.DataFrame:
        """
        Quick forecast without full analysis.

        Returns just the prediction DataFrame (forecast, lower, upper).
        Skips regime detection and driver analysis for speed.
        """
        report = self.analyze(
            assets=[asset],
            horizon=horizon,
            include_drivers=False,
            include_regime=False,
        )
        return report.to_dataframe()

    def get_regime(
        self,
        assets: Optional[List[str]] = None,
        lookback: str = "1Y",
    ) -> RegimeResult:
        """
        Get current market regime without forecasting.

        If assets not specified, uses broad market indicators (SPY, VIX).
        """
        if assets is None:
            assets = ["SPY"]

        # Fetch prices.
        try:
            prices = self.data_hub.equities.get_prices(assets, period=lookback)
        except Exception:
            prices = self._fetch_prices(assets, period=lookback)

        if prices.empty:
            return RegimeResult()

        return self.regime_analyzer.detect(prices)

    def compare_assets(
        self,
        assets: List[str],
        horizon: str = "7D",
    ) -> ComparisonReport:
        """
        Compare multiple assets with relative analysis.

        Runs individual analysis for each asset and computes correlations.
        """
        report = ComparisonReport(assets=assets)

        # Fetch all prices for correlations.
        prices_df = self._fetch_prices(assets)
        if not prices_df.empty and prices_df.shape[1] > 1:
            returns = prices_df.pct_change().dropna()
            report.correlations = returns.corr()
            # Relative strength: cumulative return over last 30 days.
            recent = returns.tail(30)
            report.relative_strength = pd.DataFrame({
                "cumulative_return": (1 + recent).prod() - 1
            })

        # Individual reports.
        for asset in assets:
            try:
                r = self.analyze([asset], horizon=horizon, include_drivers=False)
                report.individual_reports[asset] = r
            except Exception:
                pass

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_prices(
        self, assets: List[str], period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch closing prices for all assets.

        Determines the right loader (crypto vs equity) based on the ticker.
        Returns a DataFrame with one column per asset (Close prices).
        """
        frames = {}
        for asset in assets:
            try:
                if asset.endswith("-USD") or asset.lower() in ("bitcoin", "ethereum"):
                    df = self.data_hub.crypto.get_prices([asset], period=period)
                else:
                    df = self.data_hub.equities.get_prices([asset], period=period)

                # Extract Close column.
                close_col = None
                for col in df.columns:
                    if "close" in str(col).lower():
                        close_col = col
                        break
                if close_col is not None:
                    frames[asset] = df[close_col]
                elif len(df.columns) > 0:
                    frames[asset] = df.iloc[:, 0]
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames).sort_index().dropna(how="all")
        return result

    @staticmethod
    def _run_causal_analysis(prices_df: pd.DataFrame) -> CausalityResult:
        """
        Run Granger causality between asset pairs.

        Only runs if there are at least 2 assets with enough data.
        """
        from timeseries_toolkit.validation.causality import granger_causality_test

        returns = prices_df.pct_change().dropna()
        if returns.shape[1] < 2 or len(returns) < 30:
            return CausalityResult()

        relationships = []
        cols = list(returns.columns)

        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if i == j:
                    continue
                try:
                    result = granger_causality_test(
                        returns, target_col=tgt, source_cols=src, max_lags=3
                    )
                    improvement = result.get("improvement_pct", 0)
                    if improvement > 5:
                        relationships.append({
                            "source": src,
                            "target": tgt,
                            "strength": improvement / 100.0,
                            "improvement_pct": improvement,
                        })
                except Exception:
                    pass

        return CausalityResult(relationships=relationships)
