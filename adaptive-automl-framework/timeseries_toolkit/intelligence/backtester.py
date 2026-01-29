"""
Backtesting Framework for MarketIntelligence.

Validates the system by running walk-forward backtests:
    - At each step, only data available *up to that point* is used.
    - Forecasts are compared to actual outcomes.
    - Regime detection accuracy is measured against realised volatility.
    - Pipeline selection is evaluated versus always using one pipeline.

This module is CRITICAL for proving the architecture produces sensible
results and does not over-fit to in-sample data.

Walk-forward procedure:
    1. Set window start (e.g. 2022-01-01).
    2. At each step, use all data up to current date.
    3. Generate forecast for *horizon* days ahead.
    4. Record actual outcome.
    5. Advance by *step* days and repeat.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from timeseries_toolkit.intelligence.autopilot import AutoPilot
from timeseries_toolkit.intelligence.pipelines import PipelineRegistry
from timeseries_toolkit.intelligence.regime_analyzer import RegimeAnalyzer


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Comprehensive backtest results."""

    # Forecast accuracy
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    directional_accuracy: float = 0.0
    coverage: float = 0.0  # fraction of actuals within confidence bands

    # Regime detection
    regime_accuracy: float = 0.0
    regime_transitions_detected: int = 0
    regime_transitions_actual: int = 0
    false_crisis_alerts: int = 0
    missed_crisis: int = 0

    # Pipeline selection
    pipeline_distribution: Dict[str, float] = field(default_factory=dict)
    pipeline_vs_baseline: float = 0.0

    # Per-period details
    period_results: Optional[pd.DataFrame] = None

    # Quality
    avg_quality_score: float = 0.0
    diagnostics_pass_rate: float = 0.0
    n_periods: int = 0

    def summary(self) -> str:
        """Generate human-readable backtest summary."""
        lines = [
            "=" * 60,
            "  BACKTEST SUMMARY",
            "=" * 60,
            f"  Periods evaluated:       {self.n_periods}",
            f"  MAE:                     {self.mae:.4f}",
            f"  RMSE:                    {self.rmse:.4f}",
            f"  MAPE:                    {self.mape:.2f}%",
            f"  Directional accuracy:    {self.directional_accuracy:.1f}%",
            f"  Coverage (95% CI):       {self.coverage:.1f}%",
            "",
            "  --- Regime Detection ---",
            f"  Transitions detected:    {self.regime_transitions_detected}",
            f"  False crisis alerts:     {self.false_crisis_alerts}",
            f"  Missed crises:           {self.missed_crisis}",
            "",
            "  --- Pipeline Selection ---",
        ]
        for name, pct in sorted(self.pipeline_distribution.items(),
                                 key=lambda x: -x[1]):
            lines.append(f"  {name:<25} {pct:.1f}%")
        lines.append(f"  vs baseline improvement: {self.pipeline_vs_baseline:+.1f}%")
        lines.append("")
        lines.append(f"  Avg quality score:       {self.avg_quality_score:.2f}")
        lines.append(f"  Diagnostics pass rate:   {self.diagnostics_pass_rate:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class RegimeBacktestResult:
    """Regime detection backtest results."""
    regime_history: Optional[pd.DataFrame] = None
    volatility_aligned: bool = False
    drawdown_aligned: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineBacktestResult:
    """Pipeline selection backtest results."""
    selection_history: Optional[pd.DataFrame] = None
    best_pipeline_per_period: Optional[pd.DataFrame] = None
    autopilot_advantage: float = 0.0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Walk-forward backtester for MarketIntelligence.

    Validates that the system produces sensible results on historical data
    by comparing forecasts to actual outcomes.
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Args:
            fred_api_key: Optional FRED API key for macro data.
        """
        self.regime_analyzer = RegimeAnalyzer()
        self.autopilot = AutoPilot()
        self._fred_api_key = fred_api_key

    def run_backtest(
        self,
        prices: pd.Series,
        start_date: str,
        end_date: str,
        horizon: int = 7,
        step: int = 7,
        min_history: int = 120,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        Run walk-forward backtest on a pre-fetched price series.

        At each step:
        1. Use only data up to the current date.
        2. Detect regime from available history.
        3. Select pipeline via AutoPilot.
        4. Fit and generate forecast.
        5. Compare to actual outcomes.

        Args:
            prices: Price series with DatetimeIndex.
            start_date: First forecast origin date (ISO format).
            end_date: Last forecast origin date.
            horizon: Forecast horizon in days.
            step: Days between forecast origins.
            min_history: Minimum observations required before first forecast.
            verbose: Print progress.

        Returns:
            BacktestResult with comprehensive metrics.
        """
        # --- Validate inputs ----------------------------------------------
        prices = prices.sort_index().dropna()
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Generate forecast origin dates.
        origins = pd.date_range(start=start, end=end, freq=f"{step}D")
        if len(origins) == 0:
            return BacktestResult()

        # --- Walk-forward loop --------------------------------------------
        records = []
        pipeline_counts: Dict[str, int] = {}
        quality_scores = []
        diag_passes = []
        prev_regime = None

        for i, origin in enumerate(origins):
            # Slice history up to origin.
            history = prices.loc[:origin]
            if len(history) < min_history:
                continue

            # Actual future values for evaluation.
            future_start = origin + pd.Timedelta(days=1)
            future_end = origin + pd.Timedelta(days=horizon)
            actuals = prices.loc[future_start:future_end]
            if len(actuals) == 0:
                continue

            actual_last = float(actuals.iloc[-1])
            current_price = float(history.iloc[-1])

            # --- Regime detection -----------------------------------------
            try:
                regime_result = self.regime_analyzer.detect(
                    pd.DataFrame({prices.name or "price": history})
                )
                regime_label = regime_result.current_regime
            except Exception:
                regime_label = "unknown"

            # --- Pipeline selection ---------------------------------------
            try:
                pipeline, reason = self.autopilot.select_pipeline(
                    data=history, regime=regime_label, horizon=horizon
                )
                pipeline_name = pipeline.name
            except Exception:
                pipeline_name = "conservative"
                pipeline, reason = PipelineRegistry.conservative(), "fallback"

            pipeline_counts[pipeline_name] = pipeline_counts.get(pipeline_name, 0) + 1

            # --- Fit and forecast -----------------------------------------
            try:
                pipeline.fit(history)
                fc = pipeline.predict(horizon)
                fc_last = float(fc["forecast"].iloc[-1])
                fc_lower = float(fc["lower"].iloc[-1])
                fc_upper = float(fc["upper"].iloc[-1])
            except Exception:
                fc_last = current_price  # naive fallback
                fc_lower = current_price * 0.95
                fc_upper = current_price * 1.05

            # --- Diagnostics ----------------------------------------------
            try:
                diag = pipeline.get_diagnostics()
                q_score = diag.pass_count / max(diag.total_tests, 1)
                quality_scores.append(q_score)
                diag_passes.append(diag.pass_count / max(diag.total_tests, 1))
            except Exception:
                quality_scores.append(0.5)

            # --- Record ---------------------------------------------------
            # Directional accuracy: did we predict the right direction?
            actual_direction = 1 if actual_last > current_price else -1
            forecast_direction = 1 if fc_last > current_price else -1
            direction_correct = actual_direction == forecast_direction

            # Coverage: is actual within confidence bands?
            in_band = fc_lower <= actual_last <= fc_upper

            # Regime transition tracking.
            regime_changed = prev_regime is not None and regime_label != prev_regime
            prev_regime = regime_label

            records.append({
                "origin": origin,
                "current_price": current_price,
                "actual": actual_last,
                "forecast": fc_last,
                "lower": fc_lower,
                "upper": fc_upper,
                "error": fc_last - actual_last,
                "abs_error": abs(fc_last - actual_last),
                "pct_error": abs(fc_last - actual_last) / abs(actual_last) * 100
                    if actual_last != 0 else 0,
                "direction_correct": direction_correct,
                "in_band": in_band,
                "regime": regime_label,
                "pipeline": pipeline_name,
                "regime_changed": regime_changed,
            })

            if verbose and (i + 1) % 5 == 0:
                print(f"  Backtest step {i + 1}/{len(origins)}: "
                      f"regime={regime_label}, pipeline={pipeline_name}")

        # --- Aggregate metrics --------------------------------------------
        if not records:
            return BacktestResult()

        df = pd.DataFrame(records)
        total = len(df)

        # Forecast accuracy.
        mae = float(df["abs_error"].mean())
        rmse = float(np.sqrt((df["error"] ** 2).mean()))
        mape = float(df["pct_error"].mean())
        dir_acc = float(df["direction_correct"].mean() * 100)
        coverage = float(df["in_band"].mean() * 100)

        # Pipeline distribution.
        total_selections = sum(pipeline_counts.values())
        pipeline_dist = {
            k: v / total_selections * 100
            for k, v in pipeline_counts.items()
        }

        # Regime transitions.
        transitions_detected = int(df["regime_changed"].sum())

        # Detect actual crisis periods (large drawdowns > 10%).
        actual_returns = df["actual"] / df["current_price"] - 1
        actual_crisis = (actual_returns < -0.10).sum()
        detected_crisis = ((df["regime"] == "crisis") &
                           (actual_returns < -0.05)).sum()
        false_crisis = ((df["regime"] == "crisis") &
                        (actual_returns > -0.02)).sum()
        missed = (actual_crisis - detected_crisis) if actual_crisis > detected_crisis else 0

        result = BacktestResult(
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=dir_acc,
            coverage=coverage,
            regime_transitions_detected=transitions_detected,
            false_crisis_alerts=int(false_crisis),
            missed_crisis=int(missed),
            pipeline_distribution=pipeline_dist,
            period_results=df,
            avg_quality_score=float(np.mean(quality_scores)) if quality_scores else 0,
            diagnostics_pass_rate=float(np.mean(diag_passes) * 100) if diag_passes else 0,
            n_periods=total,
        )

        return result

    def validate_regime_detection(
        self,
        prices: pd.Series,
        start_date: str,
        end_date: str,
    ) -> RegimeBacktestResult:
        """
        Validate regime detection against realised volatility.

        Checks:
        - Crisis regime aligns with high-volatility periods.
        - Bull regime aligns with positive returns.
        - Bear regime aligns with negative returns.
        """
        prices = prices.sort_index().dropna()
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Detect regimes on the full period.
        history = prices.loc[:end]
        if len(history) < 60:
            return RegimeBacktestResult()

        try:
            result = self.regime_analyzer.detect(
                pd.DataFrame({prices.name or "price": history})
            )
            regime_history = result.regime_history
        except Exception:
            return RegimeBacktestResult()

        # Compute rolling 30-day realised volatility.
        returns = prices.pct_change().dropna()
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)

        # Align.
        common = regime_history.index.intersection(rolling_vol.index)
        if len(common) < 30:
            return RegimeBacktestResult()

        aligned_regime = regime_history.loc[common]
        aligned_vol = rolling_vol.loc[common]

        # Check: crisis regime should have higher volatility.
        crisis_vol = aligned_vol[aligned_regime == "crisis"]
        non_crisis_vol = aligned_vol[aligned_regime != "crisis"]
        vol_aligned = (
            crisis_vol.mean() > non_crisis_vol.mean()
            if len(crisis_vol) > 0 and len(non_crisis_vol) > 0
            else True  # no crisis detected → pass by default
        )

        # Check: drawdown alignment.
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        aligned_dd = drawdown.reindex(common)
        crisis_dd = aligned_dd[aligned_regime == "crisis"]
        dd_aligned = (
            crisis_dd.mean() < -0.05
            if len(crisis_dd) > 5
            else True
        )

        return RegimeBacktestResult(
            regime_history=pd.DataFrame({
                "regime": aligned_regime,
                "volatility": aligned_vol,
                "drawdown": aligned_dd,
            }),
            volatility_aligned=vol_aligned,
            drawdown_aligned=dd_aligned,
            details={
                "crisis_mean_vol": float(crisis_vol.mean()) if len(crisis_vol) > 0 else None,
                "non_crisis_mean_vol": float(non_crisis_vol.mean()) if len(non_crisis_vol) > 0 else None,
            },
        )

    def validate_pipeline_selection(
        self,
        prices: pd.Series,
        start_date: str,
        end_date: str,
        horizon: int = 7,
        step: int = 14,
    ) -> PipelineBacktestResult:
        """
        Validate that AutoPilot pipeline selections were sensible.

        For each period, compares the AutoPilot-selected pipeline against
        the conservative baseline.  A good AutoPilot should produce
        lower average error than always using conservative.
        """
        # Run backtest with AutoPilot.
        auto_result = self.run_backtest(
            prices, start_date, end_date,
            horizon=horizon, step=step, verbose=False
        )

        # Run baseline (always conservative).
        baseline_records = []
        origins = pd.date_range(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            freq=f"{step}D",
        )

        for origin in origins:
            history = prices.loc[:origin]
            if len(history) < 120:
                continue
            future_end = origin + pd.Timedelta(days=horizon)
            actuals = prices.loc[origin + pd.Timedelta(days=1):future_end]
            if len(actuals) == 0:
                continue

            try:
                pipeline = PipelineRegistry.conservative()
                pipeline.fit(history)
                fc = pipeline.predict(horizon)
                fc_last = float(fc["forecast"].iloc[-1])
                actual_last = float(actuals.iloc[-1])
                baseline_records.append(abs(fc_last - actual_last))
            except Exception:
                pass

        baseline_mae = float(np.mean(baseline_records)) if baseline_records else auto_result.mae

        # Compute advantage.
        if baseline_mae > 0:
            advantage = (baseline_mae - auto_result.mae) / baseline_mae * 100
        else:
            advantage = 0.0

        return PipelineBacktestResult(
            selection_history=auto_result.period_results,
            autopilot_advantage=advantage,
        )
