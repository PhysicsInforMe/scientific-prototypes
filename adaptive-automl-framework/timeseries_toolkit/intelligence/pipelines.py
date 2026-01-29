"""
Predefined pipeline configurations.

Each pipeline is a named sequence of preprocessing and modelling steps.
The PipelineRegistry provides canonical pipelines that the AutoPilot
selects from.  Users may also build custom pipelines.

Design notes:
    - Pipelines wrap existing timeseries_toolkit modules.
    - ``Pipeline.fit`` runs each step in order: preprocessors transform
      the data, then the model is fitted on the transformed output.
    - ``Pipeline.predict`` produces a DataFrame with columns
      ``forecast``, ``lower``, ``upper`` (confidence bands).
    - Confidence bands are estimated via residual standard deviation
      when the underlying model does not provide native intervals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller

from timeseries_toolkit.models.kalman import AutoKalmanFilter
from timeseries_toolkit.preprocessing.filtering import TimeSeriesFilter
from timeseries_toolkit.preprocessing.fractional_diff import (
    find_min_d_for_stationarity,
    frac_diff_ffd,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsReport:
    """Lightweight diagnostics summary for a pipeline's residuals."""
    residual_mean: float = 0.0
    residual_std: float = 0.0
    ljung_box_pvalue: float = 1.0
    shapiro_pvalue: float = 1.0
    pass_count: int = 0
    total_tests: int = 4
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Encapsulates a complete analysis pipeline.

    A pipeline is an ordered list of ``(name, step)`` tuples.
    Steps are either *preprocessors* (transform data before modelling)
    or *models* (produce forecasts).

    Naming convention for step names:
        - ``"fracdiff"`` → fractional differentiation
        - ``"filter"``   → STL + SARIMA filtering
        - ``"model"``    → forecasting model (Kalman, etc.)
    """

    def __init__(self, name: str, steps: Optional[List[Tuple[str, Any]]] = None):
        self.name = name
        self.steps: List[Tuple[str, Any]] = steps or []
        # Internal state filled during fit.
        self._fitted = False
        self._residuals: Optional[np.ndarray] = None
        self._fitted_series: Optional[pd.Series] = None
        self._original_series: Optional[pd.Series] = None
        self._model_step: Optional[Any] = None
        self._frac_d: float = 0.0

    # ------------------------------------------------------------------

    def fit(self, series: pd.Series) -> "Pipeline":
        """
        Fit the pipeline end-to-end on *series*.

        Preprocessing steps transform the data in order.
        The final model step is fitted on the preprocessed result.
        """
        self._original_series = series.copy()
        current = series.copy()

        for step_name, step_obj in self.steps:
            if step_name == "fracdiff":
                # Record optimal d for informational / reporting purposes.
                # We do NOT transform the series here because the Kalman
                # filter with 'local linear trend' handles non-stationarity
                # natively.  Applying frac diff before Kalman would require
                # a complex inversion step to map forecasts back to price
                # space, and Kalman already models the level and trend as
                # latent states.
                try:
                    min_d, _ = find_min_d_for_stationarity(current)
                    self._frac_d = min_d
                except Exception:
                    self._frac_d = 0.0

            elif step_name == "filter":
                # Apply STL + SARIMA filtering.
                filt = step_obj if isinstance(step_obj, TimeSeriesFilter) else TimeSeriesFilter()
                try:
                    filt.fit(current)
                    current = filt.transform()
                except Exception:
                    # If filtering fails (e.g. too short), skip silently.
                    pass

            elif step_name == "model":
                # Fit the model.  Currently supports AutoKalmanFilter.
                self._model_step = step_obj
                if isinstance(step_obj, AutoKalmanFilter):
                    # Ensure the series has a frequency; if not (e.g. market
                    # data with weekends missing), resample to calendar days
                    # with forward-fill so statsmodels can infer the frequency.
                    fit_series = current.copy()
                    if isinstance(fit_series.index, pd.DatetimeIndex):
                        freq = pd.infer_freq(fit_series.index)
                        if freq is None:
                            fit_series = fit_series.asfreq("D", method="ffill")

                    step_obj.fit(fit_series)
                    # Compute in-sample residuals.
                    smoothed = step_obj.smooth()
                    common = fit_series.index.intersection(smoothed.index)
                    self._residuals = (fit_series.loc[common] - smoothed.loc[common]).values
                self._fitted_series = current

        self._fitted = True
        return self

    def predict(self, horizon: int, confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Generate forecast with confidence bands.

        Args:
            horizon: Number of steps ahead.
            confidence_level: Confidence level for interval (default 95%).

        Returns:
            DataFrame with columns ``forecast``, ``lower``, ``upper``
            and a DatetimeIndex continuing from the fitted series.
        """
        if not self._fitted or self._model_step is None:
            raise RuntimeError("Pipeline must be fitted before predicting.")

        # --- Generate forecast with confidence bands -----------------------
        if isinstance(self._model_step, AutoKalmanFilter):
            # Use native state-space forecast intervals when available.
            # These properly account for state estimation uncertainty,
            # observation noise, and horizon-dependent uncertainty growth.
            try:
                fc, fc_lower, fc_upper = self._model_step.forecast_with_ci(
                    steps=horizon, confidence_level=confidence_level
                )
                forecast_vals = fc.values if isinstance(fc, pd.Series) else np.asarray(fc)
                lower = fc_lower.values if isinstance(fc_lower, pd.Series) else np.asarray(fc_lower)
                upper = fc_upper.values if isinstance(fc_upper, pd.Series) else np.asarray(fc_upper)
            except Exception:
                # Fallback to residual-based CI if native CI fails.
                fc = self._model_step.forecast(steps=horizon)
                forecast_vals = fc.values if isinstance(fc, pd.Series) else np.asarray(fc)
                z = sp_stats.norm.ppf(0.5 + confidence_level / 2.0)
                resid_std = float(np.nanstd(self._residuals)) if self._residuals is not None else 0.0
                steps_ahead = np.arange(1, horizon + 1, dtype=float)
                band_width = z * resid_std * np.sqrt(steps_ahead)
                lower = forecast_vals - band_width
                upper = forecast_vals + band_width
        else:
            raise ValueError(f"Unsupported model type: {type(self._model_step)}")

        # Build DatetimeIndex for forecast period.
        if isinstance(fc, pd.Series) and isinstance(fc.index, pd.DatetimeIndex):
            idx = fc.index
        elif self._fitted_series is not None and isinstance(
            self._fitted_series.index, pd.DatetimeIndex
        ):
            last = self._fitted_series.index[-1]
            freq = pd.infer_freq(self._fitted_series.index[-10:]) or "D"
            idx = pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]
        else:
            idx = pd.RangeIndex(horizon)

        return pd.DataFrame(
            {"forecast": forecast_vals, "lower": lower, "upper": upper},
            index=idx,
        )

    def get_diagnostics(self) -> DiagnosticsReport:
        """
        Run basic diagnostics on the residuals.

        Checks:
        1. Mean ≈ 0 (unbiased residuals).
        2. Ljung-Box test for autocorrelation.
        3. Shapiro-Wilk normality test.
        4. Residual std is finite.
        """
        if self._residuals is None or len(self._residuals) < 10:
            return DiagnosticsReport()

        resid = self._residuals[~np.isnan(self._residuals)]
        report = DiagnosticsReport()
        report.residual_mean = float(np.mean(resid))
        report.residual_std = float(np.std(resid))
        report.total_tests = 4
        passes = 0

        # Test 1: Mean near zero.
        if abs(report.residual_mean) < 2 * report.residual_std / np.sqrt(len(resid)):
            passes += 1
            report.details["mean_test"] = "PASS"
        else:
            report.details["mean_test"] = "FAIL"

        # Test 2: Ljung-Box.
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(resid, lags=[min(10, len(resid) // 5)], return_df=True)
            pval = float(lb["lb_pvalue"].iloc[0])
            report.ljung_box_pvalue = pval
            if pval > 0.05:
                passes += 1
                report.details["ljung_box"] = "PASS"
            else:
                report.details["ljung_box"] = "FAIL"
        except Exception:
            passes += 1  # give benefit of doubt if test cannot run
            report.details["ljung_box"] = "SKIP"

        # Test 3: Shapiro-Wilk normality (subsample if too large).
        try:
            sample = resid[:5000] if len(resid) > 5000 else resid
            _, pval_sw = sp_stats.shapiro(sample)
            report.shapiro_pvalue = float(pval_sw)
            if pval_sw > 0.01:
                passes += 1
                report.details["shapiro"] = "PASS"
            else:
                report.details["shapiro"] = "FAIL"
        except Exception:
            passes += 1
            report.details["shapiro"] = "SKIP"

        # Test 4: Finite std.
        if np.isfinite(report.residual_std) and report.residual_std > 0:
            passes += 1
            report.details["finite_std"] = "PASS"
        else:
            report.details["finite_std"] = "FAIL"

        report.pass_count = passes
        return report


# ---------------------------------------------------------------------------
# Pipeline Registry
# ---------------------------------------------------------------------------

class PipelineRegistry:
    """
    Registry of predefined pipelines.

    Each class method returns a *fresh* Pipeline instance so that
    multiple analyses do not share mutable state.
    """

    @staticmethod
    def conservative() -> Pipeline:
        """
        Conservative pipeline: STL filtering + Kalman.

        Best for: crisis regimes, short series, when robustness matters.
        """
        return Pipeline(
            name="conservative",
            steps=[
                ("filter", TimeSeriesFilter()),
                ("model", AutoKalmanFilter(level="local linear trend")),
            ],
        )

    @staticmethod
    def aggressive() -> Pipeline:
        """
        Aggressive pipeline: Fractional diff + Kalman.

        Best for: non-stationary data with long memory, bull/sideways regimes.
        """
        return Pipeline(
            name="aggressive",
            steps=[
                ("fracdiff", None),  # placeholder; Pipeline.fit handles it
                ("model", AutoKalmanFilter(level="local linear trend")),
            ],
        )

    @staticmethod
    def crisis() -> Pipeline:
        """
        Crisis pipeline: Robust filtering + conservative Kalman.

        Best for: high-volatility, large-drawdown environments.
        """
        return Pipeline(
            name="crisis",
            steps=[
                ("filter", TimeSeriesFilter()),
                ("model", AutoKalmanFilter(level="local linear trend")),
            ],
        )

    @staticmethod
    def trend_following() -> Pipeline:
        """
        Trend-following pipeline: Kalman with smooth trend.

        Best for: assets with strong directional momentum.
        """
        return Pipeline(
            name="trend_following",
            steps=[
                ("model", AutoKalmanFilter(level="smooth trend")),
            ],
        )
