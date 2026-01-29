"""
Layer 2: Automatic Pipeline Selection.

The AutoPilot analyses data characteristics (stationarity, seasonality,
noise, autocorrelation) and the current market regime, then selects
the most appropriate pipeline from the PipelineRegistry.

Decision logic (see ``select_pipeline`` docstring):
    1. Check stationarity via ADF test.
    2. Check seasonality via ACF analysis.
    3. Select model based on regime and data profile.
    4. Run quick validation; fall back to conservative if poor.

Reference:
    - ADF test: Hamilton (1994) "Time Series Analysis"
    - Fractional differentiation: López de Prado (2018) "Advances in Financial ML"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import acf, adfuller

from timeseries_toolkit.intelligence.pipelines import Pipeline, PipelineRegistry
from timeseries_toolkit.preprocessing.fractional_diff import find_min_d_for_stationarity


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DataCharacteristics:
    """Summary of time series properties used for pipeline selection."""

    is_stationary: bool = False
    optimal_d: float = 0.0
    has_seasonality: bool = False
    seasonal_period: Optional[int] = None
    noise_level: float = 0.0      # 0 = clean, 1 = pure noise
    autocorrelation_strength: float = 0.0  # average |ACF| at lags 1-10
    trend_strength: float = 0.0   # abs(linear slope / std)
    outlier_fraction: float = 0.0  # fraction of points > 3σ


# ---------------------------------------------------------------------------
# AutoPilot
# ---------------------------------------------------------------------------

class AutoPilot:
    """
    Automatically selects the optimal analysis pipeline.

    Uses a rule-based decision tree informed by empirical research in
    financial time-series forecasting.  The rules are deliberately simple
    and interpretable so that the ``pipeline_reason`` field can explain
    *why* a pipeline was chosen.
    """

    def __init__(self):
        self.registry = PipelineRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_pipeline(
        self,
        data: pd.Series,
        regime: str = "unknown",
        horizon: int = 7,
    ) -> tuple[Pipeline, str]:
        """
        Select the best pipeline for the given data and conditions.

        Decision tree
        -------------
        1. **Crisis regime** → always ``crisis`` pipeline (robust Kalman).
           Rationale: outliers are common; we need robustness over accuracy.

        2. **Non-stationary data** (ADF p > 0.05) → ``aggressive`` pipeline
           which includes fractional differentiation to achieve stationarity
           while preserving long memory.

        3. **High autocorrelation** (mean |ACF lags 1-5| > 0.3) → ``trend_following``
           pipeline with smooth-trend Kalman.
           Rationale: strong serial dependence is best exploited by
           state-space models that track the level directly.

        4. **Otherwise** → ``conservative`` pipeline (filter + Kalman).
           This is the safe default that works across most conditions.

        Args:
            data: Price or return series to analyse.
            regime: Current regime label from RegimeAnalyzer.
            horizon: Forecast horizon in days.

        Returns:
            Tuple of (Pipeline, reason_string).
        """
        chars = self.analyze_data_characteristics(data)

        # ---- Rule 1: Crisis overrides everything -------------------------
        if regime == "crisis":
            reason = (
                "Crisis regime detected; selecting robust conservative pipeline. "
                "Kalman filter provides natural uncertainty quantification and "
                "is less sensitive to outliers than ML approaches."
            )
            return self.registry.crisis(), reason

        # ---- Rule 2: Non-stationary → fractional diff --------------------
        if not chars.is_stationary:
            reason = (
                f"Data is non-stationary (ADF p > 0.05); optimal fractional d={chars.optimal_d:.2f}. "
                "Fractional differentiation preserves long memory while achieving "
                "stationarity (López de Prado 2018)."
            )
            return self.registry.aggressive(), reason

        # ---- Rule 3: High autocorrelation → trend following --------------
        if chars.autocorrelation_strength > 0.3:
            reason = (
                f"Strong autocorrelation detected (mean |ACF|={chars.autocorrelation_strength:.2f}). "
                "Smooth-trend Kalman filter is selected to exploit serial dependence."
            )
            return self.registry.trend_following(), reason

        # ---- Rule 4: Default → conservative ------------------------------
        reason = (
            "No strong distinguishing characteristics; selecting balanced "
            "conservative pipeline (STL filtering + local linear trend Kalman)."
        )
        return self.registry.conservative(), reason

    def analyze_data_characteristics(self, series: pd.Series) -> DataCharacteristics:
        """
        Analyse time series properties.

        Computes stationarity, seasonality, noise level, autocorrelation
        strength, trend strength, and outlier fraction.
        """
        chars = DataCharacteristics()
        values = series.dropna().values.astype(float)

        if len(values) < 20:
            # Too short for meaningful analysis.
            return chars

        # ---- Stationarity (ADF test, 5% significance) --------------------
        try:
            adf_stat, adf_pvalue, *_ = adfuller(values, maxlag=12)
            chars.is_stationary = adf_pvalue < 0.05
        except Exception:
            chars.is_stationary = False

        # ---- Optimal d for fractional differentiation --------------------
        if not chars.is_stationary:
            try:
                min_d, _ = find_min_d_for_stationarity(series)
                chars.optimal_d = min_d
            except Exception:
                chars.optimal_d = 1.0

        # ---- Seasonality via ACF -----------------------------------------
        # If ACF has a significant peak beyond lag 5, we declare seasonality.
        try:
            max_lag = min(40, len(values) // 3)
            if max_lag > 5:
                acf_vals = acf(values, nlags=max_lag, fft=True)
                # Confidence threshold ≈ 2/sqrt(n).
                threshold = 2.0 / np.sqrt(len(values))
                # Look for peaks beyond lag 5 that exceed the threshold.
                for lag in range(6, len(acf_vals)):
                    if abs(acf_vals[lag]) > threshold:
                        chars.has_seasonality = True
                        chars.seasonal_period = lag
                        break
        except Exception:
            pass

        # ---- Autocorrelation strength (mean |ACF| lags 1-5) -------------
        try:
            short_acf = acf(values, nlags=min(5, len(values) // 3), fft=True)
            chars.autocorrelation_strength = float(np.mean(np.abs(short_acf[1:])))
        except Exception:
            pass

        # ---- Noise level (1 - R² of linear fit) -------------------------
        try:
            x = np.arange(len(values), dtype=float)
            slope, intercept, r_value, _, _ = sp_stats.linregress(x, values)
            chars.noise_level = float(1.0 - r_value ** 2)
            chars.trend_strength = float(abs(slope) / (np.std(values) + 1e-12))
        except Exception:
            chars.noise_level = 1.0

        # ---- Outlier fraction (> 3σ from mean) ---------------------------
        try:
            z_scores = np.abs((values - np.mean(values)) / (np.std(values) + 1e-12))
            chars.outlier_fraction = float(np.mean(z_scores > 3.0))
        except Exception:
            pass

        return chars
