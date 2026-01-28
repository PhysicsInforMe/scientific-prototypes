"""
Time Series Filtering Module.

This module provides a two-stage filtering approach combining STL decomposition
with SARIMA modeling to extract signal from noisy time series data.
"""

import itertools
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore", category=UserWarning)


def _infer_seasonal_period(series: pd.Series) -> Tuple[str, int]:
    """
    Infer the seasonal period from a time series based on its frequency.

    Args:
        series: Time series with DatetimeIndex.

    Returns:
        Tuple of (frequency_name, seasonal_period).
    """
    freq = pd.infer_freq(series.index)

    if freq is None:
        # Infer from average time delta
        if len(series.index) < 2:
            return "Unknown", 1
        avg_delta = (series.index[1:] - series.index[:-1]).mean()
        if avg_delta <= pd.Timedelta(days=1.5):
            return "Daily", 7
        elif avg_delta <= pd.Timedelta(days=8):
            return "Weekly", 52
        elif avg_delta <= pd.Timedelta(days=32):
            return "Monthly", 12
        elif avg_delta <= pd.Timedelta(days=93):
            return "Quarterly", 4
        else:
            return "Unknown", 1

    freq_code = freq.split('-')[0]
    if 'D' in freq_code or 'B' in freq_code:
        return "Daily", 7
    elif 'W' in freq_code:
        return "Weekly", 52
    elif 'M' in freq_code:
        return "Monthly", 12
    elif 'Q' in freq_code:
        return "Quarterly", 4
    else:
        return f"Unknown ({freq})", 1


def _find_best_sarima(
    data: pd.Series,
    seasonal_period: int,
    max_p: int = 2,
    max_q: int = 2,
    max_P: int = 1,
    max_Q: int = 1
) -> Tuple[Optional[Any], Optional[Tuple], Optional[Tuple]]:
    """
    Grid search to find the best SARIMA model based on AIC.

    Args:
        data: Time series data to model.
        seasonal_period: Seasonal period for SARIMA.
        max_p: Maximum AR order.
        max_q: Maximum MA order.
        max_P: Maximum seasonal AR order.
        max_Q: Maximum seasonal MA order.

    Returns:
        Tuple of (best_model, best_order, best_seasonal_order).
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None

    p_range = range(max_p + 1)
    q_range = range(max_q + 1)
    P_range = range(max_P + 1)
    Q_range = range(max_Q + 1)

    orders = list(itertools.product(p_range, [0], q_range))
    seasonal_orders = list(itertools.product(P_range, [0], Q_range, [seasonal_period]))

    for order in orders:
        for s_order in seasonal_orders:
            # Skip trivial model
            if order == (0, 0, 0) and s_order[:3] == (0, 0, 0):
                continue
            try:
                model = SARIMAX(
                    data,
                    order=order,
                    seasonal_order=s_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = order
                    best_seasonal_order = s_order
                    best_model = model
            except Exception:
                continue

    return best_model, best_order, best_seasonal_order


def _compute_sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute sample entropy of a time series.

    Sample entropy measures the complexity/regularity of a time series.
    Lower values indicate more regularity.

    Args:
        data: Input time series as numpy array.
        m: Embedding dimension.
        r: Tolerance (as fraction of std).

    Returns:
        Sample entropy value.
    """
    n = len(data)
    if n < m + 1:
        return np.nan

    # Normalize r to data std
    r_val = r * np.std(data)
    if r_val == 0:
        return np.nan

    def _count_matches(template_len: int) -> int:
        """Count matching template pairs."""
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                # Check if templates match within tolerance
                if np.max(np.abs(data[i:i + template_len] - data[j:j + template_len])) <= r_val:
                    count += 1
        return count

    # Count matches for length m and m+1
    b = _count_matches(m)
    a = _count_matches(m + 1)

    if b == 0 or a == 0:
        return np.nan

    return -np.log(a / b)


class TimeSeriesFilter:
    """
    Two-stage time series filter combining STL decomposition with SARIMA.

    This filter extracts the underlying signal from noisy time series data
    using a two-stage approach:
    1. STL (Seasonal-Trend decomposition using LOESS) to extract residuals
    2. SARIMA modeling of residuals to capture remaining autocorrelation

    Attributes:
        seasonal_period: Detected or specified seasonal period.
        frequency_name: Human-readable frequency description.
        is_fitted: Whether the filter has been fitted.
        sarima_order: The (p, d, q) order of the fitted SARIMA model.
        sarima_seasonal_order: The (P, D, Q, m) seasonal order.

    Example:
        >>> filter = TimeSeriesFilter()
        >>> filter.fit(monthly_series)
        >>> filtered = filter.transform(monthly_series)
        >>> metrics = filter.get_metrics()
        >>> print(f"SNR: {metrics['snr_db']:.2f} dB")
    """

    def __init__(self, seasonal_period: Optional[int] = None):
        """
        Initialize the TimeSeriesFilter.

        Args:
            seasonal_period: Override automatic period detection. If None,
                period is inferred from the series frequency.
        """
        self._seasonal_period_override = seasonal_period
        self.seasonal_period: Optional[int] = None
        self.frequency_name: Optional[str] = None
        self.is_fitted: bool = False
        self.sarima_order: Optional[Tuple[int, int, int]] = None
        self.sarima_seasonal_order: Optional[Tuple[int, int, int, int]] = None

        self._sarima_model: Optional[Any] = None
        self._stl_result: Optional[Any] = None
        self._original_series: Optional[pd.Series] = None
        self._filtered_series: Optional[pd.Series] = None
        self._residuals: Optional[pd.Series] = None

    def fit(
        self,
        series: pd.Series,
        max_p: int = 2,
        max_q: int = 2,
        max_P: int = 1,
        max_Q: int = 1
    ) -> 'TimeSeriesFilter':
        """
        Fit the two-stage filter to a time series.

        Args:
            series: Time series with DatetimeIndex.
            max_p: Maximum AR order for SARIMA grid search.
            max_q: Maximum MA order for SARIMA grid search.
            max_P: Maximum seasonal AR order.
            max_Q: Maximum seasonal MA order.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If series is too short or has invalid index.
        """
        if series.empty:
            raise ValueError("Input series cannot be empty")

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")

        # Store original series
        self._original_series = series.copy()

        # Determine seasonal period
        if self._seasonal_period_override is not None:
            self.seasonal_period = self._seasonal_period_override
            self.frequency_name = "Custom"
        else:
            self.frequency_name, self.seasonal_period = _infer_seasonal_period(series)

        if len(series) < 2 * self.seasonal_period:
            raise ValueError(
                f"Series too short for seasonal period {self.seasonal_period}. "
                f"Need at least {2 * self.seasonal_period} observations."
            )

        # Stage 1: STL decomposition
        self._stl_result = STL(
            series,
            period=self.seasonal_period,
            robust=True
        ).fit()

        stl_residuals = self._stl_result.resid.dropna()

        # Stage 2: SARIMA on residuals
        self._sarima_model, self.sarima_order, self.sarima_seasonal_order = _find_best_sarima(
            stl_residuals,
            self.seasonal_period,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q
        )

        if self._sarima_model is None:
            raise ValueError("Could not fit any SARIMA model to the residuals")

        # Compute final residuals (trimmed to avoid edge effects)
        initial_residuals = self._sarima_model.resid
        self._residuals = initial_residuals.iloc[self.seasonal_period:]

        # Compute filtered series
        self._filtered_series = series - self._residuals
        self._filtered_series = self._filtered_series.fillna(series)

        self.is_fitted = True
        return self

    def transform(self, series: Optional[pd.Series] = None) -> pd.Series:
        """
        Extract the filtered signal from a time series.

        Args:
            series: Time series to filter. If None, uses the series from fit().

        Returns:
            Filtered time series with noise removed.

        Raises:
            ValueError: If filter has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Filter must be fitted before transform")

        if series is None:
            return self._filtered_series.copy()

        # Apply STL + SARIMA to new series
        stl_result = STL(
            series,
            period=self.seasonal_period,
            robust=True
        ).fit()

        stl_residuals = stl_result.resid.dropna()

        # Predict using fitted SARIMA model
        sarima_residuals = self._sarima_model.predict(
            start=0,
            end=len(stl_residuals) - 1
        )

        # Compute filtered series
        filtered = series - sarima_residuals
        return filtered.fillna(series)

    def get_residuals(self) -> pd.Series:
        """
        Get the residuals (noise) from the fitted filter.

        Returns:
            Series containing the residuals.

        Raises:
            ValueError: If filter has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Filter must be fitted before getting residuals")
        return self._residuals.copy()

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute quality metrics for the filter.

        Returns:
            Dictionary containing:
                - snr_db: Signal-to-noise ratio in decibels
                - variance_original: Variance of original series
                - variance_filtered: Variance of filtered series
                - variance_residuals: Variance of residuals
                - entropy_original: Sample entropy of original series
                - entropy_filtered: Sample entropy of filtered series
                - signal_dominance_index: Percentage of signal variance (IDS)

        Raises:
            ValueError: If filter has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Filter must be fitted before computing metrics")

        var_original = np.var(self._original_series)
        var_filtered = np.var(self._filtered_series)
        var_residuals = np.var(self._residuals)

        # Signal-to-Noise Ratio (SNR) in dB
        if var_residuals > 0:
            snr_db = 10 * np.log10(var_filtered / var_residuals)
        else:
            snr_db = np.inf

        # Signal Dominance Index (IDS)
        if var_original > 0:
            signal_dominance = (1 - var_residuals / var_original) * 100
        else:
            signal_dominance = np.nan

        # Sample entropy
        entropy_original = _compute_sample_entropy(self._original_series.values)
        entropy_filtered = _compute_sample_entropy(self._filtered_series.values)

        return {
            'snr_db': snr_db,
            'variance_original': var_original,
            'variance_filtered': var_filtered,
            'variance_residuals': var_residuals,
            'entropy_original': entropy_original,
            'entropy_filtered': entropy_filtered,
            'signal_dominance_index': signal_dominance,
        }

    def get_ljung_box_test(self, lags: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Ljung-Box test on residuals to check for white noise.

        Args:
            lags: Number of lags to test. Defaults to seasonal period.

        Returns:
            Dictionary containing:
                - statistic: Ljung-Box test statistic
                - p_value: p-value of the test
                - is_white_noise: True if residuals are white noise (p > 0.05)

        Raises:
            ValueError: If filter has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Filter must be fitted before running diagnostics")

        if lags is None:
            lags = self.seasonal_period

        result = acorr_ljungbox(self._residuals, lags=[lags], return_df=True)

        return {
            'statistic': result['lb_stat'].iloc[0],
            'p_value': result['lb_pvalue'].iloc[0],
            'is_white_noise': result['lb_pvalue'].iloc[0] > 0.05,
        }

    def should_filter(
        self,
        series: pd.Series,
        nsr_threshold: float = 0.1,
        ljung_box_alpha: float = 0.05
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a series needs filtering based on noise and structure.

        Uses two criteria:
        1. Noise-to-Signal Ratio (NSR): measures high-frequency noise
        2. Ljung-Box test: checks for autocorrelation structure

        Args:
            series: Time series to evaluate.
            nsr_threshold: NSR threshold above which series is considered noisy.
            ljung_box_alpha: Significance level for Ljung-Box test.

        Returns:
            Tuple of (should_filter, diagnostics_dict).
        """
        _, seasonal_period = _infer_seasonal_period(series)

        # Compute NSR from differenced series
        series_diff = series.diff().dropna()
        if np.var(series) > 0:
            nsr = np.var(series_diff) / np.var(series)
        else:
            nsr = 0

        # Ljung-Box test on differenced series
        lb_test = acorr_ljungbox(series_diff, lags=[seasonal_period], return_df=True)
        p_value = lb_test['lb_pvalue'].iloc[0]

        has_structure = p_value < ljung_box_alpha
        is_noisy = nsr > nsr_threshold

        diagnostics = {
            'nsr': nsr,
            'nsr_threshold': nsr_threshold,
            'is_noisy': is_noisy,
            'ljung_box_p_value': p_value,
            'ljung_box_alpha': ljung_box_alpha,
            'has_structure': has_structure,
        }

        # Filter if series is both structured and noisy
        return has_structure and is_noisy, diagnostics
