"""
Kalman Filter Module.

This module provides an automated Kalman filter implementation using
statsmodels UnobservedComponents for time series smoothing and forecasting.
"""

from itertools import product
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False


class AutoKalmanFilter:
    """
    Automated Kalman filter using statsmodels UnobservedComponents.

    This class provides an easy-to-use interface for state-space modeling
    with automatic component selection. It wraps the UnobservedComponents
    model which estimates trend, cycle, and seasonal components.

    Attributes:
        is_fitted: Whether the model has been fitted.
        model: The fitted UnobservedComponents model.
        results: The fitted model results.

    Example:
        >>> kf = AutoKalmanFilter()
        >>> kf.fit(monthly_series)
        >>> smoothed = kf.smooth()
        >>> forecast = kf.forecast(steps=12)
    """

    def __init__(
        self,
        level: str = 'local linear trend',
        cycle: bool = False,
        stochastic_cycle: bool = False,
        seasonal: Optional[int] = None,
        freq_seasonal: Optional[list] = None
    ):
        """
        Initialize the AutoKalmanFilter.

        Args:
            level: Type of trend component. Options:
                - 'local level': Random walk
                - 'random walk with drift': Random walk with drift
                - 'local linear trend': Local linear trend (default)
                - 'smooth trend': Smooth trend
                - 'fixed intercept': Fixed intercept
            cycle: Whether to include a cycle component.
            stochastic_cycle: Whether the cycle is stochastic.
            seasonal: Period for deterministic seasonality (e.g., 12 for monthly).
            freq_seasonal: List of dicts for frequency-domain seasonality.
                Example: [{'period': 365.25, 'harmonics': 4}]
        """
        self.level = level
        self.cycle = cycle
        self.stochastic_cycle = stochastic_cycle
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal

        self.is_fitted: bool = False
        self.model: Optional[Any] = None
        self.results: Optional[Any] = None
        self._series: Optional[pd.Series] = None
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0

    def fit(
        self,
        series: pd.Series,
        standardize: bool = True,
        method: str = 'lbfgs',
        maxiter: int = 1000
    ) -> 'AutoKalmanFilter':
        """
        Fit the Kalman filter to a time series.

        Args:
            series: Time series with DatetimeIndex.
            standardize: Whether to standardize the series before fitting.
                Recommended for numerical stability.
            method: Optimization method for fitting ('lbfgs', 'powell', 'nm').
            maxiter: Maximum iterations for optimization.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If series is empty or has insufficient data.
        """
        if series.empty:
            raise ValueError("Input series cannot be empty")

        self._series = series.copy()

        # Standardize for numerical stability
        if standardize:
            self._scaler_mean = series.mean()
            self._scaler_std = series.std()
            if self._scaler_std == 0:
                self._scaler_std = 1.0
            standardized = (series - self._scaler_mean) / self._scaler_std
        else:
            self._scaler_mean = 0.0
            self._scaler_std = 1.0
            standardized = series

        # Build model
        model_kwargs = {
            'level': self.level,
            'cycle': self.cycle,
            'stochastic_cycle': self.stochastic_cycle,
        }

        if self.seasonal is not None:
            model_kwargs['seasonal'] = self.seasonal

        if self.freq_seasonal is not None:
            model_kwargs['freq_seasonal'] = self.freq_seasonal

        self.model = sm.tsa.UnobservedComponents(
            standardized.astype(float),
            **model_kwargs
        )

        # Fit model
        self.results = self.model.fit(method=method, maxiter=maxiter, disp=False)
        self.is_fitted = True

        return self

    def smooth(self) -> pd.Series:
        """
        Get the smoothed (filtered) series.

        Returns:
            Series with smoothed values at original time points.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before smoothing")

        smoothed_std = self.results.smoothed_state[0]
        smoothed = smoothed_std * self._scaler_std + self._scaler_mean

        return pd.Series(smoothed, index=self._series.index, name='smoothed')

    def forecast(self, steps: int) -> pd.Series:
        """
        Generate out-of-sample forecasts.

        Args:
            steps: Number of periods to forecast.

        Returns:
            Series with forecasted values.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        # Get forecast
        forecast_result = self.results.get_forecast(steps=steps)
        forecast_std = forecast_result.predicted_mean

        # Reverse standardization
        forecast = forecast_std * self._scaler_std + self._scaler_mean

        return forecast

    def predict(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Get predictions for a date range (in-sample and/or out-of-sample).

        Args:
            start: Start date for predictions. Defaults to series start.
            end: End date for predictions. Defaults to series end.

        Returns:
            Series with predicted values.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        if start is None:
            start = self._series.index[0]
        if end is None:
            end = self._series.index[-1]

        predicted_std = self.results.predict(start=start, end=end)
        predicted = predicted_std * self._scaler_std + self._scaler_mean

        return predicted

    def get_components(self) -> Dict[str, pd.Series]:
        """
        Extract the estimated components (trend, cycle, seasonal).

        Returns:
            Dictionary with component names as keys and Series as values.
            Available components depend on model specification.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting components")

        components = {}

        # Level/Trend component
        if hasattr(self.results, 'level') and self.results.level is not None:
            level = self.results.level['smoothed'] * self._scaler_std + self._scaler_mean
            components['level'] = pd.Series(level, index=self._series.index)

        # Trend slope (if local linear trend)
        if hasattr(self.results, 'trend') and self.results.trend is not None:
            trend = self.results.trend['smoothed'] * self._scaler_std
            components['trend'] = pd.Series(trend, index=self._series.index)

        # Cycle component
        if self.cycle and hasattr(self.results, 'cycle'):
            if self.results.cycle is not None:
                cycle = self.results.cycle['smoothed'] * self._scaler_std
                components['cycle'] = pd.Series(cycle, index=self._series.index)

        # Seasonal component
        if hasattr(self.results, 'seasonal') and self.results.seasonal is not None:
            seasonal = self.results.seasonal['smoothed'] * self._scaler_std
            components['seasonal'] = pd.Series(seasonal, index=self._series.index)

        # Frequency seasonal components
        if hasattr(self.results, 'freq_seasonal') and self.results.freq_seasonal is not None:
            for i, fs in enumerate(self.results.freq_seasonal):
                if fs is not None:
                    freq_seas = fs['smoothed'] * self._scaler_std
                    components[f'freq_seasonal_{i}'] = pd.Series(
                        freq_seas, index=self._series.index
                    )

        return components

    def get_residuals(self) -> pd.Series:
        """
        Get the model residuals.

        Returns:
            Series with residual values.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting residuals")

        residuals = self.results.resid * self._scaler_std
        return pd.Series(residuals, index=self._series.index, name='residuals')

    def summary(self) -> str:
        """
        Get a summary of the fitted model.

        Returns:
            String summary of model parameters and fit statistics.
        """
        if not self.is_fitted:
            return "Model not fitted"
        return str(self.results.summary())


def compare_kalman_vs_arima(
    series: pd.Series,
    holdout: int = 4,
    kalman_kwargs: Optional[Dict] = None,
    arima_order_range: Optional[Tuple] = None
) -> Dict[str, Any]:
    """
    Compare Kalman filter against ARIMA with grid search.

    This function splits the series into training and holdout sets,
    fits both models, and computes forecast accuracy metrics.

    Args:
        series: Time series with DatetimeIndex.
        holdout: Number of observations to hold out for testing.
        kalman_kwargs: Keyword arguments for AutoKalmanFilter.
            Defaults to {'level': 'random walk with drift'}.
        arima_order_range: Tuple of (p_range, d_range, q_range) for grid search.
            Defaults to (range(0, 4), range(0, 3), range(0, 4)).

    Returns:
        Dictionary containing:
            - 'kalman_forecast': Kalman forecasts for holdout period
            - 'arima_forecast': ARIMA forecasts for holdout period
            - 'actual': Actual values in holdout period
            - 'kalman_rmse': Kalman RMSE
            - 'kalman_mae': Kalman MAE
            - 'arima_rmse': ARIMA RMSE
            - 'arima_mae': ARIMA MAE
            - 'arima_order': Best ARIMA order found
            - 'winner': 'kalman' or 'arima' based on RMSE

    Raises:
        ValueError: If series is too short for holdout.
        ImportError: If statsmodels ARIMA is not available.

    Example:
        >>> results = compare_kalman_vs_arima(gdp_series, holdout=4)
        >>> print(f"Winner: {results['winner']}")
        >>> print(f"Kalman RMSE: {results['kalman_rmse']:.4f}")
    """
    if not HAS_ARIMA:
        raise ImportError("statsmodels ARIMA is required for comparison")

    if len(series) <= holdout:
        raise ValueError(f"Series too short ({len(series)}) for holdout ({holdout})")

    # Split data
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    # Default kwargs
    if kalman_kwargs is None:
        kalman_kwargs = {'level': 'random walk with drift'}

    if arima_order_range is None:
        arima_order_range = (range(0, 4), range(0, 3), range(0, 4))

    # Fit Kalman filter
    kf = AutoKalmanFilter(**kalman_kwargs)
    kf.fit(train)
    kalman_forecast = kf.forecast(steps=holdout)
    kalman_forecast.index = test.index

    # Grid search for best ARIMA
    p_range, d_range, q_range = arima_order_range
    pdq_combinations = list(product(p_range, d_range, q_range))

    best_aic = float('inf')
    best_arima_model = None
    best_order = None

    for order in pdq_combinations:
        try:
            model = ARIMA(train, order=order).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_arima_model = model
                best_order = order
        except Exception:
            continue

    if best_arima_model is None:
        raise ValueError("Could not fit any ARIMA model")

    arima_forecast = best_arima_model.forecast(steps=holdout)
    arima_forecast.index = test.index

    # Compute metrics
    kalman_rmse = np.sqrt(mean_squared_error(test, kalman_forecast))
    kalman_mae = mean_absolute_error(test, kalman_forecast)
    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
    arima_mae = mean_absolute_error(test, arima_forecast)

    winner = 'kalman' if kalman_rmse < arima_rmse else 'arima'

    return {
        'kalman_forecast': kalman_forecast,
        'arima_forecast': arima_forecast,
        'actual': test,
        'kalman_rmse': kalman_rmse,
        'kalman_mae': kalman_mae,
        'arima_rmse': arima_rmse,
        'arima_mae': arima_mae,
        'arima_order': best_order,
        'winner': winner,
    }
