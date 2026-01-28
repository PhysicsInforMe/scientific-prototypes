"""Models module for time series analysis."""

from timeseries_toolkit.models.kalman import AutoKalmanFilter, compare_kalman_vs_arima

__all__ = [
    "AutoKalmanFilter",
    "compare_kalman_vs_arima",
]

# Optional imports - require additional dependencies
try:
    from timeseries_toolkit.models.regime import RegimeDetector
    __all__.append("RegimeDetector")
except ImportError:
    pass  # hmmlearn not installed

try:
    from timeseries_toolkit.models.forecaster import GlobalBoostForecaster
    __all__.append("GlobalBoostForecaster")
except ImportError:
    pass  # lightgbm not installed
