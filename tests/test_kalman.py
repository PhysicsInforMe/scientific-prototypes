"""Tests for Kalman filter module."""

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.models.kalman import (
    AutoKalmanFilter,
    compare_kalman_vs_arima,
)


def generate_trend_series(n: int = 100, seed: int = 42) -> pd.Series:
    """Generate series with trend and noise."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='ME')
    trend = np.linspace(0, 10, n)
    noise = np.random.randn(n) * 0.5
    return pd.Series(trend + noise, index=dates, name='trend_series')


def generate_seasonal_trend_series(n: int = 120, period: int = 12, seed: int = 42) -> pd.Series:
    """Generate series with trend, seasonality, and noise."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='ME')
    trend = np.linspace(0, 5, n)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / period)
    noise = np.random.randn(n) * 0.3
    return pd.Series(trend + seasonal + noise, index=dates, name='seasonal_trend')


class TestAutoKalmanFilter:
    """Tests for AutoKalmanFilter class."""

    def test_initialization_default(self):
        """Test default initialization."""
        kf = AutoKalmanFilter()
        assert kf.is_fitted is False
        assert kf.level == 'local linear trend'

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        kf = AutoKalmanFilter(
            level='random walk with drift',
            cycle=True,
            stochastic_cycle=True
        )
        assert kf.level == 'random walk with drift'
        assert kf.cycle is True

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        result = kf.fit(series)
        assert result is kf

    def test_fit_sets_is_fitted(self):
        """Test that fit sets is_fitted flag."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        assert kf.is_fitted is True

    def test_fit_stores_series(self):
        """Test that fit stores the series."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        assert kf._series is not None
        assert len(kf._series) == len(series)

    def test_smooth_without_fit_raises(self):
        """Test that smooth without fit raises error."""
        kf = AutoKalmanFilter()
        with pytest.raises(ValueError):
            kf.smooth()

    def test_smooth_returns_series(self):
        """Test that smooth returns a Series."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        smoothed = kf.smooth()

        assert isinstance(smoothed, pd.Series)
        assert len(smoothed) == len(series)

    def test_smooth_output_not_nan(self):
        """Test that smoothed output is not all NaN."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        smoothed = kf.smooth()

        assert not smoothed.isna().all()

    def test_smooth_reduces_noise(self):
        """Test that smoothing reduces noise."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='ME')
        trend = np.linspace(0, 10, n)
        noise = np.random.randn(n) * 2
        series = pd.Series(trend + noise, index=dates)

        kf = AutoKalmanFilter(level='local linear trend')
        kf.fit(series)
        smoothed = kf.smooth()

        # Smoothed should be closer to trend than original
        mse_original = np.mean((series.values - trend) ** 2)
        mse_smoothed = np.mean((smoothed.values - trend) ** 2)
        assert mse_smoothed < mse_original

    def test_forecast_without_fit_raises(self):
        """Test that forecast without fit raises error."""
        kf = AutoKalmanFilter()
        with pytest.raises(ValueError):
            kf.forecast(steps=5)

    def test_forecast_returns_series(self):
        """Test that forecast returns a Series."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        forecast = kf.forecast(steps=10)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 10

    def test_forecast_extends_index(self):
        """Test that forecast extends beyond original index."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        forecast = kf.forecast(steps=10)

        assert forecast.index[0] > series.index[-1]

    def test_predict_in_sample(self):
        """Test predict for in-sample data."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        predicted = kf.predict()

        assert isinstance(predicted, pd.Series)
        assert len(predicted) == len(series)

    def test_predict_with_range(self):
        """Test predict with custom date range."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)

        start = series.index[10]
        end = series.index[50]
        predicted = kf.predict(start=start, end=end)

        assert predicted.index[0] == start
        assert predicted.index[-1] == end

    def test_get_components_without_fit_raises(self):
        """Test that get_components without fit raises error."""
        kf = AutoKalmanFilter()
        with pytest.raises(ValueError):
            kf.get_components()

    def test_get_components_returns_dict(self):
        """Test that get_components returns a dictionary."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter(level='local linear trend')
        kf.fit(series)
        components = kf.get_components()

        assert isinstance(components, dict)

    def test_get_residuals_without_fit_raises(self):
        """Test that get_residuals without fit raises error."""
        kf = AutoKalmanFilter()
        with pytest.raises(ValueError):
            kf.get_residuals()

    def test_get_residuals_returns_series(self):
        """Test that get_residuals returns a Series."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        residuals = kf.get_residuals()

        assert isinstance(residuals, pd.Series)
        assert len(residuals) == len(series)

    def test_summary_not_fitted(self):
        """Test summary when not fitted."""
        kf = AutoKalmanFilter()
        summary = kf.summary()
        assert 'not fitted' in summary.lower()

    def test_summary_fitted(self):
        """Test summary when fitted."""
        series = generate_trend_series(100)
        kf = AutoKalmanFilter()
        kf.fit(series)
        summary = kf.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_empty_series_raises(self):
        """Test that empty series raises error."""
        series = pd.Series(dtype=float)
        series.index = pd.DatetimeIndex([])
        kf = AutoKalmanFilter()

        with pytest.raises(ValueError):
            kf.fit(series)

    def test_different_level_types(self):
        """Test different level specifications."""
        series = generate_trend_series(100)

        for level in ['local level', 'random walk with drift', 'local linear trend']:
            kf = AutoKalmanFilter(level=level)
            kf.fit(series)
            smoothed = kf.smooth()
            assert not smoothed.isna().all()


class TestCompareKalmanVsArima:
    """Tests for compare_kalman_vs_arima function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        series = generate_trend_series(100)
        result = compare_kalman_vs_arima(series, holdout=10)

        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Test that result contains required keys."""
        series = generate_trend_series(100)
        result = compare_kalman_vs_arima(series, holdout=10)

        required_keys = [
            'kalman_forecast', 'arima_forecast', 'actual',
            'kalman_rmse', 'kalman_mae', 'arima_rmse', 'arima_mae',
            'arima_order', 'winner'
        ]
        for key in required_keys:
            assert key in result

    def test_forecasts_correct_length(self):
        """Test that forecasts have correct length."""
        series = generate_trend_series(100)
        holdout = 10
        result = compare_kalman_vs_arima(series, holdout=holdout)

        assert len(result['kalman_forecast']) == holdout
        assert len(result['arima_forecast']) == holdout
        assert len(result['actual']) == holdout

    def test_metrics_are_positive(self):
        """Test that error metrics are positive."""
        series = generate_trend_series(100)
        result = compare_kalman_vs_arima(series, holdout=10)

        assert result['kalman_rmse'] >= 0
        assert result['kalman_mae'] >= 0
        assert result['arima_rmse'] >= 0
        assert result['arima_mae'] >= 0

    def test_winner_is_valid(self):
        """Test that winner is either 'kalman' or 'arima'."""
        series = generate_trend_series(100)
        result = compare_kalman_vs_arima(series, holdout=10)

        assert result['winner'] in ['kalman', 'arima']

    def test_arima_order_is_tuple(self):
        """Test that ARIMA order is a tuple."""
        series = generate_trend_series(100)
        result = compare_kalman_vs_arima(series, holdout=10)

        assert isinstance(result['arima_order'], tuple)
        assert len(result['arima_order']) == 3

    def test_short_series_raises(self):
        """Test that series too short raises error."""
        series = generate_trend_series(10)

        with pytest.raises(ValueError):
            compare_kalman_vs_arima(series, holdout=15)

    def test_custom_kalman_kwargs(self):
        """Test with custom Kalman parameters."""
        series = generate_trend_series(100)
        result = compare_kalman_vs_arima(
            series,
            holdout=10,
            kalman_kwargs={'level': 'local level'}
        )

        assert 'kalman_rmse' in result


class TestKalmanEndToEnd:
    """End-to-end tests for Kalman filter workflow."""

    def test_full_workflow(self):
        """Test complete Kalman filter workflow."""
        # Generate data
        series = generate_seasonal_trend_series(120, period=12, seed=42)

        # Create and fit filter
        kf = AutoKalmanFilter(level='local linear trend')
        kf.fit(series)

        # Get smoothed series
        smoothed = kf.smooth()
        assert len(smoothed) == len(series)
        assert not smoothed.isna().any()

        # Get forecast
        forecast = kf.forecast(steps=12)
        assert len(forecast) == 12

        # Get components
        components = kf.get_components()
        assert isinstance(components, dict)

        # Get residuals
        residuals = kf.get_residuals()
        assert len(residuals) == len(series)

        # Check residuals are small relative to original
        assert np.std(residuals) < np.std(series)

    def test_comparison_workflow(self):
        """Test Kalman vs ARIMA comparison workflow."""
        series = generate_trend_series(100, seed=42)

        result = compare_kalman_vs_arima(series, holdout=10)

        # Both should make reasonable forecasts
        assert result['kalman_rmse'] < 5  # Reasonable bound
        assert result['arima_rmse'] < 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
