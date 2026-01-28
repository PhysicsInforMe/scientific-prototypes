"""Tests for filtering module."""

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.preprocessing.filtering import (
    TimeSeriesFilter,
    _infer_seasonal_period,
    _find_best_sarima,
    _compute_sample_entropy,
)


def generate_seasonal_series(n: int = 200, period: int = 12, seed: int = 42) -> pd.Series:
    """Generate synthetic seasonal time series."""
    np.random.seed(seed)
    dates = pd.date_range('2010-01-01', periods=n, freq='ME')
    trend = np.linspace(0, 2, n)
    seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n) / period)
    noise = 0.1 * np.random.randn(n)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name='test_series')


class TestInferSeasonalPeriod:
    """Tests for _infer_seasonal_period function."""

    def test_monthly_frequency(self):
        """Test monthly frequency detection."""
        series = pd.Series(
            np.random.randn(100),
            index=pd.date_range('2020-01-01', periods=100, freq='ME')
        )
        freq_name, period = _infer_seasonal_period(series)
        assert period == 12
        assert 'Monthly' in freq_name

    def test_daily_frequency(self):
        """Test daily frequency detection."""
        series = pd.Series(
            np.random.randn(100),
            index=pd.date_range('2020-01-01', periods=100, freq='D')
        )
        freq_name, period = _infer_seasonal_period(series)
        assert period == 7
        assert 'Daily' in freq_name

    def test_quarterly_frequency(self):
        """Test quarterly frequency detection."""
        series = pd.Series(
            np.random.randn(20),
            index=pd.date_range('2020-01-01', periods=20, freq='QE')
        )
        freq_name, period = _infer_seasonal_period(series)
        assert period == 4
        assert 'Quarterly' in freq_name


class TestComputeSampleEntropy:
    """Tests for _compute_sample_entropy function."""

    def test_returns_float(self):
        """Test that entropy returns a float."""
        data = np.random.randn(100)
        entropy = _compute_sample_entropy(data)
        assert isinstance(entropy, float)

    def test_constant_series_low_entropy(self):
        """Test that constant series has low/nan entropy."""
        data = np.ones(100)
        entropy = _compute_sample_entropy(data)
        # Constant series should have nan or very low entropy
        assert np.isnan(entropy) or entropy < 0.1

    def test_random_series_higher_entropy(self):
        """Test that random series has higher entropy."""
        np.random.seed(42)
        data = np.random.randn(100)
        entropy = _compute_sample_entropy(data)
        if not np.isnan(entropy):
            assert entropy > 0


class TestTimeSeriesFilter:
    """Tests for TimeSeriesFilter class."""

    def test_initialization(self):
        """Test filter initialization."""
        filter = TimeSeriesFilter()
        assert filter.is_fitted is False
        assert filter.seasonal_period is None

    def test_initialization_with_period(self):
        """Test filter initialization with custom period."""
        filter = TimeSeriesFilter(seasonal_period=7)
        assert filter._seasonal_period_override == 7

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        result = filter.fit(series)
        assert result is filter

    def test_fit_sets_is_fitted(self):
        """Test that fit sets is_fitted flag."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        assert filter.is_fitted is True

    def test_fit_detects_seasonal_period(self):
        """Test that fit detects seasonal period."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        assert filter.seasonal_period == 12

    def test_transform_without_fit_raises(self):
        """Test that transform without fit raises error."""
        filter = TimeSeriesFilter()
        with pytest.raises(ValueError):
            filter.transform()

    def test_transform_returns_series(self):
        """Test that transform returns a Series."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        result = filter.transform()
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_transform_output_not_nan(self):
        """Test that transform output is not all NaN."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        result = filter.transform()
        assert not result.isna().all()

    def test_get_residuals_without_fit_raises(self):
        """Test that get_residuals without fit raises error."""
        filter = TimeSeriesFilter()
        with pytest.raises(ValueError):
            filter.get_residuals()

    def test_get_residuals_returns_series(self):
        """Test that get_residuals returns a Series."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        residuals = filter.get_residuals()
        assert isinstance(residuals, pd.Series)
        assert len(residuals) > 0

    def test_get_metrics_without_fit_raises(self):
        """Test that get_metrics without fit raises error."""
        filter = TimeSeriesFilter()
        with pytest.raises(ValueError):
            filter.get_metrics()

    def test_get_metrics_returns_dict(self):
        """Test that get_metrics returns a dictionary."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        metrics = filter.get_metrics()
        assert isinstance(metrics, dict)
        assert 'snr_db' in metrics
        assert 'variance_original' in metrics
        assert 'variance_filtered' in metrics

    def test_metrics_snr_is_positive(self):
        """Test that SNR is positive for filtered series."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        metrics = filter.get_metrics()
        # SNR should be positive or inf
        assert metrics['snr_db'] > 0 or metrics['snr_db'] == np.inf

    def test_ljung_box_test(self):
        """Test Ljung-Box diagnostic test."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        filter.fit(series)
        result = filter.get_ljung_box_test()
        assert 'p_value' in result
        assert 'is_white_noise' in result
        assert isinstance(result['is_white_noise'], (bool, np.bool_))

    def test_should_filter_returns_tuple(self):
        """Test should_filter returns tuple."""
        series = generate_seasonal_series(100, period=12)
        filter = TimeSeriesFilter()
        should_filter, diagnostics = filter.should_filter(series)
        assert isinstance(should_filter, (bool, np.bool_))
        assert isinstance(diagnostics, dict)

    def test_empty_series_raises(self):
        """Test that empty series raises error."""
        series = pd.Series(dtype=float)
        series.index = pd.DatetimeIndex([])
        filter = TimeSeriesFilter()
        with pytest.raises(ValueError):
            filter.fit(series)

    def test_short_series_raises(self):
        """Test that series too short for period raises error."""
        dates = pd.date_range('2020-01-01', periods=10, freq='ME')
        series = pd.Series(np.random.randn(10), index=dates)
        filter = TimeSeriesFilter()
        with pytest.raises(ValueError):
            filter.fit(series)


class TestFilterEndToEnd:
    """End-to-end tests for filtering workflow."""

    def test_full_workflow(self):
        """Test complete filtering workflow."""
        # Generate noisy seasonal data
        series = generate_seasonal_series(150, period=12, seed=42)

        # Add extra noise
        np.random.seed(42)
        noisy_series = series + 0.3 * np.random.randn(len(series))

        # Create and fit filter
        filter = TimeSeriesFilter()
        filter.fit(noisy_series)

        # Get filtered series
        filtered = filter.transform()
        assert len(filtered) == len(noisy_series)

        # Get residuals
        residuals = filter.get_residuals()
        assert len(residuals) > 0

        # Get metrics
        metrics = filter.get_metrics()
        assert metrics['variance_residuals'] < metrics['variance_original']

        # Run diagnostic
        lb_result = filter.get_ljung_box_test()
        assert 'p_value' in lb_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
