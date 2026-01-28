"""Tests for fractional differentiation module."""

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.preprocessing.fractional_diff import (
    frac_diff_ffd,
    find_min_d_for_stationarity,
    generate_random_walk,
    generate_stationary_ar1,
    generate_trend_stationary,
    get_weights_ffd,
)


class TestGetWeightsFfd:
    """Tests for get_weights_ffd function."""

    def test_weights_shape(self):
        """Test that weights are returned as column vector."""
        weights = get_weights_ffd(0.5)
        assert weights.ndim == 2
        assert weights.shape[1] == 1

    def test_first_weight_is_one(self):
        """Test that first weight is always 1."""
        for d in [0.1, 0.5, 0.9]:
            weights = get_weights_ffd(d)
            assert weights[-1, 0] == 1.0  # Last in reversed order

    def test_weights_decay(self):
        """Test that weights decay towards zero."""
        weights = get_weights_ffd(0.5, threshold=1e-5)
        assert len(weights) > 1
        # Absolute values should generally decrease
        abs_weights = np.abs(weights.flatten())
        assert abs_weights[-1] >= abs_weights[0]  # First (oldest) is smallest

    def test_invalid_threshold(self):
        """Test that negative threshold raises error."""
        with pytest.raises(ValueError):
            get_weights_ffd(0.5, threshold=-0.1)


class TestFracDiffFfd:
    """Tests for frac_diff_ffd function."""

    def test_output_shape(self):
        """Test that output has same columns as input."""
        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100)
        })
        result = frac_diff_ffd(df, d=0.5)
        assert list(result.columns) == list(df.columns)

    def test_d_zero_returns_original(self):
        """Test that d=0 returns original series."""
        df = pd.DataFrame({'x': np.arange(10, dtype=float)})
        result = frac_diff_ffd(df, d=0)
        pd.testing.assert_frame_equal(result, df)

    def test_empty_series_raises(self):
        """Test that empty series raises error."""
        df = pd.DataFrame()
        with pytest.raises(ValueError):
            frac_diff_ffd(df, d=0.5)

    def test_negative_d_raises(self):
        """Test that negative d raises error."""
        df = pd.DataFrame({'x': np.random.randn(100)})
        with pytest.raises(ValueError):
            frac_diff_ffd(df, d=-0.5)


class TestFindMinDForStationarity:
    """Tests for find_min_d_for_stationarity function."""

    def test_returns_tuple(self):
        """Test that function returns (float, DataFrame)."""
        series = generate_random_walk(200, seed=42)
        min_d, results = find_min_d_for_stationarity(series, num_steps=5)
        assert isinstance(min_d, float)
        assert isinstance(results, pd.DataFrame)

    def test_stationary_series_needs_low_d(self):
        """Test that stationary series needs low d."""
        series = generate_stationary_ar1(200, rho=0.5, seed=42)
        min_d, _ = find_min_d_for_stationarity(series, num_steps=5)
        assert min_d <= 0.3  # Should be close to 0

    def test_random_walk_needs_higher_d(self):
        """Test that random walk needs higher d than stationary series."""
        series = generate_random_walk(200, seed=42)
        min_d, _ = find_min_d_for_stationarity(series, num_steps=5)
        # Random walk should need some differentiation (d > 0)
        assert min_d > 0  # Should need more differentiation than d=0

    def test_empty_series_raises(self):
        """Test that empty series raises error."""
        series = pd.Series(dtype=float)
        with pytest.raises(ValueError):
            find_min_d_for_stationarity(series)


class TestSyntheticDataGenerators:
    """Tests for synthetic data generation functions."""

    def test_random_walk_length(self):
        """Test random walk generates correct length."""
        series = generate_random_walk(length=500)
        assert len(series) == 500

    def test_random_walk_seed_reproducibility(self):
        """Test that same seed produces same series."""
        s1 = generate_random_walk(100, seed=42)
        s2 = generate_random_walk(100, seed=42)
        pd.testing.assert_series_equal(s1, s2)

    def test_ar1_stationary(self):
        """Test AR1 with |rho|<1 is bounded."""
        series = generate_stationary_ar1(1000, rho=0.9, seed=42)
        assert np.abs(series).max() < 100  # Should stay bounded

    def test_ar1_invalid_rho_raises(self):
        """Test that |rho|>=1 raises error."""
        with pytest.raises(ValueError):
            generate_stationary_ar1(100, rho=1.0)

    def test_trend_stationary_has_trend(self):
        """Test trend stationary has positive trend."""
        series = generate_trend_stationary(100, trend=0.1, seed=42)
        # Later values should generally be larger
        assert series.iloc[-1] > series.iloc[0]

    def test_datetime_index(self):
        """Test that all generators return DatetimeIndex."""
        rw = generate_random_walk(100)
        ar1 = generate_stationary_ar1(100)
        ts = generate_trend_stationary(100)

        assert isinstance(rw.index, pd.DatetimeIndex)
        assert isinstance(ar1.index, pd.DatetimeIndex)
        assert isinstance(ts.index, pd.DatetimeIndex)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
