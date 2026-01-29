"""Tests for regime detection module."""

import numpy as np
import pandas as pd
import pytest

# Skip all tests if hmmlearn is not installed
pytest.importorskip("hmmlearn")

from timeseries_toolkit.models.regime import RegimeDetector


def generate_regime_switching_series(n: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic regime-switching time series."""
    np.random.seed(seed)
    dates = pd.date_range('2015-01-01', periods=n, freq='D')

    # Two regimes: low volatility and high volatility
    values = np.zeros(n)
    regime = 0  # Start in low volatility

    for i in range(n):
        # Regime switching with some probability
        if np.random.rand() < 0.02:
            regime = 1 - regime

        if regime == 0:
            values[i] = np.random.randn() * 0.5 + 2  # Low vol, mean 2
        else:
            values[i] = np.random.randn() * 2 + 5  # High vol, mean 5

    return pd.Series(values, index=dates, name='regime_series')


def generate_spread_series(n: int = 300, seed: int = 42) -> pd.Series:
    """Generate synthetic financial spread series."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Simulate spread with regime changes
    values = np.zeros(n)
    base = 1.5

    for i in range(n):
        if i < n // 3:
            values[i] = base + np.random.randn() * 0.2
        elif i < 2 * n // 3:
            values[i] = base + 2 + np.random.randn() * 0.5  # Crisis
        else:
            values[i] = base + 0.5 + np.random.randn() * 0.3

    return pd.Series(values, index=dates, name='spread')


class TestRegimeDetector:
    """Tests for RegimeDetector class."""

    def test_initialization_default(self):
        """Test default initialization."""
        detector = RegimeDetector()
        assert detector.is_fitted is False
        assert detector.max_states == 5
        assert detector.optimal_states is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        detector = RegimeDetector(
            max_states=3,
            n_cv_splits=3,
            covariance_type='diag'
        )
        assert detector.max_states == 3
        assert detector.n_cv_splits == 3
        assert detector.covariance_type == 'diag'

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector(max_states=3)
        result = detector.fit(series, n_states=2)
        assert result is detector

    def test_fit_sets_is_fitted(self):
        """Test that fit sets is_fitted flag."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector(max_states=3)
        detector.fit(series, n_states=2)
        assert detector.is_fitted is True

    def test_fit_with_fixed_states(self):
        """Test fit with fixed number of states."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=3, auto_select=False)
        assert detector.optimal_states == 3

    def test_fit_auto_select_states(self):
        """Test fit with automatic state selection."""
        series = generate_regime_switching_series(300)
        detector = RegimeDetector(max_states=4, n_cv_splits=3)
        detector.fit(series, auto_select=True)

        assert detector.optimal_states is not None
        assert 2 <= detector.optimal_states <= 4

    def test_predict_regimes_without_fit_raises(self):
        """Test that predict_regimes without fit raises error."""
        detector = RegimeDetector()
        with pytest.raises(ValueError):
            detector.predict_regimes()

    def test_predict_regimes_returns_series(self):
        """Test that predict_regimes returns a Series."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        regimes = detector.predict_regimes()

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(series)

    def test_predict_regimes_valid_values(self):
        """Test that regime values are valid."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=3)
        regimes = detector.predict_regimes()

        # Regimes should be 0, 1, 2 for 3 states
        unique_regimes = regimes.unique()
        assert len(unique_regimes) <= 3
        assert all(r in [0, 1, 2] for r in unique_regimes)

    def test_get_regime_probabilities_without_fit_raises(self):
        """Test that get_regime_probabilities without fit raises error."""
        detector = RegimeDetector()
        with pytest.raises(ValueError):
            detector.get_regime_probabilities()

    def test_get_regime_probabilities_returns_dataframe(self):
        """Test that get_regime_probabilities returns a DataFrame."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        probs = detector.get_regime_probabilities()

        assert isinstance(probs, pd.DataFrame)
        assert len(probs) == len(series)

    def test_regime_probabilities_sum_to_one(self):
        """Test that regime probabilities sum to 1."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        probs = detector.get_regime_probabilities()

        # Each row should sum to approximately 1
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_regime_probabilities_between_0_and_1(self):
        """Test that probabilities are between 0 and 1."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        probs = detector.get_regime_probabilities()

        assert (probs >= 0).all().all()
        assert (probs <= 1).all().all()

    def test_aggregate_to_frequency_without_fit_raises(self):
        """Test that aggregate_to_frequency without fit raises error."""
        detector = RegimeDetector()
        with pytest.raises(ValueError):
            detector.aggregate_to_frequency('ME')

    def test_aggregate_to_monthly(self):
        """Test aggregation to monthly frequency."""
        series = generate_regime_switching_series(365)  # 1 year daily
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        monthly_probs = detector.aggregate_to_frequency('ME')

        assert isinstance(monthly_probs, pd.DataFrame)
        assert len(monthly_probs) <= 12  # At most 12 months

    def test_aggregate_to_quarterly(self):
        """Test aggregation to quarterly frequency."""
        series = generate_regime_switching_series(365)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        quarterly_probs = detector.aggregate_to_frequency('QE')

        assert isinstance(quarterly_probs, pd.DataFrame)
        assert len(quarterly_probs) <= 4

    def test_get_regime_statistics_without_fit_raises(self):
        """Test that get_regime_statistics without fit raises error."""
        detector = RegimeDetector()
        with pytest.raises(ValueError):
            detector.get_regime_statistics()

    def test_get_regime_statistics_returns_dataframe(self):
        """Test that get_regime_statistics returns a DataFrame."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        stats = detector.get_regime_statistics()

        assert isinstance(stats, pd.DataFrame)
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'count' in stats.columns

    def test_get_transition_matrix_without_fit_raises(self):
        """Test that get_transition_matrix without fit raises error."""
        detector = RegimeDetector()
        with pytest.raises(ValueError):
            detector.get_transition_matrix()

    def test_get_transition_matrix_returns_dataframe(self):
        """Test that get_transition_matrix returns a DataFrame."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        trans_mat = detector.get_transition_matrix()

        assert isinstance(trans_mat, pd.DataFrame)
        assert trans_mat.shape == (2, 2)

    def test_transition_matrix_rows_sum_to_one(self):
        """Test that transition matrix rows sum to 1."""
        series = generate_regime_switching_series(200)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        trans_mat = detector.get_transition_matrix()

        row_sums = trans_mat.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_get_cv_scores_after_auto_select(self):
        """Test that CV scores are available after auto-selection."""
        series = generate_regime_switching_series(300)
        detector = RegimeDetector(max_states=4, n_cv_splits=3)
        detector.fit(series, auto_select=True)
        cv_scores = detector.get_cv_scores()

        assert cv_scores is not None
        assert isinstance(cv_scores, dict)

    def test_empty_series_raises(self):
        """Test that empty series raises error."""
        series = pd.Series(dtype=float)
        series.index = pd.DatetimeIndex([])
        detector = RegimeDetector()

        with pytest.raises(ValueError):
            detector.fit(series)

    def test_regimes_ordered_by_volatility(self):
        """Test that regimes are ordered by volatility."""
        series = generate_regime_switching_series(300)
        detector = RegimeDetector()
        detector.fit(series, n_states=2)
        stats = detector.get_regime_statistics()

        # Regime 0 should have lower std than regime 1
        if len(stats) == 2:
            assert stats.loc[0, 'std'] <= stats.loc[1, 'std']


class TestRegimeEndToEnd:
    """End-to-end tests for regime detection workflow."""

    def test_full_workflow(self):
        """Test complete regime detection workflow."""
        # Generate data
        series = generate_spread_series(300, seed=42)

        # Create and fit detector
        detector = RegimeDetector(max_states=4, n_cv_splits=3)
        detector.fit(series, auto_select=True)

        # Get regimes
        regimes = detector.predict_regimes()
        assert len(regimes) == len(series)
        assert not regimes.isna().any()

        # Get probabilities
        probs = detector.get_regime_probabilities()
        assert probs.shape[0] == len(series)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

        # Get statistics
        stats = detector.get_regime_statistics()
        assert 'mean' in stats.columns
        assert 'proportion' in stats.columns

        # Get transition matrix
        trans_mat = detector.get_transition_matrix()
        assert trans_mat.shape[0] == detector.optimal_states

        # Aggregate to quarterly
        quarterly_probs = detector.aggregate_to_frequency('QE')
        assert len(quarterly_probs) <= 4

    def test_mixed_frequency_workflow(self):
        """Test regime detection for mixed-frequency analysis."""
        # Generate daily data
        series = generate_regime_switching_series(365, seed=42)

        # Fit detector
        detector = RegimeDetector()
        detector.fit(series, n_states=2)

        # Get daily regimes
        daily_regimes = detector.predict_regimes()

        # Aggregate to quarterly for macro analysis
        quarterly_probs = detector.aggregate_to_frequency('QE', method='mean')

        assert len(quarterly_probs) <= 4
        # Quarterly probs should still sum to 1
        assert np.allclose(quarterly_probs.sum(axis=1), 1.0, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
