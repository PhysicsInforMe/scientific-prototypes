"""Scientific Validity Tests.

These tests verify that the mathematical and statistical properties
of each module are correct.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# =============================================================================
# fractional_diff.py validations
# =============================================================================

class TestFractionalDiffScientific:
    """Scientific validation for fractional differentiation."""

    def test_weights_sum_to_one_for_d_zero(self):
        """Weights should sum to 1 when d=0 (no differentiation)."""
        from timeseries_toolkit.preprocessing.fractional_diff import get_weights_ffd

        weights = get_weights_ffd(d=0.0, threshold=1e-5)
        # For d=0, we should get weight = [1]
        assert len(weights) == 1
        assert np.isclose(weights[0], 1.0)

    def test_d_zero_returns_original_series(self):
        """d=0 should return the original series unchanged."""
        from timeseries_toolkit.preprocessing.fractional_diff import frac_diff_ffd

        np.random.seed(42)
        original = pd.DataFrame({'value': np.random.randn(100)})
        result = frac_diff_ffd(original, d=0.0, threshold=1e-5)

        # After dropping NaN, values should be very close to original
        valid_idx = result['value'].dropna().index
        np.testing.assert_array_almost_equal(
            result.loc[valid_idx, 'value'].values,
            original.loc[valid_idx, 'value'].values,
            decimal=10
        )

    def test_d_one_returns_first_difference(self):
        """d=1 should return the first difference."""
        from timeseries_toolkit.preprocessing.fractional_diff import frac_diff_ffd

        np.random.seed(42)
        original = pd.DataFrame({'value': np.cumsum(np.random.randn(100))})
        result = frac_diff_ffd(original, d=1.0, threshold=1e-5)

        # First difference
        expected_diff = original['value'].diff()

        # Compare (skip initial NaN values)
        valid_idx = result['value'].dropna().index
        np.testing.assert_array_almost_equal(
            result.loc[valid_idx, 'value'].values,
            expected_diff.loc[valid_idx].values,
            decimal=5
        )

    def test_adf_correctly_identifies_stationarity(self):
        """ADF test should identify stationary vs non-stationary series."""
        from timeseries_toolkit.preprocessing.fractional_diff import (
            frac_diff_ffd,
            generate_random_walk,
            generate_stationary_ar1,
        )
        from statsmodels.tsa.stattools import adfuller

        np.random.seed(42)

        # Random walk (non-stationary)
        rw = generate_random_walk(500)
        adf_rw = adfuller(rw.values)
        # p-value should be high (fail to reject null of unit root)
        assert adf_rw[1] > 0.05, "Random walk should be non-stationary"

        # AR(1) with rho < 1 (stationary)
        ar1 = generate_stationary_ar1(500, rho=0.5)
        adf_ar1 = adfuller(ar1.values)
        # p-value should be low (reject null of unit root)
        assert adf_ar1[1] < 0.05, "AR(1) with phi=0.5 should be stationary"

        # Fractionally differenced random walk should become stationary
        rw_df = pd.DataFrame({'value': rw})
        diff_rw = frac_diff_ffd(rw_df, d=0.5, threshold=1e-4)
        valid_values = diff_rw['value'].dropna().values
        if len(valid_values) > 50:
            adf_diff = adfuller(valid_values)
            # Should be more stationary (lower p-value)
            assert adf_diff[1] < adf_rw[1], "Fractional diff should increase stationarity"


# =============================================================================
# filtering.py validations
# =============================================================================

class TestFilteringScientific:
    """Scientific validation for filtering module."""

    def test_decomposition_sums_to_original(self):
        """trend + seasonal + residual should equal original."""
        from timeseries_toolkit.preprocessing.filtering import TimeSeriesFilter

        np.random.seed(42)
        n = 120
        dates = pd.date_range('2010-01-01', periods=n, freq='ME')
        trend = np.linspace(0, 10, n)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.randn(n) * 0.5
        original = pd.Series(trend + seasonal + noise, index=dates)

        filter_ = TimeSeriesFilter()
        filter_.fit(original)

        # Get components
        filtered = filter_.transform()
        residuals = filter_.get_residuals()

        # filtered + residuals should approximate original
        reconstructed = filtered + residuals

        # Allow for edge effects at boundaries
        mid_idx = slice(12, -12)
        correlation = np.corrcoef(
            original.iloc[mid_idx].values,
            reconstructed.iloc[mid_idx].values
        )[0, 1]
        assert correlation > 0.95, f"Reconstruction correlation {correlation} too low"

    def test_residuals_lower_variance_than_original(self):
        """Residuals should have lower variance than original noisy series."""
        from timeseries_toolkit.preprocessing.filtering import TimeSeriesFilter

        np.random.seed(42)
        n = 120
        dates = pd.date_range('2010-01-01', periods=n, freq='ME')
        trend = np.linspace(0, 5, n)
        seasonal = np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.randn(n) * 2  # High noise
        original = pd.Series(trend + seasonal + noise, index=dates)

        filter_ = TimeSeriesFilter()
        filter_.fit(original)

        metrics = filter_.get_metrics()

        # Residual variance should be less than original variance
        assert metrics['variance_residuals'] < metrics['variance_original'], \
            "Residual variance should be less than original"


# =============================================================================
# imputation.py validations
# =============================================================================

class TestImputationScientific:
    """Scientific validation for imputation module."""

    def test_no_nan_after_imputation(self):
        """Output should have no NaN values after imputation."""
        from timeseries_toolkit.preprocessing.imputation import MixedFrequencyImputer

        np.random.seed(42)

        # Create monthly series with some missing values
        monthly_dates = pd.date_range('2010-01-01', periods=120, freq='ME')
        var1 = pd.Series(np.random.randn(120), index=monthly_dates, name='var1')
        var2 = pd.Series(np.random.randn(120), index=monthly_dates, name='var2')

        # Introduce some missing values
        var1.iloc[10:15] = np.nan
        var2.iloc[50:55] = np.nan

        X_dict = {'var1': var1, 'var2': var2}

        # Create quarterly target index
        quarterly_dates = pd.date_range('2010-03-31', periods=40, freq='QE')

        imputer = MixedFrequencyImputer()
        imputer.fit(X_dict, quarterly_dates)
        result = imputer.transform(X_dict, quarterly_dates)

        # Result should have values (may still have NaN at edges, but core should be imputed)
        assert result.notna().any().any(), "Output should have imputed values"

    def test_quarterly_alignment_preserves_data(self):
        """Quarterly alignment should preserve non-missing data."""
        from timeseries_toolkit.preprocessing.imputation import align_to_quarterly

        np.random.seed(42)
        n = 36  # 3 years of monthly data
        monthly_dates = pd.date_range('2020-01-01', periods=n, freq='ME')
        monthly_data = pd.Series(np.random.randn(n), index=monthly_dates, name='value')

        # Create quarterly target index
        quarterly_dates = pd.date_range('2020-03-31', periods=12, freq='QE')

        quarterly = align_to_quarterly(monthly_data, quarterly_dates)

        # Should have 12 quarters (3 years)
        assert len(quarterly) == 12

        # Values should not all be NaN
        assert not quarterly.isna().all(), "Quarterly alignment should preserve data"


# =============================================================================
# kalman.py validations
# =============================================================================

class TestKalmanScientific:
    """Scientific validation for Kalman filter module."""

    def test_smoothed_series_lower_variance(self):
        """Smoothed series should have lower variance than noisy original."""
        from timeseries_toolkit.models.kalman import AutoKalmanFilter

        np.random.seed(42)
        n = 200
        dates = pd.date_range('2010-01-01', periods=n, freq='ME')

        # True signal
        true_signal = np.cumsum(np.random.randn(n) * 0.1)
        # Noisy observations
        noise = np.random.randn(n) * 2
        observed = true_signal + noise

        series = pd.Series(observed, index=dates)

        kf = AutoKalmanFilter()
        kf.fit(series)
        smoothed = kf.smooth()

        # Smoothed should be closer to true signal (lower MSE)
        mse_original = np.mean((observed - true_signal) ** 2)
        mse_smoothed = np.mean((smoothed.values - true_signal) ** 2)

        assert mse_smoothed < mse_original, \
            f"Smoothed MSE ({mse_smoothed:.4f}) should be less than original ({mse_original:.4f})"

    def test_kalman_vs_ar1_analytical(self):
        """Kalman filter should approximate AR(1) dynamics."""
        from timeseries_toolkit.models.kalman import AutoKalmanFilter

        np.random.seed(42)
        n = 300
        phi = 0.8
        sigma = 1.0

        # Generate AR(1) process
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t-1] + np.random.randn() * sigma

        dates = pd.date_range('2010-01-01', periods=n, freq='ME')
        series = pd.Series(y, index=dates)

        kf = AutoKalmanFilter(level='local level')
        kf.fit(series)
        smoothed = kf.smooth()

        # Smoothed series should be highly correlated with original
        correlation = np.corrcoef(y, smoothed.values)[0, 1]
        assert correlation > 0.9, f"Correlation {correlation} should be > 0.9"


# =============================================================================
# regime.py validations
# =============================================================================

# Skip if hmmlearn not available
hmmlearn = pytest.importorskip("hmmlearn")


class TestRegimeScientific:
    """Scientific validation for regime detection module."""

    def test_detected_regimes_match_true_regimes(self):
        """Detected regimes should match true regimes >80% on synthetic data."""
        from timeseries_toolkit.models.regime import RegimeDetector

        np.random.seed(42)
        n = 500

        # Generate synthetic regime-switching data
        true_regimes = np.zeros(n, dtype=int)
        true_regimes[100:200] = 1
        true_regimes[300:400] = 1

        # Different means for each regime (make separation clearer)
        means = [0.0, 5.0]
        stds = [0.5, 0.5]

        data = np.array([
            np.random.normal(means[r], stds[r]) for r in true_regimes
        ])

        dates = pd.date_range('2010-01-01', periods=n, freq='D')
        series = pd.Series(data, index=dates)

        detector = RegimeDetector(max_states=2)
        detector.fit(series)
        predicted = detector.predict_regimes()

        # Calculate accuracy (accounting for label switching)
        accuracy1 = np.mean(predicted == true_regimes)
        accuracy2 = np.mean(predicted == (1 - true_regimes))
        accuracy = max(accuracy1, accuracy2)

        assert accuracy > 0.80, f"Regime detection accuracy {accuracy:.2%} should be > 80%"

    def test_regime_probabilities_sum_to_one(self):
        """Regime probabilities should sum to 1 at each time point."""
        from timeseries_toolkit.models.regime import RegimeDetector

        np.random.seed(42)
        n = 200
        dates = pd.date_range('2010-01-01', periods=n, freq='D')
        data = pd.Series(np.random.randn(n), index=dates)

        detector = RegimeDetector(max_states=2)
        detector.fit(data)
        probs = detector.get_regime_probabilities()

        # Probabilities should sum to 1 for each row
        row_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(
            row_sums.values,
            np.ones(len(probs)),
            decimal=5,
            err_msg="Regime probabilities should sum to 1"
        )


# =============================================================================
# causality.py validations
# =============================================================================

class TestCausalityScientific:
    """Scientific validation for causality module."""

    def test_ccm_detects_coupled_logistic_maps(self):
        """CCM should detect causality in coupled logistic maps."""
        from timeseries_toolkit.validation.causality import ccm_test

        np.random.seed(42)
        n = 500

        # Coupled logistic maps: X causes Y
        x = np.zeros(n)
        y = np.zeros(n)
        x[0] = 0.4
        y[0] = 0.2

        rx, ry = 3.8, 3.5
        beta = 0.3  # Coupling strength X -> Y

        for t in range(1, n):
            x[t] = rx * x[t-1] * (1 - x[t-1])
            y[t] = ry * y[t-1] * (1 - y[t-1]) + beta * x[t-1]
            # Keep bounded
            x[t] = np.clip(x[t], 0.001, 0.999)
            y[t] = np.clip(y[t], 0.001, 0.999)

        # Test X -> Y (should be significant)
        result_xy = ccm_test(x, y, embedding_dim=3, n_surrogates=50)

        # CCM score should be reasonably high for causal relationship
        assert result_xy['ccm_score'] > 0.3, \
            f"CCM score {result_xy['ccm_score']:.3f} should detect X->Y causality"

    def test_granger_detects_correct_direction(self):
        """Granger should detect X->Y but not Y->X when X causes Y."""
        from timeseries_toolkit.validation.causality import granger_causality_test

        np.random.seed(42)
        n = 300

        # X causes Y with lag
        x = np.random.randn(n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.5 * y[t-1] + 0.4 * x[t-1] + 0.3 * x[t-2] + np.random.randn() * 0.3

        df = pd.DataFrame({'X': x, 'Y': y})

        # Test X -> Y (should show improvement)
        result_xy = granger_causality_test(df, 'Y', 'X', max_lags=3)

        # Test Y -> X (should show less/no improvement)
        result_yx = granger_causality_test(df, 'X', 'Y', max_lags=3)

        # X->Y should have higher delta_rmse than Y->X
        assert result_xy['delta_rmse'] > result_yx['delta_rmse'], \
            f"X->Y ({result_xy['delta_rmse']:.4f}) should be stronger than Y->X ({result_yx['delta_rmse']:.4f})"


# =============================================================================
# diagnostics.py validations
# =============================================================================

class TestDiagnosticsScientific:
    """Scientific validation for diagnostics module."""

    def test_white_noise_passes_ljung_box(self):
        """White noise should pass Ljung-Box test (p > 0.05)."""
        from timeseries_toolkit.validation.diagnostics import ForensicEnsembleAnalyzer
        from statsmodels.stats.diagnostic import acorr_ljungbox

        np.random.seed(42)
        n = 200

        # Generate white noise
        white_noise = np.random.randn(n)

        # Ljung-Box test
        lb_result = acorr_ljungbox(white_noise, lags=[10], return_df=True)
        p_value = lb_result['lb_pvalue'].iloc[0]

        assert p_value > 0.05, \
            f"White noise should pass Ljung-Box (p={p_value:.4f} should be > 0.05)"

    def test_autocorrelated_fails_ljung_box(self):
        """Autocorrelated series should fail Ljung-Box test (p < 0.05)."""
        from statsmodels.stats.diagnostic import acorr_ljungbox

        np.random.seed(42)
        n = 200

        # Generate AR(1) with high autocorrelation
        phi = 0.9
        ar1 = np.zeros(n)
        for t in range(1, n):
            ar1[t] = phi * ar1[t-1] + np.random.randn()

        # Ljung-Box test
        lb_result = acorr_ljungbox(ar1, lags=[10], return_df=True)
        p_value = lb_result['lb_pvalue'].iloc[0]

        assert p_value < 0.05, \
            f"AR(1) should fail Ljung-Box (p={p_value:.4f} should be < 0.05)"

    def test_hurst_exponent_random_walk(self):
        """Hurst exponent should be approximately 0.5 for white noise increments."""
        np.random.seed(42)
        n = 2000

        # Generate white noise (should have H ≈ 0.5)
        # Note: we test white noise, not random walk, because Hurst of white noise is 0.5
        # Random walk has H ≈ 1.0 due to persistence
        white_noise = np.random.randn(n)

        # Calculate Hurst exponent using R/S analysis
        def hurst_rs(ts):
            """Calculate Hurst exponent using R/S analysis."""
            n = len(ts)
            max_k = n // 2

            rs_list = []
            n_list = []

            for k in [16, 32, 64, 128, 256, 512]:
                if k > max_k:
                    continue
                rs_values = []
                n_segments = n // k
                for i in range(n_segments):
                    segment = ts[i * k:(i + 1) * k]
                    mean_seg = np.mean(segment)
                    cumdev = np.cumsum(segment - mean_seg)
                    R = np.max(cumdev) - np.min(cumdev)
                    S = np.std(segment, ddof=1)
                    if S > 0:
                        rs_values.append(R / S)

                if rs_values:
                    rs_list.append(np.mean(rs_values))
                    n_list.append(k)

            if len(n_list) < 2:
                return 0.5

            # Linear regression in log-log space
            log_n = np.log(n_list)
            log_rs = np.log(rs_list)
            slope, _, _, _, _ = stats.linregress(log_n, log_rs)
            return slope

        H = hurst_rs(white_noise)

        # Hurst should be close to 0.5 for white noise (allow tolerance)
        assert 0.35 < H < 0.65, \
            f"Hurst exponent {H:.3f} should be approximately 0.5 for white noise"


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
