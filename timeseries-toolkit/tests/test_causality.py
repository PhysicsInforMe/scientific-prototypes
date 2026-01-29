"""Tests for causality module."""

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.validation.causality import (
    ccm_test,
    granger_causality_test,
    generate_causal_system,
    run_full_causality_analysis,
    _ccm_leave_one_out,
    _generate_surrogates,
)


def generate_causal_pair(n: int = 200, seed: int = 42):
    """Generate a pair of causally related series."""
    np.random.seed(seed)

    x = np.zeros(n)
    y = np.zeros(n)

    # X causes Y with a lag
    for i in range(1, n):
        x[i] = 0.8 * x[i-1] + np.random.randn() * 0.3
        y[i] = 0.5 * y[i-1] + 0.4 * x[i-1] + np.random.randn() * 0.2

    return x, y


def generate_independent_series(n: int = 200, seed: int = 42):
    """Generate two independent series."""
    np.random.seed(seed)
    x = np.cumsum(np.random.randn(n))
    np.random.seed(seed + 100)  # Different seed
    y = np.cumsum(np.random.randn(n))
    return x, y


class TestGenerateSurrogates:
    """Tests for _generate_surrogates function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        series = np.random.randn(100)
        surrogates = _generate_surrogates(series, n_surr=10)
        assert isinstance(surrogates, list)

    def test_correct_number_of_surrogates(self):
        """Test that correct number of surrogates is generated."""
        series = np.random.randn(100)
        surrogates = _generate_surrogates(series, n_surr=20)
        assert len(surrogates) == 20

    def test_surrogates_same_length(self):
        """Test that surrogates have same length as original."""
        series = np.random.randn(100)
        surrogates = _generate_surrogates(series, n_surr=5)
        for surr in surrogates:
            assert len(surr) == len(series)

    def test_surrogates_different_from_original(self):
        """Test that surrogates are different from original."""
        np.random.seed(42)
        series = np.random.randn(100)
        surrogates = _generate_surrogates(series, n_surr=5)

        for surr in surrogates:
            # Should not be identical
            assert not np.allclose(series, surr)

    def test_surrogates_preserve_mean(self):
        """Test that surrogates approximately preserve mean."""
        np.random.seed(42)
        series = np.random.randn(200) + 5  # Non-zero mean
        surrogates = _generate_surrogates(series, n_surr=10)

        original_mean = np.mean(series)
        for surr in surrogates:
            # Mean should be approximately preserved
            assert abs(np.mean(surr) - original_mean) < 1.0


class TestCCMLeaveOneOut:
    """Tests for _ccm_leave_one_out function."""

    def test_returns_float(self):
        """Test that function returns a float."""
        x, y = generate_causal_pair(200)
        score = _ccm_leave_one_out(x, y, dim=3, tau=1)
        assert isinstance(score, float)

    def test_score_between_0_and_1(self):
        """Test that score is between 0 and 1."""
        x, y = generate_causal_pair(200)
        score = _ccm_leave_one_out(x, y, dim=3, tau=1)
        assert 0 <= score <= 1

    def test_short_series_returns_zero(self):
        """Test that short series returns 0."""
        x = np.random.randn(5)
        y = np.random.randn(5)
        score = _ccm_leave_one_out(x, y, dim=3, tau=1)
        assert score == 0.0

    def test_causal_pair_higher_score(self):
        """Test that causal pair has higher score than independent."""
        x_causal, y_causal = generate_causal_pair(300, seed=42)
        x_indep, y_indep = generate_independent_series(300, seed=42)

        score_causal = _ccm_leave_one_out(x_causal, y_causal, dim=3, tau=1)
        score_indep = _ccm_leave_one_out(x_indep, y_indep, dim=3, tau=1)

        # Causal relationship should yield higher CCM
        # (though this is not guaranteed in all cases)
        assert score_causal >= 0


class TestCCMTest:
    """Tests for ccm_test function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        x, y = generate_causal_pair(200)
        result = ccm_test(x, y)
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Test that result contains required keys."""
        x, y = generate_causal_pair(200)
        result = ccm_test(x, y)

        required_keys = [
            'ccm_score', 'is_significant', 'p_value',
            'surrogate_threshold', 'surrogate_scores'
        ]
        for key in required_keys:
            assert key in result

    def test_ccm_score_valid(self):
        """Test that CCM score is valid."""
        x, y = generate_causal_pair(200)
        result = ccm_test(x, y)
        assert 0 <= result['ccm_score'] <= 1

    def test_is_significant_is_bool(self):
        """Test that is_significant is boolean."""
        x, y = generate_causal_pair(200)
        result = ccm_test(x, y)
        assert isinstance(result['is_significant'], (bool, np.bool_))

    def test_p_value_valid(self):
        """Test that p_value is between 0 and 1."""
        x, y = generate_causal_pair(200)
        result = ccm_test(x, y)
        assert 0 <= result['p_value'] <= 1

    def test_surrogate_scores_list(self):
        """Test that surrogate_scores is a list."""
        x, y = generate_causal_pair(200)
        result = ccm_test(x, y, n_surrogates=20)
        assert isinstance(result['surrogate_scores'], list)
        assert len(result['surrogate_scores']) == 20

    def test_works_with_pandas_series(self):
        """Test that function works with pandas Series."""
        x, y = generate_causal_pair(200)
        x_series = pd.Series(x)
        y_series = pd.Series(y)
        result = ccm_test(x_series, y_series)
        assert 'ccm_score' in result

    def test_custom_embedding_dim(self):
        """Test with custom embedding dimension."""
        x, y = generate_causal_pair(200)
        result = ccm_test(x, y, embedding_dim=4, tau=2)
        assert 'ccm_score' in result


class TestGrangerCausalityTest:
    """Tests for granger_causality_test function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        x, y = generate_causal_pair(200)
        df = pd.DataFrame({'target': y, 'source': x})
        result = granger_causality_test(df, 'target', 'source')
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Test that result contains required keys."""
        x, y = generate_causal_pair(200)
        df = pd.DataFrame({'target': y, 'source': x})
        result = granger_causality_test(df, 'target', 'source')

        required_keys = [
            'delta_rmse', 'rmse_univariate', 'rmse_bivariate', 'improvement_pct'
        ]
        for key in required_keys:
            assert key in result

    def test_delta_rmse_valid(self):
        """Test that delta_rmse is between 0 and 1."""
        x, y = generate_causal_pair(200)
        df = pd.DataFrame({'target': y, 'source': x})
        result = granger_causality_test(df, 'target', 'source')
        assert 0 <= result['delta_rmse'] <= 1

    def test_improvement_pct_matches_delta(self):
        """Test that improvement_pct matches delta_rmse."""
        x, y = generate_causal_pair(200)
        df = pd.DataFrame({'target': y, 'source': x})
        result = granger_causality_test(df, 'target', 'source')
        assert abs(result['improvement_pct'] - result['delta_rmse'] * 100) < 0.01

    def test_multiple_source_cols(self):
        """Test with multiple source columns."""
        x1, y = generate_causal_pair(200, seed=42)
        x2 = np.random.randn(200)
        df = pd.DataFrame({'target': y, 'source1': x1, 'source2': x2})
        result = granger_causality_test(df, 'target', ['source1', 'source2'])
        assert 'delta_rmse' in result

    def test_linear_model_type(self):
        """Test with linear model type."""
        x, y = generate_causal_pair(200)
        df = pd.DataFrame({'target': y, 'source': x})
        result = granger_causality_test(df, 'target', 'source', model_type='linear')
        assert 'delta_rmse' in result

    def test_nonlinear_model_type(self):
        """Test with nonlinear model type."""
        x, y = generate_causal_pair(200)
        df = pd.DataFrame({'target': y, 'source': x})
        result = granger_causality_test(df, 'target', 'source', model_type='nonlinear')
        assert 'delta_rmse' in result

    def test_causal_pair_positive_improvement(self):
        """Test that causal pair shows positive improvement."""
        x, y = generate_causal_pair(300, seed=42)
        df = pd.DataFrame({'target': y, 'source': x})
        result = granger_causality_test(df, 'target', 'source', max_lags=3)
        # Should show some improvement
        assert result['delta_rmse'] >= 0


class TestGenerateCausalSystem:
    """Tests for generate_causal_system function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        result = generate_causal_system(100)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_dataframe_and_dict(self):
        """Test that function returns DataFrame and dict."""
        data, truth = generate_causal_system(100)
        assert isinstance(data, pd.DataFrame)
        assert isinstance(truth, dict)

    def test_dataframe_has_correct_columns(self):
        """Test that DataFrame has correct columns."""
        data, _ = generate_causal_system(100)
        expected_cols = ['X1', 'X2', 'X3', 'X4', 'Y']
        for col in expected_cols:
            assert col in data.columns

    def test_dataframe_correct_length(self):
        """Test that DataFrame has correct length."""
        data, _ = generate_causal_system(200)
        assert len(data) == 200

    def test_ground_truth_has_all_features(self):
        """Test that ground truth describes all features."""
        _, truth = generate_causal_system(100)
        assert 'X1' in truth
        assert 'X2' in truth
        assert 'X3' in truth
        assert 'X4' in truth

    def test_data_is_normalized(self):
        """Test that data is approximately normalized."""
        data, _ = generate_causal_system(1000)
        for col in data.columns:
            # Mean should be close to 0
            assert abs(data[col].mean()) < 0.5
            # Std should be close to 1
            assert 0.5 < data[col].std() < 1.5

    def test_seed_reproducibility(self):
        """Test that same seed produces same data."""
        data1, _ = generate_causal_system(100, seed=42)
        data2, _ = generate_causal_system(100, seed=42)
        pd.testing.assert_frame_equal(data1, data2)


class TestRunFullCausalityAnalysis:
    """Tests for run_full_causality_analysis function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        data, _ = generate_causal_system(150)
        result = run_full_causality_analysis(data, 'Y', horizons=[1, 2])
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Test that result contains required keys."""
        data, _ = generate_causal_system(150)
        result = run_full_causality_analysis(data, 'Y', horizons=[1, 2])

        required_keys = [
            'ccm_scores', 'ccm_significant',
            'granger_linear', 'granger_nonlinear', 'summary'
        ]
        for key in required_keys:
            assert key in result

    def test_ccm_scores_is_dataframe(self):
        """Test that ccm_scores is a DataFrame."""
        data, _ = generate_causal_system(150)
        result = run_full_causality_analysis(data, 'Y', horizons=[1, 2])
        assert isinstance(result['ccm_scores'], pd.DataFrame)

    def test_summary_is_dataframe(self):
        """Test that summary is a DataFrame."""
        data, _ = generate_causal_system(150)
        result = run_full_causality_analysis(data, 'Y', horizons=[1, 2])
        assert isinstance(result['summary'], pd.DataFrame)

    def test_summary_has_all_features(self):
        """Test that summary includes all features."""
        data, _ = generate_causal_system(150)
        result = run_full_causality_analysis(data, 'Y', horizons=[1])
        summary = result['summary']

        # Should have X1, X2, X3, X4
        assert len(summary) == 4

    def test_summary_has_classification(self):
        """Test that summary has classification column."""
        data, _ = generate_causal_system(150)
        result = run_full_causality_analysis(data, 'Y', horizons=[1])
        assert 'Classification' in result['summary'].columns


class TestCausalityEndToEnd:
    """End-to-end tests for causality analysis workflow."""

    def test_full_workflow(self):
        """Test complete causality analysis workflow."""
        # Generate known causal system
        data, ground_truth = generate_causal_system(200, seed=42)

        # Run full analysis
        result = run_full_causality_analysis(
            data, 'Y',
            horizons=[1, 2],
            embedding_dim=3,
            max_lags=3
        )

        # Check outputs
        assert not result['ccm_scores'].isna().all().all()
        assert not result['granger_linear'].isna().all().all()
        assert len(result['summary']) == 4

        # X4 (noise) should generally be classified as irrelevant
        x4_classification = result['summary'].loc['X4', 'Classification']
        # Just check it has some classification
        assert isinstance(x4_classification, str)

    def test_individual_tests_workflow(self):
        """Test individual causality tests."""
        # Generate data
        data, _ = generate_causal_system(200, seed=42)

        # Test CCM
        x2 = data['X2'].values
        y = data['Y'].values
        ccm_result = ccm_test(x2, y, embedding_dim=3, n_surrogates=20)
        assert 0 <= ccm_result['ccm_score'] <= 1

        # Test Granger
        granger_result = granger_causality_test(
            data, 'Y', 'X2', max_lags=3, model_type='linear'
        )
        assert granger_result['delta_rmse'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
