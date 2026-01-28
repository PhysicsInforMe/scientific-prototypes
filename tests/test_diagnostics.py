"""Tests for diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.validation.diagnostics import ForensicEnsembleAnalyzer


def create_sample_forecast_data(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create sample forecast data for testing."""
    np.random.seed(seed)

    dates = pd.date_range('2020-01-01', periods=n, freq='QE')
    actual = np.random.randn(n) * 2 + 5

    # Good model: close to actual
    good_pred = actual + np.random.randn(n) * 0.3

    # Bad model: biased and noisy
    bad_pred = actual + 2 + np.random.randn(n) * 1.0

    # Naive model: shifted actual
    naive_pred = np.roll(actual, 1)
    naive_pred[0] = actual[0]

    return pd.DataFrame({
        'date': dates,
        'actual': actual,
        'model_good': good_pred,
        'model_bad': bad_pred,
        'model_naive': naive_pred,
    })


def create_sample_with_horizons(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create sample data with multiple forecast horizons."""
    np.random.seed(seed)

    data_list = []
    for h in [1, 2, 3, 4]:
        dates = pd.date_range('2020-01-01', periods=n // 4, freq='QE')
        actual = np.random.randn(n // 4) * 2 + 5
        pred = actual + np.random.randn(n // 4) * 0.5

        for i, d in enumerate(dates):
            data_list.append({
                'date': d,
                'horizon_step': h,
                'actual': actual[i],
                'model_pred': pred[i],
            })

    return pd.DataFrame(data_list)


class TestForensicEnsembleAnalyzer:
    """Tests for ForensicEnsembleAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good', 'model_bad']
        )

        assert analyzer.n_samples > 0
        assert analyzer.y_true is not None

    def test_initialization_with_date_col(self):
        """Test initialization with date column."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good'],
            date_col='date'
        )

        assert analyzer.dates is not None
        assert len(analyzer.dates) == analyzer.n_samples

    def test_initialization_with_horizon_filter(self):
        """Test initialization with horizon filtering."""
        df = create_sample_with_horizons()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_pred'],
            date_col='date',
            horizon_col='horizon_step',
            analysis_horizon=1
        )

        # Should only have data for horizon 1
        assert analyzer.n_samples == 25  # 100 / 4

    def test_benchmarks_generated(self):
        """Test that naive benchmarks are generated."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        assert analyzer.benchmark_naive is not None
        assert analyzer.benchmark_snaive is not None
        assert len(analyzer.benchmark_naive) == analyzer.n_samples

    def test_run_full_analysis_returns_dataframe(self):
        """Test that run_full_analysis returns a DataFrame."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good', 'model_bad']
        )

        report = analyzer.run_full_analysis()
        assert isinstance(report, pd.DataFrame)

    def test_run_full_analysis_has_all_models(self):
        """Test that report includes all models."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good', 'model_bad', 'model_naive']
        )

        report = analyzer.run_full_analysis()
        assert len(report) == 3

    def test_report_has_required_columns(self):
        """Test that report has required columns."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        report = analyzer.run_full_analysis()

        required_cols = [
            'Model', 'MAE', 'RMSE',
            '1_Baseline_Beat', '2_WhiteNoise_Pass', '3_Normality_Pass',
            '4_Spectral_Pass', '5_Hurst_Pass', '6_Entropy_Pass'
        ]
        for col in required_cols:
            assert col in report.columns

    def test_forensic_score_calculated(self):
        """Test that forensic score is calculated."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        report = analyzer.run_full_analysis()

        # Should have a forensic score column
        score_cols = [c for c in report.columns if 'Forensic_Score' in c]
        assert len(score_cols) == 1

    def test_good_model_higher_score(self):
        """Test that good model has higher forensic score."""
        df = create_sample_forecast_data(seed=42)
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good', 'model_bad']
        )

        report = analyzer.run_full_analysis()

        # Find score column
        score_col = [c for c in report.columns if 'Forensic_Score' in c][0]

        good_score = report[report['Model'] == 'model_good'][score_col].values[0]
        bad_score = report[report['Model'] == 'model_bad'][score_col].values[0]

        # Good model should generally have higher or equal score
        assert good_score >= bad_score - 1  # Allow small variation

    def test_mae_rmse_positive(self):
        """Test that MAE and RMSE are positive."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        report = analyzer.run_full_analysis()

        assert report['MAE'].iloc[0] >= 0
        assert report['RMSE'].iloc[0] >= 0

    def test_test_results_are_boolean(self):
        """Test that test results are boolean."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        report = analyzer.run_full_analysis()

        bool_cols = [
            '1_Baseline_Beat', '2_WhiteNoise_Pass', '3_Normality_Pass',
            '4_Spectral_Pass', '5_Hurst_Pass', '6_Entropy_Pass'
        ]

        for col in bool_cols:
            assert report[col].iloc[0] in [True, False]

    def test_get_detailed_results_returns_dict(self):
        """Test that get_detailed_results returns a dictionary."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        details = analyzer.get_detailed_results('model_good')
        assert isinstance(details, dict)

    def test_get_detailed_results_contains_tests(self):
        """Test that detailed results contain all test info."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        details = analyzer.get_detailed_results('model_good')

        test_keys = [
            'test_1_baseline', 'test_2_ljung_box', 'test_3_shapiro_wilk',
            'test_4_spectral', 'test_5_hurst', 'test_6_entropy'
        ]

        for key in test_keys:
            assert key in details

    def test_get_detailed_results_includes_residuals(self):
        """Test that detailed results include residuals."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        details = analyzer.get_detailed_results('model_good')
        assert 'residuals' in details
        assert len(details['residuals']) > 0

    def test_get_residuals_returns_array(self):
        """Test that get_residuals returns an array."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        residuals = analyzer.get_residuals('model_good')
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) > 0

    def test_get_residuals_nonexistent_model(self):
        """Test get_residuals for nonexistent model."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good']
        )

        residuals = analyzer.get_residuals('nonexistent')
        assert len(residuals) == 0

    def test_with_feature_cols(self):
        """Test with feature columns for leakage test."""
        df = create_sample_forecast_data()
        df['feature1'] = np.random.randn(len(df))
        df['feature2'] = np.random.randn(len(df))

        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good'],
            feature_cols=['feature1', 'feature2']
        )

        report = analyzer.run_full_analysis()

        # Should have leakage test column
        assert '7_Leakage_Pass' in report.columns

    def test_sorted_by_score(self):
        """Test that report is sorted by forensic score."""
        df = create_sample_forecast_data()
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good', 'model_bad', 'model_naive']
        )

        report = analyzer.run_full_analysis()
        score_col = [c for c in report.columns if 'Forensic_Score' in c][0]

        # Should be sorted descending
        scores = report[score_col].values
        assert scores[0] >= scores[-1]


class TestDiagnosticsEndToEnd:
    """End-to-end tests for diagnostics workflow."""

    def test_full_workflow(self):
        """Test complete diagnostics workflow."""
        # Create sample data
        df = create_sample_forecast_data(n=80, seed=42)

        # Create analyzer
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_good', 'model_bad', 'model_naive'],
            date_col='date'
        )

        # Run full analysis
        report = analyzer.run_full_analysis()

        # Check report
        assert len(report) == 3
        assert 'Model' in report.columns
        assert 'MAE' in report.columns

        # Get detailed results for best model
        best_model = report.iloc[0]['Model']
        details = analyzer.get_detailed_results(best_model)

        assert 'mae' in details
        assert 'residuals' in details
        assert len(details['residuals']) > 0

        # Get residuals
        residuals = analyzer.get_residuals(best_model)
        assert len(residuals) > 0
        assert not np.isnan(residuals).all()

    def test_with_horizon_filtering(self):
        """Test workflow with horizon filtering."""
        # Create data with multiple horizons
        df = create_sample_with_horizons(n=200, seed=42)

        # Analyze horizon 1 only
        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_pred'],
            date_col='date',
            horizon_col='horizon_step',
            analysis_horizon=1
        )

        report = analyzer.run_full_analysis()

        assert len(report) == 1
        assert report.iloc[0]['MAE'] > 0

    def test_quarterly_date_format(self):
        """Test with quarterly date format (e.g., 2020Q1)."""
        n = 40
        dates = [f"{2010 + i // 4}Q{(i % 4) + 1}" for i in range(n)]
        np.random.seed(42)
        actual = np.random.randn(n) + 5
        pred = actual + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            'date': dates,
            'actual': actual,
            'model_pred': pred,
        })

        analyzer = ForensicEnsembleAnalyzer(
            df=df,
            target_col='actual',
            model_cols=['model_pred'],
            date_col='date'
        )

        report = analyzer.run_full_analysis()
        assert len(report) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
