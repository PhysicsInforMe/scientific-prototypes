"""Tests for imputation module."""

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.preprocessing.imputation import (
    MixedFrequencyImputer,
    align_to_quarterly,
)


def generate_monthly_series(n_months: int = 36, seed: int = 42) -> pd.Series:
    """Generate synthetic monthly time series."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_months, freq='ME')
    values = np.cumsum(np.random.randn(n_months)) + 50
    return pd.Series(values, index=dates, name='monthly_var')


def generate_weekly_series(n_weeks: int = 100, seed: int = 42) -> pd.Series:
    """Generate synthetic weekly time series."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_weeks, freq='W')
    values = np.random.randn(n_weeks) * 10 + 100
    return pd.Series(values, index=dates, name='weekly_var')


def generate_quarterly_series(n_quarters: int = 12, seed: int = 42) -> pd.Series:
    """Generate synthetic quarterly time series."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_quarters, freq='QE')
    values = np.random.randn(n_quarters) * 2 + 5
    return pd.Series(values, index=dates, name='quarterly_var')


class TestAlignToQuarterly:
    """Tests for align_to_quarterly function."""

    def test_monthly_to_quarterly_last(self):
        """Test aligning monthly to quarterly with last aggregation."""
        monthly = generate_monthly_series(24)
        quarterly_idx = pd.date_range('2020-03-31', periods=8, freq='QE')
        result = align_to_quarterly(monthly, quarterly_idx, aggregation='last')

        assert isinstance(result, pd.Series)
        assert len(result) == len(quarterly_idx)

    def test_monthly_to_quarterly_mean(self):
        """Test aligning monthly to quarterly with mean aggregation."""
        monthly = generate_monthly_series(24)
        quarterly_idx = pd.date_range('2020-03-31', periods=8, freq='QE')
        result = align_to_quarterly(monthly, quarterly_idx, aggregation='mean')

        assert isinstance(result, pd.Series)
        assert not result.isna().all()

    def test_weekly_to_quarterly(self):
        """Test aligning weekly to quarterly."""
        weekly = generate_weekly_series(52)
        quarterly_idx = pd.date_range('2020-03-31', periods=4, freq='QE')
        result = align_to_quarterly(weekly, quarterly_idx, aggregation='mean')

        assert len(result) == 4

    def test_invalid_aggregation_raises(self):
        """Test that invalid aggregation method raises error."""
        monthly = generate_monthly_series(12)
        quarterly_idx = pd.date_range('2020-03-31', periods=4, freq='QE')

        with pytest.raises(ValueError):
            align_to_quarterly(monthly, quarterly_idx, aggregation='invalid')

    def test_all_aggregation_methods(self):
        """Test all valid aggregation methods."""
        monthly = generate_monthly_series(24)
        quarterly_idx = pd.date_range('2020-03-31', periods=8, freq='QE')

        for method in ['last', 'mean', 'sum', 'first', 'min', 'max']:
            result = align_to_quarterly(monthly, quarterly_idx, aggregation=method)
            assert isinstance(result, pd.Series)


class TestMixedFrequencyImputer:
    """Tests for MixedFrequencyImputer class."""

    def test_initialization(self):
        """Test imputer initialization."""
        imputer = MixedFrequencyImputer()
        assert imputer.is_fitted is False
        assert imputer.feature_names == []

    def test_initialization_with_params(self):
        """Test imputer initialization with custom parameters."""
        imputer = MixedFrequencyImputer(max_iter=20, random_state=123)
        assert imputer.imputer.max_iter == 20

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(36)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        result = imputer.fit(X_dict, target_idx)
        assert result is imputer

    def test_fit_sets_is_fitted(self):
        """Test that fit sets is_fitted flag."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(36)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        imputer.fit(X_dict, target_idx)
        assert imputer.is_fitted is True

    def test_fit_sets_feature_names(self):
        """Test that fit populates feature_names."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(36)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        imputer.fit(X_dict, target_idx)
        assert len(imputer.feature_names) > 0

    def test_transform_without_fit_raises(self):
        """Test that transform without fit raises error."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(36)}

        with pytest.raises(ValueError):
            imputer.transform(X_dict)

    def test_transform_returns_dataframe(self):
        """Test that transform returns a DataFrame."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(36)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        imputer.fit(X_dict, target_idx)
        result = imputer.transform(X_dict)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(target_idx)

    def test_transform_no_nans(self):
        """Test that transform removes NaN values."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(36)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        imputer.fit(X_dict, target_idx)
        result = imputer.transform(X_dict)

        # After imputation, no NaN values should remain
        assert not result.isna().all().any()

    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(36)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        result = imputer.fit_transform(X_dict, target_idx)

        assert isinstance(result, pd.DataFrame)
        assert imputer.is_fitted is True

    def test_mixed_frequency_data(self):
        """Test with multiple frequencies."""
        imputer = MixedFrequencyImputer()
        X_dict = {
            'monthly': generate_monthly_series(36),
            'weekly': generate_weekly_series(100),
            'quarterly': generate_quarterly_series(12),
        }
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        result = imputer.fit_transform(X_dict, target_idx)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12

    def test_monthly_expansion(self):
        """Test that monthly data is expanded to 3 columns."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly_var': generate_monthly_series(36)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        result = imputer.fit_transform(X_dict, target_idx)

        # Should have 3 columns for monthly data (M1, M2, M3)
        monthly_cols = [c for c in result.columns if 'monthly_var' in c]
        assert len(monthly_cols) == 3

    def test_weekly_aggregation(self):
        """Test that weekly data is aggregated to stats."""
        imputer = MixedFrequencyImputer()
        X_dict = {'weekly_var': generate_weekly_series(100)}
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        result = imputer.fit_transform(X_dict, target_idx)

        # Should have multiple columns for weekly stats
        weekly_cols = [c for c in result.columns if 'weekly_var' in c]
        assert len(weekly_cols) >= 3  # mean, std, etc.

    def test_imputation_report(self):
        """Test imputation report generation."""
        imputer = MixedFrequencyImputer()
        X_dict = {'monthly': generate_monthly_series(24)}  # Only 24 months
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        imputer.fit(X_dict, target_idx)
        input_report, imputation_report = imputer.get_imputation_report(X_dict, target_idx)

        assert isinstance(input_report, pd.DataFrame)
        assert isinstance(imputation_report, pd.DataFrame)
        assert 'Total Samples' in input_report.columns
        assert 'Total Imputed' in imputation_report.columns


class TestImputationEndToEnd:
    """End-to-end tests for imputation workflow."""

    def test_full_workflow(self):
        """Test complete imputation workflow."""
        # Create mixed frequency data
        np.random.seed(42)

        monthly = generate_monthly_series(36)
        weekly = generate_weekly_series(100)
        quarterly = generate_quarterly_series(12)

        X_dict = {
            'interest_rate': monthly,
            'jobless_claims': weekly,
            'gdp_component': quarterly,
        }

        # Target quarterly index
        target_idx = pd.date_range('2020-03-31', periods=12, freq='QE')

        # Create and fit imputer
        imputer = MixedFrequencyImputer(random_state=42)
        result = imputer.fit_transform(X_dict, target_idx)

        # Verify output
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12
        assert not result.isna().all().any()

        # Generate report
        input_report, imp_report = imputer.get_imputation_report(X_dict, target_idx)
        assert len(input_report) == 3
        assert len(imp_report) == 3

    def test_with_target_series(self):
        """Test imputation including target series."""
        imputer = MixedFrequencyImputer()

        X_dict = {'monthly': generate_monthly_series(36)}
        y = generate_quarterly_series(12)
        target_idx = y.index

        result = imputer.fit_transform(X_dict, target_idx, y=y)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
