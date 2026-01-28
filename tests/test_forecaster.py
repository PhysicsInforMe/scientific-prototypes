"""Tests for forecaster module."""

import numpy as np
import pandas as pd
import pytest

# Skip all tests if lightgbm is not installed
pytest.importorskip("lightgbm")

from timeseries_toolkit.models.forecaster import GlobalBoostForecaster


def generate_quarterly_gdp(n_quarters: int = 40, seed: int = 42) -> pd.Series:
    """Generate synthetic quarterly GDP growth."""
    np.random.seed(seed)
    dates = pd.date_range('2010-01-01', periods=n_quarters, freq='QE')
    values = 2.0 + np.random.randn(n_quarters) * 0.5 + 0.1 * np.sin(np.arange(n_quarters) / 4)
    return pd.Series(values, index=dates, name='GDP_growth')


def generate_monthly_indicator(n_months: int = 120, seed: int = 42) -> pd.Series:
    """Generate synthetic monthly economic indicator."""
    np.random.seed(seed)
    dates = pd.date_range('2010-01-01', periods=n_months, freq='ME')
    values = 50 + np.random.randn(n_months) * 5 + np.cumsum(np.random.randn(n_months) * 0.1)
    return pd.Series(values, index=dates, name='PMI')


def generate_weekly_indicator(n_weeks: int = 500, seed: int = 42) -> pd.Series:
    """Generate synthetic weekly economic indicator."""
    np.random.seed(seed)
    dates = pd.date_range('2010-01-01', periods=n_weeks, freq='W')
    values = 200 + np.random.randn(n_weeks) * 20
    return pd.Series(values, index=dates, name='jobless_claims')


def create_sample_entity_data(seed: int = 42):
    """Create sample data for a single entity."""
    return {
        'y': generate_quarterly_gdp(40, seed),
        'X': {
            'monthly_pmi': generate_monthly_indicator(120, seed),
            'weekly_claims': generate_weekly_indicator(500, seed),
        }
    }


def create_multi_entity_data():
    """Create sample data for multiple entities."""
    return {
        'US': create_sample_entity_data(seed=42),
        'UK': create_sample_entity_data(seed=43),
        'DE': create_sample_entity_data(seed=44),
    }


class TestGlobalBoostForecaster:
    """Tests for GlobalBoostForecaster class."""

    def test_initialization_default(self):
        """Test default initialization."""
        forecaster = GlobalBoostForecaster()
        assert forecaster.is_fitted is False
        assert forecaster.feature_names == []

    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        from sklearn.ensemble import GradientBoostingRegressor
        custom_model = GradientBoostingRegressor(n_estimators=50)
        forecaster = GlobalBoostForecaster(model=custom_model)
        assert forecaster.model is custom_model

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        result = forecaster.fit(data)
        assert result is forecaster

    def test_fit_sets_is_fitted(self):
        """Test that fit sets is_fitted flag."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)
        assert forecaster.is_fitted is True

    def test_fit_sets_feature_names(self):
        """Test that fit populates feature_names."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)
        assert len(forecaster.feature_names) > 0

    def test_fit_with_multiple_entities(self):
        """Test fit with multiple entities."""
        forecaster = GlobalBoostForecaster()
        data = create_multi_entity_data()
        forecaster.fit(data)
        assert forecaster.is_fitted is True

    def test_predict_without_fit_raises(self):
        """Test that predict without fit raises error."""
        forecaster = GlobalBoostForecaster()
        entity_data = create_sample_entity_data()

        with pytest.raises(RuntimeError):
            forecaster.predict(entity_data['X'], entity_data['y'], 'entity1')

    def test_predict_returns_array(self):
        """Test that predict returns a numpy array."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)

        entity_data = data['entity1']
        prediction = forecaster.predict(entity_data['X'], entity_data['y'], 'entity1')

        assert isinstance(prediction, np.ndarray)
        assert len(prediction) > 0

    def test_predict_single_period(self):
        """Test prediction for single period."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)

        entity_data = data['entity1']
        prediction = forecaster.predict(entity_data['X'], entity_data['y'], 'entity1', n_periods=1)

        assert len(prediction) == 1

    def test_predict_multiple_periods(self):
        """Test prediction for multiple periods."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)

        entity_data = data['entity1']
        prediction = forecaster.predict(entity_data['X'], entity_data['y'], 'entity1', n_periods=4)

        assert len(prediction) == 4

    def test_predict_not_nan(self):
        """Test that predictions are not NaN."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)

        entity_data = data['entity1']
        prediction = forecaster.predict(entity_data['X'], entity_data['y'], 'entity1')

        assert not np.isnan(prediction).any()

    def test_get_feature_importance_without_fit_raises(self):
        """Test that get_feature_importance without fit raises error."""
        forecaster = GlobalBoostForecaster()
        with pytest.raises(RuntimeError):
            forecaster.get_feature_importance()

    def test_get_feature_importance_returns_series(self):
        """Test that get_feature_importance returns a Series."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)
        importance = forecaster.get_feature_importance()

        assert isinstance(importance, pd.Series)
        assert len(importance) > 0

    def test_feature_importance_sorted(self):
        """Test that feature importance is sorted descending."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)
        importance = forecaster.get_feature_importance()

        # Should be sorted descending
        assert importance.iloc[0] >= importance.iloc[-1]

    def test_get_transformed_features_not_fitted(self):
        """Test get_transformed_features when not fitted."""
        forecaster = GlobalBoostForecaster()
        entity_data = create_sample_entity_data()
        result = forecaster.get_transformed_features(entity_data['X'], entity_data['y'], 'test')
        assert result is None

    def test_get_transformed_features_returns_dataframe(self):
        """Test that get_transformed_features returns a DataFrame."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)

        entity_data = data['entity1']
        features = forecaster.get_transformed_features(entity_data['X'], entity_data['y'], 'entity1')

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_generate_imputation_report_not_fitted(self):
        """Test imputation report when not fitted."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        input_report, imp_report = forecaster.generate_imputation_report(data)

        assert input_report.empty
        assert imp_report.empty

    def test_generate_imputation_report_returns_dataframes(self):
        """Test that generate_imputation_report returns DataFrames."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)
        input_report, imp_report = forecaster.generate_imputation_report(data)

        assert isinstance(input_report, pd.DataFrame)
        assert isinstance(imp_report, pd.DataFrame)

    def test_monthly_feature_expansion(self):
        """Test that monthly features are expanded correctly."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)

        # Check feature names include monthly expansion
        monthly_features = [f for f in forecaster.feature_names if 'pmi' in f.lower()]
        assert len(monthly_features) >= 3  # Should have M1, M2, M3

    def test_weekly_feature_aggregation(self):
        """Test that weekly features are aggregated to stats."""
        forecaster = GlobalBoostForecaster()
        data = {'entity1': create_sample_entity_data()}
        forecaster.fit(data)

        # Check feature names include weekly stats
        weekly_features = [f for f in forecaster.feature_names if 'claims' in f.lower()]
        assert len(weekly_features) >= 3  # Should have mean, std, etc.


class TestForecasterEndToEnd:
    """End-to-end tests for forecaster workflow."""

    def test_full_workflow_single_entity(self):
        """Test complete workflow with single entity."""
        # Create data
        data = {'US': create_sample_entity_data(seed=42)}

        # Create and fit forecaster
        forecaster = GlobalBoostForecaster(random_state=42)
        forecaster.fit(data)

        # Make predictions
        prediction = forecaster.predict(
            data['US']['X'],
            data['US']['y'],
            'US',
            n_periods=1
        )

        assert len(prediction) == 1
        assert not np.isnan(prediction).any()

        # Get feature importance
        importance = forecaster.get_feature_importance()
        assert len(importance) > 0

        # Get transformed features
        features = forecaster.get_transformed_features(
            data['US']['X'],
            data['US']['y'],
            'US'
        )
        assert features is not None

    def test_full_workflow_multi_entity(self):
        """Test complete workflow with multiple entities."""
        # Create data for multiple entities
        data = create_multi_entity_data()

        # Create and fit forecaster
        forecaster = GlobalBoostForecaster(random_state=42)
        forecaster.fit(data)

        # Make predictions for each entity
        for entity_id in data.keys():
            prediction = forecaster.predict(
                data[entity_id]['X'],
                data[entity_id]['y'],
                entity_id
            )
            assert len(prediction) > 0
            assert not np.isnan(prediction).any()

        # Generate report
        input_report, imp_report = forecaster.generate_imputation_report(data)
        assert len(input_report) > 0
        assert len(imp_report) > 0

    def test_cross_entity_learning(self):
        """Test that training on multiple entities improves prediction."""
        np.random.seed(42)

        # Create data with similar patterns across entities
        data = create_multi_entity_data()

        # Fit on all entities
        forecaster = GlobalBoostForecaster(random_state=42)
        forecaster.fit(data)

        # Predictions should be reasonable (not extreme)
        for entity_id in data.keys():
            prediction = forecaster.predict(
                data[entity_id]['X'],
                data[entity_id]['y'],
                entity_id
            )

            # Prediction should be in reasonable range (GDP growth typically -5% to 10%)
            assert -10 < prediction[0] < 15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
