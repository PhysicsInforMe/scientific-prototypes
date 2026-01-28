"""
Global Boost Forecaster Module.

This module provides a gradient boosting-based forecasting approach that
handles mixed-frequency data using MICE imputation for missing values.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class GlobalBoostForecaster:
    """
    Global forecasting model using gradient boosting with MICE imputation.

    This class implements a "global" forecasting approach where a single model
    is trained on data from multiple entities (e.g., countries, products) to
    leverage cross-entity patterns. It handles mixed-frequency features by
    transforming them to quarterly frequency and uses MICE for imputation.

    Attributes:
        is_fitted: Whether the model has been fitted.
        feature_names: List of feature names after transformation.
        model: The underlying gradient boosting model.

    Example:
        >>> forecaster = GlobalBoostForecaster()
        >>> all_data = {
        ...     'US': {'y': us_gdp, 'X': us_features},
        ...     'UK': {'y': uk_gdp, 'X': uk_features},
        ... }
        >>> forecaster.fit(all_data)
        >>> prediction = forecaster.predict(us_features, us_gdp, 'US')
    """

    def __init__(self, model: Optional[Any] = None, random_state: int = 42):
        """
        Initialize the GlobalBoostForecaster.

        Args:
            model: Custom sklearn-compatible model. If None, uses LightGBM
                if available, otherwise raises ImportError.
            random_state: Random seed for reproducibility.

        Raises:
            ImportError: If model is None and LightGBM is not installed.
        """
        if model is None:
            if not HAS_LIGHTGBM:
                raise ImportError(
                    "LightGBM is required for GlobalBoostForecaster. "
                    "Install it with: pip install lightgbm"
                )
            self.model = lgb.LGBMRegressor(random_state=random_state, verbose=-1)
        else:
            self.model = model

        self.imputer = IterativeImputer(
            random_state=random_state,
            verbose=0
        )

        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self._original_data: Optional[pd.DataFrame] = None
        self._imputed_data: Optional[pd.DataFrame] = None

    def _create_features(
        self,
        y_series: pd.Series,
        X_dict: Dict[str, pd.Series],
        entity_id: str
    ) -> pd.DataFrame:
        """
        Transform raw data into quarterly feature DataFrame.

        Args:
            y_series: Target series at quarterly frequency.
            X_dict: Dictionary of predictor series at various frequencies.
            entity_id: Identifier for the entity (e.g., country code).

        Returns:
            DataFrame with all features aligned to quarterly frequency.
        """
        result_df = pd.DataFrame(index=y_series.index)

        for name, series in X_dict.items():
            freq = pd.infer_freq(series.index)

            if freq and 'M' in freq:
                # Monthly: expand to 3 columns per quarter
                monthly_features = self._expand_monthly(series, name)
                result_df = result_df.join(monthly_features, how='left')

            elif freq and ('W' in freq or 'D' in freq or 'B' in freq):
                # Weekly/Daily: compute aggregate statistics
                stats = series.resample('QE').agg(['mean', 'std', 'min', 'max', 'last'])
                stats.columns = [f"{name}_{agg}" for agg in stats.columns]
                result_df = result_df.join(stats, how='left')

            elif freq and 'Q' in freq:
                # Already quarterly
                result_df = result_df.join(series.to_frame(name), how='left')

            else:
                # Unknown: try quarterly resampling
                try:
                    resampled = series.resample('QE').last()
                    result_df = result_df.join(resampled.to_frame(name), how='left')
                except Exception:
                    result_df[name] = np.nan

        result_df['entity_id'] = entity_id
        result_df['entity_id'] = result_df['entity_id'].astype('category')
        result_df = result_df.join(y_series, how='left')

        return result_df

    def _expand_monthly(self, series: pd.Series, name: str) -> pd.DataFrame:
        """
        Expand monthly series to quarterly with month-specific columns.

        Args:
            series: Monthly time series.
            name: Base name for columns.

        Returns:
            DataFrame with columns {name}_M1, {name}_M2, {name}_M3.
        """
        df = series.to_frame(name='value')
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df['month_in_quarter'] = ((df.index.month - 1) % 3) + 1

        pivoted = df.pivot_table(
            index=['year', 'quarter'],
            columns='month_in_quarter',
            values='value'
        )
        pivoted.columns = [f"{name}_M{int(c)}" for c in pivoted.columns]

        # Create quarter-end date index
        pivoted = pivoted.reset_index()
        pivoted['quarter_end'] = pd.to_datetime(
            pivoted.apply(
                lambda r: f"{int(r['year'])}-{int(r['quarter'] * 3):02d}-01",
                axis=1
            )
        ) + pd.offsets.MonthEnd(0)

        return pivoted.drop(columns=['year', 'quarter']).set_index('quarter_end')

    def fit(
        self,
        all_entities_data: Dict[str, Dict[str, Union[pd.Series, Dict]]]
    ) -> 'GlobalBoostForecaster':
        """
        Train a global model on data from multiple entities.

        Args:
            all_entities_data: Dictionary mapping entity IDs to their data.
                Each entity's data should have:
                - 'y': Target Series (quarterly)
                - 'X': Dict of predictor Series (mixed frequency)

        Returns:
            Self for method chaining.

        Example:
            >>> data = {
            ...     'US': {'y': us_gdp, 'X': {'rate': us_rate, 'pmi': us_pmi}},
            ...     'UK': {'y': uk_gdp, 'X': {'rate': uk_rate, 'pmi': uk_pmi}},
            ... }
            >>> forecaster.fit(data)
        """
        # Create global training set
        all_features = []
        for entity_id, data in all_entities_data.items():
            entity_df = self._create_features(data['y'], data['X'], entity_id)
            all_features.append(entity_df)

        full_df = pd.concat(all_features)
        full_df['entity_id'] = full_df['entity_id'].astype('category')

        # Separate target and features
        # Use specific patterns to identify target column
        # Match columns containing 'gdp' or 'target', or columns starting/ending with 'y'
        def is_target_col(col):
            col_lower = col.lower()
            if col == 'entity_id':
                return False
            # Match specific substrings
            if 'gdp' in col_lower or 'target' in col_lower:
                return True
            # Match column names that are exactly 'y' or start with 'y_' or end with '_y'
            if col_lower == 'y' or col_lower.startswith('y_') or col_lower.endswith('_y'):
                return True
            return False

        y_cols = [col for col in full_df.columns if is_target_col(col)]
        if not y_cols:
            # Use last non-entity column as target
            non_entity_cols = [c for c in full_df.columns if c != 'entity_id']
            y_cols = [non_entity_cols[-1]] if non_entity_cols else []

        y_train = full_df[y_cols]
        X_train_raw = full_df.drop(columns=y_cols)

        # Store original data for analysis
        self._original_data = X_train_raw.copy()

        # Handle completely empty columns
        all_nan_cols = X_train_raw.columns[X_train_raw.isna().all()].tolist()
        if all_nan_cols:
            X_train_raw[all_nan_cols] = 0

        # Apply MICE imputation
        X_impute_ready = X_train_raw.copy()
        X_impute_ready['entity_id'] = X_impute_ready['entity_id'].cat.codes

        X_imputed_array = self.imputer.fit_transform(X_impute_ready)
        X_imputed = pd.DataFrame(
            X_imputed_array,
            index=X_train_raw.index,
            columns=X_impute_ready.columns
        )
        X_imputed['entity_id'] = pd.Series(
            X_train_raw['entity_id'].values,
            index=X_imputed.index
        )

        self._imputed_data = X_imputed.copy()
        self.feature_names = list(X_imputed.columns)

        # Train model
        self.model.fit(X_imputed, y_train.values.ravel())
        self.is_fitted = True

        return self

    def predict(
        self,
        X_dict: Dict[str, pd.Series],
        y_series: pd.Series,
        entity_id: str,
        n_periods: int = 1
    ) -> np.ndarray:
        """
        Generate predictions for a single entity.

        Args:
            X_dict: Dictionary of predictor series.
            y_series: Historical target series.
            entity_id: Entity identifier.
            n_periods: Number of periods to predict (from end of data).

        Returns:
            Array of predictions.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Create features
        features = self._create_features(y_series, X_dict, entity_id)
        last_rows = features.iloc[-n_periods:]

        # Get target column name
        y_col_name = y_series.name
        X_pred_raw = last_rows.drop(columns=[y_col_name], errors='ignore')

        # Handle empty columns
        all_nan_cols = X_pred_raw.columns[X_pred_raw.isna().all()].tolist()
        if all_nan_cols:
            X_pred_raw[all_nan_cols] = 0

        # Prepare for imputation
        X_pred_impute = X_pred_raw.copy()
        X_pred_impute['entity_id'] = X_pred_impute['entity_id'].astype('category').cat.codes

        # Ensure same columns
        for col in self.feature_names:
            if col not in X_pred_impute.columns:
                X_pred_impute[col] = np.nan

        X_pred_impute = X_pred_impute[self.feature_names]

        # Impute and predict
        X_pred_imputed = self.imputer.transform(X_pred_impute)
        X_pred = pd.DataFrame(
            X_pred_imputed,
            index=X_pred_raw.index,
            columns=self.feature_names
        )
        X_pred['entity_id'] = pd.Series(
            X_pred_raw['entity_id'].values,
            index=X_pred.index
        )

        return self.model.predict(X_pred[self.feature_names])

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from the trained model.

        Returns:
            Series with feature names as index and importance as values.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            return importance
        else:
            raise AttributeError("Model does not have feature_importances_")

    def get_transformed_features(
        self,
        X_dict: Dict[str, pd.Series],
        y_series: pd.Series,
        entity_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Get the final transformed and imputed features for an entity.

        Args:
            X_dict: Dictionary of predictor series.
            y_series: Target series.
            entity_id: Entity identifier.

        Returns:
            DataFrame with transformed features, or None if not fitted.
        """
        if not self.is_fitted:
            return None

        # Create raw features
        raw_features = self._create_features(y_series, X_dict, entity_id)

        # Separate target
        y_col_name = y_series.name
        X_raw = raw_features.drop(columns=[y_col_name], errors='ignore')

        # Handle empty columns
        all_nan_cols = X_raw.columns[X_raw.isna().all()].tolist()
        if all_nan_cols:
            X_raw[all_nan_cols] = 0

        # Impute
        X_impute_ready = X_raw.copy()
        X_impute_ready['entity_id'] = X_impute_ready['entity_id'].astype('category').cat.codes

        X_imputed_array = self.imputer.transform(X_impute_ready)
        X_imputed = pd.DataFrame(
            X_imputed_array,
            index=X_raw.index,
            columns=X_impute_ready.columns
        )
        X_imputed['entity_id'] = pd.Series(
            X_raw['entity_id'].values,
            index=X_imputed.index
        )

        return X_imputed[self.feature_names]

    def generate_imputation_report(
        self,
        all_entities_data: Dict[str, Dict[str, Union[pd.Series, Dict]]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate reports on data coverage and imputation.

        Args:
            all_entities_data: The training data dictionary.

        Returns:
            Tuple of (input_report, imputation_report) DataFrames.
        """
        if self._original_data is None:
            return pd.DataFrame(), pd.DataFrame()

        # Get reference entity
        ref_entity = list(all_entities_data.keys())[0]
        X_dict_ref = all_entities_data[ref_entity]['X']
        y_series_ref = all_entities_data[ref_entity]['y']

        # Input data report
        input_data = []
        for name, series in X_dict_ref.items():
            freq_str = pd.infer_freq(series.index)
            if freq_str:
                freq_char = ''.join(filter(str.isalpha, freq_str.split('-')[0]))
            else:
                freq_char = 'N/A'

            input_data.append({
                'Variable': name,
                'Total Samples': len(series),
                'Start Date': series.index.min(),
                'End Date': series.index.max(),
                'Inferred Frequency': freq_char
            })

        input_df = pd.DataFrame(input_data).set_index('Variable')

        # Imputation report
        target_index = y_series_ref.index
        imputation_data = []

        for name, series in X_dict_ref.items():
            imputed_start = len(target_index[target_index < series.index.min()])
            imputed_end = len(target_index[target_index > series.index.max()])

            imputation_data.append({
                'Variable': name,
                'Imputed (Start)': imputed_start,
                'Imputed (End)': imputed_end,
                'Total Imputed': imputed_start + imputed_end,
                'Target Frequency': target_index.freqstr
            })

        imputation_df = pd.DataFrame(imputation_data).set_index('Variable')

        return input_df, imputation_df
