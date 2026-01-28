"""
Mixed Frequency Imputation Module.

This module provides tools for handling missing values in mixed-frequency
time series data using MICE (Multiple Imputation by Chained Equations).
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def align_to_quarterly(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    aggregation: str = 'last'
) -> pd.Series:
    """
    Align a time series of any frequency to a quarterly frequency.

    Args:
        series: Input time series with DatetimeIndex. Can be daily, weekly,
            monthly, or any other frequency.
        target_index: Target quarterly DatetimeIndex to align to.
        aggregation: How to aggregate values within each quarter.
            Options: 'last', 'mean', 'sum', 'first', 'min', 'max'.

    Returns:
        Series with quarterly frequency aligned to target_index.

    Raises:
        ValueError: If aggregation method is not recognized.

    Example:
        >>> monthly = pd.Series([1, 2, 3, 4, 5, 6],
        ...     index=pd.date_range('2020-01', periods=6, freq='M'))
        >>> quarterly_index = pd.date_range('2020-03', periods=2, freq='Q')
        >>> aligned = align_to_quarterly(monthly, quarterly_index, 'mean')
    """
    valid_aggregations = ['last', 'mean', 'sum', 'first', 'min', 'max']
    if aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {valid_aggregations}")

    # Resample to quarterly
    if aggregation == 'last':
        quarterly = series.resample('QE').last()
    elif aggregation == 'mean':
        quarterly = series.resample('QE').mean()
    elif aggregation == 'sum':
        quarterly = series.resample('QE').sum()
    elif aggregation == 'first':
        quarterly = series.resample('QE').first()
    elif aggregation == 'min':
        quarterly = series.resample('QE').min()
    else:  # max
        quarterly = series.resample('QE').max()

    # Reindex to target
    return quarterly.reindex(target_index)


class MixedFrequencyImputer:
    """
    MICE-based imputer for mixed-frequency time series data.

    This class wraps scikit-learn's IterativeImputer to handle missing values
    in time series data with different frequencies (daily, weekly, monthly,
    quarterly). It first aligns all series to a common (quarterly) frequency,
    then applies MICE imputation.

    MICE (Multiple Imputation by Chained Equations) imputes missing values
    by modeling each feature as a function of other features in a round-robin
    fashion.

    Attributes:
        is_fitted: Whether the imputer has been fitted.
        feature_names: Names of features after transformation.

    Example:
        >>> imputer = MixedFrequencyImputer()
        >>> X_dict = {
        ...     'monthly_var': monthly_series,
        ...     'weekly_var': weekly_series,
        ... }
        >>> imputer.fit(X_dict, quarterly_target_index)
        >>> imputed_df = imputer.transform(X_dict)
    """

    def __init__(
        self,
        max_iter: int = 10,
        random_state: Optional[int] = 42,
        verbose: int = 0
    ):
        """
        Initialize the MixedFrequencyImputer.

        Args:
            max_iter: Maximum number of imputation rounds.
            random_state: Random seed for reproducibility.
            verbose: Verbosity level (0=silent, 1=progress).
        """
        self.imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose
        )
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self._target_index: Optional[pd.DatetimeIndex] = None

    def _transform_to_quarterly(
        self,
        X_dict: Dict[str, pd.Series],
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Transform mixed-frequency data to quarterly DataFrame.

        Args:
            X_dict: Dictionary of series with various frequencies.
            target_index: Quarterly index to align to.

        Returns:
            DataFrame with all features aligned to quarterly frequency.
        """
        result_df = pd.DataFrame(index=target_index)

        for name, series in X_dict.items():
            freq = pd.infer_freq(series.index)

            if freq is None:
                # Try to infer from time deltas
                if len(series.index) < 2:
                    result_df[name] = series.reindex(target_index)
                    continue
                avg_delta = (series.index[1:] - series.index[:-1]).mean()
                if avg_delta <= pd.Timedelta(days=8):
                    freq = 'D'  # Treat as daily/weekly
                elif avg_delta <= pd.Timedelta(days=32):
                    freq = 'M'
                else:
                    freq = 'Q'

            if freq and 'M' in freq:
                # Monthly data: expand to 3 columns (one per month in quarter)
                monthly_features = self._expand_monthly_to_quarterly(series, name)
                result_df = result_df.join(monthly_features, how='left')

            elif freq and ('W' in freq or 'D' in freq or 'B' in freq):
                # Weekly/Daily data: compute aggregate statistics
                stats = series.resample('QE').agg(['mean', 'std', 'min', 'max', 'last'])
                stats.columns = [f"{name}_{agg}" for agg in stats.columns]
                result_df = result_df.join(stats, how='left')

            elif freq and 'Q' in freq:
                # Already quarterly
                result_df = result_df.join(series.to_frame(name), how='left')

            else:
                # Unknown frequency: try direct reindex
                result_df[name] = series.reindex(target_index)

        return result_df

    def _expand_monthly_to_quarterly(
        self,
        series: pd.Series,
        name: str
    ) -> pd.DataFrame:
        """
        Expand monthly series to quarterly with one column per month.

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

        # Pivot to get one column per month
        pivoted = df.pivot_table(
            index=['year', 'quarter'],
            columns='month_in_quarter',
            values='value'
        )
        pivoted.columns = [f"{name}_M{int(c)}" for c in pivoted.columns]

        # Create quarter-end date index
        pivoted['quarter_end'] = pd.to_datetime(
            pivoted.reset_index().apply(
                lambda r: f"{int(r['year'])}-{int(r['quarter'] * 3):02d}-01",
                axis=1
            )
        ) + pd.offsets.MonthEnd(0)

        pivoted = pivoted.reset_index(drop=True).set_index('quarter_end')
        return pivoted

    def fit(
        self,
        X_dict: Dict[str, pd.Series],
        target_index: pd.DatetimeIndex,
        y: Optional[pd.Series] = None
    ) -> 'MixedFrequencyImputer':
        """
        Fit the imputer on mixed-frequency data.

        Args:
            X_dict: Dictionary mapping feature names to time series.
            target_index: Quarterly DatetimeIndex to align to.
            y: Optional target series (included in imputation).

        Returns:
            Self for method chaining.
        """
        self._target_index = target_index

        # Transform to quarterly
        df = self._transform_to_quarterly(X_dict, target_index)

        if y is not None:
            df = df.join(y, how='left')

        # Handle completely empty columns
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            df[all_nan_cols] = 0

        # Fit imputer
        self.imputer.fit(df)
        self.feature_names = list(df.columns)
        self.is_fitted = True

        return self

    def transform(
        self,
        X_dict: Dict[str, pd.Series],
        target_index: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Transform and impute mixed-frequency data.

        Args:
            X_dict: Dictionary mapping feature names to time series.
            target_index: Quarterly DatetimeIndex to align to.
                Uses the index from fit() if not provided.

        Returns:
            DataFrame with imputed values at quarterly frequency.

        Raises:
            ValueError: If imputer has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")

        if target_index is None:
            target_index = self._target_index

        # Transform to quarterly
        df = self._transform_to_quarterly(X_dict, target_index)

        # Handle completely empty columns
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            df[all_nan_cols] = 0

        # Ensure same columns as fit
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_names]

        # Impute
        imputed_array = self.imputer.transform(df)
        imputed_df = pd.DataFrame(
            imputed_array,
            index=target_index,
            columns=self.feature_names
        )

        return imputed_df

    def fit_transform(
        self,
        X_dict: Dict[str, pd.Series],
        target_index: pd.DatetimeIndex,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X_dict: Dictionary mapping feature names to time series.
            target_index: Quarterly DatetimeIndex to align to.
            y: Optional target series.

        Returns:
            DataFrame with imputed values at quarterly frequency.
        """
        self.fit(X_dict, target_index, y)
        return self.transform(X_dict, target_index)

    def get_imputation_report(
        self,
        X_dict: Dict[str, pd.Series],
        target_index: pd.DatetimeIndex
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a report on data coverage and imputation.

        Args:
            X_dict: Dictionary of original series.
            target_index: Target quarterly index.

        Returns:
            Tuple of (input_report, imputation_report) DataFrames.
        """
        # Input data report
        input_data = []
        for name, series in X_dict.items():
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
                'Inferred Frequency': freq_char,
                'Missing Values': series.isna().sum()
            })

        input_df = pd.DataFrame(input_data).set_index('Variable')

        # Imputation report
        imputation_data = []
        for name, series in X_dict.items():
            # Count quarters needing imputation at start
            imputed_start = len(target_index[target_index < series.index.min()])
            # Count quarters needing imputation at end
            imputed_end = len(target_index[target_index > series.index.max()])

            imputation_data.append({
                'Variable': name,
                'Imputed (Start)': imputed_start,
                'Imputed (End)': imputed_end,
                'Total Imputed': imputed_start + imputed_end
            })

        imputation_df = pd.DataFrame(imputation_data).set_index('Variable')

        return input_df, imputation_df
