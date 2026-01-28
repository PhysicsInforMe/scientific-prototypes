"""
Fractional Differentiation Module.

This module provides functions for fractional differentiation of time series,
which allows transforming non-stationary series to stationary while preserving
more memory than integer differentiation.

The Fixed-Width Window (FFD) method is used, as described in:
"Advances in Financial Machine Learning" by Marcos López de Prado.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute weights for fractional differentiation using Fixed-Width Window method.

    The weights are computed using the formula:
        w_k = -w_{k-1} * (d - k + 1) / k

    Args:
        d: Fractional differentiation order. Typically 0 < d < 1, but can extend
           to d < 2 for more aggressive differentiation. Higher values result
           in more stationary but less memory-preserving transformations.
        threshold: Minimum absolute weight value to include. Weights below this
            threshold are discarded, controlling the effective window size.
            Smaller values increase precision but require more computation.

    Returns:
        Array of fractional differentiation weights in reverse chronological
        order (column vector shape: (n, 1)).

    Raises:
        ValueError: If threshold is not positive.

    Example:
        >>> weights = get_weights_ffd(0.5, threshold=1e-4)
        >>> print(f"Window size: {len(weights)}")
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive")

    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) <= threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights[::-1]).reshape(-1, 1)


def frac_diff_ffd(
    series: pd.DataFrame,
    d: float,
    threshold: float = 1e-5
) -> pd.DataFrame:
    """
    Apply fractional differentiation to a time series using Fixed-Width Window method.

    This method applies a weighted moving average filter where weights are
    determined by the fractional differentiation parameter d.

    Args:
        series: DataFrame containing time series data to differentiate.
            Each column is treated as a separate series. Index should be
            a DatetimeIndex or similar for proper alignment.
        d: Fractional differentiation order. Use values in (0, 1) for
            partial differentiation. d=0 returns the original series,
            d=1 is equivalent to first differencing.
        threshold: Weight cutoff threshold. Controls the effective window
            size - lower values use more historical data but increase
            computation time.

    Returns:
        DataFrame with fractionally differentiated series. Initial values
        will be NaN due to the window requirement. Column names are preserved
        from input.

    Raises:
        ValueError: If series is empty or d is negative.

    Example:
        >>> df = pd.DataFrame({'price': [100, 101, 102, 103, 104]})
        >>> diff_df = frac_diff_ffd(df, d=0.5)
    """
    if series.empty:
        raise ValueError("Input series cannot be empty")
    if d < 0:
        raise ValueError("Differentiation order d must be non-negative")

    # Handle d=0 case (no differentiation)
    if d == 0:
        return series.copy()

    # Compute weights and window size
    weights = get_weights_ffd(d, threshold)
    width = len(weights) - 1

    # Initialize output structure
    result_dict = {}

    for name in series.columns:
        # Forward-fill and drop NaN for computation
        series_filled = series[name].ffill().dropna()

        if len(series_filled) <= width:
            # Not enough data for the window size
            result_dict[name] = pd.Series(dtype=float, name=name)
            continue

        # Initialize output series
        output = pd.Series(dtype=float, name=name)

        # Apply fractional differentiation
        for iloc in range(width, len(series_filled)):
            loc_start = series_filled.index[iloc - width]
            loc_end = series_filled.index[iloc]

            # Skip if original value is NaN
            if not np.isfinite(series.loc[loc_end, name]):
                continue

            # Apply weighted sum (fractional differentiation filter)
            window = series_filled.loc[loc_start:loc_end].values
            output.loc[loc_end] = np.dot(weights.T, window)[0]

        result_dict[name] = output

    return pd.concat(result_dict, axis=1)


def find_min_d_for_stationarity(
    series: pd.Series,
    max_d: float = 1.0,
    num_steps: int = 11,
    threshold: float = 0.01,
    significance_level: float = 0.05
) -> Tuple[float, pd.DataFrame]:
    """
    Find minimum fractional differentiation order that makes series stationary.

    Uses ADF (Augmented Dickey-Fuller) test to determine stationarity.
    Tests d values from 0 to max_d and returns the minimum d where the
    ADF test rejects the null hypothesis of unit root.

    Args:
        series: Input time series to test for stationarity.
        max_d: Maximum differentiation order to test. Default is 1.0.
            Can be set up to 2.0 for very persistent series.
        num_steps: Number of d values to test between 0 and max_d.
            More steps give finer granularity but take longer.
        threshold: Weight threshold for FFD computation. Higher values
            speed up computation but reduce precision.
        significance_level: Significance level for ADF test. Default 0.05
            means we reject null hypothesis if p-value < 0.05.

    Returns:
        Tuple containing:
            - min_d: Minimum d value achieving stationarity, or max_d if
              no stationary d found in the tested range.
            - results_df: DataFrame with ADF test results for each d value.
              Columns: ['adf_stat', 'p_value', 'lags', 'nobs',
                       'critical_value', 'correlation']

    Raises:
        ValueError: If series is empty or has insufficient data.

    Example:
        >>> series = pd.Series(np.cumsum(np.random.randn(1000)))
        >>> min_d, results = find_min_d_for_stationarity(series)
        >>> print(f"Minimum d for stationarity: {min_d:.2f}")
    """
    if series.empty:
        raise ValueError("Input series cannot be empty")

    # Ensure series has a name
    series_name = series.name if series.name is not None else 'value'
    series = series.copy()
    series.name = series_name

    # Determine critical value column based on significance level
    critical_value_key = f'{int((1 - significance_level) * 100)}%'

    results = pd.DataFrame(
        columns=['adf_stat', 'p_value', 'lags', 'nobs', 'critical_value', 'correlation']
    )

    d_values = np.linspace(0, max_d, num_steps)

    for d in d_values:
        # Apply fractional differentiation
        df_input = pd.DataFrame({series_name: series})
        diff_df = frac_diff_ffd(df_input, d, threshold=threshold)
        diff_series = diff_df.dropna().iloc[:, 0]

        if len(diff_series) < 20:
            # Not enough data for ADF test
            continue

        # Align original and differentiated series for correlation
        aligned = pd.DataFrame({
            'original': series.reindex(diff_series.index),
            'diff': diff_series
        }).dropna()

        if len(aligned) < 20:
            continue

        # Compute correlation between original and differentiated
        corr = np.corrcoef(aligned['original'], aligned['diff'])[0, 1]

        # Perform ADF test
        try:
            adf_result = adfuller(aligned['diff'], autolag='AIC')
            critical_value = adf_result[4].get(critical_value_key, adf_result[4]['5%'])
            results.loc[d] = [
                adf_result[0],  # ADF statistic
                adf_result[1],  # p-value
                adf_result[2],  # lags used
                adf_result[3],  # number of observations
                critical_value,
                corr
            ]
        except Exception:
            # Skip if ADF test fails
            continue

    # Find minimum d for stationarity
    if results.empty:
        return max_d, results

    stationary_mask = results['adf_stat'] < results['critical_value']
    if stationary_mask.any():
        min_d = results[stationary_mask].index.min()
    else:
        min_d = max_d

    return min_d, results


def generate_random_walk(
    length: int = 1000,
    start: float = 0.0,
    seed: Optional[int] = None,
    name: str = 'random_walk'
) -> pd.Series:
    """
    Generate a random walk time series.

    Random walk is non-stationary by definition (has unit root).
    Formula: X_t = X_{t-1} + epsilon_t, where epsilon_t ~ N(0, 1)

    Args:
        length: Number of observations to generate.
        start: Starting value of the series.
        seed: Random seed for reproducibility. If None, results will vary.
        name: Name for the resulting Series.

    Returns:
        Series with random walk values and DatetimeIndex starting from 2000-01-01.

    Example:
        >>> rw = generate_random_walk(1000, seed=42)
        >>> print(f"Final value: {rw.iloc[-1]:.2f}")
    """
    if seed is not None:
        np.random.seed(seed)

    steps = np.random.normal(size=length)
    values = start + np.cumsum(steps)
    dates = pd.date_range(start='2000-01-01', periods=length, freq='D')

    return pd.Series(values, index=dates, name=name)


def generate_stationary_ar1(
    length: int = 1000,
    rho: float = 0.5,
    sigma: float = 0.1,
    seed: Optional[int] = None,
    name: str = 'ar1'
) -> pd.Series:
    """
    Generate a stationary AR(1) process.

    AR(1) is stationary when |rho| < 1.
    Formula: X_t = rho * X_{t-1} + epsilon_t, where epsilon_t ~ N(0, sigma^2)

    Args:
        length: Number of observations to generate.
        rho: Autocorrelation parameter. Must satisfy |rho| < 1 for stationarity.
        sigma: Standard deviation of the noise term.
        seed: Random seed for reproducibility.
        name: Name for the resulting Series.

    Returns:
        Series with AR(1) values and DatetimeIndex starting from 2000-01-01.

    Raises:
        ValueError: If |rho| >= 1 (non-stationary case).

    Example:
        >>> ar1 = generate_stationary_ar1(1000, rho=0.7, seed=42)
        >>> print(f"Mean: {ar1.mean():.4f}")
    """
    if abs(rho) >= 1:
        raise ValueError("rho must satisfy |rho| < 1 for stationarity")

    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start='2000-01-01', periods=length, freq='D')
    values = np.zeros(length)

    for t in range(1, length):
        values[t] = rho * values[t - 1] + np.random.normal(scale=sigma)

    return pd.Series(values, index=dates, name=name)


def generate_trend_stationary(
    length: int = 1000,
    trend: float = 0.01,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
    name: str = 'trend_stationary'
) -> pd.Series:
    """
    Generate a trend-stationary process.

    Trend-stationary series have a deterministic trend but stationary
    deviations from that trend.
    Formula: X_t = trend * t + epsilon_t, where epsilon_t ~ N(0, noise_std^2)

    Note: These series appear non-stationary but become stationary after
    detrending (not differencing).

    Args:
        length: Number of observations to generate.
        trend: Slope of the linear trend (per time step).
        noise_std: Standard deviation of the noise term.
        seed: Random seed for reproducibility.
        name: Name for the resulting Series.

    Returns:
        Series with trend-stationary values and DatetimeIndex starting
        from 2000-01-01.

    Example:
        >>> ts = generate_trend_stationary(1000, trend=0.05, seed=42)
        >>> print(f"Final value: {ts.iloc[-1]:.2f}")
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start='2000-01-01', periods=length, freq='D')
    time_index = np.arange(length)
    values = time_index * trend + np.random.normal(scale=noise_std, size=length)

    return pd.Series(values, index=dates, name=name)
