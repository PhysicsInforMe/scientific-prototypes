"""
Causality Testing Module.

This module provides tools for testing causal relationships between
time series using both traditional (Granger) and modern (CCM) methods.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


def ccm_test(
    source: Union[np.ndarray, pd.Series],
    target: Union[np.ndarray, pd.Series],
    embedding_dim: int = 3,
    tau: int = 1,
    n_surrogates: int = 30,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Convergent Cross Mapping (CCM) test for causal inference.

    CCM tests whether information about the source series is encoded in the
    target series' dynamics. It works by reconstructing the source from the
    target's shadow manifold using delay embedding.

    Unlike Granger causality, CCM can detect bidirectional causality and
    works for deterministic systems where Granger may fail.

    Args:
        source: Candidate causal series (X). This is the series we're testing
            as a potential cause.
        target: Effect series (Y). This is the series that may contain
            information about the source.
        embedding_dim: Dimension of the delay embedding. Should be chosen
            based on the system's complexity. Default 3 works for many systems.
        tau: Time delay for embedding. Default 1 (consecutive observations).
        n_surrogates: Number of surrogate series for significance testing.
            More surrogates give more precise p-values but take longer.
        significance_level: Threshold for declaring significance.

    Returns:
        Dictionary containing:
            - 'ccm_score': Correlation between actual and predicted source values
            - 'is_significant': Whether the causal relationship is significant
            - 'p_value': Approximate p-value from surrogate testing
            - 'surrogate_threshold': 95th percentile of surrogate scores
            - 'surrogate_scores': All surrogate CCM scores

    Example:
        >>> result = ccm_test(temperature, ice_cream_sales, embedding_dim=3)
        >>> if result['is_significant']:
        ...     print(f"Temperature causes ice cream sales (CCM={result['ccm_score']:.3f})")
    """
    # Convert to numpy arrays
    if isinstance(source, pd.Series):
        source = source.values
    if isinstance(target, pd.Series):
        target = target.values

    # Compute CCM score
    score = _ccm_leave_one_out(source, target, embedding_dim, tau)

    # Generate surrogates and compute their scores
    surrogates = _generate_surrogates(source, n_surrogates)
    surrogate_scores = [
        _ccm_leave_one_out(surr, target, embedding_dim, tau)
        for surr in surrogates
    ]

    # Compute threshold and p-value
    threshold = np.percentile(surrogate_scores, (1 - significance_level) * 100)
    p_value = np.mean([s >= score for s in surrogate_scores])

    return {
        'ccm_score': score,
        'is_significant': score > threshold,
        'p_value': p_value,
        'surrogate_threshold': threshold,
        'surrogate_scores': surrogate_scores,
    }


def _ccm_leave_one_out(
    source: np.ndarray,
    target: np.ndarray,
    dim: int,
    tau: int
) -> float:
    """
    Compute CCM score using leave-one-out cross-validation.

    Args:
        source: Source series (candidate cause).
        target: Target series (effect).
        dim: Embedding dimension.
        tau: Time delay.

    Returns:
        CCM correlation score (0 to 1).
    """
    n = len(target)

    # Check minimum length requirement
    if n < (dim * tau + 5):
        return 0.0

    # Build shadow manifold from target
    manifold = []
    valid_indices = []

    for i in range(n - (dim - 1) * tau):
        # Extract delay embedding vector
        window = target[i:i + (dim - 1) * tau + 1:tau]
        manifold.append(window)
        valid_indices.append(i + (dim - 1) * tau)

    manifold = np.array(manifold)
    source_values = source[np.array(valid_indices)]

    # Find nearest neighbors in manifold
    n_neighbors = dim + 2  # Extra neighbor to exclude self-match
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(manifold)
    distances, indices = nbrs.kneighbors(manifold)

    # Predict source values using manifold neighbors
    predictions = []
    for i in range(len(manifold)):
        # Leave-one-out: exclude current point from neighbors
        neighbor_idx = [idx for idx in indices[i] if idx != i][:dim + 1]

        if len(neighbor_idx) < 1:
            predictions.append(0)
            continue

        # Predict as mean of neighbors' source values
        predictions.append(np.mean(source_values[neighbor_idx]))

    if len(predictions) < 2:
        return 0.0

    # Compute correlation
    corr = np.corrcoef(source_values, predictions)[0, 1]
    return max(0.0, float(corr)) if not np.isnan(corr) else 0.0


def _generate_surrogates(series: np.ndarray, n_surr: int) -> List[np.ndarray]:
    """
    Generate surrogate series preserving power spectrum (IAAFT-like).

    This method randomizes phases in Fourier domain while preserving
    the amplitude spectrum, destroying causal relationships while
    maintaining autocorrelation structure.

    Args:
        series: Original series.
        n_surr: Number of surrogates to generate.

    Returns:
        List of surrogate series.
    """
    surrogates = []
    n = len(series)
    f = np.fft.rfft(series)

    for _ in range(n_surr):
        # Randomize phases
        random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(f)))
        f_surr = f * random_phases
        f_surr[0] = f[0]  # Preserve DC component (mean)

        # Transform back
        surr = np.fft.irfft(f_surr, n=n)
        surrogates.append(surr)

    return surrogates


def granger_causality_test(
    data: pd.DataFrame,
    target_col: str,
    source_cols: Union[str, List[str]],
    max_lags: int = 4,
    model_type: str = 'linear'
) -> Dict[str, Any]:
    """
    Granger causality test for time series.

    Tests whether including past values of source variables improves
    prediction of the target variable beyond using only the target's
    own past values.

    Args:
        data: DataFrame containing all series.
        target_col: Name of the target (effect) column.
        source_cols: Name(s) of source (cause) column(s).
        max_lags: Maximum number of lags to include.
        model_type: 'linear' for Ridge regression, 'nonlinear' for KNN.

    Returns:
        Dictionary containing:
            - 'delta_rmse': Reduction in RMSE (0-1, higher = stronger causality)
            - 'rmse_univariate': RMSE using only target history
            - 'rmse_bivariate': RMSE using target + source history
            - 'improvement_pct': Percentage improvement in RMSE

    Example:
        >>> result = granger_causality_test(df, 'GDP', ['interest_rate', 'inflation'])
        >>> print(f"Improvement: {result['improvement_pct']:.1f}%")
    """
    if isinstance(source_cols, str):
        source_cols = [source_cols]

    target = data[target_col].values
    sources = data[source_cols].values

    n = len(target)
    X_uni, X_bi, y_true = [], [], []

    # Build lagged feature matrices
    for i in range(max_lags, n):
        # Univariate: only target history
        past_y = target[i - max_lags:i]
        X_uni.append(past_y)

        # Bivariate: target + sources history
        past_x = sources[i - max_lags:i].flatten()
        X_bi.append(np.concatenate([past_y, past_x]))

        y_true.append(target[i])

    X_uni = np.array(X_uni)
    X_bi = np.array(X_bi)
    y_true = np.array(y_true)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_uni_list, mse_bi_list = [], []

    if model_type == 'linear':
        model = Ridge(alpha=1.0)
    else:
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')

    for train_idx, test_idx in tscv.split(X_uni):
        # Univariate model
        model.fit(X_uni[train_idx], y_true[train_idx])
        preds_uni = model.predict(X_uni[test_idx])
        mse_uni_list.append(mean_squared_error(y_true[test_idx], preds_uni))

        # Bivariate model
        model.fit(X_bi[train_idx], y_true[train_idx])
        preds_bi = model.predict(X_bi[test_idx])
        mse_bi_list.append(mean_squared_error(y_true[test_idx], preds_bi))

    rmse_uni = np.sqrt(np.mean(mse_uni_list))
    rmse_bi = np.sqrt(np.mean(mse_bi_list))

    # Compute improvement
    if rmse_uni == 0:
        delta = 0.0
    else:
        delta = max(0.0, 1 - (rmse_bi / rmse_uni))

    return {
        'delta_rmse': delta,
        'rmse_univariate': rmse_uni,
        'rmse_bivariate': rmse_bi,
        'improvement_pct': delta * 100,
    }


def generate_causal_system(
    n: int = 200,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Generate synthetic data with known causal structure for testing.

    Creates a system with:
    - X1: Primary chaotic driver (logistic map)
    - X2: Linearly depends on X1
    - X3: Non-linearly depends on X1 (sinusoidal)
    - X4: Pure noise (no causal relationship)
    - Y: Target depending on X2 (linear) and X3 (quadratic)

    This ground truth allows validating causality detection methods.

    Args:
        n: Number of time points to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (data_df, ground_truth_dict) where ground_truth_dict
        describes the true causal relationships.

    Example:
        >>> data, truth = generate_causal_system(200)
        >>> print(truth)
        {'X1': 'indirect_cause', 'X2': 'direct_linear', ...}
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize arrays
    x1 = np.zeros(n)  # Chaotic driver
    x2 = np.zeros(n)  # Linear dependent
    x3 = np.zeros(n)  # Non-linear dependent
    x4 = np.random.normal(0, 1, n)  # Pure noise
    y = np.zeros(n)   # Target

    # Initial conditions
    x1[0], x2[0], x3[0], y[0] = 0.1, 0.1, 0.1, 0.1

    # Generate dynamics
    for t in range(1, n):
        # X1: Logistic map (chaotic)
        val = 3.9 * x1[t - 1] * (1 - x1[t - 1])
        x1[t] = np.clip(val, 0.0, 1.0)

        # X2: Linear relationship with X1
        x2[t] = 0.8 * x2[t - 1] + 0.5 * x1[t - 1] + np.random.normal(0, 0.05)

        # X3: Non-linear relationship with X1
        x3[t] = 0.5 * x3[t - 1] + 0.6 * np.sin(5 * x1[t - 1])

        # Y: Depends on X2 (linear) and X3 (quadratic)
        y[t] = (0.5 * y[t - 1] + 0.4 * x2[t - 1] +
                0.5 * (x3[t - 1] ** 2) + np.random.normal(0, 0.05))

    # Create DataFrame
    data = pd.DataFrame({
        'X1': x1,
        'X2': x2,
        'X3': x3,
        'X4': x4,
        'Y': y
    })

    # Normalize
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns
    )

    # Ground truth
    ground_truth = {
        'X1': 'indirect_cause (driver of X2 and X3)',
        'X2': 'direct_linear_cause',
        'X3': 'direct_nonlinear_cause',
        'X4': 'noise (no causal relationship)',
    }

    return data_scaled, ground_truth


def run_full_causality_analysis(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    horizons: List[int] = [1, 2, 3, 4],
    embedding_dim: int = 3,
    max_lags: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive causality analysis using multiple methods.

    Combines CCM and Granger causality (linear and non-linear) to provide
    a robust assessment of causal relationships.

    Args:
        data: DataFrame with time series data.
        target_col: Name of target variable.
        feature_cols: List of feature columns to test. If None, uses all
            columns except target.
        horizons: Forecast horizons to test (e.g., [1, 2, 3] for t+1, t+2, t+3).
        embedding_dim: Embedding dimension for CCM.
        max_lags: Maximum lags for Granger test.

    Returns:
        Dictionary with DataFrames:
            - 'ccm_scores': CCM scores for each feature and horizon
            - 'ccm_significant': Boolean mask of significant CCM results
            - 'granger_linear': Linear Granger improvements
            - 'granger_nonlinear': Non-linear Granger improvements
            - 'summary': Summary classification of each feature

    Example:
        >>> results = run_full_causality_analysis(df, 'GDP')
        >>> print(results['summary'])
    """
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != target_col]

    # Initialize result DataFrames
    ccm_scores = pd.DataFrame(index=feature_cols, columns=horizons, dtype=float)
    ccm_significant = pd.DataFrame(index=feature_cols, columns=horizons, dtype=bool)
    granger_linear = pd.DataFrame(index=feature_cols, columns=horizons, dtype=float)
    granger_nonlinear = pd.DataFrame(index=feature_cols, columns=horizons, dtype=float)

    for h in horizons:
        # Shift target for horizon h
        y_shifted = data[target_col].shift(-h).dropna()
        common_idx = y_shifted.index

        for feat in feature_cols:
            x_source = data.loc[common_idx, feat].values
            y_target = y_shifted.values

            # CCM test
            ccm_result = ccm_test(x_source, y_target, embedding_dim=embedding_dim)
            ccm_scores.loc[feat, h] = ccm_result['ccm_score']
            ccm_significant.loc[feat, h] = ccm_result['is_significant']

            # Granger tests
            temp_df = pd.DataFrame({
                'target': y_target,
                'source': x_source
            })

            lin_result = granger_causality_test(
                temp_df, 'target', 'source',
                max_lags=max_lags, model_type='linear'
            )
            granger_linear.loc[feat, h] = lin_result['delta_rmse']

            nonlin_result = granger_causality_test(
                temp_df, 'target', 'source',
                max_lags=max_lags, model_type='nonlinear'
            )
            granger_nonlinear.loc[feat, h] = nonlin_result['delta_rmse']

    # Create summary
    summary_data = []
    for feat in feature_cols:
        ccm_avg = ccm_scores.loc[feat].mean()
        ccm_sig_any = ccm_significant.loc[feat].any()
        nonlin_avg = granger_nonlinear.loc[feat].mean()

        # Classification logic
        if ccm_sig_any:
            if nonlin_avg > 0.01:
                classification = 'Strong Causal Driver'
            else:
                classification = 'Structural Cause (Hidden)'
        else:
            if nonlin_avg > 0.01:
                classification = 'Predictive Proxy'
            else:
                classification = 'Noise/Irrelevant'

        summary_data.append({
            'Feature': feat,
            'Classification': classification,
            'CCM Score (avg)': round(ccm_avg, 3),
            'CCM Significant': ccm_sig_any,
            'Granger Improvement (avg)': round(nonlin_avg * 100, 1),
        })

    summary = pd.DataFrame(summary_data).set_index('Feature')

    return {
        'ccm_scores': ccm_scores,
        'ccm_significant': ccm_significant,
        'granger_linear': granger_linear,
        'granger_nonlinear': granger_nonlinear,
        'summary': summary,
    }
