# Code Documentation — Adaptive AutoML Framework

> Complete reference for every class, function, and component in `timeseries_toolkit/`.
> Each entry explains **what** it does, **how** it works, and **why** it is designed that way.

---

## Table of Contents

1. [Package Overview](#1-package-overview)
2. [preprocessing/](#2-preprocessing)
   - 2.1 [fractional_diff.py](#21-fractional_diffpy)
   - 2.2 [filtering.py](#22-filteringpy)
   - 2.3 [imputation.py](#23-imputationpy)
3. [models/](#3-models)
   - 3.1 [kalman.py](#31-kalmanpy)
   - 3.2 [regime.py](#32-regimepy)
   - 3.3 [forecaster.py](#33-forecasterpy)
4. [validation/](#4-validation)
   - 4.1 [causality.py](#41-causalitypy)
   - 4.2 [diagnostics.py](#42-diagnosticspy)
5. [intelligence/](#5-intelligence)
   - 5.1 [market_intelligence.py](#51-market_intelligencepy)
   - 5.2 [regime_analyzer.py](#52-regime_analyzerpy)
   - 5.3 [autopilot.py](#53-autopilotpy)
   - 5.4 [pipelines.py](#54-pipelinespy)
   - 5.5 [explainer.py](#55-explainerpy)
   - 5.6 [backtester.py](#56-backtesterpy)
6. [data_sources/](#6-data_sources)
   - 6.1 [hub.py](#61-hubpy)
   - 6.2 [crypto.py](#62-cryptopy)
   - 6.3 [equities.py](#63-equitiespy)
   - 6.4 [volatility.py](#64-volatilitypy)
   - 6.5 [rates.py](#65-ratespy)
   - 6.6 [liquidity.py](#66-liquiditypy)
   - 6.7 [macro.py](#67-macropy)
   - 6.8 [european.py](#68-europeanpy)
7. [utils/](#7-utils)
   - 7.1 [data_loader.py](#71-data_loaderpy)

---

## 1. Package Overview

The `timeseries_toolkit` package is an adaptive AutoML framework for financial and macroeconomic time series. It is organized into six sub-packages:

| Sub-package | Purpose |
|---|---|
| `preprocessing/` | Data transformation: fractional differentiation, STL+SARIMA filtering, mixed-frequency MICE imputation |
| `models/` | Core models: Kalman filter (state-space), HMM regime detection, gradient-boosted global forecaster |
| `validation/` | Statistical validation: CCM/Granger causality testing, 7-test forensic diagnostics |
| `intelligence/` | Orchestration layer: MarketIntelligence orchestrator, regime analyzer, autopilot pipeline selector, explainer, backtester |
| `data_sources/` | Data loaders: DataHub facade, crypto, equities, volatility, rates, liquidity, macro, European data |
| `utils/` | Utilities: CSV/Excel loading with quarterly date parsing |

**Key design principles:**

- **Modularity** — Each module is self-contained; users can import any layer independently.
- **Method chaining** — All `fit()` methods return `self` for fluent APIs.
- **Graceful degradation** — Optional dependencies (hmmlearn, lightgbm, fredapi, yfinance) are checked at import time with helpful error messages.
- **Standardized interfaces** — Consistent `fit/transform/predict` pattern across models and preprocessors.

---

## 2. preprocessing/

### 2.1 fractional_diff.py

Fractional differentiation transforms non-stationary series to stationary while preserving more memory than integer differencing. Based on the Fixed-Width Window (FFD) method from López de Prado (2018).

---

#### `get_weights_ffd`

```python
def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray
```

**What**: Computes the fractional differentiation weight vector using the FFD recurrence.

**How**: Starting from w₀ = 1, iteratively computes wₖ = −wₖ₋₁ · (d − k + 1) / k until |wₖ| falls below `threshold`. Returns weights in reverse chronological order as a column vector.

**Why**: The FFD method truncates the infinite binomial series to a finite window, making the computation tractable while maintaining a controlled approximation error. The threshold parameter provides a principled trade-off between precision and window length.

| Parameter | Type | Description |
|---|---|---|
| `d` | `float` | Fractional differentiation order (typically 0 < d < 1) |
| `threshold` | `float` | Minimum absolute weight to include (default 1e-5) |
| **Returns** | `np.ndarray` | Weight vector, shape `(n, 1)`, reverse chronological order |

**Dependencies**: `numpy`

---

#### `frac_diff_ffd`

```python
def frac_diff_ffd(series: pd.DataFrame, d: float, threshold: float = 1e-5) -> pd.DataFrame
```

**What**: Applies fractional differentiation to each column of a DataFrame using the FFD method.

**How**: Computes FFD weights via `get_weights_ffd`, then for each column applies a weighted dot product over a sliding window of length equal to the weight vector. Forward-fills NaN values before computation. Initial observations (before the window is full) are NaN.

**Why**: Fractional differentiation with d ∈ (0, 1) achieves stationarity while retaining long-range memory — a critical property for financial time series where integer differencing (d=1) destroys too much signal. The per-column processing allows batch transformation of multivariate datasets.

| Parameter | Type | Description |
|---|---|---|
| `series` | `pd.DataFrame` | Input time series (one column per series) |
| `d` | `float` | Fractional order (0 = identity, 1 = first difference) |
| `threshold` | `float` | Weight truncation threshold |
| **Returns** | `pd.DataFrame` | Fractionally differenced series |

**Dependencies**: `get_weights_ffd`, `numpy`, `pandas`

---

#### `find_min_d_for_stationarity`

```python
def find_min_d_for_stationarity(
    series: pd.Series, max_d: float = 1.0, num_steps: int = 11,
    threshold: float = 0.01, significance_level: float = 0.05
) -> Tuple[float, pd.DataFrame]
```

**What**: Searches for the minimum fractional differentiation order `d` that makes a series stationary according to the ADF test.

**How**: Tests `num_steps` values of d linearly spaced from 0 to `max_d`. For each d, applies `frac_diff_ffd`, runs the Augmented Dickey-Fuller test, and records the ADF statistic, p-value, lags, observation count, critical value, and correlation with the original series. Returns the smallest d where the ADF statistic is below the critical value.

**Why**: This automates the manual search for optimal differentiation order. By testing a grid and returning the minimum viable d, it ensures maximum memory preservation. The correlation column in the results DataFrame lets users inspect the memory-stationarity trade-off visually.

| Parameter | Type | Description |
|---|---|---|
| `series` | `pd.Series` | Input time series |
| `max_d` | `float` | Maximum d to test (default 1.0) |
| `num_steps` | `int` | Grid resolution (default 11) |
| `threshold` | `float` | Weight threshold for FFD |
| `significance_level` | `float` | ADF significance level (default 0.05) |
| **Returns** | `Tuple[float, pd.DataFrame]` | (min_d, results DataFrame with ADF statistics) |

**Dependencies**: `frac_diff_ffd`, `statsmodels.tsa.stattools.adfuller`

---

#### `generate_random_walk`

```python
def generate_random_walk(
    length: int = 1000, start: float = 0.0,
    seed: Optional[int] = None, name: str = 'random_walk'
) -> pd.Series
```

**What**: Generates a synthetic random walk (unit root) series for testing.

**How**: Produces i.i.d. N(0,1) increments and cumulates them from `start`. Returns a Series with daily DatetimeIndex from 2000-01-01.

**Why**: Provides a canonical non-stationary test case for validating that `find_min_d_for_stationarity` correctly identifies d > 0.

| Parameter | Type | Description |
|---|---|---|
| `length` | `int` | Number of observations |
| `start` | `float` | Starting value |
| `seed` | `Optional[int]` | Random seed |
| `name` | `str` | Series name |
| **Returns** | `pd.Series` | Random walk with DatetimeIndex |

---

#### `generate_stationary_ar1`

```python
def generate_stationary_ar1(
    length: int = 1000, rho: float = 0.5, sigma: float = 0.1,
    seed: Optional[int] = None, name: str = 'ar1'
) -> pd.Series
```

**What**: Generates a stationary AR(1) process: Xₜ = ρ·Xₜ₋₁ + εₜ.

**How**: Iteratively generates values with the AR(1) recurrence. Requires |ρ| < 1.

**Why**: Provides a stationary test case to verify that `find_min_d_for_stationarity` returns d ≈ 0.

| Parameter | Type | Description |
|---|---|---|
| `rho` | `float` | Autocorrelation parameter (|ρ| < 1) |
| `sigma` | `float` | Noise standard deviation |
| **Returns** | `pd.Series` | Stationary AR(1) series |

---

#### `generate_trend_stationary`

```python
def generate_trend_stationary(
    length: int = 1000, trend: float = 0.01, noise_std: float = 0.1,
    seed: Optional[int] = None, name: str = 'trend_stationary'
) -> pd.Series
```

**What**: Generates a trend-stationary process: Xₜ = trend · t + εₜ.

**How**: Linear trend plus i.i.d. Gaussian noise.

**Why**: Tests the boundary case — these series look non-stationary but become stationary after detrending (not differencing). Useful for validating that fractional differentiation handles this case without over-differencing.

---

### 2.2 filtering.py

Two-stage filtering combining STL decomposition with SARIMA modeling for noise removal.

---

#### `_infer_seasonal_period`

```python
def _infer_seasonal_period(series: pd.Series) -> Tuple[str, int]
```

**What**: Infers the seasonal period from a series' frequency or average time delta.

**How**: First tries `pd.infer_freq()`. If that returns None, computes the mean inter-observation delta and maps it to known periods: daily→7, weekly→52, monthly→12, quarterly→4.

**Why**: Automatic period inference removes a key hyperparameter from the filtering pipeline. The fallback via time delta handles irregular series that `pd.infer_freq` cannot classify.

| Parameter | Type | Description |
|---|---|---|
| `series` | `pd.Series` | Series with DatetimeIndex |
| **Returns** | `Tuple[str, int]` | (frequency name, seasonal period) |

---

#### `_find_best_sarima`

```python
def _find_best_sarima(
    data: pd.Series, seasonal_period: int,
    max_p: int = 2, max_q: int = 2, max_P: int = 1, max_Q: int = 1
) -> Tuple[Optional[Any], Optional[Tuple], Optional[Tuple]]
```

**What**: Grid-searches for the best SARIMA model on the residuals by AIC.

**How**: Iterates over all combinations of (p, 0, q) × (P, 0, Q, m) where d=D=0 (residuals are already stationary post-STL). Fits each with `SARIMAX`, tracks the lowest AIC. Skips the trivial (0,0,0)(0,0,0,m) model.

**Why**: AIC balances model fit against complexity. Fixing d=D=0 is appropriate because STL already removes trend and seasonality; the SARIMA stage only needs to model residual autocorrelation.

| Parameter | Type | Description |
|---|---|---|
| `data` | `pd.Series` | Residual series from STL |
| `seasonal_period` | `int` | Seasonal period (e.g. 12) |
| `max_p/q/P/Q` | `int` | Upper bounds for grid search |
| **Returns** | `Tuple` | (best_model, best_order, best_seasonal_order) |

**Dependencies**: `statsmodels.tsa.statespace.sarimax.SARIMAX`

---

#### `_compute_sample_entropy`

```python
def _compute_sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float
```

**What**: Computes sample entropy — a measure of signal regularity/complexity.

**How**: Counts template matches of length m and m+1 within tolerance r·std(data). Returns −ln(A/B) where A = matches at m+1, B = matches at m.

**Why**: Sample entropy quantifies how much structure remains in filtered vs. original signals. Lower values mean more regularity. Used in `get_metrics()` to evaluate filtering quality.

| Parameter | Type | Description |
|---|---|---|
| `data` | `np.ndarray` | Input array |
| `m` | `int` | Embedding dimension (default 2) |
| `r` | `float` | Tolerance as fraction of std (default 0.2) |
| **Returns** | `float` | Sample entropy value |

---

#### `class TimeSeriesFilter`

```python
class TimeSeriesFilter:
    def __init__(self, seasonal_period: Optional[int] = None)
```

**What**: Two-stage noise removal filter combining STL decomposition with SARIMA residual modeling.

**How**:
1. **Stage 1 (STL)**: Decomposes the series into trend + seasonal + residual using LOESS-based STL with robust fitting.
2. **Stage 2 (SARIMA)**: Fits a SARIMA model to the STL residuals via `_find_best_sarima` to capture remaining autocorrelation.
3. The final filtered series = original − SARIMA residuals (trimmed to avoid edge effects).

**Why**: Single-stage STL leaves autocorrelated residuals because LOESS cannot model AR/MA dynamics. The SARIMA stage captures this remaining structure, producing cleaner (more white-noise-like) final residuals. This is validated by the Ljung-Box test in `get_ljung_box_test()`.

| Attribute | Type | Description |
|---|---|---|
| `seasonal_period` | `Optional[int]` | Detected seasonal period |
| `frequency_name` | `Optional[str]` | Human-readable frequency |
| `is_fitted` | `bool` | Fit status |
| `sarima_order` | `Optional[Tuple]` | Best (p,d,q) |
| `sarima_seasonal_order` | `Optional[Tuple]` | Best (P,D,Q,m) |

**Methods:**

##### `fit(series, max_p=2, max_q=2, max_P=1, max_Q=1) -> TimeSeriesFilter`

Fits both stages. Requires DatetimeIndex. Series must have ≥ 2× seasonal period observations.

##### `transform(series=None) -> pd.Series`

Returns the filtered series. If `series` is None, returns the fit-time result. Otherwise applies the fitted STL+SARIMA to new data.

##### `get_residuals() -> pd.Series`

Returns the noise component removed by the filter.

##### `get_metrics() -> Dict[str, float]`

Returns quality metrics: SNR (dB), variances (original/filtered/residuals), sample entropies, and Signal Dominance Index (percentage of variance attributable to signal).

##### `get_ljung_box_test(lags=None) -> Dict[str, Any]`

Runs Ljung-Box test on residuals. Returns statistic, p-value, and `is_white_noise` boolean (p > 0.05).

##### `should_filter(series, nsr_threshold=0.1, ljung_box_alpha=0.05) -> Tuple[bool, Dict]`

Static decision function: recommends filtering if the series has both autocorrelation structure (Ljung-Box p < α) and high noise-to-signal ratio (NSR > threshold).

---

### 2.3 imputation.py

Mixed-frequency imputation using MICE (Multiple Imputation by Chained Equations).

---

#### `align_to_quarterly`

```python
def align_to_quarterly(
    series: pd.Series, target_index: pd.DatetimeIndex,
    aggregation: str = 'last'
) -> pd.Series
```

**What**: Resamples a series of any frequency to quarterly, aligned to a target index.

**How**: Uses `pd.Series.resample('QE')` with the specified aggregation method, then reindexes to `target_index`.

**Why**: Mixed-frequency analysis requires a common time grid. Quarterly is the natural base frequency for macroeconomic data (GDP, etc.). The aggregation parameter lets users choose the appropriate summary statistic.

| Parameter | Type | Description |
|---|---|---|
| `series` | `pd.Series` | Input at any frequency |
| `target_index` | `pd.DatetimeIndex` | Quarterly target index |
| `aggregation` | `str` | 'last', 'mean', 'sum', 'first', 'min', 'max' |
| **Returns** | `pd.Series` | Quarterly-aligned series |

---

#### `class MixedFrequencyImputer`

```python
class MixedFrequencyImputer:
    def __init__(self, max_iter: int = 10, random_state: Optional[int] = 42, verbose: int = 0)
```

**What**: MICE-based imputer for mixed-frequency time series data.

**How**:
1. **Frequency detection**: Infers frequency of each input series.
2. **Feature expansion**:
   - Monthly → 3 columns per quarter (M1, M2, M3 within each quarter).
   - Daily/Weekly → 5 aggregate columns (mean, std, min, max, last) per quarter.
   - Quarterly → pass-through.
3. **MICE imputation**: Wraps `sklearn.impute.IterativeImputer` which models each feature as a function of all others in round-robin fashion.

**Why**: MICE is the gold standard for multivariate imputation because it preserves inter-variable relationships. The monthly-to-quarterly expansion (3 columns per quarter) preserves within-quarter dynamics that a simple mean would destroy. Daily/weekly aggregation into summary statistics avoids dimensionality explosion.

| Attribute | Type | Description |
|---|---|---|
| `is_fitted` | `bool` | Fit status |
| `feature_names` | `List[str]` | Column names after transformation |

**Methods:**

##### `fit(X_dict, target_index, y=None) -> MixedFrequencyImputer`

Transforms mixed-frequency series to quarterly DataFrame, then fits the MICE imputer.

##### `transform(X_dict, target_index=None) -> pd.DataFrame`

Transforms and imputes new data using the fitted model.

##### `fit_transform(X_dict, target_index, y=None) -> pd.DataFrame`

Convenience method combining fit and transform.

##### `get_imputation_report(X_dict, target_index) -> Tuple[pd.DataFrame, pd.DataFrame]`

Returns two DataFrames: (1) input coverage report (samples, date range, frequency, missing values), (2) imputation report (quarters needing imputation at start/end of each variable).

**Dependencies**: `sklearn.impute.IterativeImputer`

---

## 3. models/

### 3.1 kalman.py

Automated Kalman filter using statsmodels `UnobservedComponents` for state-space modeling.

---

#### `class AutoKalmanFilter`

```python
class AutoKalmanFilter:
    def __init__(
        self, level: str = 'local linear trend', cycle: bool = False,
        stochastic_cycle: bool = False, seasonal: Optional[int] = None,
        freq_seasonal: Optional[list] = None
    )
```

**What**: Provides an easy-to-use interface for state-space time series modeling with automatic component selection.

**How**: Wraps `sm.tsa.UnobservedComponents` which decomposes a time series into unobserved level, trend, cycle, and seasonal components estimated via the Kalman filter. Optionally standardizes the series before fitting for numerical stability. All outputs are reverse-standardized.

**Why**: The Unobserved Components model provides a principled Bayesian framework for decomposing time series. It naturally handles missing data, provides uncertainty estimates for all components, and the Kalman filter's forward-backward passes produce optimally smoothed estimates. The `level` parameter selects the structural form:

| Level Type | State Equation | Use Case |
|---|---|---|
| `'local level'` | Random walk | Stable series with no trend |
| `'random walk with drift'` | RW + constant drift | Series with steady growth |
| `'local linear trend'` | RW level + RW slope | General purpose (default) |
| `'smooth trend'` | Smooth (integrated RW) | Trend-following |
| `'fixed intercept'` | Constant level | Stationary series |

**Methods:**

##### `fit(series, standardize=True, method='lbfgs', maxiter=1000) -> AutoKalmanFilter`

Fits the state-space model. Standardization (z-score) is recommended for numerical stability of the MLE optimizer.

##### `smooth() -> pd.Series`

Returns the Kalman-smoothed series (level component × std + mean).

##### `forecast(steps) -> pd.Series`

Generates out-of-sample point forecasts for `steps` periods ahead.

##### `forecast_with_ci(steps, confidence_level=0.95) -> Tuple[pd.Series, pd.Series, pd.Series]`

Returns (forecast, lower, upper) using native state-space prediction intervals that account for state estimation uncertainty, observation noise, and horizon-dependent growth.

##### `predict(start=None, end=None) -> pd.Series`

In-sample and/or out-of-sample predictions for a date range.

##### `get_components() -> Dict[str, pd.Series]`

Extracts estimated components: level, trend slope, cycle, seasonal, frequency-domain seasonal.

##### `get_residuals() -> pd.Series`

Returns model residuals (observed − predicted) × std.

##### `summary() -> str`

Returns the statsmodels model summary string.

**Dependencies**: `statsmodels.api (UnobservedComponents)`

---

#### `compare_kalman_vs_arima`

```python
def compare_kalman_vs_arima(
    series: pd.Series, holdout: int = 4,
    kalman_kwargs: Optional[Dict] = None,
    arima_order_range: Optional[Tuple] = None
) -> Dict[str, Any]
```

**What**: Benchmarks `AutoKalmanFilter` against ARIMA with grid search on a holdout set.

**How**: Splits the series into train/holdout. Fits a Kalman filter (default: random walk with drift) and grid-searches over all (p,d,q) combinations for ARIMA. Computes RMSE and MAE on the holdout for both.

**Why**: Provides an objective comparison between state-space and Box-Jenkins approaches. The Kalman filter has advantages for short series (built-in regularization via state-space structure) while ARIMA can capture complex AR/MA dynamics.

| Parameter | Type | Description |
|---|---|---|
| `series` | `pd.Series` | Input series |
| `holdout` | `int` | Number of holdout observations (default 4) |
| `kalman_kwargs` | `Optional[Dict]` | kwargs for AutoKalmanFilter |
| `arima_order_range` | `Optional[Tuple]` | (p_range, d_range, q_range) |
| **Returns** | `Dict[str, Any]` | Forecasts, metrics, best ARIMA order, winner |

**Dependencies**: `AutoKalmanFilter`, `statsmodels.tsa.arima.model.ARIMA`, `sklearn.metrics`

---

### 3.2 regime.py

Market regime detection using Gaussian Mixture Hidden Markov Models (GMMHMM).

---

#### `class RegimeDetector`

```python
class RegimeDetector:
    def __init__(
        self, max_states: int = 5, n_cv_splits: int = 5,
        covariance_type: str = 'full', n_iter: int = 200,
        random_state: int = 42
    )
```

**What**: Identifies distinct market regimes (e.g., bull/bear, high/low volatility) from time series data.

**How**:
1. **State selection**: Uses time-series cross-validation (expanding window via `TimeSeriesSplit`) to select the optimal number of HMM states from 2 to `max_states`. Each candidate is scored by out-of-fold log-likelihood.
2. **Model fitting**: Fits a GMMHMM with the selected state count using the EM algorithm for up to `n_iter` iterations.
3. **State ordering**: Reorders regime labels by ascending volatility so that regime 0 = lowest volatility.
4. **Viterbi decoding**: Computes the most likely state sequence.

**Why**: GMMHMM is chosen over plain GaussianHMM because financial return distributions are often multi-modal within a regime (e.g., a "crisis" regime may have both sharp drops and relief rallies). The Gaussian mixture emissions capture this intra-state heterogeneity. Cross-validation for state selection avoids BIC's tendency to over-penalize complex models on short financial time series.

| Attribute | Type | Description |
|---|---|---|
| `is_fitted` | `bool` | Fit status |
| `optimal_states` | `Optional[int]` | Selected number of states |
| `model` | `Optional[GMMHMM]` | Fitted HMM model |

**Methods:**

##### `fit(series, n_states=None, auto_select=True) -> RegimeDetector`

Fits the GMMHMM. If `n_states` is provided, skips auto-selection.

##### `predict_regimes() -> pd.Series`

Returns the Viterbi path (most likely regime sequence), ordered by volatility.

##### `get_regime_probabilities() -> pd.DataFrame`

Returns smoothed posterior probabilities for each regime at each time point. Columns: `regime_0`, `regime_1`, etc.

##### `aggregate_to_frequency(target_freq='QE', method='mean') -> pd.DataFrame`

Aggregates regime probabilities to lower frequency (e.g., quarterly). Useful for mixed-frequency analysis.

##### `get_regime_statistics() -> pd.DataFrame`

Returns mean, std, min, max, count, and proportion for each regime.

##### `get_transition_matrix() -> pd.DataFrame`

Returns the estimated transition probability matrix P[i,j] = Prob(regime j at t+1 | regime i at t).

##### `get_cv_scores() -> Optional[Dict[int, float]]`

Returns cross-validation log-likelihood scores per state count.

##### `predict_proba_new(new_series) -> pd.DataFrame`

Predicts regime probabilities for new (out-of-sample) data.

**Dependencies**: `hmmlearn.hmm.GMMHMM`, `sklearn.model_selection.TimeSeriesSplit`

---

### 3.3 forecaster.py

Gradient boosting-based global forecaster with MICE imputation for mixed-frequency data.

---

#### `class GlobalBoostForecaster`

```python
class GlobalBoostForecaster:
    def __init__(self, model: Optional[Any] = None, random_state: int = 42)
```

**What**: Trains a single gradient boosting model on data from multiple entities (countries, assets) to leverage cross-entity patterns.

**How**:
1. **Feature creation**: For each entity, transforms mixed-frequency predictors to quarterly:
   - Monthly → 3 columns (M1, M2, M3 per quarter)
   - Daily/Weekly → 5 aggregates (mean, std, min, max, last)
   - Quarterly → pass-through
   - Entity ID added as categorical feature
2. **MICE imputation**: Fills missing values using `IterativeImputer`.
3. **Training**: Concatenates all entities into a global training set. Uses LightGBM by default.
4. **Prediction**: Creates features for the target entity, imputes, and predicts.

**Why**: Global models learn shared dynamics across entities (e.g., the relationship between interest rates and GDP is similar across developed economies), providing more training data and better generalization than fitting per-entity models. LightGBM handles categorical features natively and is fast on the moderate-dimensionality datasets typical in macro forecasting.

| Attribute | Type | Description |
|---|---|---|
| `is_fitted` | `bool` | Fit status |
| `feature_names` | `List[str]` | Transformed feature names |
| `model` | LGBMRegressor or custom | Underlying model |

**Methods:**

##### `fit(all_entities_data) -> GlobalBoostForecaster`

Trains on multi-entity data. Input: `{'US': {'y': series, 'X': {'rate': series, ...}}, ...}`.

##### `predict(X_dict, y_series, entity_id, n_periods=1) -> np.ndarray`

Generates predictions for a single entity.

##### `get_feature_importance() -> pd.Series`

Returns feature importance from the trained model, sorted descending.

##### `get_transformed_features(X_dict, y_series, entity_id) -> Optional[pd.DataFrame]`

Returns the final imputed feature matrix for inspection.

##### `generate_imputation_report(all_entities_data) -> Tuple[pd.DataFrame, pd.DataFrame]`

Returns input coverage and imputation reports.

**Dependencies**: `lightgbm.LGBMRegressor`, `sklearn.impute.IterativeImputer`

---

## 4. validation/

### 4.1 causality.py

Tests causal relationships between time series using Convergent Cross Mapping (CCM) and Granger causality.

---

#### `ccm_test`

```python
def ccm_test(
    source: Union[np.ndarray, pd.Series],
    target: Union[np.ndarray, pd.Series],
    embedding_dim: int = 3, tau: int = 1,
    n_surrogates: int = 30, significance_level: float = 0.05
) -> Dict[str, Any]
```

**What**: Tests whether the source series causally drives the target using Convergent Cross Mapping.

**How**:
1. **Shadow manifold**: Constructs a delay embedding of the target series with dimension `embedding_dim` and delay `tau`.
2. **Cross-mapping**: For each point on the target's manifold, finds nearest neighbors and uses their corresponding source values to predict the current source value (leave-one-out).
3. **Significance**: Generates `n_surrogates` phase-randomized surrogates of the source (preserving power spectrum, destroying causality) and computes CCM scores. The real score is compared to the 95th percentile of surrogate scores.

**Why**: CCM detects *dynamical coupling* in the Takens embedding sense, making it suitable for deterministic/chaotic systems where Granger causality (based on linear prediction) may fail. The surrogate testing provides a non-parametric significance test that accounts for autocorrelation.

| Parameter | Type | Description |
|---|---|---|
| `source` | array/Series | Candidate cause |
| `target` | array/Series | Effect series |
| `embedding_dim` | `int` | Delay embedding dimension (default 3) |
| `tau` | `int` | Time delay (default 1) |
| `n_surrogates` | `int` | Number of surrogates (default 30) |
| `significance_level` | `float` | Significance threshold (default 0.05) |
| **Returns** | `Dict` | ccm_score, is_significant, p_value, surrogate_threshold, surrogate_scores |

**Dependencies**: `sklearn.neighbors.NearestNeighbors`

---

#### `_ccm_leave_one_out`

```python
def _ccm_leave_one_out(source, target, dim, tau) -> float
```

**What**: Core CCM computation using leave-one-out cross-validation.

**How**: Builds the target's shadow manifold (delay embedding), finds dim+2 nearest neighbors for each point, excludes self-matches, and predicts source values as the mean of neighbors' source values. Returns the Pearson correlation between actual and predicted source values, clipped to [0, 1].

---

#### `_generate_surrogates`

```python
def _generate_surrogates(series: np.ndarray, n_surr: int) -> List[np.ndarray]
```

**What**: Generates phase-randomized surrogate series.

**How**: Computes the FFT, randomizes phases (preserving the DC component), and inverse transforms. This preserves the power spectrum (autocorrelation structure) while destroying any causal coupling.

**Why**: IAAFT-like surrogates are the standard null model for testing nonlinear dynamical relationships because they preserve the second-order statistics that could produce spurious correlations.

---

#### `granger_causality_test`

```python
def granger_causality_test(
    data: pd.DataFrame, target_col: str,
    source_cols: Union[str, List[str]],
    max_lags: int = 4, model_type: str = 'linear'
) -> Dict[str, Any]
```

**What**: Tests Granger causality — whether past values of source variables improve prediction of the target beyond its own history.

**How**: Constructs two lagged feature matrices: (1) univariate (target lags only), (2) bivariate (target + source lags). Evaluates both using 5-fold time-series cross-validation with either Ridge regression (`'linear'`) or KNN (`'nonlinear'`). Reports the RMSE improvement: δ = 1 − RMSE_bi/RMSE_uni.

**Why**: The cross-validation approach avoids the parametric assumptions of the classical F-test Granger formulation. The nonlinear option (KNN) can detect nonlinear causal relationships that linear Granger misses.

| Parameter | Type | Description |
|---|---|---|
| `data` | `pd.DataFrame` | DataFrame with all series |
| `target_col` | `str` | Target column name |
| `source_cols` | `str` or `List[str]` | Source column(s) |
| `max_lags` | `int` | Maximum lag order (default 4) |
| `model_type` | `str` | 'linear' (Ridge) or 'nonlinear' (KNN) |
| **Returns** | `Dict` | delta_rmse, rmse_univariate, rmse_bivariate, improvement_pct |

---

#### `generate_causal_system`

```python
def generate_causal_system(n: int = 200, seed: Optional[int] = 42) -> Tuple[pd.DataFrame, Dict[str, str]]
```

**What**: Generates synthetic data with known causal structure for testing.

**How**: Creates 5 series:
- X1: Logistic map (chaotic driver, r=3.9)
- X2: Linear function of X1 (direct linear cause)
- X3: Sinusoidal function of X1 (direct nonlinear cause)
- X4: Pure noise (no causality)
- Y: Depends on X2 (linear) and X3 (quadratic)

All series are standardized to zero mean, unit variance.

**Why**: Known ground truth enables validation of CCM and Granger tests. The chaotic driver tests CCM's ability to detect indirect causes; the noise variable tests specificity.

---

#### `run_full_causality_analysis`

```python
def run_full_causality_analysis(
    data: pd.DataFrame, target_col: str,
    feature_cols: Optional[List[str]] = None,
    horizons: List[int] = [1, 2, 3, 4],
    embedding_dim: int = 3, max_lags: int = 3
) -> Dict[str, pd.DataFrame]
```

**What**: Comprehensive causality analysis combining CCM, linear Granger, and nonlinear Granger across multiple forecast horizons.

**How**: For each feature and horizon, runs all three tests. Produces a summary classification:

| CCM Significant | Granger Improvement > 1% | Classification |
|---|---|---|
| Yes | Yes | Strong Causal Driver |
| Yes | No | Structural Cause (Hidden) |
| No | Yes | Predictive Proxy |
| No | No | Noise/Irrelevant |

**Why**: No single causality test is complete. CCM detects dynamical coupling, Granger detects predictive utility. Combining them produces a 2×2 classification that distinguishes true causes from mere correlations.

| **Returns** | Type | Description |
|---|---|---|
| `'ccm_scores'` | `pd.DataFrame` | CCM scores per feature × horizon |
| `'ccm_significant'` | `pd.DataFrame` | Boolean significance mask |
| `'granger_linear'` | `pd.DataFrame` | Linear Granger improvements |
| `'granger_nonlinear'` | `pd.DataFrame` | Nonlinear Granger improvements |
| `'summary'` | `pd.DataFrame` | Final classification per feature |

---

### 4.2 diagnostics.py

Seven-test forensic diagnostic suite for forecast model evaluation.

---

#### `class ForensicEnsembleAnalyzer`

```python
class ForensicEnsembleAnalyzer:
    def __init__(
        self, df: pd.DataFrame, target_col: str, model_cols: List[str],
        date_col: Optional[str] = None, horizon_col: Optional[str] = None,
        analysis_horizon: Optional[int] = None,
        feature_cols: Optional[List[str]] = None
    )
```

**What**: Runs 7 statistical tests on forecast residuals to evaluate whether a model's errors behave like white noise (good) or contain exploitable patterns (bad).

**How**: Accepts a DataFrame with actual values and one or more model prediction columns. Computes residuals (actual − predicted) and runs:

| # | Test | Method | Pass Condition |
|---|---|---|---|
| 1 | Baseline Check | MAE vs naive & seasonal naive | Model MAE < both benchmarks |
| 2 | Ljung-Box | `acorr_ljungbox` | p > 0.05 (no autocorrelation) |
| 3 | Shapiro-Wilk | `scipy.stats.shapiro` | p > 0.05 (normality) |
| 4 | Spectral Analysis | Welch PSD (`scipy.signal.welch`) | CV of PSD < 1.2 (no hidden periodicity) |
| 5 | Hurst Exponent | Log-log regression of lag vs. dispersion | 0.4 ≤ H ≤ 0.6 (random walk) |
| 6 | Entropy Ratio | Histogram entropy of residuals vs target | ratio > 0.75 |
| 7 | Feature Leakage | RandomForest R² on residuals from features | adj R² ≤ 0.05 |

**Why**: Each test targets a different failure mode:
- Test 1: Does the model add value over trivial benchmarks?
- Tests 2-3: Are residuals i.i.d. normal (classical assumption)?
- Test 4: Is there hidden cyclical structure the model missed?
- Test 5: Is there long-range dependence (persistent error patterns)?
- Test 6: Are residuals appropriately complex (not too predictable)?
- Test 7: Can features predict residuals (data leakage or missing features)?

The composite "Forensic Score" (count of passed tests out of 6-7) provides a single quality metric.

**Methods:**

##### `run_full_analysis() -> pd.DataFrame`

Runs all 7 tests on all models. Returns DataFrame sorted by Forensic Score (desc) then MAE (asc).

##### `get_detailed_results(model_name) -> Dict[str, Any]`

Returns detailed statistics for each test including test-specific values (p-values, Hurst exponent, adjusted R², etc.).

##### `get_residuals(model_name) -> np.ndarray`

Returns residuals for a specific model.

**Dependencies**: `scipy.signal.welch`, `scipy.stats.shapiro`, `statsmodels.stats.diagnostic.acorr_ljungbox`, `sklearn.ensemble.RandomForestRegressor`

---

## 5. intelligence/

### 5.1 market_intelligence.py

Top-level orchestrator that coordinates regime detection, pipeline selection, and report generation.

---

#### `class MarketIntelligence`

```python
class MarketIntelligence:
    def __init__(self, fred_api_key: Optional[str] = None)
```

**What**: Unified market analysis system that runs the full intelligence pipeline: data fetch → regime detection → pipeline selection → forecasting → causal analysis → report generation.

**How**: Composes four components:
1. `DataHub` — fetches price data via Yahoo Finance / CoinGecko / FRED.
2. `RegimeAnalyzer` — detects bull/bear/crisis/sideways from HMM.
3. `AutoPilot` — selects the best pipeline based on data characteristics and regime.
4. `Explainer` — compiles results into an `IntelligenceReport`.

**Why**: Provides a one-call entry point (`analyze()`) that encapsulates the entire analytical workflow. Users who need finer control can call individual components directly.

| Horizon Mapping | |
|---|---|
| `"1D"` → 1 day | `"7D"` → 7 days |
| `"14D"` → 14 days | `"30D"` → 30 days |

**Methods:**

##### `analyze(assets, horizon='7D', include_drivers=True, include_regime=True, confidence_level=0.95, verbose=False) -> IntelligenceReport`

Full analysis pipeline:
1. Fetch prices via DataHub
2. Detect regime (optional)
3. Select pipeline via AutoPilot
4. Fit and generate forecast with confidence intervals
5. Run Granger causality on asset pairs (optional)
6. Compile report via Explainer

##### `quick_forecast(asset, horizon='7D') -> pd.DataFrame`

Lightweight forecast — skips regime detection and driver analysis. Returns just the forecast DataFrame.

##### `get_regime(assets=None, lookback='1Y') -> RegimeResult`

Standalone regime detection without forecasting. Defaults to SPY if no assets specified.

##### `compare_assets(assets, horizon='7D') -> ComparisonReport`

Multi-asset comparison with correlation matrix, relative strength, and individual reports.

##### `_fetch_prices(assets, period='1y') -> pd.DataFrame`

Private helper that routes to crypto or equity loader based on ticker format (e.g., `"BTC-USD"` → crypto).

##### `_run_causal_analysis(prices_df) -> CausalityResult` (static)

Runs pairwise Granger causality on returns between all asset pairs. Only reports relationships with > 5% improvement.

**Dependencies**: `DataHub`, `RegimeAnalyzer`, `AutoPilot`, `Explainer`, `granger_causality_test`

---

### 5.2 regime_analyzer.py

Layer 1: Wraps the HMM-based RegimeDetector to classify market states into four interpretable regimes.

---

#### `@dataclass RegimeResult`

```python
@dataclass
class RegimeResult:
    current_regime: str = "unknown"
    confidence: float = 0.0
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    transition_matrix: Optional[pd.DataFrame] = None
    days_in_regime: int = 0
    transition_risk: float = 0.0
    regime_history: Optional[pd.Series] = None
```

**What**: Container for all regime detection outputs.

---

#### `class RegimeAnalyzer`

```python
class RegimeAnalyzer:
    def __init__(self, n_regimes: int = 4)
```

**What**: Detects the current market regime (bull, bear, crisis, sideways) and provides transition probabilities.

**How**:
1. Computes log returns from prices (log returns are better-behaved for HMM).
2. Fits a 4-state GMMHMM via `RegimeDetector` (auto-selection disabled).
3. Maps HMM integer states to semantic labels using return statistics:
   - Low variance + positive mean → **bull**
   - Low variance + near-zero mean → **sideways**
   - High variance + negative mean → **crisis**
   - Remaining → **bear**
4. Uses posterior probability at t_last (not Viterbi path) to determine current regime, avoiding contradictions.
5. Computes transition risk = 1 − P^window[current, current] (probability of leaving current regime within 14 days).

**Why**: 4 regimes cover the main market environments that analytical pipelines must adapt to. Using posterior probabilities rather than the Viterbi path for the current regime avoids the pathological case where Viterbi assigns a state with near-zero marginal probability.

**Methods:**

##### `detect(prices, volatility=None, macro_indicators=None) -> RegimeResult`

Main detection method. Returns current regime, confidence, probabilities, transition matrix, days in regime, transition risk, and full regime history.

##### `get_regime_history(prices, resample_freq='W') -> pd.DataFrame`

Returns one-hot encoded regime proportions resampled to target frequency. Useful for backtesting.

##### `_map_states_to_regimes(returns)` (private)

Assigns semantic labels to HMM states based on mean return and variance ranking.

##### `_count_days_in_regime(regime_series)` (static)

Counts consecutive days at the end of the series matching the last regime.

##### `_transition_risk(trans_matrix, current_state, window=14)` (static)

Computes P(leave regime within window days) = 1 − P^window[s,s] using matrix exponentiation.

**Dependencies**: `RegimeDetector`

---

### 5.3 autopilot.py

Layer 2: Automatic pipeline selection based on data characteristics and market regime.

---

#### `@dataclass DataCharacteristics`

```python
@dataclass
class DataCharacteristics:
    is_stationary: bool = False
    optimal_d: float = 0.0
    has_seasonality: bool = False
    seasonal_period: Optional[int] = None
    noise_level: float = 0.0
    autocorrelation_strength: float = 0.0
    trend_strength: float = 0.0
    outlier_fraction: float = 0.0
```

**What**: Summary of time series properties used by the decision tree.

---

#### `class AutoPilot`

```python
class AutoPilot:
    def __init__(self)
```

**What**: Selects the optimal analysis pipeline based on a rule-based decision tree.

**How**: The `select_pipeline` method implements a 4-rule decision tree:

| Priority | Condition | Pipeline | Rationale |
|---|---|---|---|
| 1 | Crisis regime | `crisis` | Robust Kalman for outlier-heavy periods |
| 2 | Non-stationary (ADF p > 0.05) | `aggressive` | Fractional diff preserves long memory |
| 3 | High autocorrelation (mean |ACF| > 0.3) | `trend_following` | Smooth-trend Kalman exploits serial dependence |
| 4 | Default | `conservative` | Balanced STL filtering + local linear trend Kalman |

**Why**: The rules are deliberately simple and interpretable so the `pipeline_reason` field explains *why* a pipeline was chosen. Complex ML-based selection would be a black box. The rule ordering reflects priority: crisis safety > stationarity handling > trend exploitation > safe default.

**Methods:**

##### `select_pipeline(data, regime='unknown', horizon=7) -> Tuple[Pipeline, str]`

Returns (Pipeline instance, human-readable reason string).

##### `analyze_data_characteristics(series) -> DataCharacteristics`

Computes: ADF stationarity, optimal fractional d, ACF seasonality detection, autocorrelation strength, noise level (1−R²), trend strength (|slope|/std), outlier fraction (> 3σ).

**Dependencies**: `PipelineRegistry`, `find_min_d_for_stationarity`, `statsmodels.tsa.stattools.adfuller`, `statsmodels.tsa.stattools.acf`

---

### 5.4 pipelines.py

Predefined pipeline configurations and the Pipeline execution engine.

---

#### `@dataclass DiagnosticsReport`

```python
@dataclass
class DiagnosticsReport:
    residual_mean: float = 0.0
    residual_std: float = 0.0
    ljung_box_pvalue: float = 1.0
    shapiro_pvalue: float = 1.0
    pass_count: int = 0
    total_tests: int = 4
    details: Dict[str, Any] = field(default_factory=dict)
```

**What**: Lightweight diagnostics container for a pipeline's residual quality.

---

#### `class Pipeline`

```python
class Pipeline:
    def __init__(self, name: str, steps: Optional[List[Tuple[str, Any]]] = None)
```

**What**: Encapsulates a complete analysis pipeline as an ordered list of named steps.

**How**:
- **Step types**:
  - `"fracdiff"` — Records optimal d but does NOT transform data. The Kalman filter handles non-stationarity natively via its level/trend state.
  - `"filter"` — Applies `TimeSeriesFilter` (STL + SARIMA).
  - `"model"` — Fits `AutoKalmanFilter`, computes in-sample residuals.
- **Frequency handling**: If the series has no inferred frequency (e.g., market data with missing weekends), it is resampled to daily with forward-fill.

**Why**: The pipeline abstraction allows mixing and matching preprocessing steps with models. The decision not to transform data with fractional diff before Kalman avoids the complex inversion step needed to map forecasts back to price space — Kalman already models the level and trend as latent states.

**Methods:**

##### `fit(series) -> Pipeline`

Runs all steps in order. Preprocessing transforms the data; the model fits on the transformed result.

##### `predict(horizon, confidence_level=0.95) -> pd.DataFrame`

Returns DataFrame with columns `forecast`, `lower`, `upper`. Uses native state-space confidence intervals from `forecast_with_ci()` when available, falling back to residual-based z-score intervals scaled by √(horizon).

##### `get_diagnostics() -> DiagnosticsReport`

Runs 4 diagnostic tests on residuals: mean ≈ 0, Ljung-Box, Shapiro-Wilk, finite std.

---

#### `class PipelineRegistry`

```python
class PipelineRegistry
```

**What**: Factory for predefined pipeline configurations. Each method returns a fresh instance.

| Method | Pipeline | Steps |
|---|---|---|
| `conservative()` | STL filtering + local linear trend Kalman | `[("filter", TSF), ("model", AKF)]` |
| `aggressive()` | Fractional diff (informational) + local linear trend Kalman | `[("fracdiff", None), ("model", AKF)]` |
| `crisis()` | Robust filtering + conservative Kalman | `[("filter", TSF), ("model", AKF)]` |
| `trend_following()` | Smooth-trend Kalman only | `[("model", AKF(level="smooth trend"))]` |

**Why**: Fresh instances per call prevent shared mutable state between analyses. The four configurations cover the main market environments identified by the regime analyzer.

---

### 5.5 explainer.py

Layer 3: Generates human-readable reports from analysis results.

---

#### `@dataclass CausalityResult`

```python
@dataclass
class CausalityResult:
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    driver_status: List[Dict[str, Any]] = field(default_factory=list)
```

**What**: Container for causal analysis output. Each relationship dict has `source`, `target`, `strength`, `improvement_pct`.

---

#### `@dataclass ComparisonReport`

```python
@dataclass
class ComparisonReport:
    assets: List[str] = field(default_factory=list)
    correlations: Optional[pd.DataFrame] = None
    relative_strength: Optional[pd.DataFrame] = None
    individual_reports: Dict[str, IntelligenceReport] = field(default_factory=dict)
```

**What**: Container for multi-asset comparison results.

---

#### `@dataclass IntelligenceReport`

```python
@dataclass
class IntelligenceReport
```

**What**: Central container for all analysis results with export methods.

**Attributes**: timestamp, assets, horizon, forecast, regime, drivers, pipeline info, quality_score (0-1), summary text, warnings, recommendations, current_prices, data_range, observations_used.

**Methods:**

##### `to_dict() -> dict`

Exports as nested dictionary suitable for JSON serialization.

##### `to_dataframe() -> pd.DataFrame`

Exports the forecast DataFrame.

##### `to_markdown(include_charts=False) -> str`

Exports as a formatted Markdown document with sections: Header, Executive Summary, Regime Analysis, Forecast Details, Key Drivers, Pipeline & Diagnostics, Warnings & Recommendations, Technical Details.

##### `save_markdown(filepath, include_charts=False) -> None`

Writes the Markdown report to a file.

---

#### `class Explainer`

```python
class Explainer
```

**What**: Generates human-readable explanations from raw analysis outputs.

**Methods:**

##### `generate_report(forecast, regime, drivers, diagnostics, ...) -> IntelligenceReport`

Compiles all inputs into an `IntelligenceReport`. Automatically computes:
- **Quality score**: fraction of diagnostic tests passed.
- **Warnings**: elevated transition risk, bear/crisis regime, diagnostic failures, wide confidence bands.
- **Recommendations**: regime-specific positioning advice.
- **Summary text**: narrative combining regime, forecast direction, confidence, and key factors.

##### `generate_summary(report, style='executive') -> str`

Produces plain-language summary in three styles:
- `"executive"`: multi-line overview with regime, forecast, confidence, warnings, quality, transition risk.
- `"technical"`: includes diagnostic details.
- `"brief"`: single paragraph.

---

#### `_quality_label(score) -> str`

Maps numeric quality score to label: ≥0.8 → HIGH, ≥0.6 → MEDIUM, ≥0.4 → LOW, else UNRELIABLE.

---

### 5.6 backtester.py

Walk-forward backtesting framework for validating the MarketIntelligence system.

---

#### `@dataclass BacktestResult`

```python
@dataclass
class BacktestResult
```

**What**: Comprehensive backtest results container.

| Field | Type | Description |
|---|---|---|
| `mae` | `float` | Mean Absolute Error |
| `rmse` | `float` | Root Mean Squared Error |
| `mape` | `float` | Mean Absolute Percentage Error |
| `directional_accuracy` | `float` | % of periods with correct direction |
| `coverage` | `float` | % of actuals within confidence bands |
| `regime_transitions_detected` | `int` | Number of detected transitions |
| `false_crisis_alerts` | `int` | Crisis detected when return > −2% |
| `missed_crisis` | `int` | Crisis not detected when return < −10% |
| `pipeline_distribution` | `Dict[str, float]` | % selection per pipeline |
| `period_results` | `pd.DataFrame` | Per-period detailed records |
| `avg_quality_score` | `float` | Mean diagnostic quality |
| `n_periods` | `int` | Total evaluation periods |

**Methods:**

##### `summary() -> str`

Formatted text summary with sections for forecast accuracy, regime detection, and pipeline selection.

---

#### `@dataclass RegimeBacktestResult`

Container for regime validation against realised volatility and drawdowns.

---

#### `@dataclass PipelineBacktestResult`

Container for pipeline selection validation with AutoPilot advantage metric.

---

#### `class Backtester`

```python
class Backtester:
    def __init__(self, fred_api_key: Optional[str] = None)
```

**What**: Walk-forward backtester that validates the full MarketIntelligence system on historical data.

**How**: At each step, uses only data available up to the current date (no look-ahead bias):
1. Slice history to current origin date.
2. Detect regime from available data.
3. Select pipeline via AutoPilot.
4. Fit and generate forecast.
5. Compare to actual outcome.
6. Advance by `step` days and repeat.

**Why**: Walk-forward backtesting is the only rigorous way to validate a time-series system. Using only past data at each step prevents data leakage. The framework validates three aspects simultaneously: forecast accuracy, regime detection quality, and pipeline selection effectiveness.

**Methods:**

##### `run_backtest(prices, start_date, end_date, horizon=7, step=7, min_history=120, verbose=True) -> BacktestResult`

Full walk-forward backtest. Tracks per-period: forecast error, directional accuracy, confidence band coverage, regime, pipeline used, regime transitions.

##### `validate_regime_detection(prices, start_date, end_date) -> RegimeBacktestResult`

Validates regime detection against realized volatility and drawdowns:
- Crisis regime should align with high-volatility periods (rolling 30-day vol × √252).
- Crisis regime should align with significant drawdowns (> 5%).

##### `validate_pipeline_selection(prices, start_date, end_date, horizon=7, step=14) -> PipelineBacktestResult`

Compares AutoPilot-selected pipeline performance against always using conservative. Computes the percentage MAE improvement (autopilot_advantage).

**Dependencies**: `RegimeAnalyzer`, `AutoPilot`, `PipelineRegistry`

---

## 6. data_sources/

### 6.1 hub.py

Unified data access facade with preset bundles for common use cases.

---

#### `class DataHub`

```python
class DataHub:
    def __init__(self, fred_api_key: Optional[str] = None)
```

**What**: Single entry point for all data sources with lazy initialization and preset bundles.

**How**: Uses Python `@property` descriptors to lazily initialize each loader only when first accessed. FRED-dependent loaders (rates, liquidity, macro, european) are imported and constructed on-demand to avoid errors when FRED key is not available.

**Why**: Lazy initialization means users who only need crypto/equity data never trigger FRED import errors. The bundle system provides curated data packages for common analyses.

| Property | Loader | Requires FRED Key |
|---|---|---|
| `.crypto` | `CryptoDataLoader` | No |
| `.equities` | `EquityDataLoader` | No |
| `.volatility` | `VolatilityDataLoader` | No |
| `.rates` | `RatesDataLoader` | Yes |
| `.liquidity` | `LiquidityDataLoader` | Yes |
| `.macro` | `MacroDataLoader` | Yes |
| `.european` | `EuropeanDataLoader` | Partial |

**Methods:**

##### `load_bundle(bundle_name, period='2y') -> Dict[str, pd.DataFrame]`

Loads predefined data bundles:

| Bundle | Contents |
|---|---|
| `"crypto_ecosystem"` | BTC/ETH prices + Fear & Greed index |
| `"risk_indicators"` | VIX, credit spreads, yield curve slope |
| `"liquidity_monitor"` | M1/M2, Fed balance sheet, repo rates |
| `"rates_dashboard"` | Treasury yields, SOFR/LIBOR, credit spreads |
| `"macro_us"` | GDP, inflation, employment, consumer sentiment |
| `"macro_europe"` | ECB rates, Euribor, Euro Area GDP |

##### `get_aligned_dataset(series_dict, target_freq='D', method='ffill') -> pd.DataFrame`

Aligns multiple series to a common frequency by resampling (last value) and forward/backward filling.

**Dependencies**: All data loader classes

---

### 6.2 crypto.py

Cryptocurrency data from Yahoo Finance and CoinGecko.

---

#### `class CryptoDataLoader`

```python
class CryptoDataLoader:
    def __init__(self, source: str = "yahoo")
```

**What**: Loads crypto OHLCV data and market metrics from free APIs.

**How**: Two backends:
- **Yahoo** (`yfinance`): Uses `yf.Ticker.history()` for single symbols, `yf.download()` for multiple. Handles MultiIndex column restructuring.
- **CoinGecko**: REST API at `api.coingecko.com/api/v3/coins/{id}/market_chart`. Maps common symbols (BTC, ETH, etc.) to CoinGecko IDs.

**Why**: Yahoo Finance provides the most complete OHLCV data but requires `yfinance`. CoinGecko provides a free alternative without dependencies but only returns price (not OHLCV).

**Methods:**

##### `get_prices(symbols, period='2y', interval='1d') -> pd.DataFrame`

OHLCV data. Yahoo format: `["BTC-USD"]`. CoinGecko format: `["bitcoin"]`.

##### `get_market_caps(symbols) -> pd.DataFrame`

365-day market cap history from CoinGecko.

##### `get_fear_greed_index(limit=365) -> pd.Series`

Crypto Fear & Greed Index (0-100) from `api.alternative.me/fng/`.

**Dependencies**: `yfinance` (Yahoo source), `requests` (CoinGecko/Fear&Greed)

---

### 6.3 equities.py

Equity and index data from Yahoo Finance.

---

#### `class EquityDataLoader`

```python
class EquityDataLoader:
    def __init__(self)
```

**What**: Loads equity/index OHLCV data and sector ETF prices.

**How**: Same yfinance pattern as `CryptoDataLoader`. Includes a mapping of 11 SPDR sector ETFs (XLF, XLK, XLE, etc.) to sector names.

**Methods:**

##### `get_prices(symbols, period='5y', interval='1d') -> pd.DataFrame`

OHLCV data for any ticker (e.g., `["SPY", "^VIX", "^GSPC"]`).

##### `get_sectors(period='2y') -> pd.DataFrame`

Close prices for all 11 S&P 500 sector ETFs, columns renamed to sector names.

**Dependencies**: `yfinance`

---

### 6.4 volatility.py

Volatility indices from Yahoo Finance.

---

#### `class VolatilityDataLoader`

```python
class VolatilityDataLoader:
    def __init__(self)
```

**What**: Loads volatility index data (VIX, VXN, OVX, SKEW).

**Methods:**

##### `get_vix(period='2y') -> pd.Series`

CBOE Volatility Index (^VIX).

##### `get_volatility_indices(period='2y') -> pd.DataFrame`

Multiple indices: VIX (equities), VXN (Nasdaq), OVX (oil).

##### `get_skew_index(period='2y') -> pd.Series`

CBOE SKEW Index — measures perceived tail risk in S&P 500.

**Dependencies**: `yfinance`

---

### 6.5 rates.py

Interest rates and fixed income data from FRED.

---

#### `class RatesDataLoader`

```python
class RatesDataLoader:
    def __init__(self, api_key: Optional[str] = None)
```

**What**: Loads US Treasury yields, Fed Funds rate, SOFR/LIBOR from the Federal Reserve Economic Data (FRED) API.

**How**: Wraps `fredapi.Fred` with error-handling. FRED series codes are stored as class-level constants.

| FRED Code | Description |
|---|---|
| `DGS1MO` – `DGS30` | Treasury yields (1M to 30Y) |
| `T10Y2Y` | 10Y-2Y spread (precomputed) |
| `FEDFUNDS` | Effective Federal Funds Rate |
| `SOFR` | Secured Overnight Financing Rate |
| `USD3MTD156N` | 3-Month LIBOR |

**Methods:**

##### `get_treasury_yields(maturities=None, start_date=None) -> pd.DataFrame`

Yields for selected maturities (default: 3M, 2Y, 10Y, 30Y).

##### `get_yield_curve_slope(start_date=None) -> pd.Series`

10Y-2Y spread (recession indicator). Uses FRED's precomputed `T10Y2Y` series, falls back to manual computation.

##### `get_fed_funds_rate(start_date=None) -> pd.Series`

Effective Federal Funds Rate.

##### `get_libor_sofr(start_date=None) -> pd.DataFrame`

SOFR and 3-Month LIBOR.

**Dependencies**: `fredapi.Fred`

---

### 6.6 liquidity.py

Liquidity and monetary indicators from FRED.

---

#### `class LiquidityDataLoader`

```python
class LiquidityDataLoader:
    def __init__(self, api_key: Optional[str] = None)
```

**What**: Loads money supply, Fed balance sheet, repo rates, and credit spreads from FRED.

**Methods:**

##### `get_money_supply(measures=None, start_date=None) -> pd.DataFrame`

M1 (`M1SL`) and M2 (`M2SL`) money supply.

##### `get_fed_balance_sheet(start_date=None) -> pd.Series`

Fed total assets (`WALCL`) in millions of dollars.

##### `get_repo_rates(start_date=None) -> pd.DataFrame`

SOFR and overnight reverse repo rate (`RRPONTSYD`).

##### `get_credit_spreads(start_date=None) -> pd.DataFrame`

ICE BofA IG spread (`BAMLC0A0CM`) and HY spread (`BAMLH0A0HYM2`).

**Dependencies**: `fredapi.Fred`

---

### 6.7 macro.py

Macroeconomic indicators from FRED.

---

#### `class MacroDataLoader`

```python
class MacroDataLoader:
    def __init__(self, api_key: Optional[str] = None)
```

**What**: Loads US macroeconomic data: GDP, inflation, employment, PMI, consumer sentiment.

**Methods:**

##### `get_gdp(real=True, start_date=None) -> pd.Series`

Real GDP (`GDPC1`) or nominal GDP (`GDP`) in billions of dollars.

##### `get_inflation(measure='CPI', start_date=None) -> pd.Series`

CPI (`CPIAUCSL`), PCE (`PCEPI`), or Core CPI (`CPILFESL`).

##### `get_employment(start_date=None) -> pd.DataFrame`

Unemployment rate (`UNRATE`), nonfarm payrolls (`PAYEMS`), initial jobless claims (`ICSA`).

##### `get_pmi(start_date=None) -> pd.DataFrame`

ISM Manufacturing and Non-Manufacturing PMI.

##### `get_consumer_sentiment(start_date=None) -> pd.DataFrame`

Michigan Consumer Sentiment (`UMCSENT`).

**Dependencies**: `fredapi.Fred`

---

### 6.8 european.py

European financial and economic data from Yahoo Finance and FRED.

---

#### `class EuropeanDataLoader`

```python
class EuropeanDataLoader:
    def __init__(self, fred_api_key: Optional[str] = None)
```

**What**: Loads ECB policy rates, Euribor, Euro Area GDP, and BTP-Bund spread.

**How**: FRED mirrors ECB data via specific series codes. BTP-Bund spread uses two approaches: (1) FRED Italian/German 10Y yields, (2) Yahoo Finance bond ETF fallback.

**Methods:**

##### `get_ecb_rates(start_date=None) -> pd.DataFrame`

Main Refinancing Rate (`ECBMRRFR`) and Deposit Facility Rate (`ECBDFR`).

##### `get_euribor(tenors=None, start_date=None) -> pd.DataFrame`

Euribor rates for 3M, 6M, 12M tenors.

##### `get_euro_area_gdp(start_date=None) -> pd.Series`

Euro Area GDP (`CLVMNACSCAB1GQEA19`).

##### `get_btp_bund_spread(period='2y') -> pd.Series`

Italian 10Y minus German 10Y yield spread. Tries FRED first (`IRLTLT01ITM156N` − `IRLTLT01DEM156N`), falls back to Yahoo Finance bond tickers.

**Dependencies**: `fredapi.Fred` (ECB/Euribor), `yfinance` (BTP-Bund fallback)

---

## 7. utils/

### 7.1 data_loader.py

Helper functions for loading time series from files.

---

#### `load_csv`

```python
def load_csv(
    filepath: str, date_col: Optional[str] = None,
    value_col: Optional[str] = None, parse_dates: bool = True,
    date_format: Optional[str] = None, freq: Optional[str] = None
) -> pd.DataFrame
```

**What**: Loads time series data from CSV with automatic date parsing.

**How**: Reads CSV, identifies date column (default: first), parses dates (auto-detecting quarterly format `YYYYQn` via `_parse_quarterly_date`), sets DatetimeIndex, and optionally selects a specific value column and frequency.

**Why**: Standardized loading ensures consistent DatetimeIndex format across all data sources. Quarterly date parsing handles the common `2020Q1` format found in economic datasets.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str` | Path to CSV file |
| `date_col` | `Optional[str]` | Date column name (default: first) |
| `value_col` | `Optional[str]` | Value column to select |
| `parse_dates` | `bool` | Auto-parse dates |
| `date_format` | `Optional[str]` | Explicit format string |
| `freq` | `Optional[str]` | Frequency to set ('D', 'M', 'Q') |
| **Returns** | `pd.DataFrame` | DataFrame with DatetimeIndex |

---

#### `load_excel`

```python
def load_excel(
    filepath: str, sheet_name: Union[str, int] = 0,
    date_col: Optional[str] = None, value_col: Optional[str] = None,
    parse_dates: bool = True, freq: Optional[str] = None
) -> pd.DataFrame
```

**What**: Loads time series from Excel with the same date parsing logic as `load_csv`.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str` | Path to .xlsx/.xls file |
| `sheet_name` | `str` or `int` | Sheet name or index |
| **Returns** | `pd.DataFrame` | DataFrame with DatetimeIndex |

---

#### `_parse_quarterly_date`

```python
def _parse_quarterly_date(date_str: str) -> pd.Timestamp
```

**What**: Parses quarterly date strings like `'2020Q1'`, `'2020-Q1'`, `'2020 Q1'` to quarter-end Timestamps.

**How**: Strips separators, matches Q1-Q4, maps to month-end dates (Q1→03-31, Q2→06-30, etc.).

---

#### `load_multiple_csv`

```python
def load_multiple_csv(
    filepaths: dict, date_col: Optional[str] = None,
    parse_dates: bool = True
) -> pd.DataFrame
```

**What**: Loads and merges multiple CSV files into a single DataFrame.

**How**: Calls `load_csv` for each file, renames columns to the key name (for single-column files) or `{key}_{col}` (for multi-column), and joins on date index with outer merge.

**Why**: Common pattern in macro analysis: loading GDP, inflation, employment from separate files and merging into a single analysis DataFrame.

| Parameter | Type | Description |
|---|---|---|
| `filepaths` | `dict` | `{'name': 'path.csv', ...}` |
| **Returns** | `pd.DataFrame` | Merged DataFrame with outer join |

---

*Documentation generated from source code analysis of `timeseries_toolkit/` (31 files, ~7,500 lines).*
