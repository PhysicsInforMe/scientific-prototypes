# Technical Documentation

## 1. Architecture Overview

### Package Structure

```
timeseries_toolkit/
├── __init__.py              # Package initialization and exports
├── preprocessing/           # Data preparation modules
│   ├── __init__.py
│   ├── fractional_diff.py   # Fractional differentiation (FFD)
│   ├── filtering.py         # STL + SARIMA filtering
│   └── imputation.py        # MICE imputation for mixed frequencies
├── models/                  # Forecasting and detection models
│   ├── __init__.py
│   ├── kalman.py            # Kalman filter / state-space models
│   ├── regime.py            # HMM-based regime detection
│   └── forecaster.py        # LightGBM global forecaster
├── validation/              # Model validation and diagnostics
│   ├── __init__.py
│   ├── causality.py         # CCM and Granger causality tests
│   └── diagnostics.py       # Forensic ensemble analyzer
├── intelligence/            # Market intelligence system
│   ├── __init__.py
│   ├── market_intelligence.py  # Main orchestrator
│   ├── regime_analyzer.py      # Layer 1: Regime detection
│   ├── autopilot.py            # Layer 2: Pipeline selection
│   ├── explainer.py            # Layer 3: Report generation
│   ├── pipelines.py            # Pipeline configurations
│   └── backtester.py           # Walk-forward backtesting
└── utils/                   # Utility functions
    ├── __init__.py
    └── data_loader.py       # CSV/Excel data loading
```

### Module Dependencies

```
                    ┌─────────────────────┐
                    │   External Input    │
                    │  (CSV, Excel, API)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   utils/data_loader │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ fractional_   │    │   filtering     │    │   imputation    │
│ diff          │    │ (STL+SARIMA)    │    │   (MICE)        │
└───────┬───────┘    └────────┬────────┘    └────────┬────────┘
        │                     │                      │
        └──────────────┬──────┴──────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌───────────┐ ┌─────────────┐
│    kalman    │ │  regime   │ │ forecaster  │
│ (state-space)│ │  (HMM)    │ │ (LightGBM)  │
└──────┬───────┘ └─────┬─────┘ └──────┬──────┘
       │               │              │
       └───────────────┼──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────┐           ┌─────────────────┐
│  causality    │           │  diagnostics    │
│ (CCM/Granger) │           │ (7 forensic     │
└───────────────┘           │  tests)         │
                            └─────────────────┘
```

### Data Flow

1. **Input Stage**: Raw data loaded via `data_loader` (handles CSV, Excel, quarterly formats)
2. **Preprocessing Stage**:
   - `fractional_diff`: Transform non-stationary series while preserving memory
   - `filtering`: Remove noise using STL decomposition + SARIMA
   - `imputation`: Handle mixed-frequency data alignment
3. **Modeling Stage**:
   - `kalman`: State-space modeling for signal extraction
   - `regime`: Detect market regimes via HMM
   - `forecaster`: Multi-entity forecasting with LightGBM
4. **Validation Stage**:
   - `causality`: Test causal relationships
   - `diagnostics`: Comprehensive model validation

---

## 2. Module Documentation

### preprocessing/fractional_diff.py

#### Mathematical Foundation

Fractional differentiation generalizes integer differentiation to non-integer orders. For a time series $X_t$, the fractionally differenced series is:

$$X_t^{(d)} = \sum_{k=0}^{\infty} w_k X_{t-k}$$

where the weights are computed using the binomial expansion:

$$w_k = -w_{k-1} \cdot \frac{d - k + 1}{k}, \quad w_0 = 1$$

#### Fixed-Width Window (FFD) Method

The FFD method truncates the infinite sum by dropping weights below a threshold:

```python
weights = [1.0]
k = 1
while abs(weights[-1]) > threshold:
    w_k = -weights[-1] * (d - k + 1) / k
    weights.append(w_k)
    k += 1
```

#### Stationarity vs Memory Trade-off

| d value | Stationarity | Memory Preservation |
|---------|--------------|---------------------|
| 0.0     | None         | 100%                |
| 0.3     | Partial      | ~70%                |
| 0.5     | Moderate     | ~50%                |
| 0.7     | Strong       | ~30%                |
| 1.0     | Full         | 0%                  |

**Optimal d selection**: Use `find_min_d_for_stationarity()` to find the minimum d that achieves stationarity (ADF test p-value < 0.05).

#### Reference
López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 5.

---

### preprocessing/filtering.py

#### Two-Stage Approach

**Stage 1: STL Decomposition**
- Seasonal-Trend decomposition using LOESS
- Extracts: Trend + Seasonal + Residual
- Robust to outliers

**Stage 2: SARIMA on Residuals**
- Models remaining autocorrelation structure
- Grid search for optimal (p,d,q)(P,D,Q,m) orders
- Selection based on AIC criterion

#### Seasonal Period Auto-Detection

| Frequency | Inferred Period |
|-----------|-----------------|
| Daily (D) | 7 (weekly)      |
| Weekly (W)| 52 (yearly)     |
| Monthly (M)| 12 (yearly)    |
| Quarterly (Q)| 4 (yearly)  |

#### Signal Dominance Index (SDI)

$$\text{SDI} = \frac{\text{Var}(\text{filtered})}{\text{Var}(\text{original})}$$

- SDI > 0.8: Strong signal, filtering recommended
- SDI < 0.3: Weak signal, filtering may remove information

#### When to Use vs Skip Filtering

**Use filtering when:**
- Series has clear seasonal patterns
- SNR (Signal-to-Noise Ratio) > 3 dB
- Residual entropy is significantly lower than original

**Skip filtering when:**
- Series is already stationary
- No discernible seasonal pattern
- Information loss exceeds noise reduction benefit

---

### preprocessing/imputation.py

#### MICE Algorithm

Multiple Imputation by Chained Equations:

1. Initialize missing values with column means
2. For each variable with missing values:
   - Fit regression model using other variables
   - Predict missing values
   - Add random noise (optional)
3. Repeat steps 2 for multiple iterations
4. Final imputed values are averages across iterations

Implementation uses `sklearn.impute.IterativeImputer` with default BayesianRidge estimator.

#### Mixed-Frequency Alignment Strategy

```
Monthly data:     Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  ...
                   │    │    │    │    │    │    │    │    │
Quarterly target:  └────┴────┴────┘    └────┴────┘    └────┴────┘
                        Q1                  Q2             Q3
```

#### Monthly to Quarterly Pivot Logic

For monthly series aligned to quarterly:
- `_M1`: First month of quarter (Jan, Apr, Jul, Oct)
- `_M2`: Second month of quarter (Feb, May, Aug, Nov)
- `_M3`: Third month of quarter (Mar, Jun, Sep, Dec)

```python
pivoted.columns = [f"{name}_M{int(c)}" for c in pivoted.columns]
```

---

### models/kalman.py

#### State-Space Representation

The Unobserved Components Model:

**Observation equation:**
$$y_t = \mu_t + \gamma_t + \epsilon_t$$

**State equations:**
$$\mu_t = \mu_{t-1} + \beta_{t-1} + \eta_t \quad \text{(local linear trend)}$$
$$\gamma_t = -\sum_{j=1}^{s-1} \gamma_{t-j} + \omega_t \quad \text{(seasonal)}$$

where:
- $y_t$: Observed series
- $\mu_t$: Trend component
- $\beta_t$: Trend slope (drift)
- $\gamma_t$: Seasonal component
- $\epsilon_t, \eta_t, \omega_t$: Noise terms

#### UnobservedComponents Specification

```python
model = UnobservedComponents(
    series,
    level='local linear trend',  # or 'local level'
    seasonal=12,                  # seasonal period
    stochastic_level=True,
    stochastic_trend=True,
    stochastic_seasonal=True
)
```

#### Forecast Confidence Intervals

The `forecast_with_ci()` method produces prediction intervals via statsmodels' native state-space forecast covariance, which accounts for:
- State estimation uncertainty
- Observation noise
- Forecast horizon uncertainty growth

This replaces the earlier residual-based approach (`z × resid_std × √h`) which underestimates uncertainty because in-sample smoothing residuals are smaller than true forecast errors. The Market Intelligence pipelines use `forecast_with_ci()` automatically when the model is an `AutoKalmanFilter`.

#### Comparison with ARIMA

| Aspect | Kalman Filter | ARIMA |
|--------|---------------|-------|
| Interpretability | Components (trend, seasonal) | Coefficients |
| Missing data | Native handling | Requires preprocessing |
| Non-stationarity | Flexible | Requires differencing |
| Computation | O(n) per update | O(n³) for MLE |
| Forecasting | Probabilistic | Point + intervals |

---

### models/regime.py

#### Hidden Markov Model Theory

**Components:**
- Hidden states: $S = \{s_1, s_2, ..., s_K\}$ (e.g., bull/bear markets)
- Observations: $O = \{o_1, o_2, ..., o_T\}$ (e.g., returns)
- Transition matrix: $A_{ij} = P(S_{t+1} = j | S_t = i)$
- Emission probabilities: $B_j(o) = P(O_t = o | S_t = j)$

#### GMMHMM vs Standard HMM

| Feature | Gaussian HMM | GMMHMM |
|---------|--------------|--------|
| Emission | Single Gaussian | Mixture of Gaussians |
| Flexibility | Limited | Can model complex distributions |
| Parameters | Fewer | More (but more expressive) |
| Use case | Simple regimes | Complex market states |

#### BIC-Based Model Selection

$$\text{BIC} = -2 \ln(L) + k \ln(n)$$

where:
- $L$: Likelihood
- $k$: Number of parameters
- $n$: Number of observations

Lower BIC indicates better model fit with appropriate complexity.

#### Viterbi Decoding

The Viterbi algorithm finds the most likely state sequence:

$$S^* = \arg\max_{S} P(S | O, \lambda)$$

Uses dynamic programming with O(T × K²) complexity.

---

### models/forecaster.py

#### GlobalBoostForecaster Design

**Global Model Approach:**
- Train single model on data from multiple entities
- Share patterns across entities (transfer learning)
- Regularization through cross-entity information

**Architecture:**
```
Entity 1 data ─┐
Entity 2 data ─┼─► Feature Engineering ─► LightGBM ─► Predictions
Entity 3 data ─┘
```

#### Feature Engineering Approach

**Monthly features (expanded to quarterly):**
```
monthly_pmi → [pmi_M1, pmi_M2, pmi_M3]  # 3 features per quarter
```

**Weekly/Daily features (aggregated):**
```
weekly_claims → [claims_mean, claims_std, claims_min, claims_max, claims_last]
```

#### Why LightGBM for Time Series

1. **Handles mixed types**: Categorical (entity_id) + numeric features
2. **Missing values**: Native support (no imputation needed in model)
3. **Non-linear patterns**: Captures complex relationships
4. **Fast training**: Histogram-based algorithm
5. **Feature importance**: Built-in interpretability

---

### validation/causality.py

#### CCM Theory (Sugihara et al. 2012)

**Core Idea:** If X causes Y, then the state of X is encoded in Y's dynamics. We can recover X from Y's shadow manifold.

**Shadow Manifold Reconstruction:**
Using Takens' embedding theorem:
$$M_Y = \{(y_t, y_{t-\tau}, y_{t-2\tau}, ..., y_{t-(E-1)\tau})\}$$

where:
- $E$: Embedding dimension
- $\tau$: Time delay

**CCM Algorithm:**
1. Reconstruct shadow manifold $M_Y$ from target Y
2. For each point in $M_Y$, find nearest neighbors
3. Use neighbors' weights to predict corresponding X values
4. CCM score = correlation(actual X, predicted X)

#### Granger Causality Assumptions

1. **Stationarity**: Both series should be stationary
2. **Linearity**: Assumes linear relationships
3. **No confounders**: No common cause affecting both series
4. **Temporal precedence**: Cause must precede effect

#### When CCM vs Granger

| Use CCM when: | Use Granger when: |
|---------------|-------------------|
| Nonlinear dynamics | Linear relationships |
| Deterministic chaos | Stochastic processes |
| Bidirectional causality suspected | Unidirectional testing |
| Short time series | Long time series |

---

### validation/diagnostics.py

#### Seven Diagnostic Tests

| # | Test | What it checks | Pass criterion |
|---|------|----------------|----------------|
| 1 | Baseline Beat | Outperforms naive forecast | RMSE < naive RMSE |
| 2 | Ljung-Box | Residuals are white noise | p-value > 0.05 |
| 3 | Shapiro-Wilk | Residuals are normally distributed | p-value > 0.05 |
| 4 | Spectral Analysis | No periodic patterns in residuals | No dominant frequencies |
| 5 | Hurst Exponent | Residuals have no long memory | H ≈ 0.5 |
| 6 | Entropy Ratio | Residuals are unpredictable | High entropy |
| 7 | Feature Leakage | No future information used | Low correlation with future |

#### Scoring System Logic

```python
forensic_score = sum([
    baseline_beat,      # +1 if passes
    ljung_box_pass,     # +1 if passes
    normality_pass,     # +1 if passes
    spectral_pass,      # +1 if passes
    hurst_pass,         # +1 if passes
    entropy_pass,       # +1 if passes
    leakage_pass        # +1 if passes (optional)
])
# Maximum score: 6 or 7
```

#### Interpretation Guidelines

| Score | Interpretation |
|-------|----------------|
| 6-7   | Excellent: Model captures signal well |
| 4-5   | Good: Minor issues, generally reliable |
| 2-3   | Fair: Significant concerns, investigate |
| 0-1   | Poor: Model likely flawed or overfitted |

---

## 3. Usage Examples

### End-to-End Workflow

```python
import pandas as pd
from timeseries_toolkit.preprocessing import fractional_diff, filtering
from timeseries_toolkit.models import kalman
from timeseries_toolkit.validation import diagnostics

# 1. Load data
from timeseries_toolkit.utils.data_loader import load_csv
df = load_csv('gdp_quarterly.csv', date_col='date', value_col='gdp')

# 2. Check stationarity and apply fractional differentiation
from timeseries_toolkit.preprocessing.fractional_diff import find_min_d_for_stationarity
d_opt, results = find_min_d_for_stationarity(df['gdp'])
print(f"Optimal d for stationarity: {d_opt}")

# 3. Apply filtering to extract signal
from timeseries_toolkit.preprocessing.filtering import TimeSeriesFilter
filter_ = TimeSeriesFilter()
filter_.fit(df['gdp'])
filtered = filter_.transform()
metrics = filter_.get_metrics()
print(f"SNR: {metrics['snr_db']:.2f} dB")

# 4. Fit Kalman filter for smoothing
from timeseries_toolkit.models.kalman import AutoKalmanFilter
kf = AutoKalmanFilter()
kf.fit(filtered)
smoothed = kf.smooth()
forecast = kf.forecast(steps=4)

# 5. Validate model
from timeseries_toolkit.validation.diagnostics import ForensicEnsembleAnalyzer
# Create validation DataFrame
val_df = pd.DataFrame({
    'actual': df['gdp'],
    'model_pred': smoothed
})
analyzer = ForensicEnsembleAnalyzer(
    df=val_df,
    target_col='actual',
    model_cols=['model_pred']
)
report = analyzer.run_full_analysis()
print(report)
```

### Common Use Cases

#### Use Case 1: Regime Detection for Trading

```python
from timeseries_toolkit.models.regime import RegimeDetector

detector = RegimeDetector(max_states=3)
detector.fit(returns_series)
regimes = detector.predict_regimes()
probs = detector.get_regime_probabilities()

# Map regimes to actions
regime_labels = {0: 'Bear', 1: 'Neutral', 2: 'Bull'}
current_regime = regime_labels[regimes[-1]]
```

#### Use Case 2: Causal Feature Selection

```python
from timeseries_toolkit.validation.causality import run_full_causality_analysis

result = run_full_causality_analysis(
    feature_df,
    target_col='sales',
    horizons=[1, 2, 3, 4]
)

# Get features that Granger-cause the target
causal_features = result['summary'][
    result['summary']['Classification'] == 'Causal'
].index.tolist()
```

#### Use Case 3: Mixed-Frequency Nowcasting

```python
from timeseries_toolkit.preprocessing.imputation import MixedFrequencyImputer
from timeseries_toolkit.models.forecaster import GlobalBoostForecaster

# Prepare data
X_dict = {
    'monthly_pmi': pmi_series,
    'weekly_claims': claims_series,
}
quarterly_index = pd.date_range('2020-01-01', periods=20, freq='QE')

# Impute and align
imputer = MixedFrequencyImputer()
imputer.fit(X_dict, quarterly_index)
X_aligned = imputer.transform(X_dict)

# Forecast
forecaster = GlobalBoostForecaster()
forecaster.fit({'US': {'y': gdp_series, 'X': X_dict}})
prediction = forecaster.predict(X_dict, gdp_series, 'US', n_periods=1)
```

---

## 4. API Reference

### preprocessing.fractional_diff

#### `get_weights_ffd(d, threshold=1e-5)`
Compute FFD weights.
- **d** (float): Differentiation order [0, 2]
- **threshold** (float): Weight cutoff
- **Returns**: np.ndarray of weights

#### `frac_diff_ffd(series, d, threshold=1e-5)`
Apply fractional differentiation.
- **series** (pd.DataFrame): Input data
- **d** (float): Differentiation order
- **Returns**: pd.DataFrame with differentiated values

#### `find_min_d_for_stationarity(series, max_d=1.0, step=0.05)`
Find minimum d for ADF stationarity.
- **series** (pd.Series): Input series
- **Returns**: Tuple[float, pd.DataFrame]

### preprocessing.filtering

#### `TimeSeriesFilter`
Two-stage STL + SARIMA filter.

**Methods:**
- `fit(series)`: Fit the filter
- `transform()`: Get filtered series
- `get_residuals()`: Get residuals
- `get_metrics()`: Get SNR, variance metrics
- `should_filter(series)`: Check if filtering recommended

### preprocessing.imputation

#### `MixedFrequencyImputer`
MICE-based imputation for mixed frequencies.

**Methods:**
- `fit(X_dict, target_index)`: Fit imputer
- `transform(X_dict)`: Transform and impute

#### `align_to_quarterly(series, target_index, aggregation='last')`
Align monthly/weekly series to quarterly.

### models.kalman

#### `AutoKalmanFilter`
Wrapper for statsmodels UnobservedComponents.

**Methods:**
- `fit(series)`: Fit state-space model
- `smooth()`: Kalman smoothing
- `forecast(steps)`: Out-of-sample point forecast
- `forecast_with_ci(steps, confidence_level=0.95)`: Forecast with native confidence intervals using statsmodels' state-space forecast covariance. Returns `(forecast, lower, upper)` as pd.Series tuple.
- `predict(start, end)`: In-sample and out-of-sample predictions
- `get_components()`: Extract trend, seasonal, etc.
- `get_residuals()`: Model residuals

### models.regime

#### `RegimeDetector`
HMM-based regime detection.

**Methods:**
- `fit(series)`: Fit GMMHMM with state selection
- `predict_regimes()`: Viterbi decoding
- `get_regime_probabilities()`: Posterior probabilities

### models.forecaster

#### `GlobalBoostForecaster`
LightGBM-based global forecaster.

**Methods:**
- `fit(all_entities_data)`: Train on multiple entities
- `predict(X_dict, y_series, entity_id, n_periods)`: Forecast
- `get_feature_importance()`: Feature importance scores

### validation.causality

#### `ccm_test(source, target, embedding_dim=3, tau=1, n_surrogates=30)`
Convergent Cross Mapping test.
- **Returns**: Dict with ccm_score, is_significant, p_value

#### `granger_causality_test(data, target_col, source_cols, max_lags=4)`
Granger causality test.
- **Returns**: Dict with delta_rmse, improvement_pct

### validation.diagnostics

#### `ForensicEnsembleAnalyzer`
Comprehensive model diagnostics.

**Methods:**
- `run_full_analysis()`: Run all 7 tests
- `get_detailed_results(model_name)`: Detailed test results
- `get_residuals(model_name)`: Model residuals

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Sugihara, G., et al. (2012). "Detecting Causality in Complex Ecosystems." *Science*, 338(6106), 496-500.
3. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2), 357-384.
4. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd edition.
5. Durbin, J., & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*, 2nd edition. Oxford University Press.
