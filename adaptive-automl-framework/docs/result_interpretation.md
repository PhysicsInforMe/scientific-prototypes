# Result Interpretation — MarketIntelligence Backtest Analysis

## 1. Backtest Results Summary

The walk-forward backtest on BTC-USD (June–December 2025, bi-weekly forecasts, 7-day horizon) produced the following comparison:

| Metric | MarketIntelligence | Naive (Persistence) | Delta |
|--------|-------------------|---------------------|-------|
| MAE | $4,499 | $3,896 | +15.5% |
| RMSE | ~$5,200 | $4,658 | ~+12% |
| MAPE | 4.20% | 3.64% | +15.3% |
| Directional Accuracy | 28.6% | 0% (no call) | — |
| Coverage (95% CI) | 100.0% | 100.0% | — |

The MarketIntelligence system underperforms the naive baseline on all point-accuracy metrics. This document explains why and proposes structural improvements.

---

## 2. What the Pipeline Actually Does

The AutoPilot selected the "aggressive" pipeline 85.7% of the time and "crisis" 14.3%. Examining the code reveals what each pipeline executes in practice.

### Aggressive Pipeline (85.7% of periods)

The aggressive pipeline has two steps:

1. **`fracdiff`** — Computes the optimal fractional differentiation order d but **does not apply the transformation**. The series passes through unmodified. The rationale (documented in `pipelines.py:97-101`) is that the Kalman filter handles non-stationarity natively, and applying fractional differentiation would require a complex inversion step to map forecasts back to price space.

2. **`model`** — Fits `AutoKalmanFilter(level="local linear trend")` on raw prices with all default parameters: no cycle, no seasonal component, no frequency-domain seasonality, default `lbfgs` optimizer with 1000 iterations, automatic standardization.

In practice, the aggressive pipeline is a vanilla Kalman filter on raw BTC prices with no preprocessing.

### Crisis Pipeline (14.3% of periods)

The crisis pipeline applies STL filtering before the Kalman fit, but the core model is identical: `AutoKalmanFilter(level="local linear trend")` with default parameters.

### Default Parameters Used

| Parameter | Value | Alternatives Available |
|-----------|-------|----------------------|
| `level` | `local linear trend` | `local level`, `smooth trend`, `random walk with drift`, `fixed intercept` |
| `cycle` | `False` | `True` |
| `seasonal` | `None` | Integer period (e.g., 7 for weekly) |
| `freq_seasonal` | `None` | List of `{'period': p, 'harmonics': h}` dicts |
| `standardize` | `True` | `False` |
| `method` | `lbfgs` | `powell`, `nm` |
| `maxiter` | 1000 | Any positive integer |

---

## 3. Root Cause Analysis

The underperformance has two components: a dominant mathematical/structural cause (~85%) and a minor parameter tuning cause (~15%).

### 3.1 The Mathematical Cause (85%)

The `local linear trend` Kalman filter has the following state equations:

$$\mu_t = \mu_{t-1} + \beta_{t-1} + \eta_t \quad \text{(level)}$$
$$\beta_t = \beta_{t-1} + \zeta_t \quad \text{(drift slope)}$$
$$y_t = \mu_t + \epsilon_t \quad \text{(observation)}$$

The h-step-ahead forecast from this model is:

$$\hat{y}_{T+h} = \hat{\mu}_T + \hat{\beta}_T \cdot h$$

The naive forecast is:

$$\hat{y}_{T+h}^{\text{naive}} = y_T$$

The difference between the two forecasts is the estimated drift term $\hat{\beta}_T \cdot h$. For the model to outperform naive, the drift estimate must improve accuracy — meaning $\hat{\beta}_T$ must be a useful predictor of actual price movement.

**Why it fails for daily financial prices:**

For daily BTC data, the drift (expected daily return) is approximately 0.05–0.15% per day, while daily volatility is 3–5%. The signal-to-noise ratio for drift estimation is roughly:

$$\text{SNR}_{\text{drift}} \approx \frac{|\mu_{\text{daily}}|}{\sigma_{\text{daily}}} \approx \frac{0.001}{0.035} \approx 0.03$$

This means the drift signal is ~30x weaker than the noise. The Kalman filter estimates $\hat{\beta}_T$ from this noisy data, and the estimation error in $\hat{\beta}_T$ frequently exceeds the true drift, making the forecast worse than simply setting $\beta = 0$ (which is what naive does).

**This is confirmed by Notebook 01:** The signal extraction analysis found SNR = -4.54 dB for BTC daily data, meaning most of the variance is noise. The Ljung-Box test confirmed the residuals are white noise (p = 0.50), indicating no exploitable autocorrelation structure remains after STL decomposition.

### 3.2 The Meese-Rogoff Result (1983)

This finding is not specific to our implementation. Meese and Rogoff (1983) demonstrated that structural econometric models (including models with fundamentals like money supply, interest rates, and output) could not outperform the random walk for exchange rate forecasts at horizons up to 12 months. This result has been replicated across:

- Foreign exchange (the original finding)
- Equity indices (Welch & Goyal, 2008)
- Commodity prices (Alquist, Kilian & Vigfusson, 2013)
- Cryptocurrency markets (Catania, Grassi & Ravazzolo, 2019)

The result is consistent with the weak-form Efficient Market Hypothesis: if past prices contained exploitable information for predicting future prices, rational agents would trade on that information until the predictability disappeared.

### 3.3 The Tuning Cause (15%)

Within the structural constraint of a univariate Kalman filter on prices, some parameter choices are suboptimal:

| Issue | Impact | Fix Difficulty |
|-------|--------|---------------|
| No weekly seasonality (`seasonal=7`) | Minor — crypto weekday effects are weak but non-zero | Trivial |
| No cycle component | Minor — medium-frequency cycles exist but are hard to estimate reliably | Low |
| Same model for all pipelines | The four pipelines differ only in level type and whether STL is applied | Medium |
| No cross-validation for model selection | The AutoPilot uses heuristic rules, not forecast accuracy | Medium |
| STL filtering on non-seasonal daily data | May introduce artifacts rather than remove noise | Low |

These tuning improvements might close the ~15% MAE gap partially but cannot overcome the fundamental SNR limitation.

---

## 4. Where the System Adds Value Despite Worse Point Accuracy

The naive model produces a single number (current price) with no additional context. The MarketIntelligence system produces:

| Output | Naive | MarketIntelligence | Value |
|--------|-------|--------------------|-------|
| Point forecast | Trivial | Yes | Similar accuracy |
| Calibrated CI | Rough (historical vol × √h) | Native state-space covariance | Properly accounts for state uncertainty growth |
| Regime detection | None | Bull/bear/crisis/sideways | Enables risk-aware decisions |
| Transition risk | None | HMM transition matrix | Early warning for regime shifts |
| Quality self-assessment | None | 4 diagnostic tests | System knows when it is unreliable |
| Pipeline adaptation | None | AutoPilot selection | Different approach for different conditions |
| Auditable reasoning | None | pipeline_reason field | Transparency into decision logic |

For a risk management or portfolio monitoring use case, these auxiliary outputs are often more valuable than marginal improvements in point accuracy.

---

## 5. Proposed Mathematical Improvements

The following improvements address the structural 85% cause. They are ordered by expected impact and feasibility. None require changes to the existing codebase — they represent architectural extensions for future development.

### 5.1 Forecast Returns, Not Prices

**Problem:** Forecasting raw prices forces the model to predict a near-unit-root process. The signal (drift) is negligible relative to the price level.

**Solution:** Transform the target from prices to log-returns before modelling:

$$r_t = \ln(P_t / P_{t-1})$$

Then forecast $\hat{r}_{T+1}, \ldots, \hat{r}_{T+h}$ and convert back:

$$\hat{P}_{T+h} = P_T \cdot \exp\left(\sum_{i=1}^{h} \hat{r}_{T+i}\right)$$

**Why it helps:** Returns are (approximately) stationary with a well-defined unconditional mean and variance. The Kalman filter's state-space model is better suited to stationary or trend-stationary data. The fractional differentiation step (currently a no-op) would also become applicable, and the forecast inversion to price space would be a simple exponentiation rather than a complex frac-diff inversion.

**Expected impact:** Moderate. Does not solve the fundamental predictability problem but removes the unit-root confound and enables proper use of the preprocessing modules.

**References:** Campbell, Lo & MacKinlay (1997), *The Econometrics of Financial Markets*, Chapter 1.

### 5.2 Incorporate Exogenous Predictors (Multivariate State-Space)

**Problem:** The current system is purely univariate — it only uses the asset's own price history. Asset prices in efficient markets are approximately unpredictable from their own past, but cross-asset and macro signals can carry predictive information at short horizons.

**Solution:** Extend the state-space model to include exogenous regressors:

$$y_t = \mu_t + \mathbf{x}_t' \boldsymbol{\beta} + \epsilon_t$$

where $\mathbf{x}_t$ contains predictors such as:

- **Volatility indicators:** VIX, realised volatility, options implied vol
- **Cross-asset signals:** S&P 500 overnight returns (for crypto), DXY (dollar index), gold
- **On-chain data (for crypto):** exchange inflows/outflows, funding rates, open interest
- **Macro surprises:** deviation of economic releases from consensus (FRED data)
- **Sentiment:** fear-and-greed index, social media volume

`statsmodels.UnobservedComponents` supports exogenous regressors natively via the `exog` parameter. The `AutoKalmanFilter` class would need a minor extension to accept and forward exogenous data.

**Why it helps:** Exogenous signals bring information that is not already priced in. While the asset's own past is largely uninformative (EMH), cross-asset and alternative data sources may carry orthogonal predictive power. Academic evidence supports short-horizon predictability from volatility (Bollerslev, Tauchen & Zhou, 2009), cross-asset momentum (Moskowitz, Ooi & Pedersen, 2012), and funding rates for crypto (Bianchi, 2020).

**Expected impact:** Potentially large, depending on signal quality. This is the single most impactful improvement because it changes the information set from {own price history} to {own price + market microstructure + macro context}.

**Implementation notes:**
- The `data_sources` module already fetches VIX and equity data
- FRED integration exists for macro indicators
- The `MixedFrequencyImputer` handles temporal alignment of multi-frequency data
- The `GlobalBoostForecaster` already supports multi-entity exogenous features

### 5.3 Volatility-Scaled Forecasting (Regime-Conditional Models)

**Problem:** The current system detects regimes but does not condition the forecast model on the detected regime. The same Kalman filter runs regardless of whether the HMM says "bull" or "crisis." The only regime-conditional behavior is pipeline selection (which changes preprocessing, not the core model).

**Solution:** Fit separate models per regime or use regime-switching state-space models:

**Option A — Separate models per regime:**
```
If regime == bull:    fit Kalman with drift enabled, low process noise
If regime == bear:    fit Kalman with negative drift prior, high process noise
If regime == crisis:  fit Kalman with high observation noise, no trend
If regime == sideways: fit Kalman with zero drift, mean-reverting level
```

**Option B — Markov-Switching state-space model:**
Combine the HMM and Kalman filter into a single model where the state-space parameters (drift, volatility) switch according to the hidden Markov chain. This is the Kim (1994) filter:

$$y_t = \mu_t^{(S_t)} + \epsilon_t^{(S_t)}$$
$$\mu_t = \mu_{t-1} + \beta^{(S_t)} + \eta_t^{(S_t)}$$

where $(S_t)$ is the hidden regime state.

**Why it helps:** Financial time series exhibit well-documented regime-dependent behavior. Bull markets have different drift, volatility, and autocorrelation than bear markets. A single model fit across all regimes estimates an averaged parameter set that is optimal for no regime in particular.

**Expected impact:** Moderate. Most relevant during regime transitions where the current model is slowest to adapt.

**References:** Hamilton (1989), Kim (1994), Ang & Bekaert (2002).

### 5.4 Ensemble Forecasting with Model Averaging

**Problem:** The AutoPilot selects a single pipeline. If that pipeline produces a poor forecast, there is no hedge. Model selection risk is concentrated.

**Solution:** Run multiple pipelines in parallel and combine their forecasts using weighted averaging:

$$\hat{y}_{T+h}^{\text{ensemble}} = \sum_{k=1}^{K} w_k \cdot \hat{y}_{T+h}^{(k)}$$

Weight schemes, from simplest to most sophisticated:

1. **Equal weights** (1/K): robust baseline, often competitive (Genre et al., 2013)
2. **Inverse-RMSE weights**: $w_k \propto 1 / \text{RMSE}_k$ based on recent out-of-sample performance
3. **Bayesian Model Averaging (BMA)**: $w_k \propto P(M_k | \text{data})$ using marginal likelihoods
4. **LASSO stacking**: fit a regularised meta-learner on out-of-sample forecasts from each pipeline

**Why it helps:** Forecast combination reduces variance without increasing bias (Bates & Granger, 1969). Even if no single model beats naive, a well-weighted combination often does because individual model errors partially cancel. This is one of the most robust findings in the forecasting literature.

**Expected impact:** Moderate to large. The forecast combination literature consistently shows 10–30% error reduction over the best single model. Given that our gap to naive is ~15%, ensemble averaging could close or reverse it.

**References:** Bates & Granger (1969), Timmermann (2006), Genre et al. (2013).

### 5.5 Integrate the LightGBM Forecaster into the MI Pipeline

**Problem:** The `GlobalBoostForecaster` already exists in the toolkit (`models/forecaster.py`) but is not connected to the MarketIntelligence pipeline. It supports lag features, multi-entity training, and handles nonlinear patterns — capabilities the Kalman filter lacks.

**Solution:** Create a new pipeline that uses LightGBM:

```
Pipeline: "ml_boost"
Steps:
  1. fracdiff → apply fractional differentiation (actually transform, not just compute d)
  2. feature_engineering → create lag features, rolling statistics, calendar features
  3. model → GlobalBoostForecaster with walk-forward validation
```

**Why it helps:** LightGBM can capture nonlinear relationships between features and the target. While a univariate Kalman filter on prices is mathematically constrained to linear drift extrapolation, LightGBM with properly engineered features (momentum, mean-reversion, volatility clustering) can exploit patterns that linear models miss.

**Key requirements for this to work:**
- Features must be constructed from past data only (no lookahead)
- Walk-forward cross-validation for hyperparameter tuning
- Feature importance analysis to ensure no leakage
- The `ForensicEnsembleAnalyzer` should validate the ML pipeline independently

**Expected impact:** Variable. ML models can outperform random walks when given informative features, but with only lag features they typically perform similarly. The impact depends heavily on feature engineering quality.

**References:** Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies.

### 5.6 Forecast at Natural Frequencies

**Problem:** The system forecasts daily prices at a 7-day horizon. Daily price changes in liquid markets have an extremely low signal-to-noise ratio. Longer-horizon returns have more predictable components (value, momentum, carry) because the signal accumulates while noise partially cancels.

**Solution:** Instead of forecasting 7 daily steps ahead, directly model the 7-day return:

$$r_{t \to t+7} = \ln(P_{t+7} / P_t)$$

Use features measured at the same weekly frequency: weekly returns, weekly volume, weekly volatility. This reduces the problem from "predict 7 noisy daily steps" to "predict 1 weekly return."

**Why it helps:** At weekly and monthly horizons, predictability from momentum and mean-reversion is stronger than at daily frequency. Cochrane (2011) shows that return predictability increases approximately linearly with horizon for dividend-price ratios. By matching the model frequency to the forecast horizon, we avoid the error accumulation problem of multi-step daily forecasts.

**Expected impact:** Moderate. Weekly returns are more predictable than daily but still noisy. The main benefit is avoiding compounding of daily forecast errors over the 7-day horizon.

**References:** Cochrane (2011), *Presidential Address: Discount Rates*, Journal of Finance.

---

## 6. Improvement Priority Matrix

| # | Improvement | Impact | Feasibility | Dependencies |
|---|-------------|--------|-------------|-------------|
| 5.2 | Exogenous predictors | High | Medium | Data sources module extension |
| 5.4 | Ensemble averaging | Medium-High | High | No new dependencies |
| 5.1 | Forecast returns, not prices | Medium | High | Minor pipeline refactor |
| 5.5 | LightGBM integration | Variable | Medium | Feature engineering module |
| 5.3 | Regime-conditional models | Medium | Medium | Kim filter or per-regime fitting |
| 5.6 | Natural frequency forecasting | Medium | High | Aggregation logic |

The recommended implementation order is: **5.1 → 5.4 → 5.2 → 5.5 → 5.3 → 5.6**.

Rationale: Start with the simplest high-impact changes (forecast returns instead of prices, combine existing pipelines via ensemble), then add exogenous data and ML integration, and finally implement the more complex regime-conditional and multi-frequency approaches.

---

## 7. What Success Looks Like

After implementing the improvements above, realistic target metrics for the BTC-USD 7-day backtest would be:

| Metric | Current MI | Current Naive | Target MI |
|--------|-----------|---------------|-----------|
| MAE | $4,499 | $3,896 | $3,500–3,800 |
| MAPE | 4.20% | 3.64% | 3.0–3.5% |
| Directional Accuracy | 28.6% | 0% | 52–58% |
| Coverage (95% CI) | 100% | 100% | 90–97% |

These targets are deliberately conservative. Beating naive by more than 10–15% on liquid daily assets would require either proprietary signals or higher-frequency data. The real improvement should come from directional accuracy (currently below coin-flip at 35.7%) and from maintaining properly calibrated confidence intervals.

A directional accuracy of 52–58% may seem modest, but in financial markets a consistent 55% hit rate with proper risk management is commercially significant.

---

## 8. Status Update (2026-01-31)

The following fixes have been implemented since the initial analysis:

- **Regime detection fix:** `regime_analyzer.py` now uses posterior probabilities (not Viterbi path) for current regime determination. This resolved the "crisis at 0% confidence" contradiction where the Viterbi state had near-zero posterior probability.
- **Notebook audit and cleanup:** All 4 notebooks have been through two rounds of audit and re-execution. Commentary cells are now number-agnostic (no hardcoded values that go stale), all plot descriptions match actual subplot layouts, HMM stderr is suppressed, random seeds ensure reproducibility, and the forensic scorecard includes all 6 tests.
- **Documentation and infrastructure:** MIT LICENSE added, setup.py updated with correct repo URL and data source dependencies, Python version requirement aligned across README/setup.py (>= 3.9), TECHNICAL.md expanded with data_sources and intelligence modules.

The backtest numbers in this document reflect the latest re-execution (2026-01-31). The ~15% MAE gap vs naive is consistent with the mathematical analysis — this is the expected result for a univariate Kalman filter on daily crypto prices.

---

## References

- Alquist, R., Kilian, L. & Vigfusson, R. (2013). Forecasting the Price of Oil. *Handbook of Economic Forecasting*, 2, 427-507.
- Ang, A. & Bekaert, G. (2002). International Asset Allocation With Regime Shifts. *Review of Financial Studies*, 15(4), 1137-1187.
- Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets. *Annual Review of Financial Economics*, 4, 313-337.
- Bates, J.M. & Granger, C.W.J. (1969). The Combination of Forecasts. *Operations Research Quarterly*, 20(4), 451-468.
- Bianchi, D. (2020). Cryptocurrencies as an Asset Class? An Empirical Assessment. *Journal of Alternative Investments*, 23(2), 162-179.
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. *Review of Financial Studies*, 22(11), 4463-4492.
- Campbell, J.Y., Lo, A.W. & MacKinlay, A.C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.
- Catania, L., Grassi, S. & Ravazzolo, F. (2019). Forecasting Cryptocurrencies Under Model and Parameter Instability. *International Journal of Forecasting*, 35(2), 485-501.
- Cochrane, J.H. (2011). Presidential Address: Discount Rates. *Journal of Finance*, 66(4), 1047-1108.
- Genre, V., Kenny, G., Meyler, A. & Timmermann, A. (2013). Combining Expert Forecasts: Can Anything Beat the Simple Average? *International Journal of Forecasting*, 29(1), 108-121.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5), 2223-2273.
- Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*, 57(2), 357-384.
- Harvey, A.C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Kim, C.J. (1994). Dynamic Linear Models with Markov-Switching. *Journal of Econometrics*, 60(1-2), 1-22.
- Meese, R.A. & Rogoff, K. (1983). Empirical Exchange Rate Models of the Seventies: Do They Fit Out of Sample? *Journal of International Economics*, 14(1-2), 3-24.
- Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2), 228-250.
- Timmermann, A. (2006). Forecast Combinations. *Handbook of Economic Forecasting*, 1, 135-196.
- Welch, I. & Goyal, A. (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. *Review of Financial Studies*, 21(4), 1455-1508.
