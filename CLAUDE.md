# Project Instructions for Claude Code

## Overview

Refactor Python notebooks from `notebooks_raw/` into a clean package called `timeseries_toolkit`. All code, comments, and documentation must be in English.

## Step 1: Create Structure

Create this folder structure (if not already present):

```
timeseries_toolkit/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   ├── fractional_diff.py
│   ├── filtering.py
│   └── imputation.py
├── models/
│   ├── __init__.py
│   ├── kalman.py
│   ├── regime.py
│   └── forecaster.py
├── validation/
│   ├── __init__.py
│   ├── diagnostics.py
│   └── causality.py
└── utils/
    ├── __init__.py
    └── data_loader.py

notebooks/
examples/sample_data/
tests/
```

## Step 2: Refactor Notebooks

Source notebooks are in `notebooks_raw/`. Process them in this order:

| Source (notebooks_raw/) | Target Module |
|-------------------------|---------------|
| Fractional_differentiation.ipynb | preprocessing/fractional_diff.py |
| filtraggio.ipynb | preprocessing/filtering.py |
| MICE.ipynb | preprocessing/imputation.py + models/forecaster.py |
| Kalman_filter.ipynb | models/kalman.py |
| Modello_di_Regime.ipynb | models/regime.py |
| metriche_causali.ipynb | validation/causality.py |
| diagnostica.ipynb | validation/diagnostics.py |

## Refactoring Rules

For each notebook:

1. Extract all functions and classes
2. Add type hints to all signatures
3. Add Google-style docstrings in English
4. Remove Colab-specific code (!pip install, file uploads)
5. Remove hardcoded paths
6. Verify mathematical formulas are correct
7. Handle edge cases (empty arrays, NaN values)
8. Create corresponding test file in tests/

## Module Specifications

### preprocessing/fractional_diff.py

Source: `notebooks_raw/Fractional_differentiation.ipynb`

Functions to extract:
- `getWeights_FFD` → `get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray`
- `fracDiff_FFD` → `frac_diff_ffd(series: pd.DataFrame, d: float, threshold: float = 1e-5) -> pd.DataFrame`
- `plotMinFFD` → `find_min_d_for_stationarity(series: pd.Series, max_d: float = 1.0) -> Tuple[float, pd.DataFrame]`
- Include synthetic data generators: `generate_random_walk`, `generate_stationary_ar1`, `generate_trend_stationary`

### preprocessing/filtering.py

Source: `notebooks_raw/filtraggio.ipynb`

Create class `TimeSeriesFilter` with:
- `fit(series)` - fit STL + SARIMA
- `transform(series)` - extract signal
- `get_residuals()` - return residuals
- `get_metrics()` - return dict with SNR, entropy, variance
- Auto-detect seasonal period from frequency

### preprocessing/imputation.py + models/forecaster.py

Source: `notebooks_raw/MICE.ipynb`

Extract and split into two modules:
- `MixedFrequencyImputer` class (MICE wrapper) → imputation.py
- `GlobalBoostForecaster` class (LightGBM) → forecaster.py
- `align_to_quarterly` function → imputation.py

Do NOT include any MIDAS polynomial weighting code (no Beta polynomials, no Almon polynomials).

### models/kalman.py

Source: `notebooks_raw/Kalman_filter.ipynb`

Create class `AutoKalmanFilter` wrapping statsmodels UnobservedComponents:
- `fit(series)`, `smooth()`, `forecast(steps)`, `get_components()`
- Function `compare_kalman_vs_arima(series, holdout)` for benchmarking

### models/regime.py

Source: `notebooks_raw/Modello_di_Regime.ipynb`

Create class `RegimeDetector` wrapping hmmlearn GMMHMM:
- `fit(series)` with BIC-based model selection
- `predict_regimes()` returns Viterbi path
- `get_regime_probabilities()` returns smoothed probs
- `aggregate_to_frequency(target_freq)` for mixed-frequency use

### validation/causality.py

Source: `notebooks_raw/metriche_causali.ipynb`

Functions:
- `ccm_test(source, target, embedding_dim, tau)` - Convergent Cross Mapping
- `granger_causality_test(data, target_col, source_cols, max_lags)`
- `generate_causal_system(n)` - synthetic data with known causal structure

### validation/diagnostics.py

Source: `notebooks_raw/diagnostica.ipynb`

Create class `ForensicEnsembleAnalyzer` with 7 tests:
1. Baseline check (beat naive)
2. Ljung-Box white noise
3. Shapiro-Wilk normality
4. Spectral analysis (Welch)
5. Hurst exponent
6. Entropy ratio
7. Feature leakage detection

## After All Modules

1. Create `requirements.txt` with dependencies
2. Create `setup.py` for pip install
3. Create `README.md` with usage examples
4. Create demo notebooks in `notebooks/`
