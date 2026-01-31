# Time Series Toolkit

A comprehensive Python library for time series analysis, preprocessing, and forecasting.

## Features

### Preprocessing
- **Fractional Differentiation**: Transform non-stationary series while preserving memory
- **Filtering**: Two-stage STL + SARIMA filtering for noise removal
- **Imputation**: MICE-based imputation for mixed-frequency data

### Models
- **Kalman Filter**: Automated state-space modeling with UnobservedComponents
- **Regime Detection**: Hidden Markov Models for market regime identification
- **Forecaster**: Global gradient boosting with mixed-frequency support

### Validation
- **Causality Testing**: CCM and Granger causality tests
- **Diagnostics**: 7-test forensic analysis suite for forecast evaluation

## Installation

```bash
pip install -e .
```

Or with all optional dependencies:

```bash
pip install -e ".[all]"
```

## Quick Start

### Fractional Differentiation

```python
from timeseries_toolkit.preprocessing import frac_diff_ffd, find_min_d_for_stationarity

# Find minimum d for stationarity
min_d, results = find_min_d_for_stationarity(price_series)
print(f"Minimum d: {min_d}")

# Apply fractional differentiation
import pandas as pd
df = pd.DataFrame({'price': price_series})
diff_df = frac_diff_ffd(df, d=min_d)
```

### Time Series Filtering

```python
from timeseries_toolkit.preprocessing import TimeSeriesFilter

filter = TimeSeriesFilter()
filter.fit(noisy_series)
clean_series = filter.transform()
metrics = filter.get_metrics()
print(f"SNR: {metrics['snr_db']:.2f} dB")
```

### Kalman Filter

```python
from timeseries_toolkit.models import AutoKalmanFilter

kf = AutoKalmanFilter(level='local linear trend')
kf.fit(series)
smoothed = kf.smooth()
forecast = kf.forecast(steps=4)
```

### Regime Detection

```python
from timeseries_toolkit.models import RegimeDetector

detector = RegimeDetector(max_states=4)
detector.fit(spread_series)
regimes = detector.predict_regimes()
probs = detector.get_regime_probabilities()
```

### Causality Testing

```python
from timeseries_toolkit.validation import ccm_test, granger_causality_test

# CCM test
result = ccm_test(x_series, y_series, embedding_dim=3)
if result['is_significant']:
    print(f"Causal relationship detected (CCM={result['ccm_score']:.3f})")

# Granger test
result = granger_causality_test(df, 'target', ['feature1', 'feature2'])
print(f"Improvement: {result['improvement_pct']:.1f}%")
```

### Forecast Diagnostics

```python
from timeseries_toolkit.validation import ForensicEnsembleAnalyzer

analyzer = ForensicEnsembleAnalyzer(
    df=results_df,
    target_col='actual',
    model_cols=['model_A', 'model_B'],
    date_col='date'
)
report = analyzer.run_full_analysis()
print(report)
```

### Market Intelligence

```python
from timeseries_toolkit.intelligence import MarketIntelligence

mi = MarketIntelligence()

# Full analysis with regime detection and forecast
report = mi.analyze(["BTC-USD"], horizon="7D")
print(report.summary)

# Quick forecast
fc = mi.quick_forecast("SPY", horizon="7D")

# Regime detection only
regime = mi.get_regime(["SPY"])
print(f"Regime: {regime.current_regime} ({regime.confidence:.0%})")

# Export as Markdown
report.save_markdown("analysis.md")
```

## Module Structure

```
timeseries_toolkit/
├── preprocessing/
│   ├── fractional_diff.py  # Fractional differentiation
│   ├── filtering.py        # STL + SARIMA filtering
│   └── imputation.py       # MICE imputation
├── models/
│   ├── kalman.py           # Kalman filter
│   ├── regime.py           # HMM regime detection
│   └── forecaster.py       # Global boosting forecaster
├── validation/
│   ├── diagnostics.py      # Forensic analysis
│   └── causality.py        # Causality tests
├── intelligence/
│   ├── market_intelligence.py  # Unified orchestrator
│   ├── regime_analyzer.py      # Regime detection wrapper
│   ├── autopilot.py            # Automatic pipeline selection
│   ├── explainer.py            # Report generation
│   ├── pipelines.py            # Pipeline configurations
│   └── backtester.py           # Walk-forward backtesting
├── data_sources/
│   └── (7 data loader modules + DataHub)
└── utils/
    └── data_loader.py      # Data loading utilities
```

## Documentation

- [Technical Documentation](docs/TECHNICAL.md) - Mathematical foundations, API reference, and usage examples
- [User Guide](docs/USER_GUIDE.md) - MarketIntelligence usage guide
- [Result Interpretation](docs/result_interpretation.md) - Analysis of forecast performance vs naive baselines and proposed improvements

## Test Coverage

| Module | Test File | Description |
|--------|-----------|-------------|
| fractional_diff | test_fractional_diff.py | FFD stationarity transformation |
| filtering | test_filtering.py | STL + SARIMA filtering |
| imputation | test_imputation.py | MICE mixed-frequency imputation |
| kalman | test_kalman.py | State-space models |
| regime | test_regime.py | HMM regime detection |
| forecaster | test_forecaster.py | LightGBM global forecasting |
| causality | test_causality.py | CCM + Granger tests |
| diagnostics | test_diagnostics.py | 7-test forensic analysis |
| utils | test_utils.py | Data loading utilities |
| data_sources | test_data_sources.py | Yahoo Finance + FRED loaders |
| intelligence | test_intelligence.py | MarketIntelligence orchestrator |
| integration | test_intelligence_integration.py | End-to-end pipeline tests |
| backtest | test_backtest.py | Walk-forward backtesting |
| scientific | test_scientific_validity.py | Mathematical validity tests |
| **Total** | **325 tests** | **All passing** |

Run tests:
```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_filtering.py -v

# With coverage
python -m pytest tests/ --cov=timeseries_toolkit
```

## API Keys Setup

Some data sources require free API keys:

### FRED (Federal Reserve Economic Data)
1. Register at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable:
   - Windows: `set FRED_API_KEY=your_key`
   - Linux/Mac: `export FRED_API_KEY=your_key`
   - Or create a `.env` file (see `.env.example`)

Data sources that require FRED: `RatesDataLoader`, `LiquidityDataLoader`, `MacroDataLoader`, `EuropeanDataLoader`.

Yahoo Finance-based sources (`CryptoDataLoader`, `EquityDataLoader`, `VolatilityDataLoader`) work without any API key.

## Requirements

- Python >= 3.9
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0

Optional:
- lightgbm >= 3.0.0 (for GlobalBoostForecaster)
- hmmlearn >= 0.2.7 (for RegimeDetector)
- openpyxl (for Excel support)
- matplotlib, seaborn (for visualization)

## License

MIT License
