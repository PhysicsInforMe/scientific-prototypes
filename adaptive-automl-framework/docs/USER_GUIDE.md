# MarketIntelligence User Guide

## Table of Contents
1. [Quick Start](#1-quick-start)
2. [Installation and Setup](#2-installation-and-setup)
3. [Basic Usage](#3-basic-usage)
4. [Understanding the Output](#4-understanding-the-output)
5. [Advanced Configuration](#5-advanced-configuration)
6. [Interpreting Results](#6-interpreting-results)
7. [Common Use Cases](#7-common-use-cases)
8. [Troubleshooting](#8-troubleshooting)
9. [API Reference](#9-api-reference)

---

## 1. Quick Start

```python
from timeseries_toolkit.intelligence import MarketIntelligence

mi = MarketIntelligence()
report = mi.analyze(["BTC-USD"], horizon="7D")
print(report.summary)
```

This will:
- Fetch 1 year of BTC-USD price data from Yahoo Finance
- Detect the current market regime (bull/bear/crisis/sideways)
- Automatically select the best analytical pipeline
- Generate a 7-day forecast with confidence bands
- Run diagnostics to validate the forecast quality
- Produce a human-readable summary

---

## 2. Installation and Setup

### Requirements
- Python 3.8+
- Dependencies installed via `pip install -e .`

### API Keys

**Yahoo Finance**: No key required. Crypto, equity, and volatility data work out of the box.

**FRED (optional)**: For macro-economic indicators (GDP, CPI, interest rates).
1. Register at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable:
   - Windows: `set FRED_API_KEY=your_key`
   - Linux/Mac: `export FRED_API_KEY=your_key`
   - Or pass directly: `MarketIntelligence(fred_api_key="your_key")`

### Verification

```python
from timeseries_toolkit.intelligence import MarketIntelligence
mi = MarketIntelligence()
report = mi.analyze(["SPY"], horizon="7D")
print(f"Quality: {report.quality_score:.2f}")
# Should print a score between 0 and 1
```

---

## 3. Basic Usage

### Single Asset Analysis

```python
mi = MarketIntelligence()

# Full analysis with all features
report = mi.analyze(
    assets=["BTC-USD"],
    horizon="7D",          # Forecast 7 days ahead
    include_drivers=True,  # Analyse causal relationships
    include_regime=True,   # Detect market regime
    confidence_level=0.95, # 95% confidence bands
    verbose=True           # Print progress
)

print(report.summary)
print(report.forecast)
```

### Multi-Asset Comparison

```python
comparison = mi.compare_assets(["BTC-USD", "ETH-USD"], horizon="7D")

# Correlation matrix
print(comparison.correlations)

# Relative strength
print(comparison.relative_strength)

# Individual reports
for asset, r in comparison.individual_reports.items():
    print(f"{asset}: regime={r.regime.current_regime}, quality={r.quality_score:.2f}")
```

### Quick Forecast (without full analysis)

```python
# When you just need numbers, no regime or driver analysis
fc = mi.quick_forecast("SPY", horizon="7D")
print(fc)  # DataFrame with forecast, lower, upper columns
```

Use `quick_forecast` when speed matters more than context.

### Regime Detection Only

```python
regime = mi.get_regime(["SPY"])
print(f"Regime: {regime.current_regime} ({regime.confidence:.0%})")
print(f"Days in regime: {regime.days_in_regime}")
print(f"Transition risk: {regime.transition_risk:.0%}")
```

---

## 4. Understanding the Output

### The IntelligenceReport Object

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | DataFrame | Columns: `forecast`, `lower`, `upper` |
| `regime` | RegimeResult | Current regime with confidence |
| `drivers` | CausalityResult | Causal relationships (if requested) |
| `quality_score` | float | 0-1 overall quality rating |
| `pipeline_used` | str | Name of the selected pipeline |
| `pipeline_reason` | str | Why this pipeline was chosen |
| `summary` | str | Human-readable text summary |
| `warnings` | list | Automated warnings |
| `recommendations` | list | Actionable suggestions |

### Reading the Summary

The summary follows a fixed structure:

```
MARKET REGIME: Bull (91% confidence)

SPY 7D Outlook: $450.00 → $455.50 (+1.2%)
Confidence: HIGH

Key Factors:
  - Regime transition risk elevated (37%)

Quality: HIGH (4/4 diagnostics pass)
```

### Interpreting Confidence Bands

- **Narrow bands** → model is confident; low recent volatility.
- **Wide bands** → high uncertainty; volatile or crisis regime.
- **95% confidence** means: in 95 out of 100 similar situations, the actual
  value should fall within these bands.
- Bands widen over longer horizons as the state-space forecast covariance
  grows with the prediction horizon. The system uses statsmodels' native
  forecast intervals, which account for state estimation uncertainty,
  observation noise, and horizon growth.

### Quality Score Breakdown

| Score | Label | Meaning |
|-------|-------|---------|
| 0.80 - 1.00 | HIGH | All or most diagnostics pass; reliable |
| 0.60 - 0.79 | MEDIUM | Some concerns; use with moderate confidence |
| 0.40 - 0.59 | LOW | Multiple diagnostic failures; interpret cautiously |
| < 0.40 | UNRELIABLE | Significant issues; do not rely on forecast |

---

## 5. Advanced Configuration

### Custom Horizons

```python
# Supported horizons
report_1d  = mi.analyze(["SPY"], horizon="1D")   # 1 day
report_7d  = mi.analyze(["SPY"], horizon="7D")   # 1 week
report_14d = mi.analyze(["SPY"], horizon="14D")  # 2 weeks
report_30d = mi.analyze(["SPY"], horizon="30D")  # 1 month
```

### Adjusting Confidence Levels

```python
# Wider bands (99% confidence)
report = mi.analyze(["SPY"], confidence_level=0.99)

# Narrower bands (90% confidence)
report = mi.analyze(["SPY"], confidence_level=0.90)
```

### Disabling Features

```python
# Skip regime detection (faster)
report = mi.analyze(["SPY"], include_regime=False)

# Skip driver analysis
report = mi.analyze(["SPY", "QQQ"], include_drivers=False)
```

### Export Options

```python
# Dictionary
d = report.to_dict()

# DataFrame (forecast only)
df = report.to_dataframe()

# Markdown string
md = report.to_markdown()

# Save to file
report.save_markdown("reports/analysis.md")
```

---

## 6. Interpreting Results

### Regime Interpretation

| Regime | Characteristics | Typical Duration |
|--------|----------------|------------------|
| **Bull** | Positive returns, low volatility | Weeks to months |
| **Bear** | Negative returns, elevated volatility | Weeks to months |
| **Crisis** | Sharp drawdowns, very high volatility | Days to weeks |
| **Sideways** | Flat returns, low directional bias | Weeks |

### Transition Risk

The **transition risk** is the probability of the regime changing within
14 days, computed from the HMM transition matrix raised to the 14th power.

- **< 20%**: Regime is stable; current conditions likely to persist.
- **20-40%**: Moderate risk; conditions may shift.
- **> 40%**: High risk; be prepared for a regime change.

### Pipeline Selection Logic

The AutoPilot selects pipelines based on:

1. **Crisis regime** → conservative pipeline (robust to outliers)
2. **Non-stationary data** → aggressive pipeline (handles trends)
3. **High autocorrelation** → trend-following pipeline
4. **Default** → conservative (safe baseline)

The `pipeline_reason` field explains the specific logic applied.

---

## 7. Common Use Cases

### Portfolio Risk Monitoring

```python
mi = MarketIntelligence()
regime = mi.get_regime(["SPY"])

if regime.current_regime == "crisis":
    print("ALERT: Crisis regime detected!")
if regime.transition_risk > 0.3:
    print(f"WARNING: {regime.transition_risk:.0%} risk of regime shift")
```

### Comparing Assets for Allocation

```python
comparison = mi.compare_assets(["SPY", "QQQ", "IWM"], horizon="7D")
print(comparison.correlations)
print(comparison.relative_strength)
```

### Detecting Regime Changes Early

```python
# Monitor daily
regime = mi.get_regime(["SPY"])
print(f"Regime: {regime.current_regime}, Risk: {regime.transition_risk:.0%}")
```

### Backtesting

```python
from timeseries_toolkit.intelligence.backtester import Backtester

bt = Backtester()
# Provide pre-fetched price series
result = bt.run_backtest(
    prices=spy_prices,
    start_date="2024-01-01",
    end_date="2025-06-01",
    horizon=7,
    step=14
)
print(result.summary())
```

---

## 8. Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Unable to fetch price data" | Yahoo Finance API failure | Check internet; retry |
| Empty forecast DataFrame | Pipeline failed on the data | Check data quality |
| Quality score = 0 | All diagnostics failed | Data may be too short or noisy |

### Data Quality Issues

- **Minimum data**: 60+ observations recommended for regime detection.
- **Gaps**: The system handles missing days (weekends) automatically.
- **Splits/dividends**: Yahoo Finance data is pre-adjusted.

### Unexpected Results

- **Regime seems wrong**: The HMM classifies based on *return statistics*,
  not absolute price levels. A market that is falling slowly with low
  volatility may be classified as "sideways" rather than "bear".
- **Wide confidence bands**: Expected during crisis/bear regimes.
  This is a feature, not a bug — the system is expressing uncertainty.

---

## 9. API Reference

### MarketIntelligence

```python
class MarketIntelligence:
    def __init__(self, fred_api_key: Optional[str] = None)
    def analyze(self, assets, horizon="7D", include_drivers=True,
                include_regime=True, confidence_level=0.95,
                verbose=False) -> IntelligenceReport
    def quick_forecast(self, asset, horizon="7D") -> pd.DataFrame
    def get_regime(self, assets=None, lookback="1Y") -> RegimeResult
    def compare_assets(self, assets, horizon="7D") -> ComparisonReport
```

### IntelligenceReport

```python
class IntelligenceReport:
    # Fields
    timestamp: datetime
    assets: List[str]
    horizon: str
    forecast: pd.DataFrame      # columns: forecast, lower, upper
    regime: RegimeResult
    drivers: CausalityResult
    quality_score: float         # 0-1
    pipeline_used: str
    pipeline_reason: str
    summary: str
    warnings: List[str]
    recommendations: List[str]

    # Methods
    def to_dict(self) -> dict
    def to_dataframe(self) -> pd.DataFrame
    def to_markdown(self, include_charts=False) -> str
    def save_markdown(self, filepath, include_charts=False) -> None
```

### RegimeResult

```python
@dataclass
class RegimeResult:
    current_regime: str          # "bull", "bear", "crisis", "sideways"
    confidence: float            # 0-1
    regime_probabilities: Dict[str, float]
    transition_matrix: pd.DataFrame
    days_in_regime: int
    transition_risk: float       # probability of change in 14 days
    regime_history: pd.Series
```

### Backtester

```python
class Backtester:
    def __init__(self, fred_api_key=None)
    def run_backtest(self, prices, start_date, end_date,
                     horizon=7, step=7, min_history=120,
                     verbose=True) -> BacktestResult
    def validate_regime_detection(self, prices, start_date,
                                  end_date) -> RegimeBacktestResult
    def validate_pipeline_selection(self, prices, start_date,
                                    end_date, horizon=7,
                                    step=14) -> PipelineBacktestResult
```

---

*Report generated by MarketIntelligence v1.0*
*This is not financial advice. Past performance does not guarantee future results.*
