# Scientific Validation Tests

Create tests/test_scientific_validity.py with these validations:

## fractional_diff.py
- Weights must sum to 1
- d=0 returns original series
- d=1 returns first difference
- ADF test correctly identifies stationarity

## filtering.py
- trend + seasonal + residual = original
- Residuals have lower variance than original

## imputation.py
- No NaN in output after imputation
- Quarterly alignment preserves data integrity

## kalman.py
- Smoothed series has lower variance than original
- Compare with AR(1) analytical solution

## regime.py
- Detected regimes match true regimes >80% accuracy on synthetic data
- Regime probabilities sum to 1

## causality.py
- CCM detects causality on coupled logistic maps
- Granger detects X->Y but not Y->X when X causes Y

## diagnostics.py
- White noise passes Ljung-Box
- Non-white-noise fails Ljung-Box
- Hurst exponent ≈ 0.5 for random walk

Run all validations and fix any issues found.
