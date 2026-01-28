"""
Time Series Toolkit - A comprehensive library for time series analysis.

This package provides tools for:
- Preprocessing: Fractional differentiation, filtering, imputation
- Models: Kalman filters, regime detection, forecasting
- Validation: Causality testing, diagnostics
"""

__version__ = "0.1.0"

# Import submodules - these handle their own optional dependency errors
from timeseries_toolkit import preprocessing
from timeseries_toolkit import models
from timeseries_toolkit import validation
from timeseries_toolkit import utils

__all__ = [
    "preprocessing",
    "models",
    "validation",
    "utils",
]
