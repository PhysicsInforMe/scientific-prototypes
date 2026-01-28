"""
Macroeconomic Data Loader.

Load macroeconomic indicators from FRED.
"""

import os
import warnings
from typing import Dict, List, Optional

import pandas as pd


class MacroDataLoader:
    """
    Load macroeconomic indicators from FRED.

    Requires a free FRED API key from:
    https://fred.stlouisfed.org/docs/api/api_key.html

    Example:
        >>> loader = MacroDataLoader()
        >>> gdp = loader.get_gdp()
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MacroDataLoader.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.

        Raises:
            ImportError: If fredapi is not installed.
            ValueError: If no API key is provided or found.
        """
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError(
                "fredapi is required. Install with: pip install fredapi"
            )

        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise ValueError(
                "FRED API key required. Get one free at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.fred = Fred(api_key=key)

    def _get_series(self, series_id: str, **kwargs) -> pd.Series:
        """Fetch a single FRED series with error handling."""
        try:
            data = self.fred.get_series(series_id, **kwargs)
            data.name = series_id
            return data.dropna()
        except Exception as e:
            warnings.warn(f"Failed to fetch {series_id}: {e}")
            return pd.Series(dtype=float, name=series_id)

    def get_gdp(
        self,
        real: bool = True,
        start_date: Optional[str] = None,
    ) -> pd.Series:
        """
        Get US GDP.

        Args:
            real: If True, fetch real GDP (GDPC1). If False, nominal GDP.
            start_date: Start date string.

        Returns:
            Series with GDP values (billions of dollars).
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        code = "GDPC1" if real else "GDP"
        series = self._get_series(code, **kwargs)
        series.name = "real_gdp" if real else "nominal_gdp"
        return series

    def get_inflation(
        self,
        measure: str = "CPI",
        start_date: Optional[str] = None,
    ) -> pd.Series:
        """
        Get inflation measure.

        Args:
            measure: "CPI" (CPIAUCSL), "PCE" (PCEPI), or "Core_CPI" (CPILFESL).
            start_date: Start date string.

        Returns:
            Series with inflation index values.
        """
        codes = {
            "CPI": "CPIAUCSL",
            "PCE": "PCEPI",
            "Core_CPI": "CPILFESL",
        }

        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        code = codes.get(measure, "CPIAUCSL")
        series = self._get_series(code, **kwargs)
        series.name = measure.lower()
        return series

    def get_employment(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get employment indicators.

        Includes: Unemployment rate, Nonfarm payrolls, Initial jobless claims.

        Args:
            start_date: Start date string.

        Returns:
            DataFrame with employment indicator columns.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        codes = {
            "Unemployment_Rate": "UNRATE",
            "Nonfarm_Payrolls": "PAYEMS",
            "Initial_Claims": "ICSA",
        }

        frames = {}
        for name, code in codes.items():
            series = self._get_series(code, **kwargs)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()

    def get_pmi(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get ISM PMI indices.

        Includes Manufacturing and Non-Manufacturing (Services) PMI.

        Args:
            start_date: Start date string.

        Returns:
            DataFrame with PMI columns.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        codes = {
            "ISM_Manufacturing": "MANEMP",
            "ISM_NonManufacturing": "NMFBACTIVITYY",
        }

        frames = {}
        for name, code in codes.items():
            series = self._get_series(code, **kwargs)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()

    def get_consumer_sentiment(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get consumer sentiment indices.

        Includes Michigan Consumer Sentiment.

        Args:
            start_date: Start date string.

        Returns:
            DataFrame with sentiment indicator columns.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        codes = {
            "Michigan_Sentiment": "UMCSENT",
        }

        frames = {}
        for name, code in codes.items():
            series = self._get_series(code, **kwargs)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()
