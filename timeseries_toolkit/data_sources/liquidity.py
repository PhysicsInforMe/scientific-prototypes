"""
Liquidity Data Loader.

Load liquidity and monetary indicators from FRED.
"""

import os
import warnings
from typing import Dict, List, Optional

import pandas as pd


class LiquidityDataLoader:
    """
    Load liquidity and monetary indicators from FRED.

    Requires a free FRED API key from:
    https://fred.stlouisfed.org/docs/api/api_key.html

    Example:
        >>> loader = LiquidityDataLoader()
        >>> m2 = loader.get_money_supply(["M2"])
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LiquidityDataLoader.

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

    def get_money_supply(
        self,
        measures: Optional[List[str]] = None,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get money supply measures.

        Args:
            measures: List of measures (e.g., ["M1", "M2"]). Default: ["M1", "M2"].
            start_date: Start date string.

        Returns:
            DataFrame with money supply series.
        """
        if measures is None:
            measures = ["M1", "M2"]

        codes = {"M1": "M1SL", "M2": "M2SL"}
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        frames = {}
        for m in measures:
            code = codes.get(m)
            if code:
                series = self._get_series(code, **kwargs)
                if not series.empty:
                    frames[m] = series

        return pd.DataFrame(frames).sort_index()

    def get_fed_balance_sheet(
        self, start_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get Federal Reserve total assets (balance sheet size).

        FRED series: WALCL

        Args:
            start_date: Start date string.

        Returns:
            Series with Fed total assets (millions of dollars).
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        series = self._get_series("WALCL", **kwargs)
        series.name = "fed_total_assets"
        return series

    def get_repo_rates(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get repo and reverse repo rates.

        Args:
            start_date: Start date string.

        Returns:
            DataFrame with repo rate series.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        codes = {
            "SOFR": "SOFR",
            "ON_RRP_Rate": "RRPONTSYD",
        }

        frames = {}
        for name, code in codes.items():
            series = self._get_series(code, **kwargs)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()

    def get_credit_spreads(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get investment grade and high yield credit spreads.

        FRED codes:
            - BAMLC0A0CM: US Corporate IG Option-Adjusted Spread
            - BAMLH0A0HYM2: US High Yield Option-Adjusted Spread

        Args:
            start_date: Start date string.

        Returns:
            DataFrame with credit spread series.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        codes = {
            "IG_Spread": "BAMLC0A0CM",
            "HY_Spread": "BAMLH0A0HYM2",
        }

        frames = {}
        for name, code in codes.items():
            series = self._get_series(code, **kwargs)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()
