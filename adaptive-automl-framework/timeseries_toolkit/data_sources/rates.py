"""
Interest Rates Data Loader.

Load interest rates and fixed income data from FRED.
"""

import os
import warnings
from typing import Dict, List, Optional

import pandas as pd


class RatesDataLoader:
    """
    Load interest rates and fixed income data from FRED.

    Requires a free FRED API key from:
    https://fred.stlouisfed.org/docs/api/api_key.html

    Attributes:
        fred: fredapi.Fred client instance.

    Example:
        >>> loader = RatesDataLoader()  # uses FRED_API_KEY env var
        >>> yields = loader.get_treasury_yields()
    """

    # FRED series codes for treasury yields
    _YIELD_CODES = {
        "1M": "DGS1MO",
        "3M": "DGS3MO",
        "6M": "DGS6MO",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "3Y": "DGS3",
        "5Y": "DGS5",
        "7Y": "DGS7",
        "10Y": "DGS10",
        "20Y": "DGS20",
        "30Y": "DGS30",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RatesDataLoader.

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
                "https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "Set via FRED_API_KEY env var or pass api_key parameter."
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

    def get_treasury_yields(
        self,
        maturities: Optional[List[str]] = None,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get US Treasury yields.

        Args:
            maturities: List of maturities (e.g., ["3M", "2Y", "10Y", "30Y"]).
                Default: ["3M", "2Y", "10Y", "30Y"].
            start_date: Start date string (e.g., "2020-01-01").

        Returns:
            DataFrame with yield series for each maturity.
        """
        if maturities is None:
            maturities = ["3M", "2Y", "10Y", "30Y"]

        frames = {}
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        for mat in maturities:
            code = self._YIELD_CODES.get(mat)
            if code:
                series = self._get_series(code, **kwargs)
                if not series.empty:
                    frames[mat] = series

        return pd.DataFrame(frames).sort_index()

    def get_yield_curve_slope(
        self, start_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get 10Y-2Y Treasury spread (recession indicator).

        Args:
            start_date: Start date string.

        Returns:
            Series with yield curve slope values.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        # Use FRED's precomputed spread
        series = self._get_series("T10Y2Y", **kwargs)
        if not series.empty:
            series.name = "yield_curve_slope_10Y_2Y"
            return series

        # Fallback: compute manually
        y10 = self._get_series("DGS10", **kwargs)
        y2 = self._get_series("DGS2", **kwargs)
        if not y10.empty and not y2.empty:
            spread = y10 - y2
            spread.name = "yield_curve_slope_10Y_2Y"
            return spread.dropna()

        return pd.Series(dtype=float, name="yield_curve_slope_10Y_2Y")

    def get_fed_funds_rate(
        self, start_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get effective Federal Funds Rate.

        Args:
            start_date: Start date string.

        Returns:
            Series with Fed Funds rate.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        series = self._get_series("FEDFUNDS", **kwargs)
        series.name = "fed_funds_rate"
        return series

    def get_libor_sofr(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get SOFR and historical LIBOR rates.

        Args:
            start_date: Start date string.

        Returns:
            DataFrame with SOFR and LIBOR columns.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        codes = {
            "SOFR": "SOFR",
            "LIBOR_3M": "USD3MTD156N",
        }

        frames = {}
        for name, code in codes.items():
            series = self._get_series(code, **kwargs)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()
