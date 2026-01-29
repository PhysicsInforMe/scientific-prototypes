"""
European Data Loader.

Load European financial and economic data from Yahoo Finance and FRED.
"""

import os
import warnings
from typing import Dict, List, Optional

import pandas as pd
import requests


class EuropeanDataLoader:
    """
    Load European data from Yahoo Finance and FRED.

    ECB rates and Euribor are fetched from FRED (which mirrors ECB data).
    Bond data uses Yahoo Finance.

    Example:
        >>> loader = EuropeanDataLoader()
        >>> spread = loader.get_btp_bund_spread()
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the EuropeanDataLoader.

        Args:
            fred_api_key: FRED API key for ECB/Euribor data.
                If None, reads from FRED_API_KEY env var.
                FRED-dependent methods will raise if no key is available.
        """
        self._fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")
        self._fred = None

        try:
            import yfinance  # noqa: F401
            self._has_yfinance = True
        except ImportError:
            self._has_yfinance = False

    def _get_fred(self):
        """Lazily initialize FRED client."""
        if self._fred is None:
            if not self._fred_api_key:
                raise ValueError(
                    "FRED API key required for ECB/Euribor data. "
                    "Get one free at: https://fred.stlouisfed.org/docs/api/api_key.html"
                )
            from fredapi import Fred
            self._fred = Fred(api_key=self._fred_api_key)
        return self._fred

    def _get_fred_series(self, series_id: str, **kwargs) -> pd.Series:
        """Fetch a single FRED series with error handling."""
        try:
            fred = self._get_fred()
            data = fred.get_series(series_id, **kwargs)
            data.name = series_id
            return data.dropna()
        except Exception as e:
            warnings.warn(f"Failed to fetch {series_id}: {e}")
            return pd.Series(dtype=float, name=series_id)

    def get_ecb_rates(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get ECB policy rates.

        Includes main refinancing rate and deposit facility rate.

        Args:
            start_date: Start date string.

        Returns:
            DataFrame with ECB rate columns.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        codes = {
            "Main_Refinancing_Rate": "ECBMRRFR",
            "Deposit_Facility_Rate": "ECBDFR",
        }

        frames = {}
        for name, code in codes.items():
            series = self._get_fred_series(code, **kwargs)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()

    def get_euribor(
        self,
        tenors: Optional[List[str]] = None,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get Euribor rates.

        Args:
            tenors: List of tenors (e.g., ["3M", "6M", "12M"]).
                Default: ["3M", "6M", "12M"].
            start_date: Start date string.

        Returns:
            DataFrame with Euribor rate columns.
        """
        if tenors is None:
            tenors = ["3M", "6M", "12M"]

        codes = {
            "1M": "IR3TIB01EZM156N",
            "3M": "IR3TIB01EZM156N",
            "6M": "EURONTD156N",
            "12M": "EURONTD156N",
        }

        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        frames = {}
        for tenor in tenors:
            code = codes.get(tenor)
            if code:
                series = self._get_fred_series(code, **kwargs)
                if not series.empty:
                    frames[f"Euribor_{tenor}"] = series

        return pd.DataFrame(frames).sort_index()

    def get_euro_area_gdp(
        self, start_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get Euro Area GDP growth.

        Args:
            start_date: Start date string.

        Returns:
            Series with Euro Area GDP values.
        """
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date

        series = self._get_fred_series("CLVMNACSCAB1GQEA19", **kwargs)
        series.name = "euro_area_gdp"
        return series

    def get_btp_bund_spread(self, period: str = "2y") -> pd.Series:
        """
        Get BTP-Bund spread (Italian 10Y - German 10Y yield).

        Uses Yahoo Finance bond ETFs or FRED data as proxy.

        Args:
            period: Time period ("1y", "2y", "5y").

        Returns:
            Series with BTP-Bund spread in basis points.
        """
        # Try FRED first (Italian 10Y - German 10Y)
        try:
            fred = self._get_fred()
            italy_10y = self._get_fred_series("IRLTLT01ITM156N")
            germany_10y = self._get_fred_series("IRLTLT01DEM156N")

            if not italy_10y.empty and not germany_10y.empty:
                spread = (italy_10y - germany_10y).dropna()
                spread.name = "btp_bund_spread"
                return spread
        except (ValueError, Exception):
            pass

        # Fallback: use Yahoo Finance
        if self._has_yfinance:
            import yfinance as yf
            try:
                # Use 10Y bond yield tickers
                italy = yf.Ticker("IT10Y.MI")
                germany = yf.Ticker("DE10Y.MI")

                italy_hist = italy.history(period=period)
                germany_hist = germany.history(period=period)

                if not italy_hist.empty and not germany_hist.empty:
                    spread = italy_hist["Close"] - germany_hist["Close"]
                    spread.name = "btp_bund_spread"
                    if spread.index.tz is not None:
                        spread.index = spread.index.tz_localize(None)
                    return spread.dropna()
            except Exception:
                pass

        warnings.warn("Could not fetch BTP-Bund spread from any source")
        return pd.Series(dtype=float, name="btp_bund_spread")
