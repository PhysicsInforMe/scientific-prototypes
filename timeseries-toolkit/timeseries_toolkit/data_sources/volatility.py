"""
Volatility Data Loader.

Load volatility indices from Yahoo Finance.
"""

import warnings
from typing import List, Optional

import pandas as pd


class VolatilityDataLoader:
    """
    Load volatility indices from Yahoo Finance.

    Example:
        >>> loader = VolatilityDataLoader()
        >>> vix = loader.get_vix()
    """

    # Known volatility tickers
    _VOL_TICKERS = {
        "VIX": "^VIX",
        "VXN": "^VXN",
        "OVX": "^OVX",
    }

    def __init__(self):
        """
        Initialize the VolatilityDataLoader.

        Raises:
            ImportError: If yfinance is not installed.
        """
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )

    def _fetch_close(self, ticker: str, period: str) -> pd.Series:
        """Fetch close price for a single ticker."""
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            if hist.empty:
                return pd.Series(dtype=float)
            series = hist["Close"]
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            return series
        except Exception as e:
            warnings.warn(f"Failed to fetch {ticker}: {e}")
            return pd.Series(dtype=float)

    def get_vix(self, period: str = "2y") -> pd.Series:
        """
        Get CBOE VIX Index.

        Args:
            period: Time period ("1y", "2y", "5y", "max").

        Returns:
            Series with VIX close values.
        """
        series = self._fetch_close("^VIX", period)
        series.name = "VIX"
        return series

    def get_volatility_indices(self, period: str = "2y") -> pd.DataFrame:
        """
        Get multiple volatility indices.

        Fetches VIX (equities), VXN (Nasdaq), OVX (oil).
        MOVE index is not available on Yahoo Finance.

        Args:
            period: Time period.

        Returns:
            DataFrame with volatility index columns.
        """
        frames = {}
        for name, ticker in self._VOL_TICKERS.items():
            series = self._fetch_close(ticker, period)
            if not series.empty:
                frames[name] = series

        return pd.DataFrame(frames).sort_index()

    def get_skew_index(self, period: str = "2y") -> pd.Series:
        """
        Get CBOE SKEW Index.

        The SKEW Index measures perceived tail risk in the S&P 500 distribution.
        Higher values indicate greater perceived risk of outlier returns.

        Args:
            period: Time period.

        Returns:
            Series with SKEW index values.
        """
        series = self._fetch_close("^SKEW", period)
        series.name = "SKEW"
        return series
