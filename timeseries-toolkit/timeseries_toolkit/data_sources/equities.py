"""
Equity Data Loader.

Load equity and index data from Yahoo Finance.
"""

from typing import List

import pandas as pd


class EquityDataLoader:
    """
    Load equity/index data from Yahoo Finance.

    Attributes:
        _sector_etfs: Mapping of sector names to ETF tickers.

    Example:
        >>> loader = EquityDataLoader()
        >>> spy = loader.get_prices(["SPY"], period="1y")
    """

    _SECTOR_ETFS = {
        "Financials": "XLF",
        "Technology": "XLK",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication": "XLC",
    }

    def __init__(self):
        """
        Initialize the EquityDataLoader.

        Raises:
            ImportError: If yfinance is not installed.
        """
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )

    def get_prices(
        self,
        symbols: List[str],
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get OHLCV data for equities and indices.

        Args:
            symbols: Ticker symbols (e.g., ["SPY", "QQQ", "^VIX", "^GSPC"]).
            period: Time period ("1y", "2y", "5y", "10y", "max").
            interval: Data interval ("1d", "1wk", "1mo").

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns.
        """
        import yfinance as yf

        if len(symbols) == 1:
            ticker = yf.Ticker(symbols[0])
            df = ticker.history(period=period, interval=interval)
            df.columns = [f"{symbols[0]}_{c}" for c in df.columns]
        else:
            data = yf.download(
                symbols, period=period, interval=interval,
                progress=False, auto_adjust=True
            )
            if isinstance(data.columns, pd.MultiIndex):
                df = data.stack(level=0, future_stack=True).unstack(level=-1)
                df.columns = [f"{sym}_{col}" for col, sym in df.columns]
            else:
                df = data

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()

    def get_sectors(self, period: str = "2y") -> pd.DataFrame:
        """
        Get sector ETF close prices.

        Fetches: XLF, XLK, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC.

        Args:
            period: Time period ("1y", "2y", "5y").

        Returns:
            DataFrame with close prices for each sector ETF.
        """
        import yfinance as yf

        tickers = list(self._SECTOR_ETFS.values())
        data = yf.download(
            tickers, period=period, progress=False, auto_adjust=True
        )

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
        else:
            close = data

        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)

        # Rename columns to sector names
        inv_map = {v: k for k, v in self._SECTOR_ETFS.items()}
        close = close.rename(columns=inv_map)
        return close.sort_index()
