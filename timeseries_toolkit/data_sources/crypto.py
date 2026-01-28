"""
Crypto Data Loader.

Load cryptocurrency market data from free APIs (Yahoo Finance, CoinGecko).
"""

import warnings
from typing import Dict, List, Optional

import pandas as pd
import requests


class CryptoDataLoader:
    """
    Load crypto data from free APIs.

    Supports Yahoo Finance (via yfinance) and CoinGecko free tier.

    Attributes:
        source: Data source backend ("yahoo" or "coingecko").

    Example:
        >>> loader = CryptoDataLoader()
        >>> btc = loader.get_prices(["BTC-USD"], period="1y")
        >>> print(btc.head())
    """

    # CoinGecko ID mapping for common symbols
    _COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "DOT": "polkadot",
        "AVAX": "avalanche-2",
        "LINK": "chainlink",
        "MATIC": "matic-network",
        "UNI": "uniswap",
        "XRP": "ripple",
    }

    def __init__(self, source: str = "yahoo"):
        """
        Initialize the CryptoDataLoader.

        Args:
            source: "yahoo" (yfinance) or "coingecko" (free tier).

        Raises:
            ImportError: If yfinance is not installed and source is "yahoo".
            ValueError: If source is not recognized.
        """
        if source not in ("yahoo", "coingecko"):
            raise ValueError(f"Unknown source: {source}. Use 'yahoo' or 'coingecko'.")

        self.source = source

        if source == "yahoo":
            try:
                import yfinance  # noqa: F401
            except ImportError:
                raise ImportError(
                    "yfinance is required for Yahoo Finance source. "
                    "Install with: pip install yfinance"
                )

    def get_prices(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get OHLCV data for crypto assets.

        Args:
            symbols: Ticker symbols.
                Yahoo format: ["BTC-USD", "ETH-USD"]
                CoinGecko format: ["bitcoin", "ethereum"]
            period: Time period. Yahoo: "1y", "2y", "5y", "max".
                CoinGecko: mapped to days (365, 730, 1825, "max").
            interval: Data interval (Yahoo only). "1d", "1wk", "1mo".

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns per symbol.
        """
        if self.source == "yahoo":
            return self._get_prices_yahoo(symbols, period, interval)
        else:
            return self._get_prices_coingecko(symbols, period)

    def _get_prices_yahoo(
        self, symbols: List[str], period: str, interval: str
    ) -> pd.DataFrame:
        """Fetch prices from Yahoo Finance."""
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

    def _get_prices_coingecko(
        self, symbols: List[str], period: str
    ) -> pd.DataFrame:
        """Fetch prices from CoinGecko free API."""
        period_days = {
            "1y": 365, "2y": 730, "5y": 1825, "max": "max",
            "1Y": 365, "2Y": 730, "5Y": 1825,
        }
        days = period_days.get(period, 365)

        frames = {}
        for symbol in symbols:
            coin_id = self._COINGECKO_IDS.get(symbol.upper(), symbol.lower())
            url = (
                f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                f"/market_chart?vs_currency=usd&days={days}"
            )

            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "prices" in data:
                prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
                prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
                prices = prices.set_index("timestamp")
                frames[symbol] = prices["price"]

        if frames:
            return pd.DataFrame(frames).sort_index()
        return pd.DataFrame()

    def get_market_caps(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get market capitalization history.

        Uses CoinGecko API regardless of source setting.

        Args:
            symbols: Coin IDs (e.g., ["bitcoin", "ethereum"]).

        Returns:
            DataFrame with market cap time series per symbol.
        """
        frames = {}
        for symbol in symbols:
            coin_id = self._COINGECKO_IDS.get(symbol.upper(), symbol.lower())
            url = (
                f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                f"/market_chart?vs_currency=usd&days=365"
            )

            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if "market_caps" in data:
                    mc = pd.DataFrame(
                        data["market_caps"], columns=["timestamp", "market_cap"]
                    )
                    mc["timestamp"] = pd.to_datetime(mc["timestamp"], unit="ms")
                    mc = mc.set_index("timestamp")
                    frames[symbol] = mc["market_cap"]
            except Exception:
                warnings.warn(f"Failed to fetch market cap for {symbol}")

        if frames:
            return pd.DataFrame(frames).sort_index()
        return pd.DataFrame()

    def get_fear_greed_index(self, limit: int = 365) -> pd.Series:
        """
        Get Crypto Fear & Greed Index from alternative.me.

        Args:
            limit: Number of days of data to fetch.

        Returns:
            Series with Fear & Greed values (0-100), DatetimeIndex.
        """
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("data", [])
        if not records:
            return pd.Series(dtype=float, name="fear_greed_index")

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["value"] = df["value"].astype(float)
        series = df.set_index("timestamp")["value"].sort_index()
        series.name = "fear_greed_index"
        return series
