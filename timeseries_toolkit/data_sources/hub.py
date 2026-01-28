"""
DataHub - Unified Data Access.

Provides a single interface for accessing all data sources
with preset bundles for common use cases.
"""

import os
import warnings
from typing import Dict, List, Optional

import pandas as pd

from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
from timeseries_toolkit.data_sources.equities import EquityDataLoader
from timeseries_toolkit.data_sources.volatility import VolatilityDataLoader


class DataHub:
    """
    Unified interface for all data sources.

    Provides lazy initialization of loaders (only created when accessed)
    and preset bundles for common data combinations.

    Attributes:
        crypto: CryptoDataLoader instance.
        equities: EquityDataLoader instance.
        rates: RatesDataLoader instance (requires FRED key).
        liquidity: LiquidityDataLoader instance (requires FRED key).
        volatility: VolatilityDataLoader instance.
        macro: MacroDataLoader instance (requires FRED key).
        european: EuropeanDataLoader instance.

    Example:
        >>> hub = DataHub()
        >>> btc = hub.crypto.get_prices(["BTC-USD"])
        >>> vix = hub.volatility.get_vix()
        >>> risk_data = hub.load_bundle("risk_indicators")
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the DataHub.

        Args:
            fred_api_key: FRED API key. If None, reads from FRED_API_KEY
                env var. Required for rates, liquidity, macro, and some
                European data. Get one free at:
                https://fred.stlouisfed.org/docs/api/api_key.html
        """
        self._fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")

        # Lazy-initialized loaders
        self._crypto = None
        self._equities = None
        self._rates = None
        self._liquidity = None
        self._volatility = None
        self._macro = None
        self._european = None

    @property
    def crypto(self) -> CryptoDataLoader:
        """Access the CryptoDataLoader."""
        if self._crypto is None:
            self._crypto = CryptoDataLoader()
        return self._crypto

    @property
    def equities(self) -> EquityDataLoader:
        """Access the EquityDataLoader."""
        if self._equities is None:
            self._equities = EquityDataLoader()
        return self._equities

    @property
    def rates(self):
        """Access the RatesDataLoader."""
        if self._rates is None:
            from timeseries_toolkit.data_sources.rates import RatesDataLoader
            self._rates = RatesDataLoader(api_key=self._fred_api_key)
        return self._rates

    @property
    def liquidity(self):
        """Access the LiquidityDataLoader."""
        if self._liquidity is None:
            from timeseries_toolkit.data_sources.liquidity import LiquidityDataLoader
            self._liquidity = LiquidityDataLoader(api_key=self._fred_api_key)
        return self._liquidity

    @property
    def volatility(self) -> VolatilityDataLoader:
        """Access the VolatilityDataLoader."""
        if self._volatility is None:
            self._volatility = VolatilityDataLoader()
        return self._volatility

    @property
    def macro(self):
        """Access the MacroDataLoader."""
        if self._macro is None:
            from timeseries_toolkit.data_sources.macro import MacroDataLoader
            self._macro = MacroDataLoader(api_key=self._fred_api_key)
        return self._macro

    @property
    def european(self):
        """Access the EuropeanDataLoader."""
        if self._european is None:
            from timeseries_toolkit.data_sources.european import EuropeanDataLoader
            self._european = EuropeanDataLoader(fred_api_key=self._fred_api_key)
        return self._european

    def load_bundle(
        self,
        bundle_name: str,
        period: str = "2y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load predefined data bundles.

        Available bundles:
            - "crypto_ecosystem": BTC, ETH prices + Fear & Greed index
            - "risk_indicators": VIX, credit spreads, yield curve slope
            - "liquidity_monitor": M2, Fed balance sheet, repo rates
            - "rates_dashboard": Treasury yields, SOFR, credit spreads
            - "macro_us": GDP, CPI, employment, consumer sentiment
            - "macro_europe": ECB rates, Euribor, Euro Area GDP

        Args:
            bundle_name: Name of the data bundle to load.
            period: Time period for price data ("1y", "2y", "5y").

        Returns:
            Dictionary mapping data names to DataFrames/Series.

        Raises:
            ValueError: If bundle_name is not recognized.
        """
        loaders = {
            "crypto_ecosystem": self._bundle_crypto_ecosystem,
            "risk_indicators": self._bundle_risk_indicators,
            "liquidity_monitor": self._bundle_liquidity_monitor,
            "rates_dashboard": self._bundle_rates_dashboard,
            "macro_us": self._bundle_macro_us,
            "macro_europe": self._bundle_macro_europe,
        }

        loader_func = loaders.get(bundle_name)
        if loader_func is None:
            available = ", ".join(sorted(loaders.keys()))
            raise ValueError(
                f"Unknown bundle: '{bundle_name}'. Available: {available}"
            )

        return loader_func(period)

    def _bundle_crypto_ecosystem(self, period: str) -> Dict[str, pd.DataFrame]:
        """Load crypto ecosystem bundle."""
        result = {}

        try:
            result["prices"] = self.crypto.get_prices(
                ["BTC-USD", "ETH-USD"], period=period
            )
        except Exception as e:
            warnings.warn(f"Failed to load crypto prices: {e}")

        try:
            result["fear_greed"] = self.crypto.get_fear_greed_index()
        except Exception as e:
            warnings.warn(f"Failed to load Fear & Greed: {e}")

        return result

    def _bundle_risk_indicators(self, period: str) -> Dict[str, pd.DataFrame]:
        """Load risk indicators bundle."""
        result = {}

        try:
            result["volatility"] = self.volatility.get_volatility_indices(period)
        except Exception as e:
            warnings.warn(f"Failed to load volatility: {e}")

        try:
            result["credit_spreads"] = self.liquidity.get_credit_spreads()
        except Exception as e:
            warnings.warn(f"Failed to load credit spreads: {e}")

        try:
            result["yield_curve"] = self.rates.get_yield_curve_slope()
        except Exception as e:
            warnings.warn(f"Failed to load yield curve: {e}")

        return result

    def _bundle_liquidity_monitor(self, period: str) -> Dict[str, pd.DataFrame]:
        """Load liquidity monitor bundle."""
        result = {}

        try:
            result["money_supply"] = self.liquidity.get_money_supply()
        except Exception as e:
            warnings.warn(f"Failed to load money supply: {e}")

        try:
            result["fed_balance_sheet"] = self.liquidity.get_fed_balance_sheet()
        except Exception as e:
            warnings.warn(f"Failed to load Fed balance sheet: {e}")

        try:
            result["repo_rates"] = self.liquidity.get_repo_rates()
        except Exception as e:
            warnings.warn(f"Failed to load repo rates: {e}")

        return result

    def _bundle_rates_dashboard(self, period: str) -> Dict[str, pd.DataFrame]:
        """Load rates dashboard bundle."""
        result = {}

        try:
            result["treasury_yields"] = self.rates.get_treasury_yields()
        except Exception as e:
            warnings.warn(f"Failed to load treasury yields: {e}")

        try:
            result["sofr_libor"] = self.rates.get_libor_sofr()
        except Exception as e:
            warnings.warn(f"Failed to load SOFR/LIBOR: {e}")

        try:
            result["credit_spreads"] = self.liquidity.get_credit_spreads()
        except Exception as e:
            warnings.warn(f"Failed to load credit spreads: {e}")

        return result

    def _bundle_macro_us(self, period: str) -> Dict[str, pd.DataFrame]:
        """Load US macro bundle."""
        result = {}

        try:
            result["gdp"] = self.macro.get_gdp()
        except Exception as e:
            warnings.warn(f"Failed to load GDP: {e}")

        try:
            result["inflation"] = self.macro.get_inflation()
        except Exception as e:
            warnings.warn(f"Failed to load inflation: {e}")

        try:
            result["employment"] = self.macro.get_employment()
        except Exception as e:
            warnings.warn(f"Failed to load employment: {e}")

        try:
            result["sentiment"] = self.macro.get_consumer_sentiment()
        except Exception as e:
            warnings.warn(f"Failed to load sentiment: {e}")

        return result

    def _bundle_macro_europe(self, period: str) -> Dict[str, pd.DataFrame]:
        """Load European macro bundle."""
        result = {}

        try:
            result["ecb_rates"] = self.european.get_ecb_rates()
        except Exception as e:
            warnings.warn(f"Failed to load ECB rates: {e}")

        try:
            result["euribor"] = self.european.get_euribor()
        except Exception as e:
            warnings.warn(f"Failed to load Euribor: {e}")

        try:
            result["gdp"] = self.european.get_euro_area_gdp()
        except Exception as e:
            warnings.warn(f"Failed to load Euro GDP: {e}")

        return result

    def get_aligned_dataset(
        self,
        series_dict: Dict[str, pd.Series],
        target_freq: str = "D",
        method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Align multiple series to a common frequency.

        Args:
            series_dict: Dictionary mapping names to pd.Series.
            target_freq: Target frequency ("D", "W", "M", "Q").
            method: Fill method for missing values ("ffill", "bfill", None).

        Returns:
            DataFrame with all series aligned to the target frequency.
        """
        frames = {}
        for name, series in series_dict.items():
            resampled = series.resample(target_freq).last()
            if method == "ffill":
                resampled = resampled.ffill()
            elif method == "bfill":
                resampled = resampled.bfill()
            frames[name] = resampled

        return pd.DataFrame(frames).sort_index()
