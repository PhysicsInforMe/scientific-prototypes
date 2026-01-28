"""
Data Sources Module.

Provides connectors to free public APIs for financial and economic data.

Loaders:
    - CryptoDataLoader: Crypto prices via Yahoo Finance / CoinGecko
    - EquityDataLoader: Equity/index data via Yahoo Finance
    - RatesDataLoader: Interest rates via FRED
    - LiquidityDataLoader: Monetary indicators via FRED
    - VolatilityDataLoader: Volatility indices via Yahoo Finance
    - MacroDataLoader: Macroeconomic indicators via FRED
    - EuropeanDataLoader: European data via ECB / Yahoo Finance
    - DataHub: Unified interface for all sources
"""

from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
from timeseries_toolkit.data_sources.equities import EquityDataLoader
from timeseries_toolkit.data_sources.rates import RatesDataLoader
from timeseries_toolkit.data_sources.liquidity import LiquidityDataLoader
from timeseries_toolkit.data_sources.volatility import VolatilityDataLoader
from timeseries_toolkit.data_sources.macro import MacroDataLoader
from timeseries_toolkit.data_sources.european import EuropeanDataLoader
from timeseries_toolkit.data_sources.hub import DataHub

__all__ = [
    'CryptoDataLoader',
    'EquityDataLoader',
    'RatesDataLoader',
    'LiquidityDataLoader',
    'VolatilityDataLoader',
    'MacroDataLoader',
    'EuropeanDataLoader',
    'DataHub',
]
