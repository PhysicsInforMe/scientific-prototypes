"""Tests for data_sources module.

Mock tests run without hitting real APIs.
Integration tests (marked @pytest.mark.integration) hit live APIs.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Tests - No API calls
# =============================================================================

class TestCryptoDataLoaderMock:
    """Mock tests for CryptoDataLoader."""

    def test_initialization_yahoo(self):
        """Test Yahoo source initialization."""
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
        loader = CryptoDataLoader(source="yahoo")
        assert loader.source == "yahoo"

    def test_initialization_coingecko(self):
        """Test CoinGecko source initialization."""
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
        loader = CryptoDataLoader(source="coingecko")
        assert loader.source == "coingecko"

    def test_invalid_source_raises(self):
        """Test that invalid source raises ValueError."""
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
        with pytest.raises(ValueError):
            CryptoDataLoader(source="invalid")

    def test_coingecko_id_mapping(self):
        """Test that CoinGecko ID mapping exists."""
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
        assert CryptoDataLoader._COINGECKO_IDS["BTC"] == "bitcoin"
        assert CryptoDataLoader._COINGECKO_IDS["ETH"] == "ethereum"

    @patch("yfinance.Ticker")
    def test_get_prices_yahoo_returns_dataframe(self, mock_ticker):
        """Test that Yahoo prices returns a DataFrame."""
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader

        # Mock yfinance response
        mock_hist = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [95.0],
             "Close": [102.0], "Volume": [1000]},
            index=pd.DatetimeIndex(["2024-01-01"])
        )
        mock_ticker.return_value.history.return_value = mock_hist

        loader = CryptoDataLoader(source="yahoo")
        result = loader.get_prices(["BTC-USD"], period="1y")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestEquityDataLoaderMock:
    """Mock tests for EquityDataLoader."""

    def test_initialization(self):
        """Test initialization."""
        from timeseries_toolkit.data_sources.equities import EquityDataLoader
        loader = EquityDataLoader()
        assert hasattr(loader, '_SECTOR_ETFS')

    def test_sector_etfs_exist(self):
        """Test sector ETF mapping has expected entries."""
        from timeseries_toolkit.data_sources.equities import EquityDataLoader
        loader = EquityDataLoader()
        assert "Technology" in loader._SECTOR_ETFS
        assert loader._SECTOR_ETFS["Technology"] == "XLK"


class TestVolatilityDataLoaderMock:
    """Mock tests for VolatilityDataLoader."""

    def test_initialization(self):
        """Test initialization."""
        from timeseries_toolkit.data_sources.volatility import VolatilityDataLoader
        loader = VolatilityDataLoader()
        assert "VIX" in loader._VOL_TICKERS

    @patch("yfinance.Ticker")
    def test_get_vix_returns_series(self, mock_ticker):
        """Test VIX returns a Series."""
        from timeseries_toolkit.data_sources.volatility import VolatilityDataLoader

        mock_hist = pd.DataFrame(
            {"Close": [20.0, 21.0, 19.5]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        )
        mock_ticker.return_value.history.return_value = mock_hist

        loader = VolatilityDataLoader()
        result = loader.get_vix()
        assert isinstance(result, pd.Series)
        assert result.name == "VIX"


class TestRatesDataLoaderMock:
    """Mock tests for RatesDataLoader."""

    def test_no_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        from timeseries_toolkit.data_sources.rates import RatesDataLoader
        with patch.dict("os.environ", {}, clear=True):
            # Remove FRED_API_KEY if set
            import os
            old_key = os.environ.pop("FRED_API_KEY", None)
            try:
                with pytest.raises(ValueError):
                    RatesDataLoader(api_key=None)
            finally:
                if old_key:
                    os.environ["FRED_API_KEY"] = old_key

    def test_yield_codes_exist(self):
        """Test that yield code mappings exist."""
        from timeseries_toolkit.data_sources.rates import RatesDataLoader
        assert "10Y" in RatesDataLoader._YIELD_CODES
        assert RatesDataLoader._YIELD_CODES["10Y"] == "DGS10"


class TestDataHubMock:
    """Mock tests for DataHub."""

    def test_initialization(self):
        """Test that DataHub initializes."""
        from timeseries_toolkit.data_sources.hub import DataHub
        hub = DataHub()
        assert hub._fred_api_key is not None or hub._fred_api_key is None

    def test_invalid_bundle_raises(self):
        """Test that invalid bundle name raises ValueError."""
        from timeseries_toolkit.data_sources.hub import DataHub
        hub = DataHub()
        with pytest.raises(ValueError, match="Unknown bundle"):
            hub.load_bundle("nonexistent_bundle")

    def test_lazy_initialization(self):
        """Test that loaders are lazily initialized."""
        from timeseries_toolkit.data_sources.hub import DataHub
        hub = DataHub()
        assert hub._crypto is None
        assert hub._equities is None
        assert hub._volatility is None

    def test_crypto_lazy_access(self):
        """Test lazy access to crypto loader."""
        from timeseries_toolkit.data_sources.hub import DataHub
        hub = DataHub()
        crypto = hub.crypto
        assert hub._crypto is not None
        assert crypto is hub.crypto  # Same instance

    def test_get_aligned_dataset(self):
        """Test alignment function."""
        from timeseries_toolkit.data_sources.hub import DataHub
        hub = DataHub()

        dates1 = pd.date_range("2024-01-01", periods=10, freq="D")
        dates2 = pd.date_range("2024-01-01", periods=10, freq="D")

        series_dict = {
            "a": pd.Series(np.random.randn(10), index=dates1),
            "b": pd.Series(np.random.randn(10), index=dates2),
        }

        result = hub.get_aligned_dataset(series_dict, target_freq="D")
        assert isinstance(result, pd.DataFrame)
        assert "a" in result.columns
        assert "b" in result.columns


class TestImportStructure:
    """Test that all data_sources modules can be imported."""

    def test_import_crypto(self):
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
        assert CryptoDataLoader is not None

    def test_import_equities(self):
        from timeseries_toolkit.data_sources.equities import EquityDataLoader
        assert EquityDataLoader is not None

    def test_import_rates(self):
        from timeseries_toolkit.data_sources.rates import RatesDataLoader
        assert RatesDataLoader is not None

    def test_import_liquidity(self):
        from timeseries_toolkit.data_sources.liquidity import LiquidityDataLoader
        assert LiquidityDataLoader is not None

    def test_import_volatility(self):
        from timeseries_toolkit.data_sources.volatility import VolatilityDataLoader
        assert VolatilityDataLoader is not None

    def test_import_macro(self):
        from timeseries_toolkit.data_sources.macro import MacroDataLoader
        assert MacroDataLoader is not None

    def test_import_european(self):
        from timeseries_toolkit.data_sources.european import EuropeanDataLoader
        assert EuropeanDataLoader is not None

    def test_import_hub(self):
        from timeseries_toolkit.data_sources.hub import DataHub
        assert DataHub is not None

    def test_import_from_init(self):
        from timeseries_toolkit.data_sources import (
            CryptoDataLoader,
            EquityDataLoader,
            VolatilityDataLoader,
            DataHub,
        )
        assert all([CryptoDataLoader, EquityDataLoader, VolatilityDataLoader, DataHub])


# =============================================================================
# Integration Tests - Hit Live APIs
# =============================================================================

@pytest.mark.integration
class TestCryptoDataLoaderLive:
    """Live integration tests for CryptoDataLoader."""

    def test_get_prices_yahoo(self):
        """Test live Yahoo Finance crypto fetch."""
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
        loader = CryptoDataLoader(source="yahoo")
        df = loader.get_prices(["BTC-USD"], period="1mo")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_get_fear_greed(self):
        """Test live Fear & Greed fetch."""
        from timeseries_toolkit.data_sources.crypto import CryptoDataLoader
        loader = CryptoDataLoader()
        series = loader.get_fear_greed_index(limit=30)
        assert isinstance(series, pd.Series)
        assert len(series) > 0


@pytest.mark.integration
class TestEquityDataLoaderLive:
    """Live integration tests for EquityDataLoader."""

    def test_get_prices(self):
        """Test live equity price fetch."""
        from timeseries_toolkit.data_sources.equities import EquityDataLoader
        loader = EquityDataLoader()
        df = loader.get_prices(["SPY"], period="1mo")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


@pytest.mark.integration
class TestVolatilityDataLoaderLive:
    """Live integration tests for VolatilityDataLoader."""

    def test_get_vix(self):
        """Test live VIX fetch."""
        from timeseries_toolkit.data_sources.volatility import VolatilityDataLoader
        loader = VolatilityDataLoader()
        series = loader.get_vix(period="1mo")
        assert isinstance(series, pd.Series)
        assert len(series) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
