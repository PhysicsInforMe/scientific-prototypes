# Data Sources Setup

## Task 1: Create .gitignore

Create .gitignore in root with:

```
# Data files - do not commit
data/
*.csv
*.xlsx
*.xls
*.parquet
notebooks_raw/

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

## Task 2: Create Data Sources Module

Create timeseries_toolkit/data_sources/ with connectors to FREE public APIs:

### data_sources/__init__.py
Export all loaders

### data_sources/crypto.py
```python
class CryptoDataLoader:
    """Load crypto data from free APIs."""
    
    def __init__(self, source: str = "yahoo"):
        """
        Args:
            source: "yahoo" (yfinance) or "coingecko" (free tier)
        """
    
    def get_prices(self, symbols: List[str], period: str = "2Y") -> pd.DataFrame:
        """
        Get OHLCV data for crypto assets.
        
        Args:
            symbols: e.g. ["BTC-USD", "ETH-USD"] for Yahoo, ["bitcoin", "ethereum"] for CoinGecko
            period: "1Y", "2Y", "5Y", "max"
        
        Returns:
            DataFrame with datetime index, columns: open, high, low, close, volume per symbol
        """
    
    def get_market_caps(self, symbols: List[str]) -> pd.DataFrame:
        """Get market cap history."""
    
    def get_fear_greed_index(self) -> pd.Series:
        """Get Crypto Fear & Greed Index (alternative.me API)."""
```

### data_sources/equities.py
```python
class EquityDataLoader:
    """Load equity/index data from Yahoo Finance."""
    
    def get_prices(self, symbols: List[str], period: str = "5Y") -> pd.DataFrame:
        """
        Args:
            symbols: e.g. ["SPY", "QQQ", "^VIX", "^GSPC"]
        """
    
    def get_sectors(self, period: str = "2Y") -> pd.DataFrame:
        """Get sector ETFs: XLF, XLK, XLE, XLV, etc."""
```

### data_sources/rates.py
```python
class RatesDataLoader:
    """Load interest rates and fixed income data from FRED."""
    
    def __init__(self):
        """Uses fredapi - free with API key from https://fred.stlouisfed.org/docs/api/api_key.html"""
    
    def get_treasury_yields(self, maturities: List[str] = ["3M", "2Y", "10Y", "30Y"]) -> pd.DataFrame:
        """
        Get US Treasury yields.
        FRED codes: DGS3MO, DGS2, DGS10, DGS30
        """
    
    def get_yield_curve_slope(self) -> pd.Series:
        """10Y - 2Y spread (recession indicator)."""
    
    def get_fed_funds_rate(self) -> pd.Series:
        """FRED: FEDFUNDS"""
    
    def get_libor_sofr(self) -> pd.DataFrame:
        """LIBOR (historical) and SOFR rates."""
```

### data_sources/liquidity.py
```python
class LiquidityDataLoader:
    """Load liquidity and monetary indicators from FRED."""
    
    def get_money_supply(self, measures: List[str] = ["M1", "M2"]) -> pd.DataFrame:
        """FRED: M1SL, M2SL"""
    
    def get_fed_balance_sheet(self) -> pd.Series:
        """FRED: WALCL (total assets)"""
    
    def get_repo_rates(self) -> pd.DataFrame:
        """Repo and reverse repo rates."""
    
    def get_credit_spreads(self) -> pd.DataFrame:
        """
        Investment grade and high yield spreads.
        FRED: BAMLC0A0CM (IG), BAMLH0A0HYM2 (HY)
        """
```

### data_sources/volatility.py
```python
class VolatilityDataLoader:
    """Load volatility indices."""
    
    def get_vix(self) -> pd.Series:
        """Yahoo: ^VIX"""
    
    def get_volatility_indices(self) -> pd.DataFrame:
        """VIX, VXN (Nasdaq), MOVE (bonds), OVX (oil)"""
    
    def get_skew_index(self) -> pd.Series:
        """CBOE SKEW Index."""
```

### data_sources/macro.py
```python
class MacroDataLoader:
    """Load macroeconomic indicators from FRED."""
    
    def get_gdp(self, country: str = "US", frequency: str = "Q") -> pd.Series:
        """FRED: GDP, GDPC1 (real)"""
    
    def get_inflation(self, measure: str = "CPI") -> pd.Series:
        """FRED: CPIAUCSL, PCEPI, CPILFESL (core)"""
    
    def get_employment(self) -> pd.DataFrame:
        """Unemployment rate, NFP, initial claims."""
    
    def get_pmi(self) -> pd.DataFrame:
        """ISM Manufacturing and Services PMI."""
    
    def get_consumer_sentiment(self) -> pd.DataFrame:
        """Michigan Consumer Sentiment, Conference Board."""
```

### data_sources/european.py
```python
class EuropeanDataLoader:
    """Load European data from ECB Statistical Data Warehouse."""
    
    def get_ecb_rates(self) -> pd.DataFrame:
        """Main refinancing rate, deposit facility rate."""
    
    def get_euribor(self, tenors: List[str] = ["3M", "6M", "12M"]) -> pd.DataFrame:
        """Euribor rates."""
    
    def get_euro_area_gdp(self) -> pd.Series:
        """Euro area GDP growth."""
    
    def get_btp_bund_spread(self) -> pd.Series:
        """Italian 10Y - German 10Y spread (via Yahoo Finance)."""
```

## Task 3: Create Unified DataHub

Create timeseries_toolkit/data_sources/hub.py:

```python
class DataHub:
    """
    Unified interface for all data sources.
    
    Example:
        hub = DataHub()
        
        # Quick access
        btc = hub.crypto.get_prices(["BTC-USD"])
        rates = hub.rates.get_treasury_yields()
        vix = hub.volatility.get_vix()
        
        # Or load preset bundles
        risk_data = hub.load_bundle("risk_indicators")
        # Returns: VIX, credit spreads, yield curve, TED spread
        
        crypto_data = hub.load_bundle("crypto_ecosystem")
        # Returns: BTC, ETH, top 10 alts, fear/greed, dominance
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Args:
            fred_api_key: Get free from https://fred.stlouisfed.org/docs/api/api_key.html
                         Can also set FRED_API_KEY environment variable
        """
        self.crypto = CryptoDataLoader()
        self.equities = EquityDataLoader()
        self.rates = RatesDataLoader()
        self.liquidity = LiquidityDataLoader()
        self.volatility = VolatilityDataLoader()
        self.macro = MacroDataLoader()
        self.european = EuropeanDataLoader()
    
    def load_bundle(self, bundle_name: str, period: str = "2Y") -> Dict[str, pd.DataFrame]:
        """
        Load predefined data bundles.
        
        Bundles:
        - "crypto_ecosystem": BTC, ETH, alts, fear/greed
        - "risk_indicators": VIX, credit spreads, yield curve
        - "liquidity_monitor": M2, Fed balance sheet, repo
        - "rates_dashboard": Treasuries, SOFR, credit spreads
        - "macro_us": GDP, CPI, employment, PMI
        - "macro_europe": ECB rates, Euribor, BTP-Bund
        """
    
    def get_aligned_dataset(
        self, 
        series_dict: Dict[str, pd.Series],
        target_freq: str = "D",
        method: str = "ffill"
    ) -> pd.DataFrame:
        """Align multiple series to common frequency."""
```

## Task 4: Requirements Update

Add to requirements.txt:
```
yfinance>=0.2.0
fredapi>=0.5.0
requests>=2.28.0
```

## Task 5: Create Example

Create examples/data_loading_demo.py showing how to use DataHub.

## Task 6: Tests

Create tests/test_data_sources.py with:
- Mock tests (don't hit real APIs in CI)
- Integration tests marked with @pytest.mark.integration for manual runs

## Task 7: Live API Testing and Validation

After implementing all data sources, run LIVE tests on every API:

1. Test each data loader individually:
   - CryptoDataLoader: fetch BTC-USD, ETH-USD for 1Y
   - EquityDataLoader: fetch SPY, ^VIX for 1Y
   - RatesDataLoader: fetch all treasury yields
   - LiquidityDataLoader: fetch M2, Fed balance sheet
   - VolatilityDataLoader: fetch VIX, MOVE
   - MacroDataLoader: fetch GDP, CPI
   - EuropeanDataLoader: fetch ECB rates, BTP-Bund spread

2. For each API call verify:
   - No exceptions raised
   - Data is not empty
   - No NaN-only columns
   - Dates are properly parsed
   - Data types are correct (float for prices, datetime for index)

3. Test DataHub bundles:
   - load_bundle("crypto_ecosystem")
   - load_bundle("risk_indicators")
   - load_bundle("liquidity_monitor")
   - load_bundle("rates_dashboard")
   - load_bundle("macro_us")
   - load_bundle("macro_europe")

4. Fix ALL errors encountered during live testing

5. Fix ALL warnings - suppress only if truly unavoidable, otherwise fix the root cause

6. Create examples/data_sources_demo.py that demonstrates all working loaders

7. Run final test and confirm:
   - All API calls succeed
   - Zero errors
   - Zero warnings (or minimal unavoidable ones documented)
   - Print summary table showing each data source status

Run all tasks in order, do not skip Task 7.
