"""Tests for utils module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from timeseries_toolkit.utils.data_loader import (
    load_csv,
    load_excel,
    load_multiple_csv,
    _parse_quarterly_date,
)

# Check if openpyxl is available for Excel tests
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("date,value\n")
        f.write("2020-01-01,100\n")
        f.write("2020-02-01,101\n")
        f.write("2020-03-01,102\n")
        f.write("2020-04-01,103\n")
        f.write("2020-05-01,104\n")
        filepath = f.name

    yield filepath

    os.unlink(filepath)


@pytest.fixture
def quarterly_csv_file():
    """Create a temporary CSV with quarterly dates."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("date,gdp\n")
        f.write("2020Q1,2.5\n")
        f.write("2020Q2,2.3\n")
        f.write("2020Q3,2.4\n")
        f.write("2020Q4,2.6\n")
        filepath = f.name

    yield filepath

    os.unlink(filepath)


@pytest.fixture
def sample_excel_file():
    """Create a temporary Excel file for testing."""
    if not HAS_OPENPYXL:
        pytest.skip("openpyxl not installed")

    filepath = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False).name

    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='ME'),
        'value': [100, 101, 102, 103, 104]
    })
    df.to_excel(filepath, index=False)

    yield filepath

    os.unlink(filepath)


class TestParseQuarterlyDate:
    """Tests for _parse_quarterly_date function."""

    def test_parse_2020Q1(self):
        """Test parsing 2020Q1 format."""
        result = _parse_quarterly_date('2020Q1')
        assert result.year == 2020
        assert result.month == 3
        assert result.day == 31

    def test_parse_2020Q2(self):
        """Test parsing 2020Q2 format."""
        result = _parse_quarterly_date('2020Q2')
        assert result.year == 2020
        assert result.month == 6
        assert result.day == 30

    def test_parse_2020Q3(self):
        """Test parsing 2020Q3 format."""
        result = _parse_quarterly_date('2020Q3')
        assert result.year == 2020
        assert result.month == 9
        assert result.day == 30

    def test_parse_2020Q4(self):
        """Test parsing 2020Q4 format."""
        result = _parse_quarterly_date('2020Q4')
        assert result.year == 2020
        assert result.month == 12
        assert result.day == 31

    def test_parse_with_dash(self):
        """Test parsing 2020-Q1 format."""
        result = _parse_quarterly_date('2020-Q1')
        assert result.year == 2020
        assert result.month == 3

    def test_parse_with_space(self):
        """Test parsing 2020 Q1 format."""
        result = _parse_quarterly_date('2020 Q1')
        assert result.year == 2020
        assert result.month == 3


class TestLoadCSV:
    """Tests for load_csv function."""

    def test_load_basic_csv(self, sample_csv_file):
        """Test loading a basic CSV file."""
        df = load_csv(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_with_date_col(self, sample_csv_file):
        """Test loading with specified date column."""
        df = load_csv(sample_csv_file, date_col='date')
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_with_value_col(self, sample_csv_file):
        """Test loading with specified value column."""
        df = load_csv(sample_csv_file, date_col='date', value_col='value')
        assert 'value' in df.columns
        assert len(df.columns) == 1

    def test_index_is_datetime(self, sample_csv_file):
        """Test that index is DatetimeIndex."""
        df = load_csv(sample_csv_file, parse_dates=True)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_parse_dates_false(self, sample_csv_file):
        """Test loading without date parsing."""
        df = load_csv(sample_csv_file, parse_dates=False)
        # Index should not be DatetimeIndex
        assert not isinstance(df.index, pd.DatetimeIndex)

    def test_sorted_by_date(self, sample_csv_file):
        """Test that result is sorted by date."""
        df = load_csv(sample_csv_file)
        # Index should be monotonically increasing
        assert df.index.is_monotonic_increasing

    def test_quarterly_format(self, quarterly_csv_file):
        """Test loading quarterly date format."""
        df = load_csv(quarterly_csv_file, date_col='date')
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 4

    def test_quarterly_dates_correct(self, quarterly_csv_file):
        """Test that quarterly dates are parsed correctly."""
        df = load_csv(quarterly_csv_file, date_col='date')
        # First date should be end of Q1 2020
        assert df.index[0].month == 3
        assert df.index[0].year == 2020


class TestLoadExcel:
    """Tests for load_excel function."""

    def test_load_basic_excel(self, sample_excel_file):
        """Test loading a basic Excel file."""
        df = load_excel(sample_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_with_date_col(self, sample_excel_file):
        """Test loading with specified date column."""
        df = load_excel(sample_excel_file, date_col='date')
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_is_datetime(self, sample_excel_file):
        """Test that index is DatetimeIndex."""
        df = load_excel(sample_excel_file, parse_dates=True)
        assert isinstance(df.index, pd.DatetimeIndex)


class TestLoadMultipleCSV:
    """Tests for load_multiple_csv function."""

    def test_load_multiple_files(self):
        """Test loading multiple CSV files."""
        # Create temporary files
        files = {}
        filepaths = []

        for name, values in [('gdp', [1, 2, 3]), ('inflation', [4, 5, 6])]:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("date,value\n")
                for i, v in enumerate(values):
                    f.write(f"2020-0{i+1}-01,{v}\n")
                files[name] = f.name
                filepaths.append(f.name)

        try:
            df = load_multiple_csv(files, date_col='date')
            assert isinstance(df, pd.DataFrame)
            assert 'gdp' in df.columns
            assert 'inflation' in df.columns
        finally:
            for fp in filepaths:
                os.unlink(fp)

    def test_merged_on_date_index(self):
        """Test that files are merged on date index."""
        files = {}
        filepaths = []

        # Create files with same dates
        for name in ['series1', 'series2']:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("date,value\n")
                f.write("2020-01-01,100\n")
                f.write("2020-02-01,101\n")
                files[name] = f.name
                filepaths.append(f.name)

        try:
            df = load_multiple_csv(files, date_col='date')
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) == 2
        finally:
            for fp in filepaths:
                os.unlink(fp)


class TestUtilsEndToEnd:
    """End-to-end tests for utils workflow."""

    def test_full_csv_workflow(self):
        """Test complete CSV loading workflow."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,price,volume\n")
            for i in range(10):
                date = f"2020-0{(i % 9) + 1}-01"
                f.write(f"{date},{100 + i},{1000 + i * 10}\n")
            filepath = f.name

        try:
            # Load file
            df = load_csv(filepath, date_col='date')

            # Verify
            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) == 10
            assert 'price' in df.columns
            assert 'volume' in df.columns

            # Check no NaN
            assert not df.isna().all().any()

        finally:
            os.unlink(filepath)

    def test_quarterly_data_workflow(self):
        """Test workflow with quarterly data format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("period,gdp_growth\n")
            for year in [2019, 2020]:
                for q in [1, 2, 3, 4]:
                    f.write(f"{year}Q{q},{2.0 + np.random.randn() * 0.5}\n")
            filepath = f.name

        try:
            df = load_csv(filepath, date_col='period', value_col='gdp_growth')

            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) == 8

            # Verify dates are quarterly
            assert df.index[0].month in [3, 6, 9, 12]

        finally:
            os.unlink(filepath)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
