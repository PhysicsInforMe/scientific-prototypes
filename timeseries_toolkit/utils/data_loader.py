"""
Data Loading Utilities.

This module provides helper functions for loading time series data
from various file formats.
"""

from typing import Optional, Union

import pandas as pd


def load_csv(
    filepath: str,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    parse_dates: bool = True,
    date_format: Optional[str] = None,
    freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Load time series data from a CSV file.

    Args:
        filepath: Path to CSV file.
        date_col: Name of date column. If None, uses first column.
        value_col: Name of value column. If None, uses second column.
        parse_dates: Whether to parse dates automatically.
        date_format: Specific date format (e.g., '%Y-%m-%d').
        freq: Frequency to set on DatetimeIndex (e.g., 'D', 'M', 'Q').

    Returns:
        DataFrame with DatetimeIndex and value column(s).

    Example:
        >>> df = load_csv('gdp.csv', date_col='date', value_col='gdp')
        >>> print(df.head())
    """
    df = pd.read_csv(filepath)

    # Identify columns
    if date_col is None:
        date_col = df.columns[0]

    # Parse dates
    if parse_dates:
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            # Try common formats including quarterly
            date_str = str(df[date_col].iloc[0])
            if 'Q' in date_str:
                # Handle 'YYYYQn' format
                df[date_col] = df[date_col].astype(str).apply(_parse_quarterly_date)
            else:
                df[date_col] = pd.to_datetime(df[date_col])

    # Set index
    df = df.set_index(date_col).sort_index()

    # Select value column if specified
    if value_col is not None and value_col in df.columns:
        df = df[[value_col]]

    # Set frequency if specified
    if freq is not None:
        df = df.asfreq(freq)

    return df


def load_excel(
    filepath: str,
    sheet_name: Union[str, int] = 0,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    parse_dates: bool = True,
    freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Load time series data from an Excel file.

    Args:
        filepath: Path to Excel file (.xlsx or .xls).
        sheet_name: Sheet name or index to read.
        date_col: Name of date column. If None, uses first column.
        value_col: Name of value column. If None, uses second column.
        parse_dates: Whether to parse dates automatically.
        freq: Frequency to set on DatetimeIndex.

    Returns:
        DataFrame with DatetimeIndex and value column(s).

    Example:
        >>> df = load_excel('data.xlsx', sheet_name='GDP', date_col='Date')
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Identify columns
    if date_col is None:
        date_col = df.columns[0]

    # Parse dates
    if parse_dates:
        date_str = str(df[date_col].iloc[0])
        if 'Q' in date_str:
            df[date_col] = df[date_col].astype(str).apply(_parse_quarterly_date)
        else:
            df[date_col] = pd.to_datetime(df[date_col])

    # Set index
    df = df.set_index(date_col).sort_index()

    # Select value column if specified
    if value_col is not None and value_col in df.columns:
        df = df[[value_col]]

    # Set frequency
    if freq is not None:
        df = df.asfreq(freq)

    return df


def _parse_quarterly_date(date_str: str) -> pd.Timestamp:
    """
    Parse quarterly date string (e.g., '2020Q1', '2020-Q1').

    Args:
        date_str: Date string in quarterly format.

    Returns:
        Timestamp for quarter end date.
    """
    date_str = str(date_str).replace('-', '').strip()

    # Handle formats like '2020Q1' or '2020 Q1'
    date_str = date_str.replace(' ', '')

    quarter_map = {
        'Q1': '-03-31',
        'Q2': '-06-30',
        'Q3': '-09-30',
        'Q4': '-12-31',
    }

    for q, suffix in quarter_map.items():
        if q in date_str:
            year = date_str.replace(q, '')
            return pd.Timestamp(year + suffix)

    # Fallback
    return pd.to_datetime(date_str)


def load_multiple_csv(
    filepaths: dict,
    date_col: Optional[str] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load and merge multiple CSV files into a single DataFrame.

    Args:
        filepaths: Dictionary mapping column names to file paths.
        date_col: Name of date column in each file.
        parse_dates: Whether to parse dates.

    Returns:
        DataFrame with all series merged on date index.

    Example:
        >>> files = {'gdp': 'gdp.csv', 'inflation': 'cpi.csv'}
        >>> df = load_multiple_csv(files, date_col='date')
    """
    merged = None

    for name, path in filepaths.items():
        df = load_csv(path, date_col=date_col, parse_dates=parse_dates)

        # Rename columns to avoid conflicts
        if len(df.columns) == 1:
            df.columns = [name]
        else:
            df.columns = [f"{name}_{col}" for col in df.columns]

        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how='outer')

    return merged
