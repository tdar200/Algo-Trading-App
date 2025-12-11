"""
Unified Data Loader for Hybrid VectorBT + Backtrader Optimization

This module ensures both VectorBT and Backtrader receive identical,
cleaned data to prevent logic parity issues.

Key Features:
- Parquet-based caching for fast I/O
- Forward-fill NaN handling
- Strict datetime index normalization
- Support for both date-range and period-based fetching
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Cache directory relative to this file
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiry in hours
CACHE_EXPIRY_HOURS = 24


def _get_cache_path(ticker: str, start: str, end: str) -> Path:
    """Generate cache file path for a specific ticker and date range."""
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe_ticker}_{start}_{end}.parquet"


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if cached file exists and is not expired."""
    if not cache_path.exists():
        return False

    # Check file age
    file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - file_mtime

    return age < timedelta(hours=CACHE_EXPIRY_HOURS)


def fetch_data(
    ticker: str,
    start: str,
    end: str,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker with Parquet caching.

    This is the PRIMARY data source for both VectorBT and Backtrader.
    Both engines MUST use this function to ensure data alignment.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'SPY')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        force_refresh: If True, bypass cache and re-download

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone-naive)
    """
    cache_path = _get_cache_path(ticker, start, end)

    # Check cache first
    if not force_refresh and _is_cache_valid(cache_path):
        print(f"[DataLoader] Loading {ticker} from cache...")
        df = pd.read_parquet(cache_path)
        return df

    # Download from yfinance
    print(f"[DataLoader] Downloading {ticker} ({start} to {end})...")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")

    if df.empty:
        raise ValueError(f"No data returned for {ticker} in range {start} to {end}")

    # Clean and normalize the data (CRITICAL FOR PARITY)
    df = _normalize_dataframe(df)

    # Save to Parquet cache
    df.to_parquet(cache_path)
    print(f"[DataLoader] Cached {len(df)} bars to {cache_path.name}")

    return df


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame to ensure VBT/BT parity.

    This function performs critical cleaning:
    1. Flatten MultiIndex columns (yfinance sometimes returns these)
    2. Convert timezone-aware index to timezone-naive
    3. Forward-fill missing values (prevents NaN divergence)
    4. Ensure consistent column names
    5. Sort by date
    """
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only required columns
    df = df[required_cols].copy()

    # Convert timezone-aware datetime to timezone-naive (UTC)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # Sort by date
    df = df.sort_index()

    # Forward-fill then back-fill missing values
    # This prevents VBT/BT divergence from NaN handling differences
    df = df.ffill().bfill()

    # Ensure numeric types (float64 for precision)
    df = df.astype({
        'Open': 'float64',
        'High': 'float64',
        'Low': 'float64',
        'Close': 'float64',
        'Volume': 'float64'
    })

    return df


def get_data_for_backtrader(df: pd.DataFrame):
    """
    Convert DataFrame to Backtrader-compatible format.

    Backtrader expects a PandasData feed with specific column mappings.
    This function ensures the DataFrame is ready for bt.feeds.PandasData.

    Args:
        df: DataFrame from fetch_data()

    Returns:
        DataFrame ready for bt.feeds.PandasData(dataname=df)
    """
    # Backtrader expects lowercase columns and datetime index
    bt_df = df.copy()

    # Backtrader's PandasData uses the index as datetime
    # No additional transformation needed if using:
    # bt.feeds.PandasData(dataname=df, open='Open', high='High', ...)

    return bt_df


def get_data_for_vectorbt(df: pd.DataFrame):
    """
    Convert DataFrame to VectorBT-compatible format.

    VectorBT works directly with pandas DataFrames.
    This function ensures proper column naming for VBT.

    Args:
        df: DataFrame from fetch_data()

    Returns:
        DataFrame ready for VectorBT operations
    """
    # VectorBT uses the DataFrame directly
    # Ensure columns are capitalized (VBT convention)
    vbt_df = df.copy()

    return vbt_df


def clear_cache(ticker: str = None):
    """
    Clear cached data files.

    Args:
        ticker: If provided, only clear cache for this ticker.
                If None, clear all cached files.
    """
    if ticker:
        pattern = f"{ticker}_*.parquet"
        for f in CACHE_DIR.glob(pattern):
            f.unlink()
            print(f"[DataLoader] Removed {f.name}")
    else:
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
            print(f"[DataLoader] Removed {f.name}")


def get_cache_info() -> dict:
    """
    Get information about cached data files.

    Returns:
        Dict with cache statistics
    """
    files = list(CACHE_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files)

    return {
        'cache_dir': str(CACHE_DIR),
        'file_count': len(files),
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'files': [f.name for f in files]
    }


if __name__ == "__main__":
    # Test the loader
    print("Testing Data Loader...")

    df = fetch_data("SPY", "2023-01-01", "2024-01-01")
    print(f"\nLoaded {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nCache info:\n{get_cache_info()}")
