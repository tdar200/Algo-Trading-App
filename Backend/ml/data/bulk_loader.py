"""
Bulk Data Loader for S&P 500 Stocks

Parallel data fetching using ThreadPoolExecutor with parquet caching
and progress tracking via callbacks.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, List, Dict, Any
import time


CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'stock_data'
CACHE_EXPIRY_HOURS = 24


class BulkDataLoader:
    """
    Bulk loader for S&P 500 stock data with parallel fetching and caching.

    Features:
    - Parallel data fetching (10 workers by default)
    - Parquet caching with 24-hour expiry
    - Progress callback for real-time updates
    - Rate limiting to respect API limits
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        num_workers: int = 10,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize the bulk data loader.

        Args:
            cache_dir: Directory for parquet cache files
            num_workers: Number of parallel workers for data fetching
            progress_callback: Optional callback(symbol, completed, total)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.progress_callback = progress_callback

    def load_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols in parallel.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            Dict mapping symbol -> DataFrame with OHLCV data
        """
        results = {}
        failed = []
        completed = 0
        total = len(symbols)

        def load_single(symbol: str) -> tuple:
            """Load data for a single symbol."""
            try:
                df = self._load_symbol(symbol, start_date, end_date, force_refresh)
                return (symbol, df, None)
            except Exception as e:
                return (symbol, None, str(e))

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_single, sym): sym for sym in symbols}

            for future in as_completed(futures):
                symbol, df, error = future.result()
                completed += 1

                if error is None and df is not None and len(df) > 0:
                    results[symbol] = df
                else:
                    failed.append((symbol, error))

                # Report progress
                if self.progress_callback:
                    self.progress_callback(symbol, completed, total)

        if failed:
            print(f"Failed to load {len(failed)} symbols: {[s for s, _ in failed[:10]]}")

        return results

    def _load_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        force_refresh: bool
    ) -> pd.DataFrame:
        """Load data for a single symbol with caching."""
        cache_file = self._get_cache_path(symbol, start_date, end_date)

        # Check cache
        if not force_refresh and self._is_cache_valid(cache_file):
            return pd.read_parquet(cache_file)

        # Fetch from yfinance
        df = self._fetch_from_yfinance(symbol, start_date, end_date)

        if df is not None and len(df) > 0:
            # Save to cache
            df.to_parquet(cache_file)

        return df

    def _fetch_from_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch stock data from yfinance."""
        # Small delay to avoid rate limiting
        time.sleep(0.1)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            return None

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return None

        # Remove timezone info and normalize index
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = 'date'

        # Add symbol column
        df['symbol'] = symbol

        return df[required_cols + ['symbol']]

    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path for a symbol."""
        # Use just the symbol for filename to allow cache reuse
        return self.cache_dir / f"{symbol}.parquet"

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_file.exists():
            return False

        # Check file modification time
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mod_time > timedelta(hours=CACHE_EXPIRY_HOURS):
            return False

        return True

    def load_combined(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load and combine data for multiple symbols into a single DataFrame.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, ignore cache

        Returns:
            Combined DataFrame with multi-index (date, symbol)
        """
        data_dict = self.load_multiple(symbols, start_date, end_date, force_refresh)

        if not data_dict:
            return pd.DataFrame()

        # Combine all DataFrames
        dfs = []
        for symbol, df in data_dict.items():
            df = df.copy()
            df['symbol'] = symbol
            df = df.reset_index()
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.set_index(['date', 'symbol'])

        return combined

    def get_spy_benchmark(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load SPY data for benchmark comparisons.

        Args:
            start_date: Start date
            end_date: End date
            force_refresh: If True, ignore cache

        Returns:
            DataFrame with SPY OHLCV data
        """
        return self._load_symbol('SPY', start_date, end_date, force_refresh)


def create_progress_printer() -> Callable:
    """Create a simple progress callback that prints to console."""
    start_time = time.time()

    def progress_callback(symbol: str, completed: int, total: int):
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0

        print(f"\rLoading: {completed}/{total} ({100*completed/total:.1f}%) "
              f"- {symbol} - ETA: {eta:.0f}s", end='', flush=True)

        if completed == total:
            print(f"\nCompleted in {elapsed:.1f}s")

    return progress_callback


if __name__ == '__main__':
    # Test the bulk loader
    from sp500_list import get_sp500_constituents

    # Get S&P 500 symbols
    constituents = get_sp500_constituents()
    symbols = constituents['symbol'].tolist()[:20]  # Test with 20 symbols

    # Calculate date range (5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

    # Create loader with progress callback
    loader = BulkDataLoader(
        num_workers=10,
        progress_callback=create_progress_printer()
    )

    # Load data
    print(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
    data = loader.load_multiple(symbols, start_date, end_date)

    print(f"\nSuccessfully loaded {len(data)} symbols")
    if data:
        first_symbol = list(data.keys())[0]
        print(f"\nSample data for {first_symbol}:")
        print(data[first_symbol].head())
