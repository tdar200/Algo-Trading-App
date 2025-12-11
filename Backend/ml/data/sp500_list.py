"""
S&P 500 Constituent List Fetcher

Scrapes the current S&P 500 constituent list from Wikipedia,
including sector and industry metadata.
"""

import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path


CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'
SP500_CACHE_FILE = CACHE_DIR / 'sp500_constituents.json'
CACHE_EXPIRY_DAYS = 7  # Refresh constituent list weekly


def get_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """
    Get current S&P 500 constituents with sector/industry metadata.

    Args:
        force_refresh: If True, ignore cache and fetch fresh data

    Returns:
        DataFrame with columns: symbol, company, sector, sub_industry
    """
    # Check cache first
    if not force_refresh and _is_cache_valid():
        return _load_from_cache()

    # Fetch fresh data from Wikipedia
    constituents = _fetch_from_wikipedia()

    # Cache the results
    _save_to_cache(constituents)

    return constituents


def _is_cache_valid() -> bool:
    """Check if cached data exists and is not expired."""
    if not SP500_CACHE_FILE.exists():
        return False

    try:
        with open(SP500_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)

        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cache_time > timedelta(days=CACHE_EXPIRY_DAYS):
            return False

        return True
    except (json.JSONDecodeError, KeyError):
        return False


def _load_from_cache() -> pd.DataFrame:
    """Load constituent list from cache file."""
    with open(SP500_CACHE_FILE, 'r') as f:
        cache_data = json.load(f)

    return pd.DataFrame(cache_data['constituents'])


def _save_to_cache(constituents: pd.DataFrame) -> None:
    """Save constituent list to cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'constituents': constituents.to_dict(orient='records')
    }

    with open(SP500_CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)


def _fetch_from_wikipedia() -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from Wikipedia.

    Wikipedia maintains an up-to-date list at:
    https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'lxml')

    # Find the first table (current constituents)
    table = soup.find('table', {'id': 'constituents'})
    if table is None:
        # Fallback: find first wikitable
        table = soup.find('table', {'class': 'wikitable'})

    # Parse the table
    rows = table.find_all('tr')[1:]  # Skip header row

    constituents = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            symbol = cols[0].text.strip()
            company = cols[1].text.strip()
            sector = cols[2].text.strip()
            sub_industry = cols[3].text.strip()

            # Clean up symbol (remove any extra characters)
            symbol = symbol.replace('.', '-')  # BRK.B -> BRK-B for yfinance

            constituents.append({
                'symbol': symbol,
                'company': company,
                'sector': sector,
                'sub_industry': sub_industry
            })

    df = pd.DataFrame(constituents)

    print(f"Fetched {len(df)} S&P 500 constituents")

    return df


def get_sector_mapping() -> dict:
    """
    Get mapping of symbols to their sectors.

    Returns:
        Dict mapping symbol -> sector name
    """
    constituents = get_sp500_constituents()
    return dict(zip(constituents['symbol'], constituents['sector']))


def get_symbols_by_sector(sector: str) -> list:
    """
    Get all symbols in a specific sector.

    Args:
        sector: Sector name (e.g., 'Information Technology', 'Health Care')

    Returns:
        List of stock symbols in that sector
    """
    constituents = get_sp500_constituents()
    return constituents[constituents['sector'] == sector]['symbol'].tolist()


def get_all_sectors() -> list:
    """Get list of all unique sectors in S&P 500."""
    constituents = get_sp500_constituents()
    return sorted(constituents['sector'].unique().tolist())


if __name__ == '__main__':
    # Test the module
    df = get_sp500_constituents(force_refresh=True)
    print(f"\nTotal constituents: {len(df)}")
    print(f"\nSectors: {get_all_sectors()}")
    print(f"\nSample data:\n{df.head(10)}")
