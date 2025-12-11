"""
Fundamental Feature Engineering

Computes ~40 fundamental features using yfinance data:
- Valuation: P/E, P/B, P/S, EV/EBITDA
- Growth: Revenue/earnings growth rates
- Profitability: ROE, ROA, margins
- Earnings timing: Days since/until earnings
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import warnings


def compute_fundamental_features(
    df: pd.DataFrame,
    symbol: str,
    sector_fundamentals: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """
    Compute fundamental features for a stock.

    Args:
        df: DataFrame with price data (index=date)
        symbol: Stock ticker symbol
        sector_fundamentals: Optional dict of sector average fundamentals

    Returns:
        DataFrame with fundamental features added
    """
    df = df.copy()

    # Fetch fundamental data from yfinance
    fundamentals = _fetch_fundamentals(symbol)

    if fundamentals is None:
        # Return with NaN fundamental features
        df = _add_nan_features(df)
        return df

    # Add static fundamental features (repeated for all dates)
    df = _add_valuation_features(df, fundamentals)
    df = _add_growth_features(df, fundamentals)
    df = _add_profitability_features(df, fundamentals)
    df = _add_financial_health_features(df, fundamentals)

    # Add earnings timing features
    df = _add_earnings_timing(df, symbol)

    # Add sector-relative features if available
    if sector_fundamentals is not None and 'sector' in fundamentals:
        sector = fundamentals.get('sector')
        if sector in sector_fundamentals:
            df = _add_sector_relative_features(df, fundamentals, sector_fundamentals[sector])

    return df


def _fetch_fundamentals(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch fundamental data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or 'symbol' not in info:
            return None

        return info

    except Exception as e:
        warnings.warn(f"Failed to fetch fundamentals for {symbol}: {e}")
        return None


def _add_valuation_features(df: pd.DataFrame, fundamentals: Dict) -> pd.DataFrame:
    """Add valuation ratio features."""

    # P/E Ratios
    df['pe_trailing'] = fundamentals.get('trailingPE', np.nan)
    df['pe_forward'] = fundamentals.get('forwardPE', np.nan)
    df['peg_ratio'] = fundamentals.get('pegRatio', np.nan)

    # Price to Book
    df['price_to_book'] = fundamentals.get('priceToBook', np.nan)

    # Price to Sales
    df['price_to_sales'] = fundamentals.get('priceToSalesTrailing12Months', np.nan)

    # Enterprise Value metrics
    df['ev_to_revenue'] = fundamentals.get('enterpriseToRevenue', np.nan)
    df['ev_to_ebitda'] = fundamentals.get('enterpriseToEbitda', np.nan)

    # Market cap related
    market_cap = fundamentals.get('marketCap', np.nan)
    df['log_market_cap'] = np.log10(market_cap) if not np.isnan(market_cap) and market_cap > 0 else np.nan

    # Earnings yield (inverse of P/E)
    pe = fundamentals.get('trailingPE', np.nan)
    df['earnings_yield'] = 1 / pe if pe and pe > 0 else np.nan

    # Free cash flow yield
    fcf = fundamentals.get('freeCashflow', np.nan)
    if market_cap and market_cap > 0 and fcf:
        df['fcf_yield'] = fcf / market_cap
    else:
        df['fcf_yield'] = np.nan

    return df


def _add_growth_features(df: pd.DataFrame, fundamentals: Dict) -> pd.DataFrame:
    """Add growth-related features."""

    # Revenue growth
    df['revenue_growth'] = fundamentals.get('revenueGrowth', np.nan)

    # Earnings growth
    df['earnings_growth'] = fundamentals.get('earningsGrowth', np.nan)
    df['earnings_quarterly_growth'] = fundamentals.get('earningsQuarterlyGrowth', np.nan)

    # Revenue per share growth proxy
    revenue = fundamentals.get('totalRevenue', np.nan)
    shares = fundamentals.get('sharesOutstanding', np.nan)
    if revenue and shares and shares > 0:
        df['revenue_per_share'] = revenue / shares
    else:
        df['revenue_per_share'] = np.nan

    # Book value growth proxy
    df['book_value'] = fundamentals.get('bookValue', np.nan)

    return df


def _add_profitability_features(df: pd.DataFrame, fundamentals: Dict) -> pd.DataFrame:
    """Add profitability features."""

    # Margins
    df['gross_margin'] = fundamentals.get('grossMargins', np.nan)
    df['operating_margin'] = fundamentals.get('operatingMargins', np.nan)
    df['profit_margin'] = fundamentals.get('profitMargins', np.nan)
    df['ebitda_margin'] = fundamentals.get('ebitdaMargins', np.nan)

    # Return metrics
    df['roe'] = fundamentals.get('returnOnEquity', np.nan)
    df['roa'] = fundamentals.get('returnOnAssets', np.nan)

    # EPS
    df['eps_trailing'] = fundamentals.get('trailingEps', np.nan)
    df['eps_forward'] = fundamentals.get('forwardEps', np.nan)

    # EPS growth rate
    trailing_eps = fundamentals.get('trailingEps', np.nan)
    forward_eps = fundamentals.get('forwardEps', np.nan)
    if trailing_eps and forward_eps and trailing_eps > 0:
        df['eps_growth_rate'] = (forward_eps - trailing_eps) / abs(trailing_eps)
    else:
        df['eps_growth_rate'] = np.nan

    return df


def _add_financial_health_features(df: pd.DataFrame, fundamentals: Dict) -> pd.DataFrame:
    """Add financial health features."""

    # Debt metrics
    df['debt_to_equity'] = fundamentals.get('debtToEquity', np.nan)

    total_debt = fundamentals.get('totalDebt', np.nan)
    total_cash = fundamentals.get('totalCash', np.nan)
    if total_debt is not None and total_cash is not None:
        df['net_debt'] = total_debt - total_cash
    else:
        df['net_debt'] = np.nan

    # Current ratio
    df['current_ratio'] = fundamentals.get('currentRatio', np.nan)

    # Quick ratio
    df['quick_ratio'] = fundamentals.get('quickRatio', np.nan)

    # Cash per share
    cash_per_share = fundamentals.get('totalCashPerShare', np.nan)
    df['cash_per_share'] = cash_per_share

    # Dividend info
    df['dividend_yield'] = fundamentals.get('dividendYield', np.nan)
    df['payout_ratio'] = fundamentals.get('payoutRatio', np.nan)

    # Beta (market risk)
    df['beta'] = fundamentals.get('beta', np.nan)

    return df


def _add_earnings_timing(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add features related to earnings announcement timing."""
    try:
        ticker = yf.Ticker(symbol)

        # Get earnings dates
        earnings_dates = ticker.earnings_dates

        if earnings_dates is None or len(earnings_dates) == 0:
            df['days_to_earnings'] = np.nan
            df['days_since_earnings'] = np.nan
            df['earnings_period'] = 0
            return df

        # Convert to list of dates
        earnings_list = pd.to_datetime(earnings_dates.index).tz_localize(None)

        days_to_earnings = []
        days_since_earnings = []
        earnings_period = []

        for date in df.index:
            date_val = pd.to_datetime(date)
            if hasattr(date_val, 'tz') and date_val.tz is not None:
                date_val = date_val.tz_localize(None)

            # Find next earnings date
            future_dates = earnings_list[earnings_list > date_val]
            if len(future_dates) > 0:
                next_earnings = future_dates.min()
                days_to = (next_earnings - date_val).days
                days_to_earnings.append(days_to)
            else:
                days_to_earnings.append(np.nan)

            # Find previous earnings date
            past_dates = earnings_list[earnings_list <= date_val]
            if len(past_dates) > 0:
                prev_earnings = past_dates.max()
                days_since = (date_val - prev_earnings).days
                days_since_earnings.append(days_since)
            else:
                days_since_earnings.append(np.nan)

            # Earnings period flag (within 5 days of earnings)
            if len(future_dates) > 0 and days_to_earnings[-1] <= 5:
                earnings_period.append(1)
            elif len(past_dates) > 0 and days_since_earnings[-1] <= 5:
                earnings_period.append(1)
            else:
                earnings_period.append(0)

        df['days_to_earnings'] = days_to_earnings
        df['days_since_earnings'] = days_since_earnings
        df['earnings_period'] = earnings_period

    except Exception as e:
        warnings.warn(f"Failed to fetch earnings dates for {symbol}: {e}")
        df['days_to_earnings'] = np.nan
        df['days_since_earnings'] = np.nan
        df['earnings_period'] = 0

    return df


def _add_sector_relative_features(
    df: pd.DataFrame,
    fundamentals: Dict,
    sector_avg: Dict
) -> pd.DataFrame:
    """Add features relative to sector averages."""

    # P/E relative to sector
    pe = fundamentals.get('trailingPE', np.nan)
    sector_pe = sector_avg.get('pe', np.nan)
    if pe and sector_pe and sector_pe > 0:
        df['pe_vs_sector'] = pe / sector_pe
    else:
        df['pe_vs_sector'] = np.nan

    # P/B relative to sector
    pb = fundamentals.get('priceToBook', np.nan)
    sector_pb = sector_avg.get('pb', np.nan)
    if pb and sector_pb and sector_pb > 0:
        df['pb_vs_sector'] = pb / sector_pb
    else:
        df['pb_vs_sector'] = np.nan

    # ROE relative to sector
    roe = fundamentals.get('returnOnEquity', np.nan)
    sector_roe = sector_avg.get('roe', np.nan)
    if roe and sector_roe:
        df['roe_vs_sector'] = roe - sector_roe
    else:
        df['roe_vs_sector'] = np.nan

    # Margin relative to sector
    margin = fundamentals.get('profitMargins', np.nan)
    sector_margin = sector_avg.get('margin', np.nan)
    if margin and sector_margin:
        df['margin_vs_sector'] = margin - sector_margin
    else:
        df['margin_vs_sector'] = np.nan

    return df


def _add_nan_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NaN values for all fundamental features when data unavailable."""
    fundamental_features = [
        'pe_trailing', 'pe_forward', 'peg_ratio', 'price_to_book', 'price_to_sales',
        'ev_to_revenue', 'ev_to_ebitda', 'log_market_cap', 'earnings_yield', 'fcf_yield',
        'revenue_growth', 'earnings_growth', 'earnings_quarterly_growth',
        'revenue_per_share', 'book_value',
        'gross_margin', 'operating_margin', 'profit_margin', 'ebitda_margin',
        'roe', 'roa', 'eps_trailing', 'eps_forward', 'eps_growth_rate',
        'debt_to_equity', 'net_debt', 'current_ratio', 'quick_ratio',
        'cash_per_share', 'dividend_yield', 'payout_ratio', 'beta',
        'days_to_earnings', 'days_since_earnings', 'earnings_period'
    ]

    for feature in fundamental_features:
        df[feature] = np.nan

    return df


def get_sector_averages(symbols: list, sector_mapping: Dict[str, str]) -> Dict[str, Dict]:
    """
    Calculate sector average fundamentals.

    Args:
        symbols: List of stock symbols
        sector_mapping: Dict mapping symbol -> sector

    Returns:
        Dict mapping sector -> average fundamentals
    """
    sector_data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector = sector_mapping.get(symbol, info.get('sector', 'Unknown'))

            if sector not in sector_data:
                sector_data[sector] = {'pe': [], 'pb': [], 'roe': [], 'margin': []}

            if info.get('trailingPE'):
                sector_data[sector]['pe'].append(info['trailingPE'])
            if info.get('priceToBook'):
                sector_data[sector]['pb'].append(info['priceToBook'])
            if info.get('returnOnEquity'):
                sector_data[sector]['roe'].append(info['returnOnEquity'])
            if info.get('profitMargins'):
                sector_data[sector]['margin'].append(info['profitMargins'])

        except Exception:
            continue

    # Calculate averages
    sector_averages = {}
    for sector, data in sector_data.items():
        sector_averages[sector] = {
            'pe': np.median(data['pe']) if data['pe'] else np.nan,
            'pb': np.median(data['pb']) if data['pb'] else np.nan,
            'roe': np.median(data['roe']) if data['roe'] else np.nan,
            'margin': np.median(data['margin']) if data['margin'] else np.nan
        }

    return sector_averages


def get_feature_names() -> list:
    """Get list of all fundamental feature names."""
    return [
        'pe_trailing', 'pe_forward', 'peg_ratio', 'price_to_book', 'price_to_sales',
        'ev_to_revenue', 'ev_to_ebitda', 'log_market_cap', 'earnings_yield', 'fcf_yield',
        'revenue_growth', 'earnings_growth', 'earnings_quarterly_growth',
        'revenue_per_share', 'book_value',
        'gross_margin', 'operating_margin', 'profit_margin', 'ebitda_margin',
        'roe', 'roa', 'eps_trailing', 'eps_forward', 'eps_growth_rate',
        'debt_to_equity', 'net_debt', 'current_ratio', 'quick_ratio',
        'cash_per_share', 'dividend_yield', 'payout_ratio', 'beta',
        'days_to_earnings', 'days_since_earnings', 'earnings_period',
        'pe_vs_sector', 'pb_vs_sector', 'roe_vs_sector', 'margin_vs_sector'
    ]


if __name__ == '__main__':
    print("Fundamental feature names:")
    features = get_feature_names()
    print(f"Total features: {len(features)}")
    for f in features:
        print(f"  - {f}")
