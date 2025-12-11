"""
Cross-Sectional Feature Engineering

Computes ~40 features comparing each stock to the market and peers:
- Relative strength vs SPY (beta, alpha)
- Sector-relative performance
- Percentile rank in universe
- Correlation to sector/market
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional


def compute_cross_sectional_features(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    sector_data: Optional[Dict[str, pd.DataFrame]] = None,
    universe_data: Optional[Dict[str, pd.DataFrame]] = None,
    sector: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute cross-sectional features comparing stock to market/peers.

    Args:
        df: DataFrame with columns: open, high, low, close, volume (index=date)
        benchmark_df: SPY benchmark DataFrame (optional)
        sector_data: Dict of sector peer DataFrames (optional)
        universe_data: Dict of all stock DataFrames for ranking (optional)
        sector: The stock's sector name (optional)

    Returns:
        DataFrame with cross-sectional features added
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Compute returns if not present
    if 'return_1d' not in df.columns:
        df['return_1d'] = df['close'].pct_change()

    # Add benchmark-relative features if benchmark available
    if benchmark_df is not None:
        df = _add_benchmark_features(df, benchmark_df)

    # Add sector-relative features if sector data available
    if sector_data is not None and sector is not None:
        df = _add_sector_features(df, sector_data, sector)

    # Add universe ranking features if universe data available
    if universe_data is not None:
        df = _add_universe_rankings(df, universe_data)

    # Add standalone cross-sectional proxies (work without external data)
    df = _add_standalone_features(df)

    return df


def _add_benchmark_features(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Add features relative to SPY benchmark."""

    benchmark_df = benchmark_df.copy()
    benchmark_df.columns = [c.lower() for c in benchmark_df.columns]

    # Ensure benchmark has returns
    if 'return_1d' not in benchmark_df.columns:
        benchmark_df['return_1d'] = benchmark_df['close'].pct_change()

    # Align dates
    common_dates = df.index.intersection(benchmark_df.index)
    if len(common_dates) < 20:
        return df

    stock_ret = df.loc[common_dates, 'return_1d']
    market_ret = benchmark_df.loc[common_dates, 'return_1d']

    # Rolling beta and alpha
    for window in [20, 50, 100]:
        betas = []
        alphas = []

        for i in range(len(common_dates)):
            if i < window:
                betas.append(np.nan)
                alphas.append(np.nan)
                continue

            y = stock_ret.iloc[i-window:i].values
            x = market_ret.iloc[i-window:i].values

            # Remove NaN values
            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < window // 2:
                betas.append(np.nan)
                alphas.append(np.nan)
                continue

            y_clean = y[mask]
            x_clean = x[mask]

            try:
                slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
                betas.append(slope)
                alphas.append(intercept * 252)  # Annualized alpha
            except Exception:
                betas.append(np.nan)
                alphas.append(np.nan)

        df.loc[common_dates, f'beta_{window}'] = betas
        df.loc[common_dates, f'alpha_{window}'] = alphas

    # Relative strength (cumulative outperformance)
    for window in [5, 10, 20, 50]:
        stock_cum = (1 + stock_ret).rolling(window).apply(lambda x: x.prod() - 1, raw=True)
        market_cum = (1 + market_ret).rolling(window).apply(lambda x: x.prod() - 1, raw=True)

        df.loc[common_dates, f'rel_strength_{window}'] = stock_cum - market_cum

    # Correlation to market
    for window in [20, 50]:
        corr = stock_ret.rolling(window).corr(market_ret)
        df.loc[common_dates, f'market_corr_{window}'] = corr

    # Tracking error
    diff = stock_ret - market_ret
    df.loc[common_dates, 'tracking_error_20'] = diff.rolling(20).std() * np.sqrt(252)
    df.loc[common_dates, 'tracking_error_50'] = diff.rolling(50).std() * np.sqrt(252)

    # Information ratio
    df.loc[common_dates, 'info_ratio_20'] = (
        diff.rolling(20).mean() * 252 / (diff.rolling(20).std() * np.sqrt(252) + 1e-10)
    )

    # Up/Down capture ratio
    for window in [50]:
        up_days = market_ret > 0
        down_days = market_ret < 0

        up_capture = []
        down_capture = []

        for i in range(len(common_dates)):
            if i < window:
                up_capture.append(np.nan)
                down_capture.append(np.nan)
                continue

            period_stock = stock_ret.iloc[i-window:i]
            period_market = market_ret.iloc[i-window:i]
            period_up = up_days.iloc[i-window:i]
            period_down = down_days.iloc[i-window:i]

            # Up capture
            if period_up.sum() > 0:
                stock_up = period_stock[period_up].mean()
                market_up = period_market[period_up].mean()
                up_capture.append(stock_up / (market_up + 1e-10) * 100)
            else:
                up_capture.append(np.nan)

            # Down capture
            if period_down.sum() > 0:
                stock_down = period_stock[period_down].mean()
                market_down = period_market[period_down].mean()
                down_capture.append(stock_down / (market_down + 1e-10) * 100)
            else:
                down_capture.append(np.nan)

        df.loc[common_dates, f'up_capture_{window}'] = up_capture
        df.loc[common_dates, f'down_capture_{window}'] = down_capture

    return df


def _add_sector_features(
    df: pd.DataFrame,
    sector_data: Dict[str, pd.DataFrame],
    sector: str
) -> pd.DataFrame:
    """Add features relative to sector peers."""

    if sector not in sector_data or len(sector_data[sector]) == 0:
        return df

    peers = sector_data[sector]

    # Calculate sector average returns
    peer_returns = []
    for symbol, peer_df in peers.items():
        if 'return_1d' not in peer_df.columns:
            peer_df = peer_df.copy()
            peer_df['return_1d'] = peer_df['close'].pct_change()
        peer_returns.append(peer_df['return_1d'].rename(symbol))

    if len(peer_returns) < 2:
        return df

    # Combine peer returns
    sector_returns = pd.concat(peer_returns, axis=1)
    sector_avg = sector_returns.mean(axis=1)

    # Align dates
    common_dates = df.index.intersection(sector_avg.index)
    if len(common_dates) < 20:
        return df

    stock_ret = df.loc[common_dates, 'return_1d']

    # Relative to sector performance
    for window in [5, 10, 20]:
        stock_cum = (1 + stock_ret).rolling(window).apply(lambda x: x.prod() - 1, raw=True)
        sector_cum = (1 + sector_avg.loc[common_dates]).rolling(window).apply(
            lambda x: x.prod() - 1, raw=True
        )
        df.loc[common_dates, f'sector_rel_{window}'] = stock_cum - sector_cum

    # Correlation to sector
    df.loc[common_dates, 'sector_corr_20'] = stock_ret.rolling(20).corr(sector_avg.loc[common_dates])

    return df


def _add_universe_rankings(
    df: pd.DataFrame,
    universe_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Add percentile rankings across the stock universe."""

    if len(universe_data) < 10:
        return df

    # Get symbol for this stock
    if 'symbol' in df.columns:
        symbol = df['symbol'].iloc[0]
    else:
        return df

    # Calculate rankings for each date
    dates = df.index

    # Pre-calculate all returns
    all_returns = {}
    all_volatility = {}

    for sym, sym_df in universe_data.items():
        if 'return_1d' not in sym_df.columns:
            sym_df = sym_df.copy()
            sym_df['return_1d'] = sym_df['close'].pct_change()

        ret_20 = sym_df['return_1d'].rolling(20).sum()
        vol_20 = sym_df['return_1d'].rolling(20).std()

        all_returns[sym] = ret_20
        all_volatility[sym] = vol_20

    # Calculate rankings
    return_rank = []
    volatility_rank = []

    for date in dates:
        # Get all stock values for this date
        date_returns = []
        date_vols = []
        stock_ret = np.nan
        stock_vol = np.nan

        for sym, ret_series in all_returns.items():
            if date in ret_series.index:
                val = ret_series.loc[date]
                if not np.isnan(val):
                    date_returns.append(val)
                    if sym == symbol:
                        stock_ret = val

        for sym, vol_series in all_volatility.items():
            if date in vol_series.index:
                val = vol_series.loc[date]
                if not np.isnan(val):
                    date_vols.append(val)
                    if sym == symbol:
                        stock_vol = val

        # Calculate percentile
        if len(date_returns) > 10 and not np.isnan(stock_ret):
            return_rank.append(stats.percentileofscore(date_returns, stock_ret) / 100)
        else:
            return_rank.append(np.nan)

        if len(date_vols) > 10 and not np.isnan(stock_vol):
            volatility_rank.append(stats.percentileofscore(date_vols, stock_vol) / 100)
        else:
            volatility_rank.append(np.nan)

    df['return_percentile_rank'] = return_rank
    df['volatility_percentile_rank'] = volatility_rank

    return df


def _add_standalone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional proxy features that don't require external data."""

    # Momentum momentum (acceleration)
    returns_5 = df['close'].pct_change(5)
    returns_20 = df['close'].pct_change(20)
    df['momentum_acceleration'] = returns_5 - returns_5.shift(5)

    # Price relative to 52-week range (percentile position)
    high_252 = df['high'].rolling(252).max()
    low_252 = df['low'].rolling(252).min()
    df['price_52w_percentile'] = (df['close'] - low_252) / (high_252 - low_252 + 1e-10)

    # Volume relative to 52-week average
    vol_avg_252 = df['volume'].rolling(252).mean()
    df['volume_52w_ratio'] = df['volume'] / (vol_avg_252 + 1)

    # New high/low indicators
    df['at_52w_high'] = (df['close'] >= df['close'].rolling(252).max()).astype(int)
    df['at_52w_low'] = (df['close'] <= df['close'].rolling(252).min()).astype(int)

    # Distance from 52-week high/low
    df['dist_52w_high'] = df['close'] / df['high'].rolling(252).max() - 1
    df['dist_52w_low'] = df['close'] / df['low'].rolling(252).min() - 1

    # Momentum score (composite)
    mom_1m = df['close'].pct_change(21)
    mom_3m = df['close'].pct_change(63)
    mom_6m = df['close'].pct_change(126)
    mom_12m = df['close'].pct_change(252)

    # Exclude most recent month from 12m momentum (momentum strategy)
    df['momentum_12_1'] = mom_12m - mom_1m

    # Weighted momentum score
    df['momentum_score'] = 0.4 * mom_3m + 0.3 * mom_6m + 0.3 * mom_12m

    return df


def get_feature_names() -> list:
    """Get list of standalone cross-sectional feature names."""
    dummy_data = pd.DataFrame({
        'open': np.random.randn(300) + 100,
        'high': np.random.randn(300) + 101,
        'low': np.random.randn(300) + 99,
        'close': np.random.randn(300) + 100,
        'volume': np.random.randint(1000000, 10000000, 300)
    })

    features_df = compute_cross_sectional_features(dummy_data)

    original_cols = ['open', 'high', 'low', 'close', 'volume', 'return_1d']
    feature_cols = [c for c in features_df.columns if c not in original_cols]

    return feature_cols


if __name__ == '__main__':
    print("Cross-sectional feature names (standalone):")
    features = get_feature_names()
    print(f"Total features: {len(features)}")
    for f in features:
        print(f"  - {f}")
