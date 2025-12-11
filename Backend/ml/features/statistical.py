"""
Statistical Feature Engineering

Computes ~50 statistical features:
- Rolling statistics: mean, std, skewness, kurtosis
- Z-scores: price, volume, returns
- Hurst exponent (trend persistence)
- Autocorrelation (lag 1-5)
- Drawdown metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all statistical features for a stock DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume (index=date)

    Returns:
        DataFrame with all statistical features added
    """
    df = df.copy()

    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]

    # Compute returns if not present
    if 'return_1d' not in df.columns:
        df['return_1d'] = df['close'].pct_change()

    # Add all feature categories
    df = _add_rolling_statistics(df)
    df = _add_zscore_features(df)
    df = _add_autocorrelation_features(df)
    df = _add_drawdown_features(df)
    df = _add_distribution_features(df)
    df = _add_trend_persistence_features(df)

    return df


def _add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistical measures."""

    windows = [5, 10, 20, 50]

    for window in windows:
        # Rolling mean of returns
        df[f'return_mean_{window}'] = df['return_1d'].rolling(window).mean()

        # Rolling standard deviation of returns
        df[f'return_std_{window}'] = df['return_1d'].rolling(window).std()

        # Rolling skewness
        df[f'return_skew_{window}'] = df['return_1d'].rolling(window).skew()

        # Rolling kurtosis
        df[f'return_kurt_{window}'] = df['return_1d'].rolling(window).kurt()

        # Rolling min/max
        df[f'return_min_{window}'] = df['return_1d'].rolling(window).min()
        df[f'return_max_{window}'] = df['return_1d'].rolling(window).max()

        # Rolling range (max - min)
        df[f'return_range_{window}'] = df[f'return_max_{window}'] - df[f'return_min_{window}']

        # Price volatility (using log returns)
        log_returns = np.log(df['close'] / df['close'].shift(1))
        df[f'volatility_{window}'] = log_returns.rolling(window).std() * np.sqrt(252)  # Annualized

        # Volume statistics
        df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['volume'].rolling(window).std()

    return df


def _add_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Z-score normalized features."""

    windows = [20, 50]

    for window in windows:
        # Price Z-score
        rolling_mean = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        df[f'price_zscore_{window}'] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)

        # Volume Z-score
        vol_mean = df['volume'].rolling(window).mean()
        vol_std = df['volume'].rolling(window).std()
        df[f'volume_zscore_{window}'] = (df['volume'] - vol_mean) / (vol_std + 1e-10)

        # Return Z-score
        ret_mean = df['return_1d'].rolling(window).mean()
        ret_std = df['return_1d'].rolling(window).std()
        df[f'return_zscore_{window}'] = (df['return_1d'] - ret_mean) / (ret_std + 1e-10)

        # High-Low range Z-score
        hl_range = df['high'] - df['low']
        range_mean = hl_range.rolling(window).mean()
        range_std = hl_range.rolling(window).std()
        df[f'range_zscore_{window}'] = (hl_range - range_mean) / (range_std + 1e-10)

    return df


def _add_autocorrelation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add autocorrelation features for different lags."""

    window = 20

    for lag in range(1, 6):
        # Return autocorrelation
        df[f'autocorr_ret_lag{lag}'] = df['return_1d'].rolling(window).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
            raw=False
        )

        # Volume autocorrelation
        df[f'autocorr_vol_lag{lag}'] = df['volume'].rolling(window).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
            raw=False
        )

    # Sum of absolute autocorrelations (momentum indicator)
    autocorr_cols = [f'autocorr_ret_lag{i}' for i in range(1, 6)]
    df['autocorr_sum'] = df[autocorr_cols].abs().sum(axis=1)

    return df


def _add_drawdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add drawdown-related features."""

    # Running maximum
    df['running_max'] = df['close'].cummax()

    # Current drawdown
    df['drawdown'] = (df['close'] - df['running_max']) / df['running_max']

    # Drawdown duration (days since last peak)
    df['peak_date'] = df['close'].expanding().apply(
        lambda x: len(x) - 1 - np.argmax(x.values[::-1]),
        raw=False
    )
    df['drawdown_duration'] = df.index.to_series().reset_index(drop=True).index - df['peak_date']

    # Rolling max drawdown
    for window in [20, 50, 100]:
        rolling_max = df['close'].rolling(window).max()
        rolling_drawdown = (df['close'] - rolling_max) / rolling_max
        df[f'max_drawdown_{window}'] = rolling_drawdown.rolling(window).min()

    # Recovery rate (rate of recovery from drawdowns)
    df['recovery_rate'] = df['return_1d'].where(df['drawdown'] < 0, 0).rolling(10).mean()

    # Clean up temporary columns
    df = df.drop(columns=['running_max', 'peak_date'], errors='ignore')

    return df


def _add_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features related to return distribution."""

    window = 50

    # Percentile rank of current return
    df['return_percentile'] = df['return_1d'].rolling(window).apply(
        lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100 if len(x) > 1 else 0.5,
        raw=False
    )

    # Tail risk metrics
    df['var_5'] = df['return_1d'].rolling(window).quantile(0.05)  # 5% VaR
    df['var_1'] = df['return_1d'].rolling(window).quantile(0.01)  # 1% VaR

    # Expected shortfall (CVaR)
    def cvar(returns, alpha=0.05):
        if len(returns) < 10:
            return np.nan
        var = np.percentile(returns, alpha * 100)
        return returns[returns <= var].mean()

    df['cvar_5'] = df['return_1d'].rolling(window).apply(lambda x: cvar(x, 0.05), raw=False)

    # Sortino ratio (downside risk adjusted)
    def sortino_ratio(returns):
        if len(returns) < 10:
            return np.nan
        mean_ret = returns.mean()
        downside = returns[returns < 0].std()
        return mean_ret / (downside + 1e-10) if downside > 0 else 0

    df['sortino_20'] = df['return_1d'].rolling(20).apply(sortino_ratio, raw=False)
    df['sortino_50'] = df['return_1d'].rolling(50).apply(sortino_ratio, raw=False)

    # Calmar ratio approximation
    df['calmar_approx'] = df['return_mean_20'] * 252 / (abs(df['max_drawdown_20']) + 1e-10)

    return df


def _add_trend_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features related to trend persistence including Hurst exponent approximation."""

    # Hurst exponent approximation using R/S analysis
    def hurst_rs(series, min_window=10):
        """
        Simplified Hurst exponent using rescaled range.
        H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk
        """
        if len(series) < min_window * 2:
            return np.nan

        series = series.dropna()
        if len(series) < min_window * 2:
            return np.nan

        try:
            n = len(series)
            max_k = min(n // 2, 100)

            rs_list = []
            n_list = []

            for k in range(min_window, max_k):
                subseries = [series.iloc[i:i+k] for i in range(0, n-k+1, k)]

                for sub in subseries:
                    if len(sub) < min_window:
                        continue
                    mean = sub.mean()
                    std = sub.std()
                    if std == 0:
                        continue
                    cumdev = (sub - mean).cumsum()
                    r = cumdev.max() - cumdev.min()
                    rs = r / std
                    rs_list.append(rs)
                    n_list.append(len(sub))

            if len(rs_list) < 3:
                return np.nan

            # Linear regression on log-log plot
            log_n = np.log(n_list)
            log_rs = np.log(rs_list)

            slope, _, _, _, _ = stats.linregress(log_n, log_rs)
            return slope

        except Exception:
            return np.nan

    # Calculate Hurst exponent on rolling windows
    df['hurst_50'] = df['close'].rolling(50).apply(hurst_rs, raw=False)
    df['hurst_100'] = df['close'].rolling(100).apply(hurst_rs, raw=False)

    # Trend consistency (fraction of positive days in window)
    df['trend_consistency_10'] = (df['return_1d'] > 0).rolling(10).mean()
    df['trend_consistency_20'] = (df['return_1d'] > 0).rolling(20).mean()

    # Trend strength (cumulative return / sum of absolute returns)
    def trend_efficiency(returns):
        total_return = returns.sum()
        total_movement = returns.abs().sum()
        return total_return / (total_movement + 1e-10)

    df['trend_efficiency_10'] = df['return_1d'].rolling(10).apply(trend_efficiency, raw=False)
    df['trend_efficiency_20'] = df['return_1d'].rolling(20).apply(trend_efficiency, raw=False)

    # ADX-like trend strength from price only
    def price_trend_strength(prices):
        if len(prices) < 5:
            return np.nan
        # Linear regression slope normalized by price level
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        return r_value ** 2  # R-squared as trend strength

    df['r_squared_10'] = df['close'].rolling(10).apply(price_trend_strength, raw=False)
    df['r_squared_20'] = df['close'].rolling(20).apply(price_trend_strength, raw=False)

    return df


def get_feature_names() -> List[str]:
    """Get list of all statistical feature names."""
    dummy_data = pd.DataFrame({
        'open': np.random.randn(200) + 100,
        'high': np.random.randn(200) + 101,
        'low': np.random.randn(200) + 99,
        'close': np.random.randn(200) + 100,
        'volume': np.random.randint(1000000, 10000000, 200)
    })

    features_df = compute_statistical_features(dummy_data)

    original_cols = ['open', 'high', 'low', 'close', 'volume', 'return_1d']
    feature_cols = [c for c in features_df.columns if c not in original_cols]

    return feature_cols


if __name__ == '__main__':
    print("Statistical feature names:")
    features = get_feature_names()
    print(f"Total features: {len(features)}")
    print(f"Sample features: {features[:20]}")
