"""
Target Variable Engineering

Computes prediction targets for ML models:
- Forward returns: 5-day, 10-day, 20-day
- Classification targets: Up/Down with threshold
- Risk-adjusted returns
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def compute_targets(
    df: pd.DataFrame,
    horizons: List[int] = [5, 10, 20],
    classification_threshold: float = 0.02
) -> pd.DataFrame:
    """
    Compute target variables for ML prediction.

    Args:
        df: DataFrame with price data (must have 'close' column)
        horizons: List of forward-looking periods in days
        classification_threshold: Threshold for up/down classification (default 2%)

    Returns:
        DataFrame with target columns added
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Forward returns for each horizon
    for horizon in horizons:
        # Simple forward return
        df[f'target_return_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1

        # Log return (better for modeling)
        df[f'target_log_return_{horizon}d'] = np.log(
            df['close'].shift(-horizon) / df['close']
        )

        # Classification target (1 = up, 0 = down)
        df[f'target_direction_{horizon}d'] = (
            df[f'target_return_{horizon}d'] > classification_threshold
        ).astype(int)

        # 3-class target (-1 = down, 0 = neutral, 1 = up)
        conditions = [
            df[f'target_return_{horizon}d'] > classification_threshold,
            df[f'target_return_{horizon}d'] < -classification_threshold
        ]
        choices = [1, -1]
        df[f'target_class_{horizon}d'] = np.select(conditions, choices, default=0)

    # Risk-adjusted forward returns
    df = _add_risk_adjusted_targets(df, horizons)

    # Max drawdown during forward period
    df = _add_drawdown_targets(df, horizons)

    # Volatility during forward period
    df = _add_volatility_targets(df, horizons)

    return df


def _add_risk_adjusted_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Add risk-adjusted return targets."""

    for horizon in horizons:
        # Calculate forward volatility
        forward_vol = []

        for i in range(len(df) - horizon):
            period_returns = df['close'].iloc[i:i+horizon].pct_change().dropna()
            if len(period_returns) > 1:
                vol = period_returns.std() * np.sqrt(252)  # Annualized
                forward_vol.append(vol)
            else:
                forward_vol.append(np.nan)

        # Pad with NaN for last horizon days
        forward_vol.extend([np.nan] * horizon)

        df[f'target_vol_{horizon}d'] = forward_vol

        # Sharpe-like ratio (return / volatility)
        # Annualized return / annualized volatility
        ann_return = df[f'target_return_{horizon}d'] * (252 / horizon)
        df[f'target_sharpe_{horizon}d'] = ann_return / (df[f'target_vol_{horizon}d'] + 1e-10)

        # Clip extreme Sharpe values
        df[f'target_sharpe_{horizon}d'] = df[f'target_sharpe_{horizon}d'].clip(-10, 10)

    return df


def _add_drawdown_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Add max drawdown targets for the forward period."""

    for horizon in horizons:
        max_drawdowns = []

        for i in range(len(df) - horizon):
            # Get forward price series
            forward_prices = df['close'].iloc[i:i+horizon+1].values

            # Calculate drawdown series
            running_max = np.maximum.accumulate(forward_prices)
            drawdowns = (forward_prices - running_max) / running_max

            max_dd = drawdowns.min()
            max_drawdowns.append(max_dd)

        # Pad with NaN for last horizon days
        max_drawdowns.extend([np.nan] * horizon)

        df[f'target_max_dd_{horizon}d'] = max_drawdowns

    return df


def _add_volatility_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """Add realized volatility targets."""

    for horizon in horizons:
        realized_vols = []

        for i in range(len(df) - horizon):
            # Get forward returns
            forward_prices = df['close'].iloc[i:i+horizon+1]
            forward_returns = forward_prices.pct_change().dropna()

            if len(forward_returns) > 1:
                # Realized volatility (annualized)
                vol = forward_returns.std() * np.sqrt(252)
                realized_vols.append(vol)
            else:
                realized_vols.append(np.nan)

        # Pad with NaN
        realized_vols.extend([np.nan] * horizon)

        df[f'target_realized_vol_{horizon}d'] = realized_vols

        # Volatility regime (high/low)
        vol_median = df[f'target_realized_vol_{horizon}d'].median()
        df[f'target_high_vol_{horizon}d'] = (
            df[f'target_realized_vol_{horizon}d'] > vol_median
        ).astype(int)

    return df


def get_target_names(horizons: List[int] = [5, 10, 20]) -> List[str]:
    """Get list of all target variable names."""
    targets = []

    for horizon in horizons:
        targets.extend([
            f'target_return_{horizon}d',
            f'target_log_return_{horizon}d',
            f'target_direction_{horizon}d',
            f'target_class_{horizon}d',
            f'target_vol_{horizon}d',
            f'target_sharpe_{horizon}d',
            f'target_max_dd_{horizon}d',
            f'target_realized_vol_{horizon}d',
            f'target_high_vol_{horizon}d'
        ])

    return targets


def get_primary_target(horizon: int = 10) -> str:
    """Get the primary regression target name."""
    return f'target_return_{horizon}d'


def get_classification_target(horizon: int = 10) -> str:
    """Get the primary classification target name."""
    return f'target_direction_{horizon}d'


def split_features_targets(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.

    Args:
        df: DataFrame with features and targets
        target_col: Name of target column
        exclude_cols: Additional columns to exclude from features

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Default columns to exclude
    default_exclude = [
        'symbol', 'open', 'high', 'low', 'close', 'volume',
        'date', 'index'
    ]

    # Get all target columns
    target_cols = get_target_names()

    # Combine exclusion lists
    exclude = set(default_exclude + target_cols)
    if exclude_cols:
        exclude.update(exclude_cols)

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in exclude]

    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Remove rows where target is NaN
    valid_idx = ~y.isna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    return X, y


if __name__ == '__main__':
    # Test the module
    print("Target variable names:")
    targets = get_target_names()
    print(f"Total targets: {len(targets)}")
    for t in targets:
        print(f"  - {t}")

    print(f"\nPrimary regression target: {get_primary_target()}")
    print(f"Primary classification target: {get_classification_target()}")
