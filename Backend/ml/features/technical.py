"""
Technical Feature Engineering

Computes ~100 technical indicators across multiple categories:
- Trend: SMA, EMA, MACD, ADX, Aroon
- Momentum: RSI, Stochastic, CCI, Williams %R, ROC
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Volume: OBV, VWAP, Chaikin Money Flow
- Price patterns: Higher highs/lows, support/resistance proximity
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features for a stock DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume (index=date)

    Returns:
        DataFrame with all technical features added
    """
    df = df.copy()

    # Ensure column names are lowercase
    df.columns = [c.lower() for c in df.columns]

    # Add all feature categories
    df = _add_trend_features(df)
    df = _add_momentum_features(df)
    df = _add_volatility_features(df)
    df = _add_volume_features(df)
    df = _add_price_pattern_features(df)
    df = _add_moving_average_features(df)

    return df


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-following indicators."""

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    df['adx_trend_strength'] = df['adx'] / 100  # Normalize

    # Aroon Indicator
    aroon = ta.trend.AroonIndicator(df['close'])
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()
    df['aroon_indicator'] = aroon.aroon_indicator()

    # Parabolic SAR
    psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
    df['psar'] = psar.psar()
    df['psar_up'] = psar.psar_up()
    df['psar_down'] = psar.psar_down()

    # CCI (Commodity Channel Index)
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

    # DPO (Detrended Price Oscillator)
    df['dpo'] = ta.trend.DPOIndicator(df['close']).dpo()

    # Mass Index
    df['mass_index'] = ta.trend.MassIndex(df['high'], df['low']).mass_index()

    # Trix
    df['trix'] = ta.trend.TRIXIndicator(df['close']).trix()

    # Vortex Indicator
    vortex = ta.trend.VortexIndicator(df['high'], df['low'], df['close'])
    df['vortex_pos'] = vortex.vortex_indicator_pos()
    df['vortex_neg'] = vortex.vortex_indicator_neg()

    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators."""

    # RSI (multiple periods)
    for period in [7, 14, 21]:
        df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)

    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(
        df['high'], df['low'], df['close']
    ).williams_r()

    # ROC (Rate of Change) - multiple periods
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()

    # Awesome Oscillator
    df['awesome_osc'] = ta.momentum.AwesomeOscillatorIndicator(
        df['high'], df['low']
    ).awesome_oscillator()

    # KAMA (Kaufman Adaptive Moving Average)
    df['kama'] = ta.momentum.KAMAIndicator(df['close']).kama()

    # PPO (Percentage Price Oscillator)
    ppo = ta.momentum.PercentagePriceOscillator(df['close'])
    df['ppo'] = ppo.ppo()
    df['ppo_signal'] = ppo.ppo_signal()
    df['ppo_hist'] = ppo.ppo_hist()

    # Ultimate Oscillator
    df['uo'] = ta.momentum.UltimateOscillator(
        df['high'], df['low'], df['close']
    ).ultimate_oscillator()

    # TSI (True Strength Index)
    df['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()

    return df


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators."""

    # Bollinger Bands (multiple periods)
    for period in [10, 20]:
        bb = ta.volatility.BollingerBands(df['close'], window=period)
        df[f'bb_high_{period}'] = bb.bollinger_hband()
        df[f'bb_low_{period}'] = bb.bollinger_lband()
        df[f'bb_mid_{period}'] = bb.bollinger_mavg()
        df[f'bb_width_{period}'] = bb.bollinger_wband()
        df[f'bb_pband_{period}'] = bb.bollinger_pband()  # % position within bands

    # ATR (Average True Range) - multiple periods
    for period in [7, 14, 21]:
        df[f'atr_{period}'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=period
        ).average_true_range()

    # ATR as percentage of price
    df['atr_pct'] = df['atr_14'] / df['close'] * 100

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_low'] = kc.keltner_channel_lband()
    df['kc_mid'] = kc.keltner_channel_mband()
    df['kc_width'] = (df['kc_high'] - df['kc_low']) / df['kc_mid']
    df['kc_pband'] = kc.keltner_channel_pband()

    # Donchian Channel
    dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
    df['dc_high'] = dc.donchian_channel_hband()
    df['dc_low'] = dc.donchian_channel_lband()
    df['dc_mid'] = dc.donchian_channel_mband()
    df['dc_width'] = dc.donchian_channel_wband()
    df['dc_pband'] = dc.donchian_channel_pband()

    # Ulcer Index
    df['ulcer_index'] = ta.volatility.UlcerIndex(df['close']).ulcer_index()

    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators."""

    # OBV (On-Balance Volume)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

    # OBV change (normalized)
    df['obv_change'] = df['obv'].pct_change()

    # Chaikin Money Flow
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
        df['high'], df['low'], df['close'], df['volume']
    ).chaikin_money_flow()

    # Force Index
    df['force_index'] = ta.volume.ForceIndexIndicator(
        df['close'], df['volume']
    ).force_index()

    # MFI (Money Flow Index)
    df['mfi'] = ta.volume.MFIIndicator(
        df['high'], df['low'], df['close'], df['volume']
    ).money_flow_index()

    # Ease of Movement
    eom = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume'])
    df['eom'] = eom.ease_of_movement()
    df['eom_sma'] = eom.sma_ease_of_movement()

    # Volume Price Trend
    df['vpt'] = ta.volume.VolumePriceTrendIndicator(
        df['close'], df['volume']
    ).volume_price_trend()

    # VWAP approximation (intraday indicator adapted for daily)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_approx'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    # Volume ratios
    df['volume_sma_5'] = df['volume'].rolling(5).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Accumulation/Distribution
    df['ad'] = ta.volume.AccDistIndexIndicator(
        df['high'], df['low'], df['close'], df['volume']
    ).acc_dist_index()

    # Negative Volume Index
    df['nvi'] = ta.volume.NegativeVolumeIndexIndicator(
        df['close'], df['volume']
    ).negative_volume_index()

    return df


def _add_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price pattern features."""

    # Daily returns
    df['return_1d'] = df['close'].pct_change()

    # Higher highs / Lower lows (trend structure)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

    # Consecutive higher highs / lower lows
    df['hh_streak'] = df['higher_high'].groupby(
        (df['higher_high'] != df['higher_high'].shift()).cumsum()
    ).cumsum() * df['higher_high']

    df['ll_streak'] = df['lower_low'].groupby(
        (df['lower_low'] != df['lower_low'].shift()).cumsum()
    ).cumsum() * df['lower_low']

    # Distance from recent high/low
    for period in [5, 10, 20, 50]:
        df[f'dist_from_high_{period}'] = (
            df['close'] / df['high'].rolling(period).max() - 1
        )
        df[f'dist_from_low_{period}'] = (
            df['close'] / df['low'].rolling(period).min() - 1
        )

    # Gap analysis
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_up'] = (df['gap'] > 0.01).astype(int)
    df['gap_down'] = (df['gap'] < -0.01).astype(int)

    # Candle body and wicks
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_pct'] = df['body'] / (df['high'] - df['low'] + 1e-10)

    # Inside/Outside bars
    df['inside_bar'] = (
        (df['high'] < df['high'].shift(1)) &
        (df['low'] > df['low'].shift(1))
    ).astype(int)

    df['outside_bar'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['low'] < df['low'].shift(1))
    ).astype(int)

    return df


def _add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving average based features."""

    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'close_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

    # Exponential Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'close_vs_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    # MA crossovers
    df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)  # Golden cross

    df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
    df['ema_20_50_cross'] = (df['ema_20'] > df['ema_50']).astype(int)

    # Price momentum relative to MAs
    df['ma_spread'] = (df['sma_20'] - df['sma_50']) / df['sma_50']

    # All MAs aligned (trend strength)
    df['ma_alignment'] = (
        (df['sma_5'] > df['sma_10']) &
        (df['sma_10'] > df['sma_20']) &
        (df['sma_20'] > df['sma_50'])
    ).astype(int) - (
        (df['sma_5'] < df['sma_10']) &
        (df['sma_10'] < df['sma_20']) &
        (df['sma_20'] < df['sma_50'])
    ).astype(int)

    return df


def get_feature_names() -> list:
    """Get list of all technical feature names."""
    # Create a dummy DataFrame to extract feature names
    dummy_data = pd.DataFrame({
        'open': np.random.randn(300) + 100,
        'high': np.random.randn(300) + 101,
        'low': np.random.randn(300) + 99,
        'close': np.random.randn(300) + 100,
        'volume': np.random.randint(1000000, 10000000, 300)
    })

    # Ensure high > low
    dummy_data['high'] = dummy_data[['open', 'close']].max(axis=1) + abs(np.random.randn(300))
    dummy_data['low'] = dummy_data[['open', 'close']].min(axis=1) - abs(np.random.randn(300))

    features_df = compute_technical_features(dummy_data)

    # Return only the computed features (not original columns)
    original_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in features_df.columns if c not in original_cols]

    return feature_cols


if __name__ == '__main__':
    # Test the module
    print("Technical feature names:")
    features = get_feature_names()
    print(f"Total features: {len(features)}")
    print(f"Sample features: {features[:20]}")
