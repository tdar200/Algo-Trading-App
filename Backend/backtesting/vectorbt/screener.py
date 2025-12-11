"""
VectorBT Screener for Support/Resistance Breakout Strategy

This module implements a vectorized version of the SupResStrategy
for fast parameter screening.

Key Design: Matches Backtrader's logic:
1. Detect swing highs/lows using percentage retracements
2. Track resistance levels dynamically
3. Entry on price crossover above resistance
4. Exit on take profit, stop loss, or support breakdown
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("[Screener] vectorbt not installed")

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def is_gpu_available() -> bool:
    """Check if CUDA GPU is available for VectorBT acceleration."""
    if not GPU_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def detect_levels_sequential(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    first_ret: float,
    second_ret: float
) -> tuple:
    """
    Detect resistance and support levels using sequential bar tracking.
    Mimics Backtrader's SupResStrategy logic.

    Returns:
        Tuple of (resistance_levels, support_levels) as lists of (bar_index, price)
    """
    n = len(high)
    resistance_levels = []
    support_levels = []

    # Convert percentages to multipliers
    first_up = 1 + (first_ret / 100)
    first_down = 1 - (first_ret / 100)
    second_up = 1 + (second_ret / 100)
    second_down = 1 - (second_ret / 100)

    # State tracking for resistance detection
    start_high_bar = {'idx': 0, 'high': high[0], 'low': low[0]}
    middle_high_bar = {'idx': 0, 'high': high[0], 'low': low[0]}

    # State tracking for support detection
    start_low_bar = {'idx': 0, 'high': high[0], 'low': low[0]}
    middle_low_bar = {'idx': 0, 'high': high[0], 'low': low[0]}

    for i in range(1, n):
        current_high = high[i]
        current_low = low[i]

        # === RESISTANCE DETECTION ===
        # First retracement: price rises from start_high_bar low
        if current_high >= start_high_bar['low'] * first_up:
            if current_high > middle_high_bar['high']:
                middle_high_bar = {'idx': i, 'high': current_high, 'low': current_low}

        # Second retracement: price drops from middle_high_bar high
        if current_low <= middle_high_bar['high'] * second_down:
            # Resistance confirmed at middle_high_bar
            resistance_levels.append((middle_high_bar['idx'], middle_high_bar['high']))
            # Reset tracking
            start_high_bar = {'idx': i, 'high': current_high, 'low': current_low}
            middle_high_bar = {'idx': i, 'high': current_high, 'low': current_low}
        elif current_low < start_high_bar['low']:
            # New low, reset start
            start_high_bar = {'idx': i, 'high': current_high, 'low': current_low}
            middle_high_bar = {'idx': i, 'high': current_high, 'low': current_low}

        # === SUPPORT DETECTION ===
        # First retracement: price drops from start_low_bar high
        if current_low <= start_low_bar['high'] * first_down:
            if current_low < middle_low_bar['low']:
                middle_low_bar = {'idx': i, 'high': current_high, 'low': current_low}

        # Second retracement: price rises from middle_low_bar low
        if current_high >= middle_low_bar['low'] * second_up:
            # Support confirmed at middle_low_bar
            support_levels.append((middle_low_bar['idx'], middle_low_bar['low']))
            # Reset tracking
            start_low_bar = {'idx': i, 'high': current_high, 'low': current_low}
            middle_low_bar = {'idx': i, 'high': current_high, 'low': current_low}
        elif current_high > start_low_bar['high']:
            # New high, reset start
            start_low_bar = {'idx': i, 'high': current_high, 'low': current_low}
            middle_low_bar = {'idx': i, 'high': current_high, 'low': current_low}

    return resistance_levels, support_levels


def generate_signals(
    df: pd.DataFrame,
    first_retracement: float = 5.0,
    second_retracement: float = 5.0,
    touch_count: int = 1,
    breakout_buffer: float = 0.0,
    take_profit: float = 10.0,
    stop_loss: float = 5.0
) -> tuple:
    """
    Generate entry/exit signals matching Backtrader's logic.

    Returns:
        Tuple of (entries, exits) boolean arrays
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    open_prices = df['Open'].values
    n = len(df)

    # Detect levels
    resistance_levels, support_levels = detect_levels_sequential(
        high, low, close, first_retracement, second_retracement
    )

    # Debug logging
    print(f"[VBT] Detected {len(resistance_levels)} resistance levels, {len(support_levels)} support levels", flush=True)

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    # Track active levels and broken resistances
    active_resistances = []  # List of resistance prices
    broken_resistances = set()
    active_supports = []  # List of support prices

    in_position = False
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        current_close = close[i]
        prev_close = close[i-1]

        # Add new resistance levels as they're detected
        for idx, price in resistance_levels:
            if idx == i:
                active_resistances.append(price)

        # Add new support levels
        for idx, price in support_levels:
            if idx == i:
                active_supports.append(price)

        if not in_position:
            # Check for breakout entry above any resistance
            for resistance in sorted(active_resistances):
                if resistance in broken_resistances:
                    continue

                # Entry trigger with buffer
                entry_trigger = resistance * (1 + breakout_buffer / 100)

                # Crossover detection: prev_close <= trigger < current_close
                if prev_close <= entry_trigger < current_close:
                    entries[i] = True
                    in_position = True
                    entry_price = open_prices[min(i + 1, n - 1)]  # Execute at next open
                    entry_bar = i
                    broken_resistances.add(resistance)
                    break
        else:
            # Check exit conditions
            pnl_pct = (current_close - entry_price) / entry_price * 100 if entry_price > 0 else 0

            # Take profit
            if pnl_pct >= take_profit:
                exits[i] = True
                in_position = False
                entry_price = 0.0
                continue

            # Stop loss
            if pnl_pct <= -stop_loss:
                exits[i] = True
                in_position = False
                entry_price = 0.0
                continue

            # Check breakdown below support
            for support in active_supports:
                if current_close < support:
                    exits[i] = True
                    in_position = False
                    entry_price = 0.0
                    break

    # Shift signals by 1 for "execute on next bar" behavior
    entries = np.roll(entries, 1)
    entries[0] = False
    exits = np.roll(exits, 1)
    exits[0] = False

    entry_count = np.sum(entries)
    exit_count = np.sum(exits)
    print(f"[VBT] Signals generated: entries={entry_count}, exits={exit_count}", flush=True)

    return entries, exits


def run_vectorbt_backtest(
    df: pd.DataFrame,
    first_retracement: float = 5.0,
    second_retracement: float = 5.0,
    touch_count: int = 1,
    breakout_buffer: float = 0.0,
    take_profit: float = 10.0,
    stop_loss: float = 5.0,
    initial_capital: float = 100000,
    commission: float = 0.001
) -> dict:
    """
    Run a single backtest with VectorBT and return statistics.
    """
    if not VBT_AVAILABLE:
        raise RuntimeError("vectorbt is not installed")

    # Generate signals
    entries, exits = generate_signals(
        df,
        first_retracement=first_retracement,
        second_retracement=second_retracement,
        touch_count=touch_count,
        breakout_buffer=breakout_buffer,
        take_profit=take_profit,
        stop_loss=stop_loss
    )

    close = df['Close'].values

    try:
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=commission,
            freq='1D'
        )

        # Extract statistics
        trades = portfolio.trades.records_readable
        total_trades = len(trades) if len(trades) > 0 else 0

        if total_trades > 0:
            winning_trades = len(trades[trades['PnL'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            gross_profit = trades[trades['PnL'] > 0]['PnL'].sum() if winning_trades > 0 else 0
            gross_loss = abs(trades[trades['PnL'] < 0]['PnL'].sum()) if (total_trades - winning_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0

        final_value = portfolio.final_value()
        net_profit = final_value - initial_capital
        net_profit_pct = (net_profit / initial_capital) * 100
        max_drawdown = portfolio.max_drawdown() * 100

        sharpe = portfolio.sharpe_ratio() if hasattr(portfolio, 'sharpe_ratio') else 0

        return {
            'net_profit': float(net_profit),
            'net_profit_percent': float(net_profit_pct),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 'Infinity',
            'max_drawdown_percent': float(max_drawdown),
            'sharpe_ratio': float(sharpe) if not np.isnan(sharpe) else 0,
            'final_value': float(final_value)
        }

    except Exception as e:
        return {
            'error': str(e),
            'net_profit': 0,
            'net_profit_percent': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown_percent': 0,
            'sharpe_ratio': 0,
            'final_value': initial_capital
        }


def screen_parameters(
    df: pd.DataFrame,
    param_ranges: dict,
    initial_capital: float = 100000,
    commission: float = 0.001,
    filters: dict = None,
    progress_callback=None
) -> list:
    """
    Screen multiple parameter combinations and return filtered results.
    """
    from itertools import product

    if filters is None:
        filters = {
            'min_sharpe': 1.0,
            'max_drawdown': 20,
            'min_win_rate': 45,
            'min_trades': 50
        }

    param_names = list(param_ranges.keys())
    param_values = []

    for name in param_names:
        r = param_ranges[name]
        values = list(np.arange(r['start'], r['end'] + r['step'], r['step']))
        param_values.append(values)

    combinations = list(product(*param_values))
    total = len(combinations)

    results = []
    candidates_found = 0

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        bt_params = {
            'first_retracement': params.get('firstRetracement', 5),
            'second_retracement': params.get('secondRetracement', 5),
            'touch_count': int(params.get('touchCount', 1)),
            'breakout_buffer': params.get('breakoutBuffer', 0),
            'take_profit': params.get('takeProfit', 10),
            'stop_loss': params.get('stopLoss', 5),
        }

        stats = run_vectorbt_backtest(
            df,
            initial_capital=initial_capital,
            commission=commission,
            **bt_params
        )

        passes_filter = True

        if stats.get('sharpe_ratio', 0) < filters.get('min_sharpe', 0):
            passes_filter = False
        if stats.get('max_drawdown_percent', 100) > filters.get('max_drawdown', 100):
            passes_filter = False
        if stats.get('win_rate', 0) < filters.get('min_win_rate', 0):
            passes_filter = False
        if stats.get('total_trades', 0) < filters.get('min_trades', 0):
            passes_filter = False

        if passes_filter:
            results.append({
                'params': {k: float(v) for k, v in params.items()},
                'statistics': stats
            })
            candidates_found += 1

        if progress_callback:
            progress_callback(i + 1, total, candidates_found)

    results.sort(key=lambda x: x['statistics'].get('net_profit', 0), reverse=True)

    return results


if __name__ == "__main__":
    print("Testing VectorBT Screener...")
    print(f"VectorBT available: {VBT_AVAILABLE}")
    print(f"GPU available: {is_gpu_available()}")

    if VBT_AVAILABLE:
        from ..data.loader import fetch_data
        df = fetch_data("SPY", "2023-01-01", "2024-01-01")

        result = run_vectorbt_backtest(df)
        print(f"\nSingle backtest result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
