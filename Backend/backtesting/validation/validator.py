"""
Backtrader Validator for Hybrid Optimization

Phase 2: Detailed validation of top candidates from VectorBT screening.

This module runs candidates through Backtrader for:
- Event-driven simulation matching real execution
- Detailed trade logging
- Full equity curve tracking
- Comparison with VectorBT results
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Optional


class ValidationStrategy(bt.Strategy):
    """
    Simplified backtest strategy for validation.

    Mirrors the VectorBT screener logic for parity comparison.
    """

    params = (
        ("FIRST_RETRACEMENT", 5),
        ("SECOND_RETRACEMENT", 5),
        ("TOUCH_COUNT", 1),
        ("BREAKOUT_BUFFER", 0),
        ("TAKE_PROFIT", 10),
        ("STOP_LOSS", 5),
        ("RISK_PERCENTAGE", 5),
        ("RES_SUP_RANGE", 0.001),
    )

    def __init__(self):
        # Level tracking
        self.resistance_clusters = {}
        self.broken_resistances = set()

        # Sequential bar tracking for resistance detection
        self.start_high_bar = {'high': 0, 'low': float('inf')}
        self.middle_high_bar = {'high': 0, 'low': 0}

        # Order tracking
        self.order = None
        self.buyprice = None

        # Statistics
        self.trades = []
        self.equity_curve = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0).isoformat()
            if order.isbuy():
                self.buyprice = order.executed.price
                self.trades.append({
                    'type': 'buy',
                    'price': order.executed.price,
                    'date': dt,
                    'size': order.executed.size
                })
            elif order.issell():
                self.trades.append({
                    'type': 'sell',
                    'price': order.executed.price,
                    'date': dt,
                    'size': order.executed.size
                })

        self.order = None

    def add_to_cluster(self, clusters, level, range_pct):
        """Add level to cluster and return cluster key and count."""
        found_cluster = None
        for base in list(clusters.keys()):
            high_range = base * (1 + range_pct)
            low_range = base * (1 - range_pct)
            if low_range <= level <= high_range:
                found_cluster = base
                break

        if found_cluster:
            clusters[found_cluster].append(level)
            return found_cluster, len(clusters[found_cluster])
        else:
            clusters[level] = [level]
            return level, 1

    def detect_resistance(self):
        """Detect swing high resistance using sequential bar tracking."""
        current_high = self.data.high[0]
        current_low = self.data.low[0]

        first_mult = 1 + (self.p.FIRST_RETRACEMENT / 100)
        second_mult = 1 - (self.p.SECOND_RETRACEMENT / 100)

        # Update start bar - track lowest point
        if current_low < self.start_high_bar['low']:
            self.start_high_bar = {'high': current_high, 'low': current_low}
            self.middle_high_bar = {'high': current_high, 'low': current_low}
            return None

        # Check for new potential swing high
        if current_high > self.middle_high_bar['high']:
            if current_high >= self.start_high_bar['low'] * first_mult:
                self.middle_high_bar = {'high': current_high, 'low': current_low}
            return None

        # Check for pullback confirmation
        if current_high <= self.middle_high_bar['high'] * second_mult:
            resistance = self.middle_high_bar['high']
            # Reset for next swing
            self.start_high_bar = {'high': current_high, 'low': current_low}
            self.middle_high_bar = {'high': current_high, 'low': current_low}
            return resistance

        return None

    def next(self):
        # Track equity
        self.equity_curve.append({
            'date': self.datas[0].datetime.date(0).isoformat(),
            'equity': self.broker.getvalue()
        })

        if self.order:
            return

        # Detect new resistance
        new_resistance = self.detect_resistance()
        if new_resistance:
            cluster_key, touch_count = self.add_to_cluster(
                self.resistance_clusters,
                new_resistance,
                self.p.RES_SUP_RANGE
            )

        if not self.position:
            # Check for breakout entry
            for key, levels in list(self.resistance_clusters.items()):
                if key in self.broken_resistances:
                    continue
                if len(levels) < self.p.TOUCH_COUNT:
                    continue

                resistance = max(levels)
                breakout_price = resistance * (1 + self.p.BREAKOUT_BUFFER / 100)

                if self.data.close[0] > breakout_price:
                    # Calculate position size
                    cash = self.broker.getcash()
                    size = int((cash * self.p.RISK_PERCENTAGE / 100) / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
                        self.broken_resistances.add(key)
                    break
        else:
            # Check exit conditions
            pnl_pct = (self.data.close[0] - self.buyprice) / self.buyprice * 100

            if pnl_pct >= self.p.TAKE_PROFIT:
                self.order = self.sell(size=self.position.size)
            elif pnl_pct <= -self.p.STOP_LOSS:
                self.order = self.sell(size=self.position.size)


def run_backtrader_validation(
    df: pd.DataFrame,
    params: dict,
    initial_capital: float = 100000,
    commission: float = 0.001
) -> dict:
    """
    Run a single Backtrader validation backtest.

    Args:
        df: OHLCV DataFrame
        params: Strategy parameters
        initial_capital: Starting capital
        commission: Commission rate

    Returns:
        Dict with backtest statistics and trade details
    """
    # Create Backtrader engine
    cerebro = bt.Cerebro()

    # Add data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)

    # Map params to strategy params
    strategy_params = {
        'FIRST_RETRACEMENT': params.get('firstRetracement', 5),
        'SECOND_RETRACEMENT': params.get('secondRetracement', 5),
        'TOUCH_COUNT': int(params.get('touchCount', 1)),
        'BREAKOUT_BUFFER': params.get('breakoutBuffer', 0),
        'TAKE_PROFIT': params.get('takeProfit', 10),
        'STOP_LOSS': params.get('stopLoss', 5),
    }

    cerebro.addstrategy(ValidationStrategy, **strategy_params)

    # Broker settings
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    try:
        results = cerebro.run()
        strat = results[0]

        # Extract statistics
        final_value = cerebro.broker.getvalue()
        net_profit = final_value - initial_capital
        net_profit_pct = (net_profit / initial_capital) * 100

        # Sharpe ratio
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe = sharpe_analysis.get('sharperatio', 0) or 0

        # Drawdown
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0) or 0

        # Trade analysis
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0) or 0

        if total_trades > 0:
            won = trade_analysis.get('won', {}).get('total', 0) or 0
            win_rate = (won / total_trades) * 100

            gross_profit = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0) or 0
            gross_loss = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0) or 0)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0

        return {
            'net_profit': float(net_profit),
            'net_profit_percent': float(net_profit_pct),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 'Infinity',
            'max_drawdown_percent': float(max_drawdown),
            'sharpe_ratio': float(sharpe),
            'final_value': float(final_value),
            'trades': strat.trades,
            'equity_curve': strat.equity_curve
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


def validate_candidates(
    df: pd.DataFrame,
    candidates: List[dict],
    initial_capital: float = 100000,
    commission: float = 0.001,
    max_workers: int = 4,
    progress_callback: Optional[Callable] = None
) -> List[dict]:
    """
    Validate multiple candidates using Backtrader.

    Args:
        df: OHLCV DataFrame
        candidates: List of {'params': {...}, 'statistics': {...}} from VBT screening
        initial_capital: Starting capital
        commission: Commission rate
        max_workers: Number of parallel workers
        progress_callback: Function(current, total) for progress updates

    Returns:
        List of validated results with BT statistics added
    """
    total = len(candidates)
    validated = []
    completed = 0

    def validate_single(candidate):
        params = candidate['params']
        vbt_stats = candidate.get('statistics', {})

        bt_stats = run_backtrader_validation(
            df=df,
            params=params,
            initial_capital=initial_capital,
            commission=commission
        )

        return {
            'params': params,
            'vbt_statistics': vbt_stats,
            'bt_statistics': bt_stats
        }

    # Use ThreadPoolExecutor for parallel validation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_single, c): c for c in candidates}

        for future in as_completed(futures):
            result = future.result()
            validated.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, total)

    # Sort by BT net profit
    validated.sort(
        key=lambda x: x['bt_statistics'].get('net_profit', 0),
        reverse=True
    )

    return validated


if __name__ == "__main__":
    print("Testing Backtrader Validator...")

    # Example usage
    from ..data.loader import fetch_data

    df = fetch_data("SPY", "2023-01-01", "2024-01-01")

    params = {
        'firstRetracement': 5,
        'secondRetracement': 5,
        'touchCount': 1,
        'breakoutBuffer': 0,
        'takeProfit': 10,
        'stopLoss': 5
    }

    result = run_backtrader_validation(df, params)
    print(f"\nBacktrader result:")
    for k, v in result.items():
        if k not in ['trades', 'equity_curve']:
            print(f"  {k}: {v}")
