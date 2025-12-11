"""
VectorBT vs Backtrader Result Comparator

This module compares results from VectorBT screening with Backtrader validation
to ensure parity and identify strategy candidates that may be sensitive to
execution timing differences.

Key Metrics Compared:
- Net Profit / Return %
- Win Rate
- Total Trades
- Max Drawdown
- Sharpe Ratio
"""

from typing import List, Dict
import numpy as np


# Discrepancy threshold: results differing by more than this are flagged
DISCREPANCY_THRESHOLD_PERCENT = 1.0  # 1%


def compare_single_result(vbt_stats: dict, bt_stats: dict) -> dict:
    """
    Compare VectorBT and Backtrader statistics for a single result.

    Args:
        vbt_stats: Statistics from VectorBT screening
        bt_stats: Statistics from Backtrader validation

    Returns:
        Comparison result with discrepancy metrics
    """
    comparisons = {}

    # Compare net profit percentage (primary metric)
    vbt_return = vbt_stats.get('net_profit_percent', 0)
    bt_return = bt_stats.get('net_profit_percent', 0)

    if bt_return != 0:
        return_discrepancy = abs(vbt_return - bt_return) / abs(bt_return) * 100
    else:
        return_discrepancy = abs(vbt_return - bt_return) * 100 if vbt_return != 0 else 0

    comparisons['return_discrepancy_percent'] = round(return_discrepancy, 2)

    # Compare total trades
    vbt_trades = vbt_stats.get('total_trades', 0)
    bt_trades = bt_stats.get('total_trades', 0)

    if bt_trades != 0:
        trades_discrepancy = abs(vbt_trades - bt_trades) / bt_trades * 100
    else:
        trades_discrepancy = 100 if vbt_trades != 0 else 0

    comparisons['trades_discrepancy_percent'] = round(trades_discrepancy, 2)

    # Compare win rate
    vbt_win_rate = vbt_stats.get('win_rate', 0)
    bt_win_rate = bt_stats.get('win_rate', 0)
    comparisons['win_rate_diff'] = round(abs(vbt_win_rate - bt_win_rate), 2)

    # Compare max drawdown
    vbt_dd = vbt_stats.get('max_drawdown_percent', 0)
    bt_dd = bt_stats.get('max_drawdown_percent', 0)
    comparisons['drawdown_diff'] = round(abs(vbt_dd - bt_dd), 2)

    # Compare Sharpe ratio
    vbt_sharpe = vbt_stats.get('sharpe_ratio', 0) or 0
    bt_sharpe = bt_stats.get('sharpe_ratio', 0) or 0
    comparisons['sharpe_diff'] = round(abs(vbt_sharpe - bt_sharpe), 2)

    # Overall status
    if return_discrepancy > DISCREPANCY_THRESHOLD_PERCENT:
        status = 'MISMATCH'
        warning = f'Return discrepancy {return_discrepancy:.2f}% exceeds {DISCREPANCY_THRESHOLD_PERCENT}% threshold'
    elif trades_discrepancy > 10:  # More than 10% difference in trade count
        status = 'WARNING'
        warning = f'Trade count differs by {trades_discrepancy:.1f}%'
    else:
        status = 'VALID'
        warning = None

    return {
        'status': status,
        'warning': warning,
        'metrics': comparisons,
        'vbt_return': vbt_return,
        'bt_return': bt_return,
        'vbt_trades': vbt_trades,
        'bt_trades': bt_trades
    }


def compare_all_results(validated_results: List[dict]) -> dict:
    """
    Compare all validated results and generate summary statistics.

    Args:
        validated_results: List of {'params': {...}, 'vbt_statistics': {...}, 'bt_statistics': {...}}

    Returns:
        Comparison summary with statistics
    """
    comparisons = []
    valid_count = 0
    mismatch_count = 0
    warning_count = 0

    return_discrepancies = []
    trades_discrepancies = []

    for result in validated_results:
        vbt_stats = result.get('vbt_statistics', {})
        bt_stats = result.get('bt_statistics', {})

        comparison = compare_single_result(vbt_stats, bt_stats)
        comparison['params'] = result.get('params', {})

        comparisons.append(comparison)

        # Count by status
        if comparison['status'] == 'VALID':
            valid_count += 1
        elif comparison['status'] == 'MISMATCH':
            mismatch_count += 1
        else:
            warning_count += 1

        # Collect discrepancies for stats
        return_discrepancies.append(comparison['metrics']['return_discrepancy_percent'])
        trades_discrepancies.append(comparison['metrics']['trades_discrepancy_percent'])

    # Calculate summary statistics
    summary = {
        'total_compared': len(validated_results),
        'valid': valid_count,
        'mismatches': mismatch_count,
        'warnings': warning_count,
        'match_rate': round((valid_count / len(validated_results)) * 100, 2) if validated_results else 0,
        'avg_return_discrepancy': round(np.mean(return_discrepancies), 2) if return_discrepancies else 0,
        'max_return_discrepancy': round(max(return_discrepancies), 2) if return_discrepancies else 0,
        'avg_trades_discrepancy': round(np.mean(trades_discrepancies), 2) if trades_discrepancies else 0,
        'threshold_used': DISCREPANCY_THRESHOLD_PERCENT
    }

    return {
        'summary': summary,
        'comparisons': comparisons
    }


def filter_valid_results(validated_results: List[dict]) -> List[dict]:
    """
    Filter results to only include those that pass parity check.

    Args:
        validated_results: List of validated results

    Returns:
        List of results that pass the < 1% discrepancy threshold
    """
    valid = []

    for result in validated_results:
        vbt_stats = result.get('vbt_statistics', {})
        bt_stats = result.get('bt_statistics', {})

        comparison = compare_single_result(vbt_stats, bt_stats)

        if comparison['status'] == 'VALID':
            result['comparison'] = comparison
            valid.append(result)

    return valid


def generate_parity_report(comparison_results: dict) -> str:
    """
    Generate a human-readable parity report.

    Args:
        comparison_results: Output from compare_all_results()

    Returns:
        Formatted report string
    """
    summary = comparison_results['summary']
    comparisons = comparison_results['comparisons']

    lines = [
        "=" * 60,
        "VectorBT vs Backtrader Parity Report",
        "=" * 60,
        "",
        "Summary:",
        f"  Total Compared: {summary['total_compared']}",
        f"  Valid (< {summary['threshold_used']}% discrepancy): {summary['valid']}",
        f"  Mismatches: {summary['mismatches']}",
        f"  Warnings: {summary['warnings']}",
        f"  Match Rate: {summary['match_rate']}%",
        "",
        f"  Average Return Discrepancy: {summary['avg_return_discrepancy']}%",
        f"  Max Return Discrepancy: {summary['max_return_discrepancy']}%",
        f"  Average Trades Discrepancy: {summary['avg_trades_discrepancy']}%",
        "",
    ]

    # List mismatches
    mismatches = [c for c in comparisons if c['status'] == 'MISMATCH']
    if mismatches:
        lines.append("Mismatched Results:")
        lines.append("-" * 40)
        for m in mismatches[:10]:  # Show first 10
            params_str = ', '.join(f"{k}={v}" for k, v in m['params'].items())
            lines.append(f"  Params: {params_str}")
            lines.append(f"    VBT Return: {m['vbt_return']:.2f}%")
            lines.append(f"    BT Return: {m['bt_return']:.2f}%")
            lines.append(f"    Discrepancy: {m['metrics']['return_discrepancy_percent']:.2f}%")
            lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


class ParityChecker:
    """
    Utility class for running parity checks on validated results.
    """

    def __init__(self, threshold: float = 1.0):
        """
        Initialize with custom discrepancy threshold.

        Args:
            threshold: Maximum allowed discrepancy percentage
        """
        global DISCREPANCY_THRESHOLD_PERCENT
        DISCREPANCY_THRESHOLD_PERCENT = threshold
        self.threshold = threshold

    def check(self, validated_results: List[dict]) -> dict:
        """
        Run parity check on validated results.

        Returns comparison summary and filtered valid results.
        """
        comparison = compare_all_results(validated_results)
        valid_results = filter_valid_results(validated_results)

        return {
            'comparison': comparison,
            'valid_results': valid_results,
            'report': generate_parity_report(comparison)
        }


if __name__ == "__main__":
    print("Testing Comparator...")

    # Example comparison
    vbt_stats = {
        'net_profit_percent': 15.5,
        'total_trades': 25,
        'win_rate': 60,
        'max_drawdown_percent': 8.5,
        'sharpe_ratio': 1.2
    }

    bt_stats = {
        'net_profit_percent': 15.3,
        'total_trades': 24,
        'win_rate': 58,
        'max_drawdown_percent': 8.8,
        'sharpe_ratio': 1.15
    }

    result = compare_single_result(vbt_stats, bt_stats)
    print(f"Comparison result: {result}")
