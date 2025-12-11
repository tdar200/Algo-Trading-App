"""
VectorBT Optimizer - GPU-Accelerated Parameter Screening

This module handles the Phase 1 screening pipeline:
1. Generate all parameter combinations
2. Run VectorBT backtest for each
3. Filter by performance criteria
4. Output top candidates for Backtrader validation
"""

import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .screener import run_vectorbt_backtest, is_gpu_available, VBT_AVAILABLE


def generate_combinations(param_ranges: dict) -> list:
    """
    Generate all parameter combinations from ranges.

    Args:
        param_ranges: Dict of parameter ranges
            {
                'firstRetracement': {'start': 3, 'end': 10, 'step': 1},
                'takeProfit': {'start': 5, 'end': 20, 'step': 5},
            }

    Returns:
        List of parameter dictionaries
    """
    param_names = list(param_ranges.keys())
    param_values = []

    for name in param_names:
        r = param_ranges[name]
        start = r.get('start', 0)
        end = r.get('end', 0)
        step = r.get('step', 1)

        # Generate range values
        values = []
        current = start
        while current <= end + 0.0001:  # Small epsilon for float comparison
            values.append(current)
            current += step

        param_values.append(values)

    # Generate cartesian product
    combinations = []
    for combo in product(*param_values):
        combinations.append(dict(zip(param_names, combo)))

    return combinations


def run_screening(
    df,
    param_ranges: dict,
    initial_capital: float = 100000,
    commission: float = 0.001,
    filters: dict = None,
    top_n: int = 100,
    progress_callback=None,
    use_parallel: bool = True,
    max_workers: int = 4
) -> dict:
    """
    Run Phase 1 VectorBT screening.

    Args:
        df: OHLCV DataFrame
        param_ranges: Parameter ranges to test
        initial_capital: Starting capital
        commission: Commission rate
        filters: Performance filters
        top_n: Maximum candidates to return
        progress_callback: Function(current, total, candidates_found) for progress
        use_parallel: Use parallel processing (slower on GPU due to GIL)
        max_workers: Number of parallel workers

    Returns:
        Dict with screening results and metadata
    """
    if not VBT_AVAILABLE:
        return {
            'error': 'vectorbt not installed',
            'candidates': [],
            'stats': {}
        }

    # Default filters
    if filters is None:
        filters = {
            'min_sharpe': 1.0,
            'max_drawdown': 20,
            'min_win_rate': 45,
            'min_trades': 50
        }

    # Generate combinations
    combinations = generate_combinations(param_ranges)
    total = len(combinations)

    if total == 0:
        return {
            'error': 'No parameter combinations generated',
            'candidates': [],
            'stats': {}
        }

    start_time = time.time()
    results = []
    candidates_found = 0
    completed = 0

    def process_single(params):
        """Process a single parameter combination."""
        # Map param names to function arguments
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

        return params, stats

    def passes_filter(stats):
        """Check if result passes all filters."""
        if stats.get('error'):
            return False
        if stats.get('sharpe_ratio', 0) < filters.get('min_sharpe', 0):
            return False
        if stats.get('max_drawdown_percent', 100) > filters.get('max_drawdown', 100):
            return False
        if stats.get('win_rate', 0) < filters.get('min_win_rate', 0):
            return False
        if stats.get('total_trades', 0) < filters.get('min_trades', 0):
            return False
        return True

    # Sequential processing (recommended for GPU - avoids GIL contention)
    if not use_parallel or is_gpu_available():
        for i, params in enumerate(combinations):
            params, stats = process_single(params)

            # Debug: Log first 3 results to see actual stats
            if i < 3:
                print(f"[VBT DEBUG] Combo {i}: params={params}")
                print(f"[VBT DEBUG] Stats: trades={stats.get('total_trades')}, sharpe={stats.get('sharpe_ratio')}, win_rate={stats.get('win_rate')}, drawdown={stats.get('max_drawdown_percent')}")

            if passes_filter(stats):
                results.append({
                    'params': {k: float(v) for k, v in params.items()},
                    'statistics': stats
                })
                candidates_found += 1

            completed += 1
            if progress_callback:
                progress_callback(completed, total, candidates_found)

    else:
        # Parallel processing (for CPU-only systems)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single, p): p for p in combinations}

            for future in as_completed(futures):
                params, stats = future.result()

                if passes_filter(stats):
                    results.append({
                        'params': {k: float(v) for k, v in params.items()},
                        'statistics': stats
                    })
                    candidates_found += 1

                completed += 1
                if progress_callback:
                    progress_callback(completed, total, candidates_found)

    # Sort by net profit descending
    results.sort(key=lambda x: x['statistics'].get('net_profit', 0), reverse=True)

    # Keep only top N
    results = results[:top_n]

    elapsed = time.time() - start_time

    return {
        'candidates': results,
        'stats': {
            'total_combinations': total,
            'candidates_found': len(results),
            'passed_filter': candidates_found,
            'elapsed_seconds': round(elapsed, 2),
            'rate_per_second': round(total / elapsed, 2) if elapsed > 0 else 0,
            'gpu_used': is_gpu_available(),
            'filters_applied': filters
        }
    }


def save_candidates(results: dict, output_path: str = None) -> str:
    """
    Save screening candidates to JSON file.

    Args:
        results: Output from run_screening()
        output_path: Optional path for output file

    Returns:
        Path to saved file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "data" / "cache"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"candidates_{timestamp}.json"

    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[Optimizer] Saved {len(results.get('candidates', []))} candidates to {output_path}")

    return str(output_path)


def load_candidates(input_path: str) -> dict:
    """
    Load screening candidates from JSON file.

    Args:
        input_path: Path to candidates JSON file

    Returns:
        Results dict with candidates
    """
    with open(input_path, 'r') as f:
        return json.load(f)


class HybridOptimizer:
    """
    Manages the full hybrid optimization pipeline.

    Phase 1: VectorBT screening
    Phase 2: Backtrader validation (delegated to validator.py)
    """

    def __init__(
        self,
        df,
        param_ranges: dict,
        initial_capital: float = 100000,
        commission: float = 0.001
    ):
        self.df = df
        self.param_ranges = param_ranges
        self.initial_capital = initial_capital
        self.commission = commission

        self.screening_results = None
        self.validation_results = None

    def run_phase1_screening(
        self,
        filters: dict = None,
        top_n: int = 100,
        progress_callback=None
    ) -> dict:
        """
        Run Phase 1: VectorBT screening.

        Returns screening results with top candidates.
        """
        self.screening_results = run_screening(
            df=self.df,
            param_ranges=self.param_ranges,
            initial_capital=self.initial_capital,
            commission=self.commission,
            filters=filters,
            top_n=top_n,
            progress_callback=progress_callback
        )

        return self.screening_results

    def get_candidates_for_validation(self) -> list:
        """
        Get candidate parameters for Phase 2 validation.

        Returns list of parameter dicts.
        """
        if not self.screening_results:
            return []

        return [c['params'] for c in self.screening_results.get('candidates', [])]

    def get_screening_stats(self) -> dict:
        """Get Phase 1 screening statistics."""
        if not self.screening_results:
            return {}
        return self.screening_results.get('stats', {})


if __name__ == "__main__":
    print("Testing VectorBT Optimizer...")
    print(f"GPU available: {is_gpu_available()}")

    # Example usage
    param_ranges = {
        'firstRetracement': {'start': 3, 'end': 10, 'step': 2},
        'secondRetracement': {'start': 3, 'end': 10, 'step': 2},
        'takeProfit': {'start': 5, 'end': 20, 'step': 5},
        'stopLoss': {'start': 3, 'end': 10, 'step': 2},
    }

    combinations = generate_combinations(param_ranges)
    print(f"Total combinations: {len(combinations)}")
    print(f"First 3: {combinations[:3]}")
