"""
ML Pipeline Orchestration

Main pipeline for S&P 500 pattern discovery:
1. Load all S&P 500 stock data
2. Compute features for all stocks
3. Train ensemble model
4. Discover universal patterns
5. Generate reports and predictions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import json
import warnings

from .data.sp500_list import get_sp500_constituents, get_sector_mapping
from .data.bulk_loader import BulkDataLoader, create_progress_printer
from .features.technical import compute_technical_features
from .features.statistical import compute_statistical_features
from .features.cross_sectional import compute_cross_sectional_features
from .features.fundamental import compute_fundamental_features
from .features.targets import compute_targets, split_features_targets
from .models.ensemble import UniversalPatternEnsemble
from .validation.time_series_cv import PurgedWalkForwardCV, evaluate_with_cv
from .discovery.shap_analysis import SHAPAnalyzer, analyze_model_shap
from .discovery.pattern_extraction import UniversalPatternExtractor


# Default paths
MODEL_DIR = Path(__file__).parent.parent / 'data' / 'models'
RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'results'


class MLPipeline:
    """
    End-to-end ML pipeline for S&P 500 pattern discovery.

    Usage:
        pipeline = MLPipeline()
        pipeline.run(progress_callback=my_callback)
        predictions = pipeline.predict('AAPL')
        patterns = pipeline.get_patterns()
    """

    def __init__(
        self,
        years_of_data: int = 5,
        target_horizon: int = 10,
        use_gpu: bool = True,
        model_dir: Path = MODEL_DIR,
        results_dir: Path = RESULTS_DIR
    ):
        """
        Initialize ML pipeline.

        Args:
            years_of_data: Years of historical data to use
            target_horizon: Prediction horizon in days
            use_gpu: Whether to use GPU acceleration
            model_dir: Directory to save models
            results_dir: Directory to save results
        """
        self.years_of_data = years_of_data
        self.target_horizon = target_horizon
        self.use_gpu = use_gpu
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline state
        self.data_loader = None
        self.ensemble = None
        self.shap_analyzer = None
        self.pattern_extractor = None

        self.feature_names: List[str] = []
        self.constituents: pd.DataFrame = None
        self.all_data: Dict[str, pd.DataFrame] = {}
        self.combined_features: pd.DataFrame = None
        self.patterns: List[Dict] = []
        self.feature_importance: pd.DataFrame = None

        # Status tracking
        self.status = {
            'stage': 'initialized',
            'progress': 0,
            'message': 'Pipeline initialized'
        }

    def run(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        max_stocks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the full ML pipeline.

        Args:
            progress_callback: Optional callback(stage, progress_pct, message)
            max_stocks: Limit number of stocks (for testing)

        Returns:
            Dict with results summary
        """
        start_time = datetime.now()

        def update_status(stage: str, progress: int, message: str):
            self.status = {'stage': stage, 'progress': progress, 'message': message}
            if progress_callback:
                progress_callback(stage, progress, message)
            print(f"[{stage}] {progress}% - {message}")

        try:
            # Stage 1: Load S&P 500 constituents
            update_status('loading', 0, 'Fetching S&P 500 constituent list...')
            self.constituents = get_sp500_constituents()
            symbols = self.constituents['symbol'].tolist()

            if max_stocks:
                symbols = symbols[:max_stocks]

            update_status('loading', 10, f'Loading data for {len(symbols)} stocks...')

            # Stage 2: Load all stock data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.years_of_data * 365)).strftime('%Y-%m-%d')

            def data_progress(symbol, completed, total):
                pct = 10 + int(30 * completed / total)
                update_status('loading', pct, f'Loading {symbol} ({completed}/{total})')

            self.data_loader = BulkDataLoader(
                num_workers=10,
                progress_callback=data_progress
            )
            self.all_data = self.data_loader.load_multiple(symbols, start_date, end_date)

            # Load SPY benchmark
            spy_data = self.data_loader.get_spy_benchmark(start_date, end_date)

            update_status('loading', 40, f'Loaded {len(self.all_data)} stocks successfully')

            # Stage 3: Compute features for all stocks
            update_status('features', 40, 'Computing features...')
            all_features = []
            total_stocks = len(self.all_data)

            for i, (symbol, df) in enumerate(self.all_data.items()):
                try:
                    features_df = self._compute_all_features(df, symbol, spy_data)
                    if features_df is not None and len(features_df) > 100:
                        features_df['symbol'] = symbol
                        all_features.append(features_df)
                except Exception as e:
                    warnings.warn(f"Failed to compute features for {symbol}: {e}")

                if (i + 1) % 50 == 0:
                    pct = 40 + int(30 * (i + 1) / total_stocks)
                    update_status('features', pct, f'Computed features for {i + 1}/{total_stocks} stocks')

            # Combine all features
            self.combined_features = pd.concat(all_features, ignore_index=True)
            self.feature_names = [c for c in self.combined_features.columns
                                 if not c.startswith('target_') and c != 'symbol']

            update_status('features', 70, f'Total samples: {len(self.combined_features)}')

            # Stage 4: Prepare training data
            update_status('training', 70, 'Preparing training data...')
            target_col = f'target_return_{self.target_horizon}d'

            # Drop rows with NaN targets
            valid_mask = self.combined_features[target_col].notna()
            valid_data = self.combined_features[valid_mask].copy()

            # Split features and target
            X = valid_data[self.feature_names].values
            y = valid_data[target_col].values

            # Handle NaN/Inf in features
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

            # Train/val split (last 20% for validation)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            update_status('training', 75, f'Training on {len(X_train)} samples...')

            # Stage 5: Train ensemble model
            def model_progress(model_name, pct):
                overall = 75 + int(15 * pct / 100)
                update_status('training', overall, f'Training {model_name}...')

            self.ensemble = UniversalPatternEnsemble(task='regression', use_gpu=self.use_gpu)
            self.ensemble.fit(
                X_train, y_train,
                X_val, y_val,
                feature_names=self.feature_names,
                progress_callback=model_progress
            )

            update_status('training', 90, 'Training complete!')

            # Stage 6: Analyze feature importance
            update_status('discovery', 90, 'Analyzing feature importance...')
            self.feature_importance = self.ensemble.get_feature_importance()

            # SHAP analysis on sample
            sample_size = min(1000, len(X_val))
            sample_idx = np.random.choice(len(X_val), sample_size, replace=False)
            X_sample = X_val[sample_idx]

            # Use XGBoost model for SHAP
            xgb_model = self.ensemble.base_models['xgboost']
            importance_shap, interactions, summary = analyze_model_shap(
                xgb_model, X_sample, self.feature_names
            )

            update_status('discovery', 93, 'Extracting universal patterns...')

            # Stage 7: Extract patterns
            predictions = self.ensemble.predict(X_val)
            self.pattern_extractor = UniversalPatternExtractor(
                self.feature_names,
                n_patterns=10
            )
            self.pattern_extractor.extract_patterns(X_val, y_val, predictions)
            self.patterns = self.pattern_extractor.to_dict()

            update_status('discovery', 96, 'Saving results...')

            # Stage 8: Save results
            self._save_results()

            update_status('complete', 100, 'Pipeline complete!')

            elapsed = (datetime.now() - start_time).total_seconds()

            return {
                'status': 'success',
                'elapsed_seconds': elapsed,
                'n_stocks': len(self.all_data),
                'n_samples': len(self.combined_features),
                'n_features': len(self.feature_names),
                'n_patterns': len(self.patterns),
                'top_features': self.feature_importance['feature'].head(10).tolist()
            }

        except Exception as e:
            update_status('error', 0, str(e))
            raise

    def _compute_all_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """Compute all features for a single stock."""
        try:
            # Technical features
            df = compute_technical_features(df)

            # Statistical features
            df = compute_statistical_features(df)

            # Cross-sectional features
            df = compute_cross_sectional_features(df, benchmark_df)

            # Fundamental features (slow, can be skipped for speed)
            # df = compute_fundamental_features(df, symbol)

            # Target variables
            df = compute_targets(df, horizons=[5, 10, 20])

            # Drop rows with too many NaN values
            df = df.dropna(thresh=len(df.columns) * 0.5)

            return df

        except Exception as e:
            warnings.warn(f"Feature computation failed for {symbol}: {e}")
            return None

    def _save_results(self):
        """Save model and results to disk."""
        # Save ensemble model
        self.ensemble.save(str(self.model_dir / 'ensemble'))

        # Save feature importance
        self.feature_importance.to_csv(self.results_dir / 'feature_importance.csv', index=False)

        # Save patterns
        with open(self.results_dir / 'patterns.json', 'w') as f:
            json.dump(self.patterns, f, indent=2)

        # Save pattern report
        report = self.pattern_extractor.generate_report()
        with open(self.results_dir / 'pattern_report.txt', 'w') as f:
            f.write(report)

        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'years_of_data': self.years_of_data,
            'target_horizon': self.target_horizon,
            'n_stocks': len(self.all_data),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        with open(self.results_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {self.results_dir}")

    def predict(self, symbol: str) -> Dict[str, Any]:
        """
        Make prediction for a single stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with predictions and explanations
        """
        if self.ensemble is None:
            raise ValueError("Model not trained. Run pipeline.run() first.")

        # Load recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        loader = BulkDataLoader()
        data = loader.load_multiple([symbol], start_date, end_date)

        if symbol not in data:
            raise ValueError(f"Failed to load data for {symbol}")

        # Compute features
        df = self._compute_all_features(data[symbol], symbol)
        if df is None or len(df) == 0:
            raise ValueError(f"Failed to compute features for {symbol}")

        # Get latest row
        latest = df.iloc[-1:]
        X = latest[self.feature_names].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Make predictions
        predictions = self.ensemble.predict_with_models(X)

        return {
            'symbol': symbol,
            'date': str(df.index[-1]),
            'predictions': {k: float(v[0]) for k, v in predictions.items()},
            'horizon_days': self.target_horizon
        }

    def get_patterns(self) -> List[Dict]:
        """Get discovered patterns."""
        return self.patterns

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importance."""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Run pipeline.run() first.")
        return self.feature_importance.head(top_n)

    def load_model(self):
        """Load saved model from disk."""
        self.ensemble = UniversalPatternEnsemble(use_gpu=self.use_gpu)
        self.ensemble.load(str(self.model_dir / 'ensemble'))

        # Load metadata
        with open(self.results_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']

        # Load patterns
        with open(self.results_dir / 'patterns.json', 'r') as f:
            self.patterns = json.load(f)

        # Load feature importance
        self.feature_importance = pd.read_csv(self.results_dir / 'feature_importance.csv')

        print("Model loaded successfully")


# Convenience function for Flask API
def run_training_pipeline(
    progress_callback: Optional[Callable] = None,
    max_stocks: Optional[int] = None
) -> Dict:
    """
    Run the ML training pipeline.

    Args:
        progress_callback: Optional progress callback
        max_stocks: Limit stocks for testing

    Returns:
        Results dict
    """
    pipeline = MLPipeline()
    return pipeline.run(progress_callback, max_stocks)


if __name__ == '__main__':
    # Test the pipeline with a small sample
    print("Testing ML Pipeline...")

    pipeline = MLPipeline(years_of_data=2, target_horizon=10)

    # Run with just 10 stocks for testing
    results = pipeline.run(max_stocks=10)

    print("\nResults:")
    print(json.dumps(results, indent=2))

    # Test prediction
    print("\nTesting prediction...")
    prediction = pipeline.predict('AAPL')
    print(json.dumps(prediction, indent=2))
