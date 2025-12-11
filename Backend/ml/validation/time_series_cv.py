"""
Time Series Cross-Validation

Purged walk-forward cross-validation for financial ML:
- Respects temporal ordering (no data leakage)
- Gap between train and test to prevent look-ahead bias
- Expanding or sliding window training
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional, Generator
from dataclasses import dataclass


@dataclass
class CVFold:
    """Container for a single CV fold."""
    fold_num: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation for time series.

    Features:
    - Temporal ordering preserved
    - Purging: Gap between train and test sets
    - Embargo: Additional gap to handle label leakage
    - Expanding or sliding window modes

    Example:
        cv = PurgedWalkForwardCV(n_splits=5, purge_gap=20, embargo_gap=5)
        for train_idx, test_idx in cv.split(X, dates):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 20,
        embargo_gap: int = 5,
        min_train_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        expanding: bool = True
    ):
        """
        Initialize walk-forward CV.

        Args:
            n_splits: Number of folds
            purge_gap: Number of observations to purge between train and test
            embargo_gap: Additional embargo after test set (for label overlap)
            min_train_size: Minimum training set size
            max_train_size: Maximum training size (for sliding window)
            test_size: Fixed test set size (if None, calculated from n_splits)
            expanding: If True, expanding window; if False, sliding window
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.expanding = expanding

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature array
            y: Target array (not used, for sklearn compatibility)
            groups: Group labels (e.g., dates) (not used)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        folds = self._generate_folds(n_samples)

        for fold in folds:
            yield fold.train_indices, fold.test_indices

    def _generate_folds(self, n_samples: int) -> List[CVFold]:
        """Generate all fold definitions."""
        # Calculate test size if not specified
        if self.test_size is None:
            # Reserve space for purge and embargo gaps
            effective_samples = n_samples - self.n_splits * (self.purge_gap + self.embargo_gap)
            test_size = effective_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # Minimum train size
        min_train = self.min_train_size or test_size * 2

        folds = []
        test_end = n_samples

        for fold_num in range(self.n_splits - 1, -1, -1):
            # Test set bounds
            test_start = test_end - test_size
            if test_start < min_train + self.purge_gap:
                break

            # Training set bounds (with purge gap)
            train_end = test_start - self.purge_gap

            if self.expanding:
                train_start = 0
            else:
                # Sliding window
                train_start = max(0, train_end - (self.max_train_size or train_end))

            # Enforce minimum training size
            if train_end - train_start < min_train:
                break

            # Create indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            fold = CVFold(
                fold_num=self.n_splits - 1 - fold_num,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            folds.append(fold)

            # Move test window back (with embargo gap)
            test_end = test_start - self.embargo_gap

        # Reverse to get chronological order
        folds.reverse()

        return folds

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splits."""
        if X is not None:
            return len(self._generate_folds(len(X)))
        return self.n_splits

    def get_fold_info(self, n_samples: int) -> pd.DataFrame:
        """
        Get detailed information about each fold.

        Args:
            n_samples: Total number of samples

        Returns:
            DataFrame with fold information
        """
        folds = self._generate_folds(n_samples)

        info = []
        for fold in folds:
            info.append({
                'fold': fold.fold_num,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'train_size': len(fold.train_indices),
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'test_size': len(fold.test_indices),
                'purge_gap': fold.test_start - fold.train_end
            })

        return pd.DataFrame(info)


class GroupedTimeSeriesCV:
    """
    Time series CV with group-aware purging.

    Use this when data has multiple time series (e.g., multiple stocks)
    and you want to properly handle temporal dependencies within groups.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 20,
        group_col: str = 'symbol'
    ):
        """
        Initialize grouped time series CV.

        Args:
            n_splits: Number of folds
            purge_gap: Number of time periods to purge
            group_col: Column name for group identifier
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.group_col = group_col

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices respecting group structure.

        Args:
            df: DataFrame with date and group columns
            date_col: Name of date column

        Yields:
            Tuple of (train_indices, test_indices)
        """
        # Get unique dates sorted
        dates = df[date_col].unique()
        dates = np.sort(dates)
        n_dates = len(dates)

        # Calculate date splits
        test_size = n_dates // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Define date boundaries
            test_start_idx = (fold + 1) * test_size + fold * self.purge_gap
            test_end_idx = test_start_idx + test_size

            if test_end_idx > n_dates:
                break

            train_end_idx = test_start_idx - self.purge_gap

            # Get dates for train and test
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            # Get indices
            train_mask = df[date_col].isin(train_dates)
            test_mask = df[date_col].isin(test_dates)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices


def evaluate_with_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: PurgedWalkForwardCV,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Evaluate model using walk-forward CV.

    Args:
        model: Model with fit/predict interface
        X: Features
        y: Targets
        cv: Cross-validation splitter
        metrics: List of metric names ('mse', 'mae', 'r2', 'accuracy')

    Returns:
        DataFrame with metrics for each fold
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
    from sklearn.base import clone

    if metrics is None:
        metrics = ['mse', 'r2']

    results = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and fit model
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        # Predict
        y_pred = fold_model.predict(X_test)

        # Calculate metrics
        fold_results = {'fold': fold_num, 'train_size': len(train_idx), 'test_size': len(test_idx)}

        if 'mse' in metrics:
            fold_results['mse'] = mean_squared_error(y_test, y_pred)
        if 'rmse' in metrics:
            fold_results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        if 'mae' in metrics:
            fold_results['mae'] = mean_absolute_error(y_test, y_pred)
        if 'r2' in metrics:
            fold_results['r2'] = r2_score(y_test, y_pred)
        if 'accuracy' in metrics:
            fold_results['accuracy'] = accuracy_score(y_test, (y_pred > 0.5).astype(int))

        results.append(fold_results)

    df = pd.DataFrame(results)

    # Add summary row
    summary = {'fold': 'mean'}
    for col in df.columns:
        if col != 'fold':
            summary[col] = df[col].mean()
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    return df


if __name__ == '__main__':
    # Test the CV splitter
    print("Testing PurgedWalkForwardCV...")

    # Create sample data
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)

    # Create CV splitter
    cv = PurgedWalkForwardCV(n_splits=5, purge_gap=20, embargo_gap=5)

    # Get fold info
    fold_info = cv.get_fold_info(n_samples)
    print("\nFold Information:")
    print(fold_info)

    # Test splitting
    print("\nSplitting data:")
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Fold {i}: Train {len(train_idx)} samples [{train_idx[0]}-{train_idx[-1]}], "
              f"Test {len(test_idx)} samples [{test_idx[0]}-{test_idx[-1]}]")
